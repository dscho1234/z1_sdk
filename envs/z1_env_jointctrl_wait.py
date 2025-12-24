import sys
import os
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional, Callable
import gym
from gym import spaces
import functools
from scipy.optimize import minimize, least_squares

# Add the lib directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import PchipInterpolator, CubicSpline

from urdf_parser_py.urdf import URDF
import xml.etree.ElementTree as ET
import open3d as o3d
# ========= Verbose logging control =========
VERBOSE = False

def vprint(*args, **kwargs) -> None:
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


def generate_waypoints_and_timepoints(current_positions, desired_positions, dt, buffer_time=0.01, next_desired_positions=None):
    """
    Generate waypoints and timepoints for smooth interpolation.
    
    If next_desired_positions is provided, creates waypoints that end with a velocity
    pointing towards the next segment, ensuring smooth transitions between segments.
    """
    if next_desired_positions is not None:
        # Calculate direction to next position
        direction_to_next = next_desired_positions - desired_positions
        
        # Create a small offset in the direction of next position at the end
        # This ensures the velocity at the end of current segment points towards next segment
        # The offset should be small enough that we're still "at" desired_positions
        # but large enough to influence the velocity direction
        velocity_hint_scale = buffer_time * 0.5  # Small offset to hint velocity direction
        end_position = desired_positions + direction_to_next * velocity_hint_scale
        
        waypoints = np.array([
            current_positions,
            desired_positions,  # Reach desired position
            end_position  # End with slight offset towards next (for velocity continuity)
        ])
        # Timepoints: reach desired position early, then transition towards next
        timepoints = np.array([0, dt - buffer_time, dt])
    else:
        # Original behavior when next position is unknown
        waypoints = np.array([current_positions, desired_positions])
        timepoints = np.array([0, dt])
    
    return waypoints, timepoints


# SO3 constraint null objective function
class SO3Constraint:
    def __init__(self, SO3_des=None):
        if SO3_des is None:
            # Default to identity matrix (no rotation preference)
            self.SO3_des = np.eye(3)
        else:
            self.SO3_des = SO3_des.copy()
    
    def evaluate(self, SO3):
        # SO3 error metric: 0.5 * (3 - trace(R * R_des^T))
        # This measures the deviation from desired rotation
        so3_err = 0.5 * (3 - np.trace(SO3 @ self.SO3_des.T))
        return so3_err


class Z1BaseEnv(gym.Env):
    """
    Base environment for Z1 robot arm following OpenAI Gym interface.
    This class provides basic initialization and common functionality.
    """
    
    def __init__(self, has_gripper: bool = True, control_frequency: float = 500.0, urdf_path = None):
        """
        Initialize the Z1 base environment.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
        """
        super().__init__()
        
        self.has_gripper = has_gripper
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        
        # Initialize the arm interface
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel
        
        # Robot state
        self.current_joint_pos = np.zeros(6)
        self.current_joint_vel = np.zeros(6)
        self.current_gripper_pos = 0.0
        self.current_gripper_vel = 0.0
        
        # Control state
        self.is_initialized = False
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # Joint limits for IK
        # NOTE: default value is different from the robot's capable joint angles, so dscho modified it
        self.joint_limits = [
            (-2.618, 2.618),   # J1: ±150°
            (0, 3.142),        # J2: 0—180°
            (-2.879, 0),       # J3: -165°—0
            (-1.75, 1.65),      # J4: ±80° # (-1.396, 1.396) (default)
            (-1.75, 1.6),      # J5: ±85° # (-1.484, 1.484) (default)
            (-2.793, 2.793)    # J6: ±160°
        ]
        
        # Define action and observation spaces (to be overridden by subclasses)
        self.action_space = None
        self.observation_space = None


        ##########################  for custom forward kinematics
        self.urdf_path = urdf_path
        self.robot = None
        self.joint_angles = np.zeros(6)  # 6개 조인트 각도
        self.link_transforms = {}  # 각 링크의 변환 행렬 저장
        

        # URDF 파싱
        self.parse_urdf()
        
        
    def reset(self, joint_angle: Optional[np.ndarray] = None, ik_type='null_space', option: str = "IK") -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            joint_angle: Optional joint angles to move to. If None, uses default reset behavior.
                        Should be a 6-element array for 6-DOF arm.
        
        Returns:
            Initial observation
        """
        if self.is_initialized:
            if joint_angle is not None:
                # Move to specified joint angles
                print(f"Moving to specified joint angles: {joint_angle}")
                self._move_to_joint_angles(joint_angle, option=option, ik_type=ik_type)
            else:
                self.arm.backToStart()
        else: # if not self.is_initialized:
            print("Initializing arm interface...")
            # Start the control loop first
            self.arm.loopOn()
            time.sleep(0.1)  # Small delay for initialization
            
            # dscho added
            # self.arm.labelRun("forward")
            
            # Skip backToStart for now to avoid hanging
            print("Skipping backToStart() to avoid potential hanging...")
            print("Using current robot position as starting point...")
            
            # If joint_angle is specified during initialization, move to it
            if joint_angle is not None:
                print(f"Moving to specified joint angles during initialization: {joint_angle}")
                self._move_to_joint_angles(joint_angle, option=option, ik_type=ik_type)
            
            self.is_initialized = True
        
        # Get initial state
        self._update_state()
        self.episode_step = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement step method")
    
    def _update_state(self):
        """Update the current robot state from the arm interface."""
        self.current_joint_pos = np.array(self.arm.lowstate.getQ())
        self.current_joint_vel = np.array(self.arm.lowstate.getQd())
        if self.has_gripper:
            self.current_gripper_pos = self.arm.lowstate.getGripperQ()
            self.current_gripper_vel = self.arm.gripperQd
    
    def _move_to_joint_angles(self, target_joint_angles: np.ndarray, option: str = "IK", ik_type='jacobian'):
        """
        Move the robot to specified joint angles using either low-level commands, MoveJ, or IK-based jointCtrlCmd.
        
        Args:
            target_joint_angles: Target joint angles (6-element array)
            option: Movement method - "lowcmd" for low-level commands, "MoveJ" for high-level MoveJ, "IK" for IK-based jointCtrlCmd
        """
        if len(target_joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(target_joint_angles)}")
        
        print(f"Moving to joint angles: {target_joint_angles} using {option} method")
        
        if option == "IK":
            # Use IK-based jointCtrlCmd (similar to _execute_step_logic)
            print("Using IK-based jointCtrlCmd...")
            
            # Ensure we're in JOINTCTRL state
            if not hasattr(self, 'arm') or self.arm.getCurrentState() != unitree_arm_interface.ArmFSMState.JOINTCTRL:
                print("Switching to JOINTCTRL state...")
                self.arm.startTrack(unitree_arm_interface.ArmFSMState.JOINTCTRL)
                self.arm.setWait(False)
            
            # Convert target joint angles to target pose using forward kinematics
            if self.fk_debug:
                T_target = self.compute_forward_kinematics(target_joint_angles)
            else:
                T_target = self.arm_model.forwardKinematics(target_joint_angles, 6)
            target_position = T_target[:3, 3]
            # Convert rotation matrix to quaternion using scipy
            target_orientation = R.from_matrix(T_target[:3, :3]).as_quat()  # [x,y,z,w]
            
            # Get joint_speed (default to 1.0 if not available)
            joint_speed = getattr(self, 'joint_speed', 1.0)
            
            # Calculate dt ratio for internal loop
            dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
            
            # Maximum iterations to prevent infinite loops (similar to lowcmd duration)
            max_iterations = 20  
            tolerance = 0.01  # Joint angle tolerance in radians
            
            # Loop until convergence or max iterations
            for iteration in range(max_iterations):
                # Update state
                self._update_state()
                
                # Check if we've reached the target
                current_joint_pos = self.current_joint_pos
                joint_error = np.linalg.norm(target_joint_angles - current_joint_pos)
                if joint_error < tolerance:
                    print(f"Reached target joint angles within tolerance ({joint_error:.6f} < {tolerance})")
                    break
                
                # Get current end-effector pose for joint direction calculation
                current_ee_pose = self._get_current_ee_pose()
                current_pos = current_ee_pose[:3]
                current_quat = current_ee_pose[3:7]
                current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
                
                
                # Keep current gripper position as target (don't change gripper)
                target_gripper = current_gripper_pos
                
                # Calculate joint directions to target pose
                joint_directions, actual_joint_speed, gripper_speed = \
                    self._calculate_joint_directions(
                        target_position, target_orientation, target_gripper,
                        current_pos, current_quat, current_gripper_pos, self.dt, ik_type=ik_type
                    )
                
                # Execute jointCtrlCmd for dt_ratio iterations
                for i in range(dt_ratio):
                    self.arm.jointCtrlCmd(joint_directions, joint_speed)
                    time.sleep(self.arm._ctrlComp.dt)
            
            # Final state update
            self._update_state()
            print("IK-based joint movement completed")
            print("Target joint angles: ", target_joint_angles)
            print("Current joint angles: ", np.array(self.arm.lowstate.getQ()))
            print(f"Final joint error: {np.linalg.norm(target_joint_angles - self.current_joint_pos):.6f}")
            
        elif option == "MoveJ":
            # Use MoveJ high-level command (similar to example_highcmd_custom.py)
            print("Using MoveJ high-level command...")
            self.arm.setWait(True)
            
            # Convert target joint angles to transformation matrix
            if self.fk_debug:
                T_target = self.compute_forward_kinematics(target_joint_angles)
            else:
                T_target = self.arm._ctrlComp.armModel.forwardKinematics(target_joint_angles, 6)
            
            # Convert transformation matrix to posture (roll, pitch, yaw, x, y, z)
            posture = unitree_arm_interface.homoToPosture(T_target)
            
            # Set gripper position (keep current gripper position)
            gripper_pos = self.arm.lowstate.getGripperQ() if self.has_gripper else 0.0
            jnt_speed = 1.0  # Joint speed (can be adjusted)
            
            print(f"MoveJ to posture: {posture}, gripper: {gripper_pos}, speed: {jnt_speed}")
            
            # Send MoveJ command
            success = self.arm.MoveJ(posture, gripper_pos, jnt_speed)
            if success:
                print("MoveJ command sent successfully")
            else:
                print("MoveJ command failed")
            self.arm.setWait(False)
                
        else:  # option == "lowcmd" (default)
            # Use low-level commands (original implementation)
            print("Using low-level commands...")
            
            # Set low-level command mode
            self.arm.setFsmLowcmd()
            
            # Get current position as starting point
            lastPos = np.array(self.arm.lowstate.getQ())
            targetPos = target_joint_angles

            # Duration for smooth movement (about 2 seconds at 500Hz)
            duration = 1000
            
            # Hold current position before starting movement
            print("Holding current position before movement...")
            for _ in range(100):  # Hold for 0.1 seconds
                # Keep sending the same position commands to maintain current position
                self.arm.setArmCmd(lastPos, np.zeros(6), np.zeros(6))
                if self.has_gripper:
                    self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
                self.arm.sendRecv()
                time.sleep(self.arm._ctrlComp.dt)
            
            # Smooth interpolation to target position
            for i in range(duration):
                # Interpolate position
                self.arm.q = lastPos * (1 - i/duration) + targetPos * (i/duration)
                
                # Calculate velocity for smooth movement
                self.arm.qd = (targetPos - lastPos) / (duration * self.arm._ctrlComp.dt)
                
                # Calculate torque using inverse dynamics
                self.arm.tau = self.arm_model.inverseDynamics(
                    self.arm.q, self.arm.qd, np.zeros(6), np.zeros(6)
                )
                
                # Set arm commands
                self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
                
                # Set gripper command (keep current gripper position)
                if self.has_gripper:
                    self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
                
                # Send commands
                self.arm.sendRecv()
                time.sleep(self.arm._ctrlComp.dt)
            
            print("Joint movement completed")
            print("Target joint angles: ", target_joint_angles)
            print("Current joint angles: ", np.array(self.arm.lowstate.getQ()))
            
            # Hold position for a moment to ensure stability
            print("Holding position for stability...")
            for _ in range(50):  # Hold for 0.1 seconds
                # Keep sending the same position commands to maintain position
                self.arm.setArmCmd(self.arm.q, np.zeros(6), self.arm.tau)
                if self.has_gripper:
                    self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
                self.arm.sendRecv()
                time.sleep(self.arm._ctrlComp.dt)
            
            # Restart the sendRecv thread and switch back to proper FSM state
            # This is crucial to prevent hanging after low-level commands
            self.arm.loopOn()  # Restart sendRecvThread
            
            # Switch to JOINTCTRL state and maintain position
            print("Switching to JOINTCTRL state...")
            self.arm.setFsm(unitree_arm_interface.ArmFSMState.JOINTCTRL)
            
            # Immediately set the current position as target in JOINTCTRL mode
            print("Setting current position as target in JOINTCTRL mode...")
            current_joint_pos = np.array(self.arm.lowstate.getQ())
            
            # Set the arm to maintain current position
            self.arm.q = current_joint_pos
            self.arm.qd = np.zeros(6)  # Zero velocity
            self.arm.tau = np.zeros(6)  # Zero torque initially
            
            # Send commands to maintain position
            for _ in range(100):  # Send commands for 0.2 seconds
                self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
                if self.has_gripper:
                    self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
                self.arm.sendRecv()
                time.sleep(self.arm._ctrlComp.dt)
            
            print("Position maintained in JOINTCTRL mode")
        
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _get_observation method")
    
    def _get_reward(self) -> float:
        """Calculate reward for current state (to be implemented by subclasses)."""
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done (to be implemented by subclasses)."""
        return self.episode_step >= self.max_episode_steps
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional implementation)."""
        pass
    
    def close(self):
        """Clean up the environment."""
        if self.is_initialized:
            self.arm.backToStart()
            self.arm.loopOff()
            self.is_initialized = False

    
    def parse_urdf(self):
        """URDF 파일을 파싱하여 로봇 구조 정보 추출"""
        try:
            self.robot = URDF.from_xml_file(self.urdf_path)
            vprint(f"URDF 파싱 완료: {len(self.robot.joints)} 개 조인트, {len(self.robot.links)} 개 링크")
            
            # 조인트 정보 출력
            for i, joint in enumerate(self.robot.joints):
                vprint(f"Joint {i+1}: {joint.name}, Type: {joint.type}, Axis: {joint.axis}")
                
        except Exception as e:
            vprint(f"URDF 파싱 오류: {e}")
            
    # for custom forward kinematics
    def compute_forward_kinematics(self, joint_angles, gripper_angle=0.0):
        """전진기구학 계산 - URDF 기반 (Unitree SDK는 절대 위치를 반환하므로 부적합)"""
        self.joint_angles = joint_angles.copy()
        # Unitree Z1 SDK는 절대 위치를 반환하므로 URDF 기반 계산 사용
        self.compute_forward_kinematics_urdf(gripper_angle)
        return self.link_transforms['z1_GripperMover'].copy()
    
    def compute_forward_kinematics_urdf(self, gripper_angle=0.0):
        """URDF 기반 전진기구학 계산 (폴백)"""
        # 조인트 정보를 딕셔너리로 저장
        joint_info = {}
        for joint in self.robot.joints:
            joint_info[joint.name] = joint
            
        # 링크 정보를 딕셔너리로 저장
        link_info = {}
        for link in self.robot.links:
            link_info[link.name] = link
            
        # 변환 행렬 초기화
        self.link_transforms = {}
        
        # world -> link00 (고정 조인트)
        self.link_transforms['world'] = np.eye(4)
        self.link_transforms['link00'] = np.eye(4)
        
        # 각 조인트에 대해 변환 행렬 계산
        joint_angle_idx = 0
        for joint in self.robot.joints:
            if joint.type == 'revolute' and joint_angle_idx < len(self.joint_angles):
                # 조인트 각도
                angle = self.joint_angles[joint_angle_idx]
                joint_angle_idx += 1
                
                # 조인트 축
                axis = np.array(joint.axis)
                
                # 조인트 원점
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산 (RPY)
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 조인트 회전 행렬 (축 주위 회전)
                joint_rot_matrix = R.from_rotvec(axis * angle).as_matrix()
                
                # 변환 행렬 구성
                T_joint = np.eye(4)
                T_joint[:3, :3] = rot_matrix @ joint_rot_matrix
                T_joint[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_joint
                
                self.link_transforms[joint.child] = child_transform
                
            elif joint.type == 'fixed':
                # 고정 조인트
                origin = joint.origin
                if origin is not None:
                    xyz = np.array(origin.xyz) if origin.xyz else np.zeros(3)
                    rpy = np.array(origin.rpy) if origin.rpy else np.zeros(3)
                else:
                    xyz = np.zeros(3)
                    rpy = np.zeros(3)
                
                # 회전 행렬 계산
                if np.any(rpy):
                    rot_matrix = R.from_euler('xyz', rpy).as_matrix()
                else:
                    rot_matrix = np.eye(3)
                
                # 변환 행렬 구성
                T_fixed = np.eye(4)
                T_fixed[:3, :3] = rot_matrix
                T_fixed[:3, 3] = xyz
                
                # 부모 링크의 변환 행렬과 결합
                parent_transform = self.link_transforms.get(joint.parent, np.eye(4))
                child_transform = parent_transform @ T_fixed
                
                self.link_transforms[joint.child] = child_transform
        
        # Gripper 위치 계산 (URDF 기반 정확한 오프셋 사용)
        if 'link06' in self.link_transforms:
            # gripperStator는 link06에서 xyz="0.051 0.0 0.0" 오프셋으로 연결
            gripper_stator_offset = np.array([0.051, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_stator_transform = self.link_transforms['link06'].copy()
            gripper_stator_transform[:3, 3] += gripper_stator_transform[:3, :3] @ gripper_stator_offset
            
            self.link_transforms['z1_GripperStator'] = gripper_stator_transform
            
            # gripperMover는 gripperStator에서 xyz="0.049 0.0 0" 오프셋으로 연결
            gripper_mover_offset = np.array([0.049, 0.0, 0.0])  # URDF에서 정의된 오프셋
            gripper_mover_transform = gripper_stator_transform.copy()
            gripper_mover_transform[:3, 3] += gripper_mover_transform[:3, :3] @ gripper_mover_offset
            
            # gripper 회전 적용 (Y축 주위 회전, URDF에서 axis xyz="0 1 0")
            if gripper_angle != 0.0:
                gripper_rotation = R.from_rotvec([0, gripper_angle, 0]).as_matrix()
                gripper_mover_transform[:3, :3] = gripper_mover_transform[:3, :3] @ gripper_rotation
            
            self.link_transforms['z1_GripperMover'] = gripper_mover_transform
                




class EEPoseCtrlJointCmdWrapper(Z1BaseEnv):
    """
    End-effector pose control wrapper for Z1 robot using jointCtrlCmd with RTC-style future sequence handling.
    This wrapper allows controlling the robot's end-effector pose by converting target poses to joint commands
    using jointCtrlCmd with support for future target pose sequences from inference.
    """
    
    def __init__(self, 
                 has_gripper: bool = True, 
                 control_frequency: float = 500.0,
                 position_tolerance: float = 0.01,
                 orientation_tolerance: float = 0.1,
                 joint_speed: float = 1.0,
                 sequence_length: int = 10,
                 use_current_joint_pos_when_ik_fails = True,
                 T_E_C: np.ndarray = None,
                 urdf_path = None,
                 fk_debug = True,
                 interpolator_option: str = 'Cubic',
                 custom_speed_factor: float = 1.0,
                 use_retargeting: bool = False,
                 ):
        """
        Initialize the end-effector pose control wrapper using joint commands with RTC support.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
            position_tolerance: Position tolerance for convergence
            orientation_tolerance: Orientation tolerance for convergence
            joint_speed: Joint speed for jointCtrlCmd commands (range: [0, π])
            sequence_length: Length of future target pose sequences from inference
            interpolator_option: Interpolator type - "Pchip" or "Cubic" (default: "Cubic")
            custom_speed_factor: Speed factor for interpolator velocities (default: 1.0)
            
        """
        super().__init__(has_gripper, control_frequency, urdf_path)
        self.use_current_joint_pos_when_ik_fails = use_current_joint_pos_when_ik_fails
        self.T_E_C = T_E_C
        self.fk_debug = fk_debug
        assert fk_debug, 'temporarily, we set fk_debug=true, its value means gripper se(3)'
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.joint_speed = joint_speed
        self.sequence_length = sequence_length
        self.interpolator_option = interpolator_option
        self.custom_speed_factor = custom_speed_factor
        self.ik_type = 'null_space' # 'jacobian' # 'null_space'
        self.use_retargeting = use_retargeting
        
        # Store previous step's joint directions for continuous movement during inference
        self.previous_joint_directions = np.zeros(7)  # [J1, J2, J3, J4, J5, J6, gripper]
        self.has_previous_directions = False
        # Store previous iteration's chosen target joint positions (for IK fallback comparison)
        self.prev_target_joint_pos = None
        self.prev_final_error_pos = None
        self.prev_final_error_ori = None
        self.prev_null_obj_val = None
        
        # For compatibility with Z1 environment interface
        # Store last step result for is_step_complete() and get_step_result()
        self.last_step_result = None
        self.last_step_complete = True  # Initially complete (no step executed yet)
        self.last_step_start_time = None  # Track when non-blocking step started
        
        # RTC-style future sequence handling
        self.current_sequence = None  # Current future target pose sequence
        self.sequence_index = 0       # Current index in the sequence
        self.sequence_step_count = 0  # Total steps executed in current sequence
        
        # Target pose (position + quaternion) - current target from sequence
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])  # x, y, z, w quaternion (identity)
        self.target_gripper = 0.0
        
        # For chunk action intermediate results tracking
        self.chunk_intermediate_results = {}  # {action_index: result_dict}
        self.chunk_completed_actions = set()  # Set of completed action indices
        self.current_chunk_horizon = 0  # Current chunk size
        
        # Define action space: [x, y, z, qx, qy, qz, qw, gripper] (8D)
        # Position: [-1, 1] meters, Orientation: [-1, 1] quaternion [x,y,z,w], Gripper: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: [joint_pos(6), joint_vel(6), ee_pos(3), ee_quat(4) in [x,y,z,w], gripper_pos(1), gripper_vel(1)]
        obs_low = np.concatenate([
            np.full(6, -np.pi),  # joint positions
            np.full(6, -10.0),   # joint velocities
            np.full(3, -2.0),    # end-effector position
            np.full(4, -1.0),    # end-effector quaternion
            np.array([-1.0]),    # gripper position
            np.array([-5.0])     # gripper velocity
        ])
        obs_high = np.concatenate([
            np.full(6, np.pi),   # joint positions
            np.full(6, 10.0),    # joint velocities
            np.full(3, 2.0),     # end-effector position
            np.full(4, 1.0),     # end-effector quaternion
            np.array([1.0]),     # gripper position
            np.array([5.0])      # gripper velocity
        ])
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
    def reset(self, joint_angle: Optional[np.ndarray] = None, reset_for_end=False, option: str = "IK") -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Args:
            joint_angle: Optional joint angles to move to. If None, uses default reset behavior.
                        Should be a 6-element array for 6-DOF arm.
        """
        assert joint_angle is not None, "joint_angle must be provided for stable initialization"
        super().reset(joint_angle, option=option)
        if reset_for_end:
            return
        # Start joint control mode
        self.arm.startTrack(unitree_arm_interface.ArmFSMState.JOINTCTRL)
        print("Started joint control mode")
        
        self.arm.setWait(False)

        # Set initial target to current end-effector pose
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        self.target_position = current_ee_pose[:3]
        self.target_orientation = current_ee_pose[3:7]
        self.target_gripper = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Reset RTC sequence handling
        self.current_sequence = None
        self.sequence_index = 0
        self.sequence_step_count = 0
        
        # Reset previous directions flag for first step calculation
        self.has_previous_directions = False
        self.prev_target_joint_pos = None
        self.prev_final_error_pos = None
        self.prev_final_error_ori = None
        self.prev_null_obj_val = None
        
        # Reset chunk tracking variables
        self.chunk_intermediate_results = {}
        self.chunk_completed_actions = set()
        self.current_chunk_horizon = 0
        
        # Reset compatibility variables
        self.last_step_result = None
        self.last_step_complete = True
        self.last_step_start_time = None
        
        return self._get_observation()
    
    
    def step(self, action: np.ndarray, wait: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using jointCtrlCmd.
        Converts target pose to joint commands and executes them.
        
        Args:
            action: [x, y, z, qx, qy, qz, qw, gripper] target pose (quaternion in [x,y,z,w] format)
                   or [horizon, 8] array for chunked execution
            wait: If True, wait for step to complete before returning. If False, execute step in background.
                  Note: wait parameter is kept for compatibility but always executes blocking.
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Mark step as not complete at the start (for is_step_complete() compatibility)
        # self.last_step_complete = False
        # self.last_step_start_time = time.time()
        
        try:
            # Check if action is chunked (2D) or single (1D)
            if len(action.shape) == 2:
                # action : [horizon, dim]
                # return self._execute_step_chunk_logic(action)
                return self._execute_step_chunk_logic_with_interpolator(action)
            elif len(action.shape) == 1:
                # Single action
                # return self._execute_step_logic(action)
                return self._execute_step_logic_with_interpolator(action)
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(21), 0.0, False, {'error': str(e)}
    
    def _execute_step_logic(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute the core step logic. This method contains the actual step execution code.
        
        Args:
            action: [x, y, z, qx, qy, qz, qw, gripper] target pose (quaternion in [x,y,z,w] format)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Note: last_step_complete and last_step_start_time are already set in step() method
        self.last_step_complete = False  # Already set in step()
        
        
        print('@@@@@@@@@@@@@ in _execute_step_logic function, action :', action)
        # Update target pose from action
        # action format: [x, y, z, qx, qy, qz, qw, gripper]
        self.target_position = action[:3]
        self.target_orientation = self._normalize_quaternion(action[3:7])  # [qx, qy, qz, qw]
        if self.has_gripper:
            self.target_gripper = action[7]
        


        if self.T_E_C is not None:
            # assume input action is T_bc
            T_bc = np.eye(4)
            T_bc[:3, :3] = R.from_quat(self.target_orientation).as_matrix()
            T_bc[:3, 3] = self.target_position
            T_be = T_bc @ np.linalg.inv(self.T_E_C)
            self.target_position = T_be[:3, 3]
            self.target_orientation = R.from_matrix(T_be[:3, :3]).as_quat()

        # Get current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_ee_pose_before_cmd = current_ee_pose.copy()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Calculate joint directions to target pose
        joint_directions, actual_joint_speed, gripper_speed = \
            self._calculate_joint_directions(
                self.target_position, self.target_orientation, self.target_gripper,
                current_pos, current_quat, current_gripper_pos, self.dt
            )
        print(f'joint_directions : {joint_directions}')
        print(f'actual_joint_speed : {actual_joint_speed}')
        

        
        self.last_step_start_time = start_time = time.time()
        
        
        end_time = start_time + self.dt
        sleep_time_list = []
        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            self.arm.jointCtrlCmd(joint_directions, self.joint_speed)
            sleep_start = time.time()
            time.sleep(self.arm._ctrlComp.dt)
            sleep_time = time.time() - sleep_start
            sleep_time_list.append(sleep_time)
            

        cmd_time = time.time() - start_time
        # print("cmd time: ", cmd_time)
        # print("sleep time mean:", np.array(sleep_time_list).mean())
        # print("sleep time std: ", np.array(sleep_time_list).std())
        # print("sleep time max: ", np.array(sleep_time_list).max())
        # print("sleep time min: ", np.array(sleep_time_list).min())





        # # Calculate dt ratio for internal loop
        # dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
        # # Note: last_step_start_time is already set in step() method
        # start_time = self.last_step_start_time
        # sleep_time_list = []
        # # Execute jointCtrlCmd for dt_ratio iterations
        # for i in range(dt_ratio):
        #     print(i)
        #     self.arm.jointCtrlCmd(joint_directions, self.joint_speed)
        #     sleep_start = time.time()
        #     time.sleep(self.arm._ctrlComp.dt)
        #     sleep_time = time.time() - sleep_start
        #     sleep_time_list.append(sleep_time)
        # cmd_time = time.time() - start_time
        # print("cmd time: ", cmd_time)
        # print("sleep time mean:", np.array(sleep_time_list).mean())
        # print("sleep time std: ", np.array(sleep_time_list).std())
        # print("sleep time max: ", np.array(sleep_time_list).max())
        # print("sleep time min: ", np.array(sleep_time_list).min())





        # Update state and get results
        self._update_state()
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        
        # Create info dictionary
        info = {
            'fsm_state': self.arm.getCurrentState(),
            'target_position': self.target_position.copy(),
            'target_orientation': self.target_orientation.copy(),
            'current_ee_pose': self._get_current_ee_pose(),
            'current_cam_pose': self._get_current_camera_pose_for_obs(),
            'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
            'orientation_error': self._quaternion_distance(
                self.target_orientation, self._get_current_ee_orientation()
            ),
            'joint_directions': joint_directions.copy(),
            'actual_joint_speed': actual_joint_speed,
            'gripper_speed': gripper_speed,
            # 'dt_ratio': dt_ratio,
            'current_ee_pose_before_cmd': current_ee_pose_before_cmd.copy(),
        }
        
        
        
        # Store result for compatibility with get_step_result()
        result = (observation, reward, done, info)
        self.last_step_result = result
        # print("last_step_result: ", self.last_step_result)



        # Mark as complete for blocking execution
        self.last_step_complete = True
        self.last_step_start_time = None

        
        self.episode_step += 1
        return observation, reward, done, info
    
    def _execute_step_logic_with_interpolator(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute the core step logic. This method contains the actual step execution code.
        
        Args:
            action: [x, y, z, qx, qy, qz, qw, gripper] target pose (quaternion in [x,y,z,w] format)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Note: last_step_complete and last_step_start_time are already set in step() method
        self.last_step_complete = False  # Already set in step()
        
        
        print('@@@@@@@@@@@@@ in _execute_step_logic_with_interpolator function, action :', action)
        # Update target pose from action
        # action format: [x, y, z, qx, qy, qz, qw, gripper]
        self.target_position = action[:3]
        self.target_orientation = self._normalize_quaternion(action[3:7])  # [qx, qy, qz, qw]
        if self.has_gripper:
            self.target_gripper = action[7]
        


        if self.T_E_C is not None:
            # assume input action is T_bc
            T_bc = np.eye(4)
            T_bc[:3, :3] = R.from_quat(self.target_orientation).as_matrix()
            T_bc[:3, 3] = self.target_position
            T_be = T_bc @ np.linalg.inv(self.T_E_C)
            self.target_position = T_be[:3, 3]
            self.target_orientation = R.from_matrix(T_be[:3, :3]).as_quat()

        # Get current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_ee_pose_before_cmd = current_ee_pose.copy()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Calculate target joint positions using IK (similar to _calculate_joint_directions)
        target_T = self._pose_to_transformation_matrix(self.target_position, self.target_orientation)
        current_joint_pos = self.current_joint_pos
        
        # Use inverse kinematics to get target joint positions
        ik_type = self.ik_type # getattr(self, 'ik_type', 'jacobian')  # Default to 'jacobian' if not set
        if ik_type == 'null_space':
            success, target_joint_pos, iterations, final_error_pos, null_obj_val = self.solve_ik_null_space(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance=1e-2,
                tolerance_null=1e-3
            )
            final_error_ori = 0.0
        elif ik_type == 'jacobian':
            success, target_joint_pos, iterations, final_error_pos, final_error_ori, null_obj_val = self.solve_ik_6d_dls_jacobian(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance_pos=1e-2,
                tolerance_ori=2e-2,
                w_pos=1.0,
                w_ori=0.1,
                lambda0=1e-3,
                use_adaptive_damping=True,
                damp_gain=1e-2,
                alpha_init=1.0,
                joint_clip=True,
            )
        
        if not success:
            print(f"Warning: Z1 IK failed to converge (error_pos: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            if self.use_current_joint_pos_when_ik_fails:
                target_joint_pos = current_joint_pos.copy()
            else:
                self.prev_final_error_pos = final_error_pos
                self.prev_final_error_ori = final_error_ori
                self.prev_null_obj_val = null_obj_val
        else:
            print(f"Z1 IK solved successfully in {iterations} iterations (error: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            self.prev_final_error_pos = final_error_pos
            self.prev_final_error_ori = final_error_ori
            self.prev_null_obj_val = null_obj_val
        
        self.prev_target_joint_pos = target_joint_pos.copy()
        
        # Get current positions (arm + gripper)
        current_positions = np.concatenate([self.current_joint_pos, [current_gripper_pos]])
        desired_positions = np.concatenate([target_joint_pos, [self.target_gripper]])
        
        # Generate waypoints and timepoints for smooth interpolation
        waypoints, timepoints = generate_waypoints_and_timepoints(
            current_positions,
            desired_positions,
            self.dt
        )
        
        # Create interpolator (using PchipInterpolator for smooth interpolation)
        if self.interpolator_option == 'Pchip':
            interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)
            interpolator_feedforward_velocity = interpolator_position.derivative()
        elif self.interpolator_option == 'Cubic':
            interpolator_position = CubicSpline(timepoints, waypoints, axis=0, bc_type='natural')
            interpolator_feedforward_velocity = interpolator_position.derivative()
        else:
            interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)
            interpolator_feedforward_velocity = interpolator_position.derivative()
        
        # Execute jointCtrlCmd with interpolator
        self.last_step_start_time = start_time = time.time()
        end_time = start_time + self.dt
        sleep_time_list = []
        
        # Initialize joint_directions, actual_joint_speed, gripper_speed for info dict
        joint_directions = np.zeros(7)
        actual_joint_speed = 0.0
        gripper_speed = 0.0
        iter =0 
        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            
            # Evaluate interpolator at current time
            eval_time = np.clip(current_time, timepoints[0], timepoints[-1])
            interpolated_positions = interpolator_position(eval_time)
            interpolated_velocities = interpolator_feedforward_velocity(eval_time) * self.custom_speed_factor
            
            # Extract interpolated joint positions, velocities and gripper position
            interpolated_joint_pos = interpolated_positions[:6]
            interpolated_joint_vel = interpolated_velocities[:6]
            interpolated_gripper_pos = interpolated_positions[6] if self.has_gripper else 0.0
            interpolated_gripper_vel = interpolated_velocities[6] if self.has_gripper else 0.0
            
            joint_direction = np.array([interpolated_joint_vel[0], 
                                        interpolated_joint_vel[1], 
                                        interpolated_joint_vel[2], 
                                        interpolated_joint_vel[3], 
                                        interpolated_joint_vel[4], 
                                        interpolated_joint_vel[5], 
                                        interpolated_gripper_vel])
            
            iter += 1
            self.arm.jointCtrlCmd(joint_direction, 1.0) # qd = direction*jointSpeed
            sleep_start = time.time()
            time.sleep(self.arm._ctrlComp.dt)
            sleep_time = time.time() - sleep_start
            sleep_time_list.append(sleep_time)
            
        cmd_time = time.time() - start_time
        # print("cmd time: ", cmd_time)
        # print("sleep time mean:", np.array(sleep_time_list).mean())
        # print("sleep time std: ", np.array(sleep_time_list).std())
        # print("sleep time max: ", np.array(sleep_time_list).max())
        # print("sleep time min: ", np.array(sleep_time_list).min())







        # Update state and get results
        self._update_state()
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        
        # Create info dictionary
        info = {
            'fsm_state': self.arm.getCurrentState(),
            'target_position': self.target_position.copy(),
            'target_orientation': self.target_orientation.copy(),
            'current_ee_pose': self._get_current_ee_pose(),
            'current_cam_pose': self._get_current_camera_pose_for_obs(),
            'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
            'orientation_error': self._quaternion_distance(
                self.target_orientation, self._get_current_ee_orientation()
            ),
            'joint_directions': joint_directions.copy(),
            'actual_joint_speed': actual_joint_speed,
            'gripper_speed': gripper_speed,
            # 'dt_ratio': dt_ratio,
            'current_ee_pose_before_cmd': current_ee_pose_before_cmd.copy(),
        }
        
        
        
        # Store result for compatibility with get_step_result()
        result = (observation, reward, done, info)
        self.last_step_result = result
        # print("last_step_result: ", self.last_step_result)



        # Mark as complete for blocking execution
        self.last_step_complete = True
        self.last_step_start_time = None

        
        self.episode_step += 1
        return observation, reward, done, info

    def _execute_step_chunk_logic(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute a chunk of steps with end-effector pose control using jointCtrlCmd.
        Processes multiple actions sequentially, executing each action for dt duration.
        
        Args:
            actions: [horizon, 8] array of target poses (quaternion in [x,y,z,w] format)
            
        Returns:
            observation: Current observation (after all actions are executed)
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information with horizon-length arrays
        """
        assert len(actions.shape) == 2, f"Expected actions shape [horizon, 8], got {actions.shape}"
        horizon = actions.shape[0]
        
        
        # Note: last_step_complete and last_step_start_time are already set in step() method
        # self.last_step_complete = False  # Already set in step()
        # self.last_step_start_time = time.time()  # Already set in step()
        
        # Initialize chunk tracking
        self.chunk_intermediate_results = {}
        self.chunk_completed_actions = set()
        self.current_chunk_horizon = horizon
        
        # Get current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Lists to collect data at each action (horizon length)
        timepoint_target_positions = []
        timepoint_target_orientations = []
        timepoint_current_ee_poses = []
        timepoint_current_cam_poses = []
        timepoint_position_errors = []
        timepoint_orientation_errors = []
        timepoint_joint_directions = []
        timepoint_actual_joint_speeds = []
        timepoint_gripper_speeds = []
        
        # Calculate dt ratio for internal loop
        dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
        # Note: last_step_start_time is already set in step() method
        
        # Execute each action in the chunk sequentially
        for i in range(horizon):
            action = actions[i]
            
            # Update target pose from action
            target_position = action[:3]
            target_orientation = self._normalize_quaternion(action[3:7])  # [qx, qy, qz, qw]
            target_gripper = action[7] if self.has_gripper else 0.0

            if self.T_E_C is not None:
                # assume input action is T_bc
                T_bc = np.eye(4)
                T_bc[:3, :3] = R.from_quat(target_orientation).as_matrix()
                T_bc[:3, 3] = target_position
                T_be = T_bc @ np.linalg.inv(self.T_E_C)
                target_position = T_be[:3, 3]
                target_orientation = R.from_matrix(T_be[:3, :3]).as_quat()




            
            # Get current state before executing this action
            self._update_state()
            current_ee_pose = self._get_current_ee_pose()
            current_pos = current_ee_pose[:3]
            current_quat = current_ee_pose[3:7]
            current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
            
            # Calculate joint directions to target pose
            joint_directions, actual_joint_speed, gripper_speed = \
                self._calculate_joint_directions(
                    target_position, target_orientation, target_gripper,
                    current_pos, current_quat, current_gripper_pos, self.dt
                )
            
            # Execute jointCtrlCmd for dt_ratio iterations
            for j in range(dt_ratio):
                self.arm.jointCtrlCmd(joint_directions, self.joint_speed)
                time.sleep(self.arm._ctrlComp.dt)
            
            # Update state after executing this action
            self._update_state()
            current_ee_pose_after = self._get_current_ee_pose()
            current_pos_after = current_ee_pose_after[:3]
            current_quat_after = current_ee_pose_after[3:7]
            
            # Calculate errors
            position_error = np.linalg.norm(target_position - current_pos_after)
            orientation_error = self._quaternion_distance(target_orientation, current_quat_after)
            
            # Store collected data
            timepoint_target_positions.append(target_position.copy())
            timepoint_target_orientations.append(target_orientation.copy())
            timepoint_current_ee_poses.append(self._get_current_ee_pose_for_obs().copy())
            timepoint_current_cam_poses.append(self._get_current_camera_pose_for_obs().copy())
            timepoint_position_errors.append(position_error)
            timepoint_orientation_errors.append(orientation_error)
            timepoint_joint_directions.append(joint_directions.copy())
            timepoint_actual_joint_speeds.append(actual_joint_speed)
            timepoint_gripper_speeds.append(gripper_speed)
            
            # Store intermediate result for this action index
            action_index = i
            self.chunk_intermediate_results[action_index] = {
                'target_position': target_position.copy(),
                'target_orientation': target_orientation.copy(),
                'current_ee_pose': self._get_current_ee_pose_for_obs().copy(),
                'current_cam_pose': self._get_current_camera_pose_for_obs().copy(),
                'position_error': position_error,
                'orientation_error': orientation_error,
                'joint_directions': joint_directions.copy(),
                'actual_joint_speed': actual_joint_speed,
                'gripper_speed': gripper_speed,
            }
            self.chunk_completed_actions.add(action_index)
        
        # Store the last target as the current target (for info dict)
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.target_gripper = target_gripper
        
        # Final state update
        self._update_state()
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        
        # Create info dictionary with horizon-length arrays
        info = {
            'fsm_state': self.arm.getCurrentState(),
            'target_position': np.array(timepoint_target_positions),  # [horizon, 3]
            'target_orientation': np.array(timepoint_target_orientations),  # [horizon, 4]
            'current_ee_pose': np.array(timepoint_current_ee_poses),  # [horizon, 8]
            'current_cam_pose': np.array(timepoint_current_cam_poses),  # [horizon, 8]
            'position_error': np.array(timepoint_position_errors),  # [horizon]
            'orientation_error': np.array(timepoint_orientation_errors),  # [horizon]
            'joint_directions': np.array(timepoint_joint_directions),  # [horizon, 7]
            'actual_joint_speed': np.array(timepoint_actual_joint_speeds),  # [horizon]
            'gripper_speed': np.array(timepoint_gripper_speeds),  # [horizon]
            'horizon': horizon,
        }
        
        
        
        # Store result for compatibility with get_step_result()
        result = (observation, reward, done, info)
        self.last_step_result = result
        # print("last_step_result: ", self.last_step_result)

        # Mark as complete for blocking execution
        self.last_step_complete = True
        self.last_step_start_time = None
        
        self.episode_step += 1
        return observation, reward, done, info

    def _se3_residual(self, T_fk: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
        """
        SE(3) residual as 6D vector:
        - translation error: p_fk - p_tgt   (in meters, assuming)
        - rotation error:    log(R_tgt^T R_fk) as rotvec (in radians)
        """
        assert T_fk.shape == (4, 4)
        assert T_tgt.shape == (4, 4)

        p_fk = T_fk[:3, 3]
        p_tgt = T_tgt[:3, 3]

        R_fk = T_fk[:3, :3]
        R_tgt = T_tgt[:3, :3]

        # rotation error: identity when fk == tgt
        R_err = R_tgt.T @ R_fk
        rotvec = R.from_matrix(R_err).as_rotvec()  # 3-vector (axis * angle)

        return np.concatenate([p_fk - p_tgt, rotvec], axis=0)

    def _retarget_se3_trajectory(
        self,
        target_T: np.ndarray,          # [T,4,4]
        q0: np.ndarray,                # [DoF]
        *,
        w_pos: float = 1.0,            # weight for position residual
        w_rot: float = 1.0,            # weight for rotation residual
        lambda_smooth: float = 1e-2,   # smoothness weight (q_t - q_{t-1})
        bounds=None,                   # (lower, upper) each can be scalar or [DoF]
        max_nfev: int = 50,
        tol: float = 1e-6,
        verbose: int = 0,
    ):
        """
        Retarget SE(3) target trajectory into joint trajectory via per-timestep IK:
            q_t = argmin_q  || e_se3(fk(q), T_target[t]) ||^2
                            + lambda_smooth * || q - q_{t-1} ||^2

        Returns:
            q_traj: [T, DoF]
            info:   dict with per-step cost and solver status
        """
        target_T = np.asarray(target_T)
        q0 = np.asarray(q0).astype(float)

        assert target_T.ndim == 3 and target_T.shape[1:] == (4, 4), "target_T must be [T,4,4]"
        T = target_T.shape[0]
        dof = q0.shape[0]

        if bounds is None:
            lb = -np.inf * np.ones(dof)
            ub = +np.inf * np.ones(dof)
        else:
            lb, ub = bounds
            lb = np.broadcast_to(np.asarray(lb, dtype=float), (dof,)).copy()
            ub = np.broadcast_to(np.asarray(ub, dtype=float), (dof,)).copy()

        # sqrt weights for least_squares residual scaling
        s_pos = np.sqrt(w_pos)
        s_rot = np.sqrt(w_rot)
        s_sm  = np.sqrt(lambda_smooth)

        q_prev = q0.copy()
        q_traj = np.zeros((T, dof), dtype=float)
        T_traj = np.zeros((T, 4, 4), dtype=float)

        costs = []
        statuses = []
        nfev_list = []  # Track number of function evaluations per timestep

        for t in range(T):
            T_tgt = target_T[t]

            def residual(q):
                T_fk = self.compute_forward_kinematics(q)  # must return (4,4)
                e6 = self._se3_residual(T_fk, T_tgt)   # [6] = [pos(3), rot(3)]
                # scale residuals
                e6_scaled = np.concatenate([s_pos * e6[:3], s_rot * e6[3:]], axis=0)
                smooth = s_sm * (q - q_prev)      # [DoF]
                return np.concatenate([e6_scaled, smooth], axis=0)

            # First timestep may need more iterations if starting from poor initial guess
            # Use more iterations for first timestep, fewer for subsequent (warm start)
            current_max_nfev = max_nfev * 2 if t == 0 else max_nfev

            res = least_squares(
                residual,
                x0=q_prev,             # warm start
                bounds=(lb, ub),
                max_nfev=current_max_nfev,
                xtol=tol,
                ftol=tol,
                gtol=tol,
                verbose=verbose,
            )

            q_prev = res.x
            q_traj[t] = q_prev
            T_traj[t] = self.compute_forward_kinematics(q_prev)
            # (optional) store diagnostics
            costs.append(res.cost)      # 0.5 * sum(residual^2)
            statuses.append({"success": bool(res.success), "status": int(res.status), "message": res.message})
            nfev_list.append(res.nfev)  # Number of function evaluations for this timestep

        info = {
            "costs": np.asarray(costs, dtype=float),
            "statuses": statuses,
            "weights": {"w_pos": w_pos, "w_rot": w_rot, "lambda_smooth": lambda_smooth},
            "nfev_list": np.asarray(nfev_list, dtype=int),  # Function evaluations per timestep
            "total_nfev": sum(nfev_list),  # Total function evaluations
        }
        return q_traj, T_traj, info

    def _execute_step_chunk_logic_with_interpolator(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute a chunk of steps with end-effector pose control using jointCtrlCmd with interpolator.
        Pre-computes all IK for the chunk, then executes control using interpolator.
        
        Args:
            actions: [horizon, 8] array of target poses (quaternion in [x,y,z,w] format)
            
        Returns:
            observation: Current observation (after all actions are executed)
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information with horizon-length arrays
        """
        assert len(actions.shape) == 2, f"Expected actions shape [horizon, 8], got {actions.shape}"
        horizon = actions.shape[0]
        
        # Note: last_step_complete and last_step_start_time are already set in step() method
        self.last_step_complete = False
        
        # Initialize chunk tracking
        self.chunk_intermediate_results = {}
        self.chunk_completed_actions = set()
        self.current_chunk_horizon = horizon
        
        # Get current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        current_joint_pos = self.current_joint_pos.copy()
        
        # Pre-compute all IK for all actions in the chunk (like step_chunk in widowx_all.py)
        target_joint_positions = []
        target_gripper_positions = []
        target_positions_list = []
        target_orientations_list = []
        target_T_list = []  # Store target SE(3) transformations for retargeting
        ik_success_list = []  # Track IK success for each action
        
        for i in range(horizon):
            action = actions[i]
            
            # Extract target pose from action
            target_position = action[:3]
            target_orientation = self._normalize_quaternion(action[3:7])  # [qx, qy, qz, qw]
            target_gripper = action[7] if self.has_gripper else 0.0
            
            # Apply T_E_C transformation if needed
            if self.T_E_C is not None:
                # assume input action is T_bc
                T_bc = np.eye(4)
                T_bc[:3, :3] = R.from_quat(target_orientation).as_matrix()
                T_bc[:3, 3] = target_position
                T_be = T_bc @ np.linalg.inv(self.T_E_C)
                target_position = T_be[:3, 3]
                target_orientation = R.from_matrix(T_be[:3, :3]).as_quat()
            
            target_positions_list.append(target_position.copy())
            target_orientations_list.append(target_orientation.copy())
            
            # Calculate target joint positions using IK (similar to _execute_step_logic_with_interpolator)
            target_T = self._pose_to_transformation_matrix(target_position, target_orientation)
            target_T_list.append(target_T.copy())
            
            # Use inverse kinematics to get target joint positions
            ik_type = self.ik_type # getattr(self, 'ik_type', 'jacobian')  # Default to 'jacobian' if not set
            if ik_type == 'null_space':
                success, target_joint_pos, iterations, final_error_pos, null_obj_val = self.solve_ik_null_space(
                    target_T, 
                    initial_guess=current_joint_pos if i == 0 else target_joint_positions[-1],
                    max_iterations=50,
                    tolerance=1e-2,
                    tolerance_null=1e-3
                )
                final_error_ori = 0.0
            elif ik_type == 'jacobian':
                success, target_joint_pos, iterations, final_error_pos, final_error_ori, null_obj_val = self.solve_ik_6d_dls_jacobian(
                    target_T, 
                    initial_guess=current_joint_pos if i == 0 else target_joint_positions[-1],
                    max_iterations=50,
                    tolerance_pos=1e-2,
                    tolerance_ori=2e-2,
                    w_pos=1.0,
                    w_ori=0.1,
                    lambda0=1e-3,
                    use_adaptive_damping=True,
                    damp_gain=1e-2,
                    alpha_init=1.0,
                    joint_clip=True,
                )
            
            ik_success_list.append(success)
            
            if not success:
                print(f"Warning: Z1 IK failed to converge for action {i} (error_pos: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
                if self.use_current_joint_pos_when_ik_fails:
                    target_joint_pos = current_joint_pos.copy() if i == 0 else target_joint_positions[-1].copy()
                else:
                    # Use the IK result even if not fully converged
                    pass
            else:
                print(f"Z1 IK solved successfully for action {i} in {iterations} iterations (error: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            
            target_joint_positions.append(target_joint_pos)
            target_gripper_positions.append(target_gripper)
        
        # Convert to numpy arrays
        target_joint_positions = np.array(target_joint_positions)  # [horizon, 6]
        target_gripper_positions = np.array(target_gripper_positions)  # [horizon]
        target_T_trajectory = np.array(target_T_list)  # [horizon, 4, 4]
        
        # Apply retargeting if enabled and IK failed at least once
        if self.use_retargeting and not all(ik_success_list):
            retarget_start_time = time.time()
            print(f"\nZ1 env retargeting enabled: IK failed for {sum(1 - np.array(ik_success_list))}/{horizon} actions. Applying retargeting...")
            
            # Use IK results as initial guess for retargeting
            q0_retarget = target_joint_positions[0].copy()
            
            # Get joint limits for bounds
            joint_limits = self.joint_limits
            q_min = np.array([limit[0] for limit in joint_limits])
            q_max = np.array([limit[1] for limit in joint_limits])
            
            q0_retarget = np.clip(q0_retarget, q_min, q_max)

            # Perform retargeting
            q_traj_retargeted, T_traj_retargeted, retarget_info = self._retarget_se3_trajectory(
                target_T_trajectory,
                q0_retarget,
                w_pos=1.0,
                w_rot=0.5,
                lambda_smooth=1e-3,
                bounds=(q_min, q_max),
                max_nfev=100,
                tol=1e-6,
                verbose=0,
            )
            
            # Replace target_joint_positions with retargeted results
            target_joint_positions = q_traj_retargeted.copy() # [horizon, 6]
            target_T_trajectory = T_traj_retargeted.copy() # [horizon, 4, 4]
            target_positions_list = list(target_T_trajectory[:, :3, 3])
            target_orientations_list = list(R.from_matrix(target_T_trajectory[:, :3, :3]).as_quat())

            retarget_time = time.time() - retarget_start_time
            print(f"Z1 env retargeting completed. Time: {retarget_time:.4f}s, Final cost: {retarget_info['costs'][-1]:.6f}, Total function evaluations: {retarget_info['total_nfev']}")
            

        T_be_target = target_T_trajectory.copy() # [horizon, 4, 4] base coordinate
        if self.T_E_C is not None:
            T_bc_target = np.einsum('hij,jk->hik', T_be_target , self.T_E_C) # [horizon, 4, 4] camera coordinate
        else:
            T_bc_target = T_be_target.copy() # [horizon, 4, 4] camera coordinate

        
        # Store the last target as the current target (for info dict)
        self.target_position = target_positions_list[-1]
        self.target_orientation = target_orientations_list[-1]
        self.target_gripper = target_gripper_positions[-1]
        
        # Build waypoints: [current, target_0, target_1, ..., target_{horizon-1}]
        # Get current positions (arm + gripper)
        current_positions = np.concatenate([self.current_joint_pos, [current_gripper_pos]])
        
        # Build waypoints array: [current, target_0, target_1, ..., target_{horizon-1}]
        waypoints = [current_positions]
        for i in range(horizon):
            desired_positions = np.concatenate([target_joint_positions[i], [target_gripper_positions[i]]])
            waypoints.append(desired_positions)
        waypoints = np.array(waypoints)  # [horizon+1, 7]
        
        # Build timepoints: [0, dt, 2*dt, ..., horizon*dt]
        timepoints = np.array([i * self.dt for i in range(horizon + 1)])
        
        # Create interpolator (using PchipInterpolator or CubicSpline for smooth interpolation)
        if self.interpolator_option == 'Pchip':
            interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)
            interpolator_feedforward_velocity = interpolator_position.derivative()
        elif self.interpolator_option == 'Cubic':
            interpolator_position = CubicSpline(timepoints, waypoints, axis=0, bc_type='natural')
            interpolator_feedforward_velocity = interpolator_position.derivative()
        else:
            interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)
            interpolator_feedforward_velocity = interpolator_position.derivative()
        
        # Lists to collect data at each action (horizon length)
        timepoint_target_positions = []
        timepoint_target_orientations = []
        timepoint_current_ee_poses = []
        timepoint_current_cam_poses = []
        timepoint_position_errors = []
        timepoint_orientation_errors = []
        timepoint_joint_directions = []
        timepoint_actual_joint_speeds = []
        timepoint_gripper_speeds = []
        
        # Execute jointCtrlCmd with interpolator
        self.last_step_start_time = start_time = time.time()
        total_duration = horizon * self.dt
        end_time = start_time + total_duration
        
        # Track which timepoints have been sampled (sample at dt, 2*dt, ..., (horizon-1)*dt)
        sampled_timepoints = set()
        target_timepoints = [i * self.dt for i in range(1, horizon)]  # [dt, 2*dt, ..., (horizon-1)*dt]
        # last timepoint is computed after while loop
        
        sleep_time_list = []
        iter = 0
        
        # Blocking execution: run interpolation loop
        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            
            # Evaluate interpolator at current time
            eval_time = np.clip(current_time, timepoints[0], timepoints[-1])
            interpolated_positions = interpolator_position(eval_time)
            interpolated_velocities = interpolator_feedforward_velocity(eval_time) * self.custom_speed_factor
            
            # Extract interpolated joint positions, velocities and gripper position
            interpolated_joint_pos = interpolated_positions[:6]
            interpolated_joint_vel = interpolated_velocities[:6]
            interpolated_gripper_pos = interpolated_positions[6] if self.has_gripper else 0.0
            interpolated_gripper_vel = interpolated_velocities[6] if self.has_gripper else 0.0
            
            joint_direction = np.array([interpolated_joint_vel[0], 
                                        interpolated_joint_vel[1], 
                                        interpolated_joint_vel[2], 
                                        interpolated_joint_vel[3], 
                                        interpolated_joint_vel[4], 
                                        interpolated_joint_vel[5], 
                                        interpolated_gripper_vel])
            
            iter += 1
            self.arm.jointCtrlCmd(joint_direction, 1.0)  # qd = direction*jointSpeed
            sleep_start = time.time()
            time.sleep(self.arm._ctrlComp.dt)
            sleep_time = time.time() - sleep_start
            sleep_time_list.append(sleep_time)
            
            # Sample at each target timepoint (dt, 2*dt, ..., horizon*dt)
            for i, target_time in enumerate(target_timepoints):
                if target_time not in sampled_timepoints and current_time >= target_time:
                    # Mark this timepoint as sampled
                    sampled_timepoints.add(target_time)
                    
                    # Update state to get current values at this timepoint
                    self._update_state()
                    current_ee_pose = self._get_current_ee_pose()
                    current_pos_after = current_ee_pose[:3]
                    current_quat_after = current_ee_pose[3:7]
                    
                    # Get target for this timepoint
                    target_pos = target_positions_list[i]
                    target_orient = target_orientations_list[i]
                    
                    # Calculate errors
                    position_error = np.linalg.norm(target_pos - current_pos_after)
                    orientation_error = self._quaternion_distance(target_orient, current_quat_after)
                    
                    # Calculate joint directions for info (using interpolated velocities at this timepoint)
                    target_eval_time = np.clip(target_time, timepoints[0], timepoints[-1])
                    target_interpolated_velocities = interpolator_feedforward_velocity(target_eval_time) * self.custom_speed_factor
                    target_interpolated_gripper_vel = target_interpolated_velocities[6] if self.has_gripper else 0.0
                    joint_directions = np.array([
                        target_interpolated_velocities[0],
                        target_interpolated_velocities[1],
                        target_interpolated_velocities[2],
                        target_interpolated_velocities[3],
                        target_interpolated_velocities[4],
                        target_interpolated_velocities[5],
                        target_interpolated_gripper_vel
                    ])
                    actual_joint_speed = np.linalg.norm(target_interpolated_velocities[:6])
                    gripper_speed = abs(target_interpolated_gripper_vel)
                    
                    # Store collected data
                    timepoint_target_positions.append(target_pos.copy())
                    timepoint_target_orientations.append(target_orient.copy())
                    timepoint_current_ee_poses.append(self._get_current_ee_pose_for_obs().copy())
                    timepoint_current_cam_poses.append(self._get_current_camera_pose_for_obs().copy())
                    timepoint_position_errors.append(position_error)
                    timepoint_orientation_errors.append(orientation_error)
                    timepoint_joint_directions.append(joint_directions.copy())
                    timepoint_actual_joint_speeds.append(actual_joint_speed)
                    timepoint_gripper_speeds.append(gripper_speed)
                    
                    # Store intermediate result for this action index (i-th action in chunk, 0-indexed)
                    action_index = i
                    self.chunk_intermediate_results[action_index] = {
                        'target_position': target_pos.copy(),
                        'target_orientation': target_orient.copy(),
                        'current_ee_pose': self._get_current_ee_pose_for_obs().copy(),
                        'current_cam_pose': self._get_current_camera_pose_for_obs().copy(),
                        'position_error': position_error,
                        'orientation_error': orientation_error,
                        'target_poses': T_bc_target.copy(),
                        'joint_directions': joint_directions.copy(),
                        'actual_joint_speed': actual_joint_speed,
                        'gripper_speed': gripper_speed,
                    }
                    self.chunk_completed_actions.add(action_index)
        
        cmd_time = time.time() - start_time
        # print("cmd time: ", cmd_time)
        # print("sleep time mean:", np.array(sleep_time_list).mean())
        # print("sleep time std: ", np.array(sleep_time_list).std())
        # print("sleep time max: ", np.array(sleep_time_list).max())
        # print("sleep time min: ", np.array(sleep_time_list).min())
        
        # Final state update
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos_after = current_ee_pose[:3]
        current_quat_after = current_ee_pose[3:7]
        
        # Calculate errors for final timepoint (last action)
        position_error = np.linalg.norm(self.target_position - current_pos_after)
        orientation_error = self._quaternion_distance(self.target_orientation, current_quat_after)
        
        # Calculate joint directions for final timepoint
        final_eval_time = np.clip(total_duration, timepoints[0], timepoints[-1])
        final_interpolated_velocities = interpolator_feedforward_velocity(final_eval_time) * self.custom_speed_factor
        final_interpolated_gripper_vel = final_interpolated_velocities[6] if self.has_gripper else 0.0
        joint_directions = np.array([
            final_interpolated_velocities[0],
            final_interpolated_velocities[1],
            final_interpolated_velocities[2],
            final_interpolated_velocities[3],
            final_interpolated_velocities[4],
            final_interpolated_velocities[5],
            final_interpolated_gripper_vel
        ])
        actual_joint_speed = np.linalg.norm(final_interpolated_velocities[:6])
        gripper_speed = abs(final_interpolated_gripper_vel)
        
        # Store final timepoint data
        timepoint_target_positions.append(self.target_position.copy())
        timepoint_target_orientations.append(self.target_orientation.copy())
        timepoint_current_ee_poses.append(self._get_current_ee_pose_for_obs().copy())
        timepoint_current_cam_poses.append(self._get_current_camera_pose_for_obs().copy())
        timepoint_position_errors.append(position_error)
        timepoint_orientation_errors.append(orientation_error)
        timepoint_joint_directions.append(joint_directions.copy())
        timepoint_actual_joint_speeds.append(actual_joint_speed)
        timepoint_gripper_speeds.append(gripper_speed)
        
        # Store final action result (last action in chunk, index = horizon - 1)
        final_action_index = horizon - 1
        self.chunk_intermediate_results[final_action_index] = {
            'target_position': self.target_position.copy(),
            'target_orientation': self.target_orientation.copy(),
            'current_ee_pose': self._get_current_ee_pose_for_obs().copy(),
            'current_cam_pose': self._get_current_camera_pose_for_obs().copy(),
            'position_error': position_error,
            'orientation_error': orientation_error,
            'target_poses': T_bc_target.copy(),
            'joint_directions': joint_directions.copy(),
            'actual_joint_speed': actual_joint_speed,
            'gripper_speed': gripper_speed,
        }
        self.chunk_completed_actions.add(final_action_index)
        
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        
        # Create info dictionary with horizon-length arrays
        info = {
            'fsm_state': self.arm.getCurrentState(),
            'target_position': np.array(timepoint_target_positions),  # [horizon, 3]
            'target_orientation': np.array(timepoint_target_orientations),  # [horizon, 4]
            'current_ee_pose': np.array(timepoint_current_ee_poses),  # [horizon, 8]
            'current_cam_pose': np.array(timepoint_current_cam_poses),  # [horizon, 8]
            'position_error': np.array(timepoint_position_errors),  # [horizon]
            'orientation_error': np.array(timepoint_orientation_errors),  # [horizon]
            'joint_directions': np.array(timepoint_joint_directions),  # [horizon, 7]
            'actual_joint_speed': np.array(timepoint_actual_joint_speeds),  # [horizon]
            'gripper_speed': np.array(timepoint_gripper_speeds),  # [horizon]
            'horizon': horizon,
        }
        
        # Store result for compatibility with get_step_result()
        result = (observation, reward, done, info)
        self.last_step_result = result

        # Mark as complete for blocking execution
        self.last_step_complete = True
        self.last_step_start_time = None
        
        self.episode_step += 1
        return observation, reward, done, info
    
    def compute_target_pos(self, target_position: np.ndarray, target_orientation: np.ndarray, 
                            last_successful_joint_pos: Optional[np.ndarray] = None, return_success: bool = False, ik_type: Optional[str] = None ): # tolerance=5e-3, tolerance_null=1e-3
        """
        Calculate target joint positions based on target and current poses using IK.
        Uses inverse kinematics to convert target pose to joint commands.
        
        Args:
            target_position: Target end-effector position
            target_orientation: Target end-effector orientation (quaternion in [x,y,z,w] format)
            tolerance: Position tolerance for IK convergence
            tolerance_null: Null objective tolerance for IK convergence
            last_successful_joint_pos: Optional last successful joint positions (used when IK fails in step_chunk)
            return_success: If True, return (target_joint_pos, success) tuple instead of just target_joint_pos
            
        Returns:
            target_joint_pos: Target joint positions (6-element array), or (target_joint_pos, success) if return_success=True
        """
        
        # Apply T_E_C transformation if needed
        if self.T_E_C is not None:
            # assume input action is T_bc
            T_bc = np.eye(4)
            T_bc[:3, :3] = R.from_quat(target_orientation).as_matrix()
            T_bc[:3, 3] = target_position
            T_be = T_bc @ np.linalg.inv(self.T_E_C)
            target_position = T_be[:3, 3]
            target_orientation = R.from_matrix(T_be[:3, :3]).as_quat()


        # Convert target pose to transformation matrix
        target_T = self._pose_to_transformation_matrix(target_position, target_orientation)
        
        # Get current joint positions
        current_joint_pos = self.current_joint_pos
        if ik_type is None:
            ik_type = self.ik_type
        # Use inverse kinematics to get target joint positions
        if ik_type == 'null_space':
            success, target_joint_pos, iterations, final_error_pos, null_obj_val = self.solve_ik_null_space(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance=1e-2,
                tolerance_null=1e-3,
            )
            final_error_ori = 0.0
        elif ik_type == 'jacobian':
            success, target_joint_pos, iterations, final_error_pos, final_error_ori, null_obj_val  = self.solve_ik_6d_dls_jacobian(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance_pos=1e-2,
                tolerance_ori=2e-2,
                w_pos=1.0,
                w_ori=0.1,
                lambda0=1e-3,
                use_adaptive_damping=True,
                damp_gain=1e-2,
                alpha_init=1.0,
            )
        else:
            raise ValueError(f"Invalid IK type: {self.ik_type}")
        
        if not success:
            print(f"Z1 Warning: IK failed to converge (error_pos: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            if self.use_current_joint_pos_when_ik_fails:
                # If last_successful_joint_pos is provided (from step_chunk), use it instead of current_joint_pos
                if last_successful_joint_pos is not None:
                    target_joint_pos = last_successful_joint_pos.copy()
                else:
                    target_joint_pos = current_joint_pos.copy()
            else:
                # Use the IK result even if not fully converged
                print(f"Z1 IK fallback: using current q from solver (curr_err_pos={final_error_pos:.6f}, curr_null={null_obj_val:.6f})")
                self.prev_final_error_pos = final_error_pos
                self.prev_final_error_ori = 0.0  # Not computed in null_space IK
                self.prev_null_obj_val = null_obj_val
        else:
            print(f"Z1 IK solved successfully in {iterations} iterations (error: {final_error_pos:.6f}, null_obj: {null_obj_val:.6f})")
            # On success, update previous metrics to current
            self.prev_final_error_pos = final_error_pos
            self.prev_final_error_ori = 0.0  # Not computed in null_space IK
            self.prev_null_obj_val = null_obj_val
        
        # Remember chosen target q for next iteration's comparison
        self.prev_target_joint_pos = target_joint_pos.copy()
        
        if return_success:
            return target_joint_pos, success
        else:
            return target_joint_pos

    def set_current_sequence(self, sequence: np.ndarray, start_index: int = 0):
        """
        Set the current target pose sequence for RTC-style execution.
        
        Args:
            sequence: Target pose sequence [sequence_length, 8]
            start_index: Index to start from in the sequence (for RTC-style skipping past poses)
        """
        self.current_sequence = sequence
        self.sequence_index = start_index
        self.sequence_step_count = 0
        print(f"RTC: Set new sequence with shape {sequence.shape}, starting from index {start_index}")
    
    def get_current_sequence(self) -> np.ndarray:
        """
        Get the current sequence.
        
        Returns:
            sequence: Current sequence
        """
        return self.current_sequence.copy()
    
    def get_next_action_from_sequence(self) -> np.ndarray:
        """
        Get the next action from the current sequence.
        
        Returns:
            action: Next action from sequence, or None if sequence is exhausted
        """
        if self.current_sequence is None:
            return None
        
        if self.sequence_index < len(self.current_sequence):
            action = self.current_sequence[self.sequence_index]
            self.sequence_index += 1
            self.sequence_step_count += 1
            print(f"RTC: Using sequence index {self.sequence_index-1}/{len(self.current_sequence)-1}")
            return action
        else:
            print("RTC: Sequence exhausted")
            return None
    
    def has_sequence_available(self) -> bool:
        """Check if there's a sequence available and not exhausted."""
        return self.current_sequence is not None and self.sequence_index < len(self.current_sequence)
    
    def is_step_complete(self) -> bool:
        """
        Check if the step execution is complete.
        For compatibility with Z1 environment interface.
        
        Returns:
            True if step is complete, False otherwise
        """
        # For blocking execution, always return True
        if self.last_step_complete:
            return True
        else:
            return False

        # # For non-blocking execution, check if dt has passed
        # if self.last_step_start_time is not None:
        #     elapsed_time = time.time() - self.last_step_start_time
        #     if elapsed_time >= self.dt:
        #         self.last_step_complete = True
        #         return True
        #     return False
        
        # # Default to complete if no step has been executed
        # return True
    
    def get_step_result(self) -> Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Get the result from the step execution if it's complete.
        For compatibility with Z1 environment interface.
        
        Returns:
            Step result tuple if complete, None if still running or no result available
        """
        if not self.is_step_complete():
            return None
        # print("in get step result, last_step_result: ", self.last_step_result)
        return self.last_step_result
    
    def is_action_in_chunk_complete(self, action_index: int) -> bool:
        """
        Check if a specific action within the current chunk has been completed.
        This allows external processes to poll for completion of individual actions
        within a chunk, enabling observation collection after each action.
        
        Args:
            action_index: Index of the action within the chunk (0-indexed)
            
        Returns:
            True if the action at action_index has been completed, False otherwise
        """
        return action_index in self.chunk_completed_actions
    
    def get_action_in_chunk_intermediate_result(self, action_index: int) -> Optional[Dict[str, Any]]:
        """
        Get the intermediate result for a specific action within the current chunk.
        This allows external processes to retrieve observation data (EE pose, etc.)
        after each individual action completes, even when using chunked execution.
        
        Args:
            action_index: Index of the action within the chunk (0-indexed)
            
        Returns:
            Dictionary containing intermediate result data for the action, or None if not available.
            The dictionary contains:
                - 'target_position': Target position for this action [3]
                - 'target_orientation': Target orientation for this action [4]
                - 'current_ee_pose': Current EE pose after this action [8]
                - 'position_error': Position error for this action
                - 'orientation_error': Orientation error for this action
                - 'joint_directions': Joint directions for this action [7]
                - 'actual_joint_speed': Actual joint speed for this action
                - 'gripper_speed': Gripper speed for this action
        """
        if action_index not in self.chunk_intermediate_results:
            return None
        
        result = self.chunk_intermediate_results[action_index].copy()
        return result
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including joint states and end-effector pose."""
        current_ee_pose = self._get_current_ee_pose()
        
        observation = np.concatenate([
            self.current_joint_pos,           # 6D joint positions
            self.current_joint_vel,           # 6D joint velocities
            current_ee_pose[:3],              # 3D end-effector position
            current_ee_pose[3:7],             # 4D end-effector quaternion [x,y,z,w]
            np.array([self.current_gripper_pos]),  # 1D gripper position
            np.array([self.current_gripper_vel])   # 1D gripper velocity
        ])
        
        return observation.astype(np.float32)
    
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        return self.episode_step >= self.max_episode_steps
    
    def _get_current_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose (position + quaternion in [x,y,z,w] format)."""
        if self.fk_debug:
            T = self.compute_forward_kinematics(self.current_joint_pos)
        else:
            # Get current transformation matrix
            T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)

        
        
        # Extract position
        position = T[:3, 3]
        
        # Extract quaternion from rotation matrix
        quaternion = self._rotation_matrix_to_quaternion(T[:3, :3])
        
        # Get gripper position
        gripper_pos = self.current_gripper_pos if self.has_gripper else 0.0
        
        return np.concatenate([position, quaternion, [gripper_pos]])
    
    def _get_current_ee_pose_for_obs(self) -> np.ndarray:
        ee_pose = self._get_current_ee_pose()
        raw_gripper_state = ee_pose[-1].copy()
        if raw_gripper_state < -0.6: # opened more than half-open
            gripper_state = 0.0 # open for hand data representation
        else: # opened less than half-open
            gripper_state = 1.0 # closed for hand data representation

        return np.concatenate([ee_pose[:-1], [gripper_state]])
    
    def _get_current_camera_pose_for_obs(self) -> np.ndarray:
        if self.fk_debug:
            T_be = self.compute_forward_kinematics(self.current_joint_pos)
            # print("@@@@@ DEBUG: T_be pose is computed by forwardKinematics, since we compute T_E_C by computing forwardkinematics")
            # T_be = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        else:
            T_be = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        T_ec = self.T_E_C.copy()
        T_bc = T_be @ T_ec

        # Extract position
        position = T_bc[:3, 3]
        
        # Extract quaternion from rotation matrix
        quaternion = self._rotation_matrix_to_quaternion(T_bc[:3, :3])
        
        return np.concatenate([position, quaternion])

        # return T_bc.copy()


    def _get_current_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        if self.fk_debug:
            T = self.compute_forward_kinematics(self.current_joint_pos)
        else:
            T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        return T[:3, 3]
    
    def _get_current_ee_se3(self, joint_pos = None) -> np.ndarray:
        """Get current end-effector SE(3) matrix."""
        if joint_pos is None:
            joint_pos = self.current_joint_pos
        if self.fk_debug:
            T = self.compute_forward_kinematics(joint_pos)
        else:
            T = self.arm_model.forwardKinematics(joint_pos, 6)
        
        return T
    
    def jacobian_position(self, q):
        """Calculate position Jacobian using numerical differentiation."""
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        T = self._get_current_ee_se3(joint_pos=q)
        p = T[:3, 3]
        jac = np.zeros([3, 6])
        for i in range(6):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            T_ = self._get_current_ee_se3(joint_pos=q_)
            p_ = T_[:3, 3]
            jac[:, i] = (p_ - p)*epsilon_inv
        return jac
    
    def solve_ik_null_space(self, target_T, initial_guess=None, max_iterations=100, tolerance=1e-2, tolerance_null=1e-5, epsilon=1e-6):
        """
        Solve IK using pseudo-inverse with null-space approach.
        
        Args:
            target_T: Target 4x4 transformation matrix
            initial_guess: Initial joint angle guess
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for position error
            tolerance_null: Convergence tolerance for null objective
            epsilon: Small value for numerical differentiation
            
        Returns:
            tuple: (success, joint_angles, iterations, final_error, null_obj_val)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        q = initial_guess.copy()
        
        # Initialize null objective with desired rotation (default: identity)
        target_SO3 = target_T[:3, :3]
        null_obj = SO3Constraint(target_SO3)
        
        iter_taken = 0
        
        while True:
            # Compute current forward kinematics
            current_T = self._get_current_ee_se3(joint_pos=q)
            
            # Compute position error only (like in the original code)
            pos_error = target_T[:3, 3] - current_T[:3, 3]
            err = np.linalg.norm(pos_error)
            
            # Compute null objective value
            current_SO3 = current_T[:3, :3]
            null_obj_val = null_obj.evaluate(current_SO3)

            # Compute Jacobian
            J = self.jacobian_position(q)
            
            # Check convergence: both position error and null objective must be satisfied
            if (err < tolerance and null_obj_val < tolerance_null) or iter_taken >= max_iterations:
                break
            else:
                iter_taken += 1
            
            # Pseudo-inverse approach
            J_dagger = np.linalg.pinv(J)
            J_null = np.eye(6) - J_dagger @ J  # null space of Jacobian
            
            # Compute null objective gradient using numerical differentiation
            phi = np.zeros(6)
            
            for i in range(6):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                # Apply joint limits to perturbed configuration
                q_perturb = np.clip(q_perturb, [limit[0] for limit in self.joint_limits], [limit[1] for limit in self.joint_limits])
                
                perturb_T = self._get_current_ee_se3(joint_pos=q_perturb)
                perturb_SO3 = perturb_T[:3, :3]
                null_obj_val_perturb = null_obj.evaluate(perturb_SO3)
                phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon
            
            # Update using pseudo-inverse + null-space approach
            # delta_x = ee_pos - x (position error)
            delta_x = pos_error
            delta_q = J_dagger @ delta_x - J_null @ phi
            q = q + delta_q
            
            # Apply joint limits
            q = np.clip(q, [limit[0] for limit in self.joint_limits], [limit[1] for limit in self.joint_limits])
        
        # Final error check (position error only, like in original code)
        current_T = self._get_current_ee_se3(joint_pos=q)
        final_error = err  # np.linalg.norm(final_pos_error)
        
        # Check if both conditions are satisfied for success
        success = (final_error < tolerance and null_obj_val < tolerance_null)
        
        return success, q, iter_taken, final_error, null_obj_val


    # jacobian ver
    # 간단한 유틸: 현재 오차/코스트 계산
    def pose_errors_and_cost(self, q_vec, target_T, w_pos, w_ori):
        T = self._get_current_ee_se3(joint_pos=q_vec)
        p = T[:3, 3]
        R_cur = T[:3, :3]
        p_d = target_T[:3, 3]
        R_d = target_T[:3, :3]

        # 위치/자세 오차
        e_p = p_d - p
        # 공간 프레임 오차: e_o = Log(R_d R^T)^\vee
        R_e = R_d @ R_cur.T
        e_o = R.from_matrix(R_e).as_rotvec()

        # SO3Constraint와 동일 척도(작을수록 좋음)
        null_obj_val = 0.5 * (3.0 - np.trace(R_cur @ R_d.T))

        # 6D 오차 벡터와 가중 코스트
        e6 = np.concatenate([w_pos * e_p, w_ori * e_o])
        cost = 0.5 * (e6 @ e6)
        return e_p, e_o, null_obj_val, cost

    # 적응 감쇠 계산(특이값/조작도 기반)
    def compute_lambda(self, J, lambda0, use_adaptive_damping, damp_gain):
        lam = lambda0
        if use_adaptive_damping:
            # 특이값 기반(최소 특이값이 작으면 감쇠 증가)
            s = np.linalg.svd(J, compute_uv=False)
            s_min = float(np.min(s)) if s.size > 0 else 0.0
            lam = lambda0 + damp_gain / (s_min + 1e-6)
        return lam

    def solve_ik_6d_qp(
        self,
        target_T: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 80,
        tolerance_pos: float = 1e-2,
        tolerance_ori: float = 2e-2,
        Kp: float = 3.0,          # 위치 게인 [1/s]
        Ko: float = 3.0,          # 자세 게인 [1/s]
        sat_w: float = 2.0,       # 각속도 포화 [rad/s]
        sat_v: float = 0.5,       # 선속도 포화 [m/s]
        dq_inf_cap: float = 0.25  # per-iter |dq|_inf 제한(라디안)
    ):
        """
        Unitree arm_model.solveQP를 이용한 6D 해결율 IK.
        반환형: (success, q, iter_taken, final_error_pos, null_obj_val)
        """
        assert target_T.shape == (4, 4)
        q = (self.current_joint_pos.copy() if initial_guess is None else initial_guess.copy())

        dt_qp = float(getattr(self.arm._ctrlComp, "dt", 0.002))  # SDK 내부 dt 사용 권장

        def errors(q_vec):
            T = self._get_current_ee_se3(joint_pos=q_vec)
            p, R_cur = T[:3, 3], T[:3, :3]
            p_d, R_d = target_T[:3, 3], target_T[:3, :3]
            e_p = p_d - p
            R_e = R_d @ R_cur.T                 # space 기준
            e_o = R.from_matrix(R_e).as_rotvec()
            null_val = 0.5 * (3.0 - np.trace(R_cur @ R_d.T))
            return e_p, e_o, null_val

        e_p, e_o, null_obj_val = errors(q)
        it = 0
        success = False

        while it < max_iterations:
            if (np.linalg.norm(e_p) < tolerance_pos) and (np.linalg.norm(e_o) < tolerance_ori):
                success = True
                break

            # 원하는 트위스트 ( [ω; v] 순서 ) + 포화
            Vdes = np.concatenate([Ko * e_o, Kp * e_p]).astype(float)
            Vdes[:3] = np.clip(Vdes[:3], -sat_w, sat_w)
            Vdes[3:] = np.clip(Vdes[3:], -sat_v, sat_v)

            # QP로 qd 풀기 (SDK 시그니처: solveQP(twist, q_near, dt))
            try:
                qd = np.array(self.arm_model.solveQP(Vdes, q, dt_qp), dtype=float).reshape(6)
            except Exception:
                # 혹시 실패 시 최소자승 대체 (매우 드뭄)
                J = np.asarray(self.arm_model.CalcJacobian(q)).reshape(6, 6).astype(float)
                qd = np.linalg.lstsq(J, Vdes, rcond=None)[0]
                print('######## Least QP fails. Least squares solution used instead.')

            # 스텝 적용
            dq = qd * dt_qp

            # per-iter |dq|_inf 제한(너무 큰 스텝 방지)
            infn = np.linalg.norm(dq, ord=np.inf)
            if infn > dq_inf_cap:
                dq *= (dq_inf_cap / (infn + 1e-12))

            q = q + dq

            # 안전망 클립(대부분 필요 없음; QP가 제약 처리)
            q = np.clip(
                q,
                np.array([lim[0] for lim in self.joint_limits]),
                np.array([lim[1] for lim in self.joint_limits]),
            )

            e_p, e_o, null_obj_val = errors(q)
            it += 1

        final_error_pos = float(np.linalg.norm(e_p))
        return success, q, it, final_error_pos, float(null_obj_val)

    def solve_ik_6d_dls_jacobian(
        self,
        target_T: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance_pos: float = 1e-2,     # 위치 수렴 허용 (m)
        tolerance_ori: float = 2e-2,     # 자세 수렴 허용 (rad; 회전벡터 노름)
        w_pos: float = 1.0,              # 위치 가중
        w_ori: float = 1.0,              # 자세 가중
        lambda0: float = 1e-3,           # 기본 감쇠
        use_adaptive_damping: bool = True,
        damp_gain: float = 1e-2,         # 적응 감쇠 이득
        alpha_init: float = 1.0,         # 초기 라인서치 스텝
        joint_clip: bool = True
    ):
        """
        Solve IK using 6D Damped Least Squares with the robot's spatial Jacobian.

        Args mirror solve_ik_null_space as much as possible. Returns:
            tuple: (success, q, iter_taken, final_error_pos, null_obj_val)
                - success: both position & orientation tolerances satisfied
                - q: solved joint angles
                - iter_taken: iterations used
                - final_error_pos: ||p_d - p(q)||
                - null_obj_val: 0.5*(3 - trace(R_current * R_des^T))  (SO3Constraint와 동일 척도)
        """
        assert target_T.shape == (4, 4), "target_T must be 4x4 SE(3) matrix"

        # 초기 q 설정
        if initial_guess is None:
            q = self.current_joint_pos.copy()
        else:
            q = initial_guess.copy()

        # 가중 행렬 W (6x6, 위치/자세 분리 가중)
        W = np.diag([w_pos, w_pos, w_pos, w_ori, w_ori, w_ori]).astype(float)

        

        

        iter_taken = 0
        success = False

        # 초기 에러/코스트
        e_p, e_o, null_obj_val, cost = self.pose_errors_and_cost(q, target_T, w_pos, w_ori)

        while iter_taken < max_iterations:
            # 수렴 체크
            if (np.linalg.norm(e_p) < tolerance_pos) and (np.linalg.norm(e_o) < tolerance_ori):
                success = True
                break

            # 6x6 공간 야코비안
            J = np.asarray(self.arm_model.CalcJacobian(q)).reshape(6, 6).astype(float)

            # 6D 오차 벡터 (가중은 정규방정식에서 W로 처리해도 되지만,
            # 여기서는 정규방정식에 W를 명시적으로 넣음)
            # e6 = np.concatenate([e_p, e_o])
            
            # 방법 A: e6/W 순서를 J에 맞춰 바꿈
            e6 = np.concatenate([e_o, e_p])  # 각속도 먼저
            W  = np.diag([w_ori, w_ori, w_ori, w_pos, w_pos, w_pos])

            # DLS 정규방정식
            lam = self.compute_lambda(J, lambda0, use_adaptive_damping, damp_gain)
            
            
            
            # A = J.T @ W @ J + (lam ** 2) * np.eye(6)
            # b = J.T @ W @ e6

            # try:
            #     dq = np.linalg.solve(A, b)
            # except np.linalg.LinAlgError:
            #     # 드물게 A가 수치적으로 불량하면 안정적 대안
            #     dq = np.linalg.lstsq(A, b, rcond=None)[0]

            



            # 2) 액티브-셋 DLS로 dq 구하기
            dq = self._dls_step_with_active_set(
                J, e6, W, lam, q,
                np.array([l[0] for l in self.joint_limits]),
                np.array([l[1] for l in self.joint_limits]),
                margin=0.03  # 0.02~0.05 rad 권장
            )

            # 3) 라인서치 + (필요시) dq 크기 제한
            dq_norm = np.linalg.norm(dq, ord=np.inf)
            if dq_norm > 0.2:   # per-iter step cap (예시)
                dq *= (0.2 / dq_norm + 1e-12)




            # 간단한 백트래킹 라인서치(발산 방지)
            alpha = alpha_init
            q_candidate = q + alpha * dq

            if joint_clip:
                q_candidate = np.clip(
                    q_candidate,
                    [lim[0] for lim in self.joint_limits],
                    [lim[1] for lim in self.joint_limits],
                )

            # _, _, _, cost_new = self.pose_errors_and_cost(q_candidate, target_T, w_pos, w_ori)

            # # 개선되지 않으면 스텝 줄이기(최대 3회)
            # backtrack = 0
            # while cost_new > cost and backtrack < 3:
            #     alpha *= 0.5
            #     q_candidate = q + alpha * dq
            #     if joint_clip:
            #         q_candidate = np.clip(
            #             q_candidate,
            #             [lim[0] for lim in self.joint_limits],
            #             [lim[1] for lim in self.joint_limits],
            #         )
            #     _, _, _, cost_new = self.pose_errors_and_cost(q_candidate, target_T, w_pos, w_ori)
            #     backtrack += 1

            # 스텝 적용
            q = q_candidate
            e_p, e_o, null_obj_val, cost = self.pose_errors_and_cost(q, target_T, w_pos, w_ori)
            iter_taken += 1

        # 최종 에러 메트릭
        final_error_pos = np.linalg.norm(e_p)
        final_error_ori = np.linalg.norm(e_o)

        return success, q, iter_taken, final_error_pos, final_error_ori, float(null_obj_val)

    def _dls_step_with_active_set(self, J, e6, W, lam, q, qmin, qmax, margin=0.02):
        """
        박스 제약(조인트 한계)을 고려한 한 스텝 DLS.
        - 한계 근처(margin)에서 그 한계 쪽으로 더 가려는 관절은 '잠금' 처리(Δq_i=0)
        - '잠금' 관절은 J의 해당 열을 제외하고 남은 자유도로 최소자승을 다시 풉니다.
        """
        n = 6
        free = np.ones(n, dtype=bool)
        # 한계에 바짝 붙어있는 관절은 먼저 후보로 체크
        near_low  = (q - qmin) < margin
        near_high = (qmax - q) < margin

        for _ in range(n):  # 최악의 경우 관절 6개 모두 잠글 수 있으니 n회 이내 종료
            Jf = J[:, free]
            Af = Jf.T @ W @ Jf + (lam**2) * np.eye(np.sum(free))
            bf = Jf.T @ W @ e6

            try:
                dqf = np.linalg.solve(Af, bf)
            except np.linalg.LinAlgError:
                dqf = np.linalg.lstsq(Af, bf, rcond=None)[0]

            dq = np.zeros(n)
            dq[free] = dqf

            q_try = q + dq

            # 한계 침범을 유발하는 관절을 잠금 후보로
            lock_low  = (q_try < qmin) & (dq < 0)
            lock_high = (q_try > qmax) & (dq > 0)
            # 또는 한계에 가까운데 그쪽으로 더 가려는 경우
            lock_low  |= near_low  & (dq < 0)
            lock_high |= near_high & (dq > 0)

            lock = (lock_low | lock_high) & free
            if not np.any(lock):
                return dq  # 제약 위반 없으면 스텝 확정

            # 잠금 적용하고 다시 풉니다
            free[lock] = False

        # 모든 관절을 잠갔다면 0 스텝
        return np.zeros(n)



    def _get_current_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as quaternion in [x,y,z,w] format."""
        if self.fk_debug:
            T = self.compute_forward_kinematics(self.current_joint_pos)
        else:
            T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        return self._rotation_matrix_to_quaternion(T[:3, :3])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix using scipy."""
        # q is already in [x,y,z,w] format, use directly
        rotation = R.from_quat(q)
        return rotation.as_matrix()
    
    def _rotation_matrix_to_quaternion(self, R_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion using scipy."""
        rotation = R.from_matrix(R_matrix)
        return rotation.as_quat()  # scipy returns [x,y,z,w] format
    
    def _pose_to_transformation_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix using scipy."""
        T = np.eye(4)
        T[:3, :3] = self._quaternion_to_rotation_matrix(quaternion)
        T[:3, 3] = position
        return T
    
    
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length using scipy."""
        # q is already in [x,y,z,w] format, use directly
        rotation = R.from_quat(q)
        # scipy automatically normalizes quaternions
        return rotation.as_quat()
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate distance between two quaternions using scipy."""
        # q1 and q2 are already in [x,y,z,w] format, use directly
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        
        # Calculate relative rotation and get magnitude
        relative_rotation = r2 * r1.inv()
        return np.linalg.norm(relative_rotation.as_rotvec())
    
    def _quaternion_error(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Calculate angular velocity error between target and current quaternions using scipy."""
        # q_target and q_current are already in [x,y,z,w] format, use directly
        r_target = R.from_quat(q_target)
        r_current = R.from_quat(q_current)
        
        # Calculate relative rotation: r_error = r_target * r_current^-1
        r_error = r_target * r_current.inv()
        
        # Convert to rotation vector (angular velocity)
        angular_velocity = r_error.as_rotvec()
        
        return angular_velocity
    
    def _calculate_joint_directions(self, target_position, target_orientation, target_gripper, 
                                  current_pos, current_quat, current_gripper_pos, 
                                  control_time, ik_type='jacobian'):
        """
        Calculate joint directions based on target and current poses.
        Uses inverse kinematics to convert target pose to joint commands.
        
        Args:
            target_position: Target end-effector position
            target_orientation: Target end-effector orientation (quaternion in [x,y,z,w] format)
            target_gripper: Target gripper position
            current_pos: Current end-effector position
            current_quat: Current end-effector orientation (quaternion in [x,y,z,w] format)
            current_gripper_pos: Current gripper position
            control_time: Time available for control
            
        Returns:
            joint_directions: [J1, J2, J3, J4, J5, J6, gripper] directions
            actual_joint_speed: Actual joint speed used
            gripper_speed: Actual gripper speed used
        """
        # Convert target pose to transformation matrix
        target_T = self._pose_to_transformation_matrix(target_position, target_orientation)
        
        # Get current joint positions
        current_joint_pos = self.current_joint_pos
        
        # Use inverse kinematics to get target joint positions
        # Try to solve IK for target pose using solve_ik_null_space
        if ik_type == 'null_space':
            # NOTE: validated for simple trajectory tracking
            success, target_joint_pos, iterations, final_error_pos, null_obj_val = self.solve_ik_null_space(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance=1e-2,
                tolerance_null=1e-2
            )
            final_error_ori = 0.0 # temporary value for debug
            
        elif ik_type == 'jacobian':
            # NOTE: validated for simple trajectory tracking
            success, target_joint_pos, iterations, final_error_pos, final_error_ori, null_obj_val = self.solve_ik_6d_dls_jacobian(
                target_T, 
                initial_guess=current_joint_pos,
                max_iterations=50,
                tolerance_pos=1e-2,
                tolerance_ori=2e-2,
                w_pos=1.0,
                w_ori=0.1,
                lambda0=1e-3,
                use_adaptive_damping=True,
                damp_gain=1e-2,
                alpha_init=1.0,
                joint_clip=True,
            )
        
        # NOTE: currently doesn't work for very simple trajectory tracking
        # success, target_joint_pos, iterations, final_error_pos, null_obj_val = self.solve_ik_6d_qp(
        #     target_T, 
        #     initial_guess=current_joint_pos,
        #     max_iterations=50,
        #     tolerance_pos=1e-2,
        #     tolerance_ori=2e-2,
        #     Kp=3.0,
        #     Ko=3.0,
        # )
        # final_error_ori = 0.0 # temporary value for debug
        

        if not success:
            print(f"Warning: Z1 IK failed to converge (error_pos: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            if self.use_current_joint_pos_when_ik_fails:
                target_joint_pos = current_joint_pos.copy()
                # Do not update prev metrics here since we didn't select a solver-produced q
            else:
                # Compare using previous stored (final_error, null_obj_val) vs current iteration's
                use_prev = False
                # if self.prev_final_error is not None and self.prev_null_obj_val is not None and self.prev_target_joint_pos is not None:
                #     # Lexicographic comparison: prioritize final_error, then null_obj_val
                #     prev_pair = (self.prev_final_error, self.prev_null_obj_val)
                #     curr_pair = (final_error, null_obj_val)
                #     if (prev_pair[0] < curr_pair[0]) or (np.isclose(prev_pair[0], curr_pair[0]) and prev_pair[1] < curr_pair[1]):
                #         use_prev = True
                # Select the q with smaller error metrics
                if use_prev:
                    print(f"IK fallback: using prev q (prev_err_pos={self.prev_final_error_pos:.6f}, prev_err_ori={self.prev_final_error_ori:.6f}, prev_null={self.prev_null_obj_val:.6f}) vs curr_err_pos={final_error_pos:.6f}, curr_err_ori={final_error_ori:.6f}, curr_null={null_obj_val:.6f}")
                    target_joint_pos = self.prev_target_joint_pos.copy()
                    # Keep previous metrics as they correspond to chosen q
                else:
                    print(f"IK fallback: using current q from solver (curr_err_pos={final_error_pos:.6f}, curr_err_ori={final_error_ori:.6f}, curr_null={null_obj_val:.6f})")
                    # Update previous metrics to current since we chose current q
                    self.prev_final_error_pos = final_error_pos
                    self.prev_final_error_ori = final_error_ori
                    self.prev_null_obj_val = null_obj_val
        else:
            print(f"Z1 IK solved successfully in {iterations} iterations (error: {final_error_pos:.6f}, error_ori: {final_error_ori:.6f}, null_obj: {null_obj_val:.6f})")
            # On success, update previous metrics to current
            self.prev_final_error_pos = final_error_pos
            self.prev_final_error_ori = final_error_ori
            self.prev_null_obj_val = null_obj_val
        # Remember chosen target q for next iteration's comparison
        self.prev_target_joint_pos = target_joint_pos.copy()
    
        
        # Calculate joint position error
        joint_error = target_joint_pos - current_joint_pos
        joint_error_norm = np.linalg.norm(joint_error)
        
        # Calculate required joint speed to reach target in control_time
        if joint_error_norm > 1e-6:
            required_joint_speed = joint_error_norm / control_time
            actual_joint_speed = min(required_joint_speed, self.joint_speed)
            joint_direction = joint_error / joint_error_norm
        else:
            actual_joint_speed = 0.0
            joint_direction = np.zeros(6)
        
        # Calculate gripper direction and speed
        gripper_error = target_gripper - current_gripper_pos if self.has_gripper else 0.0
        if abs(gripper_error) > 1e-6:
            gripper_direction = np.sign(gripper_error)
            gripper_speed = min(abs(gripper_error) / control_time, 1.0)  # Max gripper speed is 1.0
        else:
            gripper_direction = 0.0
            gripper_speed = 0.0
        
        # Create joint command: [J1, J2, J3, J4, J5, J6, gripper] directions
        # Scale directions by actual speeds
        joint_directions = np.array([
            joint_direction[0] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J1
            joint_direction[1] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J2
            joint_direction[2] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J3
            joint_direction[3] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J4
            joint_direction[4] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J5
            joint_direction[5] * (actual_joint_speed / self.joint_speed) if self.joint_speed > 0 else 0,  # J6
            gripper_direction * gripper_speed  # gripper
        ])
        
        # Clamp directions to [-1, 1] range
        joint_directions = np.clip(joint_directions, -1.0, 1.0)
        
        return joint_directions, actual_joint_speed, gripper_speed
    
    
