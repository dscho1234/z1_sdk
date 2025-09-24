import sys
import os
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
import gym
from gym import spaces

# Add the lib directory to the path
import unitree_arm_interface

# Import the base class from z1_env
sys.path.append(os.path.join(os.path.dirname(__file__)))
from z1_env import Z1BaseEnv


class EEPoseCtrlLowCmdWrapper(Z1BaseEnv):
    """
    End-effector pose control wrapper using low-level commands and inverse kinematics.
    This wrapper uses direct joint control with inverse kinematics to achieve target poses.
    """
    
    def __init__(self, has_gripper: bool = True, control_frequency: float = 500.0, 
                 position_tolerance: float = 0.01, orientation_tolerance: float = 0.1,
                 move_speed: float = 1.0, move_timeout: float = 2.0):
        """
        Initialize the end-effector pose control wrapper with low-level commands.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
            position_tolerance: Position tolerance for reaching target
            orientation_tolerance: Orientation tolerance for reaching target
            move_speed: Movement speed multiplier (0.1 to 1.0)
            move_timeout: Timeout for movement completion in seconds
            inference_time: Time taken for neural network inference (seconds)
        """
        # Initialize base class
        super().__init__(has_gripper, control_frequency)
        
        # Control parameters specific to this wrapper
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.move_speed = move_speed
        self.move_timeout = move_timeout
        # inference_time is not stored as self, it's passed per step
        
        # Target pose
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        self.target_gripper = 0.0
        
        # Movement state
        self.move_start_time = 0.0
        self.is_moving = False
        
        # Define action space: [x, y, z, qw, qx, qy, qz, gripper] (8D)
        # Position: [-1, 1] meters, Orientation: [-1, 1] quaternion, Gripper: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: [joint_pos(6), joint_vel(6), ee_pos(3), ee_quat(4), gripper_pos(1), gripper_vel(1)]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi] * 6 + [-10.0] * 6 + [-2.0] * 3 + [-1.0] * 4 + [-1.0] * 2),
            high=np.array([np.pi] * 6 + [10.0] * 6 + [2.0] * 3 + [1.0] * 4 + [1.0] * 2),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        # Call base class reset
        super().reset()

        
        # Set to low-level command mode
        self.arm.setFsmLowcmd()
        print("Set to low-level command mode")
        
        
        # Set initial target to current end-effector pose
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        self.target_position = current_ee_pose[:3]
        self.target_orientation = current_ee_pose[3:7]
        
        # Initialize movement state
        self.is_moving = False
        self.move_start_time = 0.0
        
        print(f"Initial observation shape: {self._get_observation().shape}")
        print(f"Initial joint positions: {self.current_joint_pos}")
        print(f"Initial end-effector position: {self._get_current_ee_position()}")
        print(f"Initial end-effector orientation: {self._get_current_ee_orientation()}")
        print(f"Initial gripper position: {self.current_gripper_pos}")
        
        return self._get_observation()
    
    def step(self, action: np.ndarray, inference_time: float = 0.15) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using inverse kinematics.
        This follows the example_lowcmd.py approach with real IK.
        
        Args:
            action: [x, y, z, qw, qx, qy, qz, gripper] target pose
            inference_time: Time taken for neural network inference (seconds)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Update target pose from action
        self.target_position = action[:3]
        self.target_orientation = action[3:7]
        if self.has_gripper:
            self.target_gripper = action[7]
        
        # Normalize quaternion
        self.target_orientation = self._normalize_quaternion(self.target_orientation)
        
        # Convert target pose to transformation matrix
        target_T = self._pose_to_transformation_matrix(
            self.target_position, self.target_orientation
        )
        
        # Use real inverse kinematics like example_model.py
        try:
            # Get current joint positions
            current_joint_pos = self.current_joint_pos.copy()
            
            # Solve inverse kinematics to get target joint positions
            # Using the same approach as example_model.py
            hasIK, target_joint_pos = self.arm_model.inverseKinematics(target_T, current_joint_pos, False)
            
            if not hasIK:
                print("Warning: Inverse kinematics failed, using current joint positions")
                target_joint_pos = current_joint_pos.copy()
            else:
                print(f"IK solved successfully: {target_joint_pos}")
            
            # Store target joint positions for interpolation
            self.target_joint_pos = target_joint_pos
            self.start_joint_pos = current_joint_pos.copy()
            self.step_count = 0
            self.total_steps = int(self.control_frequency * self.move_timeout)  # Total steps for movement
            
            print(f"Target EE position: {self.target_position}")
            print(f"Target joint positions: {target_joint_pos}")
            print(f"Movement duration: {self.total_steps} steps")
            
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            self.target_joint_pos = current_joint_pos.copy()
            self.start_joint_pos = current_joint_pos.copy()
            self.step_count = 0
            self.total_steps = 1
        
        # Calculate actual control time (dt - inference_time)
        actual_control_time = max(self.dt - inference_time, 0.001)  # Minimum 1ms control time
        
        # Calculate dt ratio for internal loop using actual control time
        dt_ratio = int(actual_control_time / self.arm._ctrlComp.dt)
        
        # Execute movement with proper timing (like example_lowcmd.py)
        if hasattr(self, 'target_joint_pos') and self.step_count < self.total_steps:
            # Send low-level commands with proper timing (like example_lowcmd.py)
            for i in range(dt_ratio):
                # Calculate current progress within this step (like example_lowcmd.py)
                # Total progress includes both step_count and current iteration within step
                total_progress = (self.step_count * dt_ratio + i) / (self.total_steps * dt_ratio)
                
                # Interpolate between start and target joint positions (like example_lowcmd.py)
                current_joint_pos = self.start_joint_pos * (1 - total_progress) + self.target_joint_pos * total_progress
                
                # Calculate joint velocities (like example_lowcmd.py)
                joint_vel = (self.target_joint_pos - self.start_joint_pos) / (self.total_steps * self.dt)
                
                # Calculate required torques using inverse dynamics (like example_lowcmd.py)
                joint_acc = np.zeros(6)
                gravity_comp = np.zeros(6)
                target_torque = self.arm_model.inverseDynamics(current_joint_pos, joint_vel, joint_acc, gravity_comp)
                
                # Set gripper target (like example_lowcmd.py)
                gripper_target = self.target_gripper * total_progress
                gripper_vel = (self.target_gripper - self.current_gripper_pos) / (self.total_steps * self.dt)
                gripper_torque = 0.0  # Simple gripper control
                
                # Send commands to robot
                self.arm.setArmCmd(current_joint_pos, joint_vel, target_torque)
                self.arm.setGripperCmd(gripper_target, gripper_vel, gripper_torque)
                self.arm.sendRecv()
                time.sleep(self.arm._ctrlComp.dt)
            
            # Update step count
            self.step_count += 1
            
            # Update movement state
            self.is_moving = self.step_count < self.total_steps
            
            print(f"Step {self.step_count}/{self.total_steps}: Joint pos: {current_joint_pos}")
            
        else:
            # Movement completed
            self.is_moving = False
            print("Movement completed")
        
        # Update state
        self._update_state()
        
        # Get observation, reward, and done status
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        
        # Create info dictionary
        info = {
            'is_moving': self.is_moving,
            'target_position': self.target_position.copy(),
            'target_orientation': self.target_orientation.copy(),
            'current_ee_pose': self._get_current_ee_pose(),
            'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
            'orientation_error': self._quaternion_distance(
                self.target_orientation, self._get_current_ee_orientation()
            ),
            'dt_ratio': dt_ratio,
            'actual_control_time': actual_control_time,
            'inference_time': inference_time
        }
        
        self.episode_step += 1
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including joint states and end-effector pose."""
        current_ee_pose = self._get_current_ee_pose()
        
        # Combine joint states and end-effector pose
        observation = np.concatenate([
            self.current_joint_pos,      # 6 joint positions
            self.current_joint_vel,      # 6 joint velocities
            current_ee_pose[:3],         # 3 end-effector position
            current_ee_pose[3:7],        # 4 end-effector orientation (quaternion)
            [self.current_gripper_pos],  # 1 gripper position
            [self.current_gripper_vel]   # 1 gripper velocity
        ])
        
        return observation.astype(np.float32)
    
    def _get_reward(self) -> float:
        """Calculate reward based on position and orientation error."""
        position_error = np.linalg.norm(self.target_position - self._get_current_ee_position())
        orientation_error = self._quaternion_distance(
            self.target_orientation, self._get_current_ee_orientation()
        )
        
        # Reward is negative error (closer to target = higher reward)
        reward = -(position_error + orientation_error)
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Check if max episode steps reached
        if self.episode_step >= self.max_episode_steps:
            return True
        
        # Check if target reached with tolerance
        position_error = np.linalg.norm(self.target_position - self._get_current_ee_position())
        orientation_error = self._quaternion_distance(
            self.target_orientation, self._get_current_ee_orientation()
        )
        
        if position_error < self.position_tolerance and orientation_error < self.orientation_tolerance:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including joint states and end-effector pose."""
        current_ee_pose = self._get_current_ee_pose()
        
        # Combine joint states and end-effector pose
        observation = np.concatenate([
            self.current_joint_pos,      # 6 joint positions
            self.current_joint_vel,      # 6 joint velocities
            current_ee_pose[:3],         # 3 end-effector position
            current_ee_pose[3:7],        # 4 end-effector orientation (quaternion)
            [self.current_gripper_pos],  # 1 gripper position
            [self.current_gripper_vel]   # 1 gripper velocity
        ])
        
        return observation.astype(np.float32)
    
    def _get_current_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose (position + quaternion)."""
        # Get current transformation matrix
        T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        
        # Extract position
        position = T[:3, 3]
        
        # Extract quaternion from rotation matrix
        quaternion = self._rotation_matrix_to_quaternion(T[:3, :3])
        
        # Get gripper position
        gripper_pos = self.current_gripper_pos if self.has_gripper else 0.0
        
        return np.concatenate([position, quaternion, [gripper_pos]])
    
    def _get_current_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        return T[:3, 3]
    
    def _get_current_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as quaternion."""
        T = self.arm_model.forwardKinematics(self.current_joint_pos, 6)
        return self._rotation_matrix_to_quaternion(T[:3, :3])
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _pose_to_transformation_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self._quaternion_to_rotation_matrix(quaternion)
        T[:3, 3] = position
        return T
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate distance between two quaternions."""
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product
        dot_product = np.dot(q1, q2)
        
        # Ensure dot product is in valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle between quaternions
        angle = 2 * np.arccos(abs(dot_product))
        
        return angle
    
    def _normalize_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        return quaternion / np.linalg.norm(quaternion)
    
    def close(self):
        """Close the environment and clean up resources."""
        # if hasattr(self, 'arm'):
        #     self.arm.backToStart()
        #     self.arm.loopOff()
        #     print("Stopped control loop (loopOff)")
        # super().close()
        
        self.arm.loopOn()
        self.arm.backToStart()
        self.arm.loopOff()
