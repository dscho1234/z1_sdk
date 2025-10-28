import sys
import os
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import threading
from typing import Tuple, Dict, Any, Optional, Callable
import gym
from gym import spaces
import functools
from scipy.optimize import minimize

# Add the lib directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface
from scipy.spatial.transform import Rotation as R


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


def _step_worker_loop(env_instance):
    """Worker loop that continuously processes step requests."""
    while True:
        try:
            # Wait for step request
            if hasattr(env_instance, 'step_request_event') and env_instance.step_request_event.wait():
                env_instance.step_request_event.clear()
                
                # Check if we should stop (when close() is called)
                if not hasattr(env_instance, 'is_initialized') or not env_instance.is_initialized:
                    break
                
                # Execute the step logic
                start = time.time()
                # print('start step execution in background thread')
                result = env_instance._execute_step_logic(env_instance.pending_action)
                env_instance.step_result_container['result'] = result
                env_instance.step_result_container['completed'] = True
                print(f"Step execution completed in background thread in {time.time() - start:.6f}s")
                
                # Signal completion
                env_instance.step_ready.set()
                
        except Exception as e:
            print(f"Error in step execution: {e}")
            # Put a dummy result to unblock the system
            dummy_result = (np.zeros(21), 0.0, False, {'error': str(e)})
            env_instance.step_result_container['result'] = dummy_result
            env_instance.step_result_container['completed'] = True
            env_instance.step_ready.set()


class Z1BaseEnv(gym.Env):
    """
    Base environment for Z1 robot arm following OpenAI Gym interface.
    This class provides basic initialization and common functionality.
    """
    
    def __init__(self, has_gripper: bool = True, control_frequency: float = 500.0):
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
        self.joint_limits = [
            (-2.618, 2.618),   # J1: ±150°
            (0, 3.142),        # J2: 0—180°
            (-2.879, 0),       # J3: -165°—0
            (-1.396, 1.396),   # J4: ±80°
            (-1.484, 1.484),   # J5: ±85°
            (-2.793, 2.793)    # J6: ±160°
        ]
        
        # Define action and observation spaces (to be overridden by subclasses)
        self.action_space = None
        self.observation_space = None
        
    def reset(self, joint_angle: Optional[np.ndarray] = None, option: str = "lowcmd") -> np.ndarray:
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
                self._move_to_joint_angles(joint_angle, option=option)
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
                self._move_to_joint_angles(joint_angle, option=option)
            
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
    
    def _move_to_joint_angles(self, target_joint_angles: np.ndarray, option: str = "lowcmd"):
        """
        Move the robot to specified joint angles using either low-level commands or MoveJ.
        
        Args:
            target_joint_angles: Target joint angles (6-element array)
            option: Movement method - "lowcmd" for low-level commands, "MoveJ" for high-level MoveJ
        """
        if len(target_joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(target_joint_angles)}")
        
        print(f"Moving to joint angles: {target_joint_angles} using {option} method")
        
        if option == "MoveJ":
            # Use MoveJ high-level command (similar to example_highcmd_custom.py)
            print("Using MoveJ high-level command...")
            self.arm.setWait(True)
            
            # Convert target joint angles to transformation matrix
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
            # Wait for inference process to finish if running
            if hasattr(self, 'inference_process') and self.inference_process and self.inference_process.is_alive():
                print("Waiting for inference process to finish...")
                self.inference_process.join(timeout=1.0)
                if self.inference_process.is_alive():
                    self.inference_process.terminate()
            
            # Wait for step thread to finish if running
            if hasattr(self, 'step_thread') and self.step_thread and self.step_thread.is_alive():
                print("Waiting for step thread to finish...")
                # Signal the thread to stop by setting a stop flag
                if hasattr(self, 'step_request_event'):
                    self.step_request_event.set()  # This will cause the thread to exit
                self.step_thread.join(timeout=1.0)
                if self.step_thread.is_alive():
                    print("Warning: Step thread did not finish within timeout")
            
            self.arm.backToStart()
            self.arm.loopOff()
            self.is_initialized = False




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
                 use_current_joint_pos_when_ik_fails = False,
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
            
        """
        super().__init__(has_gripper, control_frequency)
        self.use_current_joint_pos_when_ik_fails = use_current_joint_pos_when_ik_fails
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.joint_speed = joint_speed
        self.sequence_length = sequence_length
        
        # Store previous step's joint directions for continuous movement during inference
        self.previous_joint_directions = np.zeros(7)  # [J1, J2, J3, J4, J5, J6, gripper]
        self.has_previous_directions = False
        
        # Non-blocking inference system using multiprocessing
        self.action_queue = Queue()
        self.inference_process = None
        self.inference_ready = Event()
        
        # Non-blocking step execution system using threading
        self.step_thread = None
        self.step_result_container = {}
        self.step_ready = threading.Event()
        self.step_thread_created = False
        self.step_request_event = threading.Event()
        self.pending_action = None
        
        # RTC-style future sequence handling
        self.current_sequence = None  # Current future target pose sequence
        self.sequence_index = 0       # Current index in the sequence
        self.sequence_step_count = 0  # Total steps executed in current sequence
        
        # Target pose (position + quaternion) - current target from sequence
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])  # x, y, z, w quaternion (identity)
        self.target_gripper = 0.0
        
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
        
    def reset(self, joint_angle: Optional[np.ndarray] = None, option: str = "lowcmd") -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Args:
            joint_angle: Optional joint angles to move to. If None, uses default reset behavior.
                        Should be a 6-element array for 6-DOF arm.
        """
        assert joint_angle is not None, "joint_angle must be provided for stable initialization"
        super().reset(joint_angle, option=option)
        
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
        
        # Create step thread once during reset for reuse
        if not self.step_thread_created:
            self.step_thread = threading.Thread(
                target=_step_worker_loop,
                args=(self,),
                daemon=True
            )
            self.step_thread_created = True
            self.step_thread.start()
            print("Created reusable step thread")
        
        # Initialize step ready event for first step
        self.step_ready.set()  # Set initially so first step can proceed
        
        return self._get_observation()
    
    
    def step(self, action: np.ndarray, wait: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using jointCtrlCmd.
        Converts target pose to joint commands and executes them.
        
        Args:
            action: [x, y, z, qx, qy, qz, qw, gripper] target pose (quaternion in [x,y,z,w] format)
            wait: If True, wait for step to complete before returning. If False, execute step in background.
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        

        if wait:
            # Blocking execution - execute step logic directly
            return self._execute_step_logic(action)
        else:
            # Non-blocking execution - use reusable background thread
            if not self.step_ready.is_set():
                print("Warning: Step is already running in background. Ignoring new step request.")
                # Return current state if step is already running
                self._update_state()
                observation = self._get_observation()
                reward = self._get_reward()
                done = self._is_done()
                info = {'fsm_state': self.arm.getCurrentState(), 'step_running': True}
                return observation, reward, done, info
            
            # Reset result container and event
            self.step_result_container = {'result': None, 'completed': False}
            self.step_ready.clear()
            
            # Set pending action and signal the worker thread
            self.pending_action = action
            self.step_request_event.set()
            
            # Return immediately with current state
            self._update_state()
            observation = self._get_observation()
            reward = self._get_reward()
            done = self._is_done()
            info = {'fsm_state': self.arm.getCurrentState(), 'step_running': True}
            return observation, reward, done, info
    
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
        print('@@@@@@@@@@@@@ in _execute_step_logic function, action :', action)
        # Update target pose from action
        # action format: [x, y, z, qx, qy, qz, qw, gripper]
        self.target_position = action[:3]
        self.target_orientation = self._normalize_quaternion(action[3:7])  # [qx, qy, qz, qw]
        if self.has_gripper:
            self.target_gripper = action[7]
        
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
        
        # Calculate dt ratio for internal loop
        dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
        start = time.time()
        sleep_time_list = []
        # Execute jointCtrlCmd for dt_ratio iterations
        for i in range(dt_ratio):
            self.arm.jointCtrlCmd(joint_directions, self.joint_speed)
            sleep_start = time.time()
            time.sleep(self.arm._ctrlComp.dt)
            sleep_time = time.time() - sleep_start
            sleep_time_list.append(sleep_time)
        cmd_time = time.time() - start
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
            'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
            'orientation_error': self._quaternion_distance(
                self.target_orientation, self._get_current_ee_orientation()
            ),
            'joint_directions': joint_directions.copy(),
            'actual_joint_speed': actual_joint_speed,
            'gripper_speed': gripper_speed,
            'dt_ratio': dt_ratio,
            'current_ee_pose_before_cmd': current_ee_pose_before_cmd.copy(),
        }
        
        self.episode_step += 1
        return observation, reward, done, info
    
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
        """Check if the background step execution is complete."""
        return self.step_ready.is_set()
    
    def get_step_result(self) -> Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Get the result from the background step execution if it's complete.
        
        Returns:
            Step result tuple if complete, None if still running or no result available
        """
        if not self.step_ready.is_set():
            return None  # Still running
        
        if self.step_result_container.get('completed', False):
            return self.step_result_container.get('result', None)
        
        return None
    
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
    
    def _get_current_ee_se3(self, joint_pos = None) -> np.ndarray:
        """Get current end-effector SE(3) matrix."""
        if joint_pos is None:
            joint_pos = self.current_joint_pos
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

    def _get_current_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as quaternion in [x,y,z,w] format."""
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
                                  control_time):
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
        
        success, target_joint_pos, iterations, final_error, null_obj_val = self.solve_ik_null_space(
            target_T, 
            initial_guess=current_joint_pos,
            max_iterations=20,
            tolerance=1e-2,
            tolerance_null=1e-4
        )
        if not success:
            print(f"Warning: IK failed to converge (error: {final_error:.6f}, null_obj: {null_obj_val:.6f}), using current joint positions")
            if self.use_current_joint_pos_when_ik_fails:
                target_joint_pos = current_joint_pos.copy()
            
        else:
            print(f"IK solved successfully in {iterations} iterations (error: {final_error:.6f}, null_obj: {null_obj_val:.6f})")
    
        
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
    
    
