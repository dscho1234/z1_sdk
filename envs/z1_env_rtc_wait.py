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

# Add the lib directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface



def _step_worker_thread(env_instance, action, result_container, step_ready):
    """Worker function that executes a step in a separate thread."""
    try:
        # Execute the step logic in the thread
        start = time.time()
        print('start step execution in background thread')
        result = env_instance._execute_step_logic(action)
        result_container['result'] = result
        result_container['completed'] = True
        print(f"Step execution completed in background thread in {time.time() - start:.6f}s")
    except Exception as e:
        print(f"Error in step execution: {e}")
        # Put a dummy result to unblock the system
        dummy_result = (np.zeros(21), 0.0, False, {'error': str(e)})
        result_container['result'] = dummy_result
        result_container['completed'] = True
    finally:
        step_ready.set()


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
        
        # Define action and observation spaces (to be overridden by subclasses)
        self.action_space = None
        self.observation_space = None
        
    def reset(self, joint_angle: Optional[np.ndarray] = None) -> np.ndarray:
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
                self._move_to_joint_angles(joint_angle)
            else:
                self.arm.backToStart()
        else: # if not self.is_initialized:
            print("Initializing arm interface...")
            # Start the control loop first
            self.arm.loopOn()
            time.sleep(0.1)  # Small delay for initialization
            
            # dscho added
            self.arm.labelRun("forward")
            
            # Skip backToStart for now to avoid hanging
            print("Skipping backToStart() to avoid potential hanging...")
            print("Using current robot position as starting point...")
            
            # If joint_angle is specified during initialization, move to it
            if joint_angle is not None:
                print(f"Moving to specified joint angles during initialization: {joint_angle}")
                self._move_to_joint_angles(joint_angle)
            
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
    
    def _move_to_joint_angles(self, target_joint_angles: np.ndarray):
        """
        Move the robot to specified joint angles using low-level commands.
        
        Args:
            target_joint_angles: Target joint angles (6-element array)
        """
        if len(target_joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(target_joint_angles)}")
        
        print(f"Moving to joint angles: {target_joint_angles}")
        
        # Set low-level command mode
        self.arm.setFsmLowcmd()
        
        # Get current position as starting point
        lastPos = np.array(self.arm.lowstate.getQ())
        targetPos = target_joint_angles
        
        # Duration for smooth movement (about 2 seconds at 500Hz)
        duration = 1000
        
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
                self.step_thread.join(timeout=1.0)
                if self.step_thread.is_alive():
                    print("Warning: Step thread did not finish within timeout")
            
            self.arm.backToStart()
            self.arm.loopOff()
            self.is_initialized = False




class EEPoseCtrlCartesianCmdWrapper(Z1BaseEnv):
    """
    End-effector pose control wrapper for Z1 robot using cartesianCtrlCmd with RTC-style future sequence handling.
    This wrapper allows controlling the robot's end-effector pose directly using cartesian velocity commands
    with support for future target pose sequences from inference.
    """
    
    def __init__(self, 
                 has_gripper: bool = True, 
                 control_frequency: float = 500.0,
                 position_tolerance: float = 0.01,
                 orientation_tolerance: float = 0.1,
                 angular_vel: float = 0.3,
                 linear_vel: float = 0.3,
                 sequence_length: int = 10,
                 ):
        """
        Initialize the end-effector pose control wrapper using cartesian commands with RTC support.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
            position_tolerance: Position tolerance for convergence
            orientation_tolerance: Orientation tolerance for convergence
            angular_vel: Angular velocity for cartesian commands
            linear_vel: Linear velocity for cartesian commands
            sequence_length: Length of future target pose sequences from inference
            
        """
        super().__init__(has_gripper, control_frequency)
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.angular_vel = angular_vel
        self.linear_vel = linear_vel
        self.sequence_length = sequence_length
        
        # Store previous step's directions for continuous movement during inference
        self.previous_cartesian_directions = np.zeros(7)  # [roll, pitch, yaw, x, y, z, gripper]
        self.previous_angular_vel = angular_vel
        self.previous_linear_vel = linear_vel
        self.has_previous_directions = False
        
        # Non-blocking inference system using multiprocessing
        self.action_queue = Queue()
        self.inference_process = None
        self.inference_ready = Event()
        
        # Non-blocking step execution system using threading
        self.step_thread = None
        self.step_result_container = {}
        self.step_ready = threading.Event()
        
        # RTC-style future sequence handling
        self.current_sequence = None  # Current future target pose sequence
        self.sequence_index = 0       # Current index in the sequence
        self.sequence_step_count = 0  # Total steps executed in current sequence
        
        # Target pose (position + quaternion) - current target from sequence
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        self.target_gripper = 0.0
        
        # Define action space: [x, y, z, qw, qx, qy, qz, gripper] (8D)
        # Position: [-1, 1] meters, Orientation: [-1, 1] quaternion, Gripper: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: [joint_pos(6), joint_vel(6), ee_pos(3), ee_quat(4), gripper_pos(1), gripper_vel(1)]
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
        
    def reset(self, joint_angle: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Args:
            joint_angle: Optional joint angles to move to. If None, uses default reset behavior.
                        Should be a 6-element array for 6-DOF arm.
        """
        super().reset(joint_angle)
        
        # Start cartesian control mode
        self.arm.startTrack(unitree_arm_interface.ArmFSMState.CARTESIAN)
        print("Started cartesian control mode")
        
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
        
        return self._get_observation()
    
    
    def step(self, action: np.ndarray, wait: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using cartesianCtrlCmd.
        Simple version that just executes the given action for dt_ratio iterations.
        
        Args:
            action: [x, y, z, qw, qx, qy, qz, gripper] target pose
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
            # Non-blocking execution - start step in background thread
            if self.step_thread and self.step_thread.is_alive():
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
            
            # Start step execution in background thread
            self.step_thread = threading.Thread(
                target=_step_worker_thread,
                args=(self, action, self.step_result_container, self.step_ready),
                daemon=True
            )
            self.step_thread.start()
            
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
            action: [x, y, z, qw, qx, qy, qz, gripper] target pose
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Update target pose from action
        self.target_position = action[:3]
        self.target_orientation = self._normalize_quaternion(action[3:7])
        if self.has_gripper:
            self.target_gripper = action[7]
        
        # Get current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Calculate cartesian directions to target
        cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
            self._calculate_cartesian_directions(
                self.target_position, self.target_orientation, self.target_gripper,
                current_pos, current_quat, current_gripper_pos, self.dt
            )
        
        # Calculate dt ratio for internal loop
        dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
        
        # Execute cartesianCtrlCmd for dt_ratio iterations
        for i in range(dt_ratio):
            self.arm.cartesianCtrlCmd(cartesian_directions, self.angular_vel, self.linear_vel)
            time.sleep(self.arm._ctrlComp.dt)
        
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
            'cartesian_directions': cartesian_directions.copy(),
            'actual_linear_speed': actual_linear_speed,
            'actual_angular_speed': actual_angular_speed,
            'gripper_speed': gripper_speed,
            'dt_ratio': dt_ratio
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
        return self.step_thread is None or not self.step_thread.is_alive()
    
    def get_step_result(self) -> Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
        """
        Get the result from the background step execution if it's complete.
        
        Returns:
            Step result tuple if complete, None if still running or no result available
        """
        if self.step_thread and self.step_thread.is_alive():
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
            current_ee_pose[3:7],             # 4D end-effector quaternion
            np.array([self.current_gripper_pos]),  # 1D gripper position
            np.array([self.current_gripper_vel])   # 1D gripper velocity
        ])
        
        return observation.astype(np.float32)
    
    def _get_reward(self) -> float:
        """Calculate reward based on pose tracking accuracy."""
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        
        # Position error reward
        position_error = np.linalg.norm(self.target_position - current_pos)
        position_reward = -position_error
        
        # Orientation error reward
        orientation_error = self._quaternion_distance(self.target_orientation, current_quat)
        orientation_reward = -orientation_error
        
        # Gripper reward
        gripper_error = abs(self.target_gripper - self.current_gripper_pos) if self.has_gripper else 0.0
        gripper_reward = -gripper_error
        
        # Total reward
        total_reward = position_reward + 0.1 * orientation_reward + 0.1 * gripper_reward
        
        return total_reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        return self.episode_step >= self.max_episode_steps
    
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
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.array([1, 0, 0, 0])
        return q / norm
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate distance between two quaternions."""
        # Ensure quaternions are normalized
        q1 = self._normalize_quaternion(q1)
        q2 = self._normalize_quaternion(q2)
        
        # Calculate dot product
        dot_product = np.abs(np.dot(q1, q2))
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, 0.0, 1.0)
        
        # Calculate angle between quaternions
        angle = 2 * np.arccos(dot_product)
        
        return angle
    
    def _quaternion_error(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Calculate angular velocity error between target and current quaternions."""
        # Ensure quaternions are normalized
        q_target = self._normalize_quaternion(q_target)
        q_current = self._normalize_quaternion(q_current)
        
        # Calculate quaternion error: q_error = q_target * q_current^-1
        q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Quaternion multiplication
        w = q_target[0] * q_current_inv[0] - q_target[1] * q_current_inv[1] - q_target[2] * q_current_inv[2] - q_target[3] * q_current_inv[3]
        x = q_target[0] * q_current_inv[1] + q_target[1] * q_current_inv[0] + q_target[2] * q_current_inv[3] - q_target[3] * q_current_inv[2]
        y = q_target[0] * q_current_inv[2] - q_target[1] * q_current_inv[3] + q_target[2] * q_current_inv[0] + q_target[3] * q_current_inv[1]
        z = q_target[0] * q_current_inv[3] + q_target[1] * q_current_inv[2] - q_target[2] * q_current_inv[1] + q_target[3] * q_current_inv[0]
        
        q_error = np.array([w, x, y, z])
        
        # Convert quaternion error to angular velocity (simplified)
        # For small errors, angular velocity is approximately 2 * [x, y, z]
        angular_velocity = 2.0 * q_error[1:4]  # [wx, wy, wz]
        
        return angular_velocity
    
    def _calculate_cartesian_directions(self, target_position, target_orientation, target_gripper, 
                                      current_pos, current_quat, current_gripper_pos, 
                                      control_time):
        """
        Calculate cartesian directions based on target and current poses.
        
        Args:
            target_position: Target end-effector position
            target_orientation: Target end-effector orientation (quaternion)
            target_gripper: Target gripper position
            current_pos: Current end-effector position
            current_quat: Current end-effector orientation (quaternion)
            current_gripper_pos: Current gripper position
            control_time: Time available for control
            
        Returns:
            cartesian_directions: [roll, pitch, yaw, x, y, z, gripper] directions
            actual_linear_speed: Actual linear speed used
            actual_angular_speed: Actual angular speed used
            gripper_speed: Actual gripper speed used
        """
        # Calculate position error and desired movement direction
        position_error = target_position - current_pos
        position_error_norm = np.linalg.norm(position_error)
        
        # Calculate orientation error and desired rotation direction
        orientation_error = self._quaternion_error(target_orientation, current_quat)
        orientation_error_norm = np.linalg.norm(orientation_error)
        
        # Calculate required speeds to reach target in control_time
        if position_error_norm > 1e-6:
            required_linear_speed = position_error_norm / control_time
            actual_linear_speed = min(required_linear_speed, self.linear_vel)
            linear_direction = position_error / position_error_norm
        else:
            actual_linear_speed = 0.0
            linear_direction = np.zeros(3)
        
        if orientation_error_norm > 1e-6:
            required_angular_speed = orientation_error_norm / control_time
            actual_angular_speed = min(required_angular_speed, self.angular_vel)
            angular_direction = orientation_error / orientation_error_norm
        else:
            actual_angular_speed = 0.0
            angular_direction = np.zeros(3)
        
        # Calculate gripper direction and speed
        gripper_error = target_gripper - current_gripper_pos if self.has_gripper else 0.0
        if abs(gripper_error) > 1e-6:
            gripper_direction = np.sign(gripper_error)
            gripper_speed = min(abs(gripper_error) / control_time, 1.0)  # Max gripper speed is 1.0
        else:
            gripper_direction = 0.0
            gripper_speed = 0.0
        
        # Create cartesian command: [roll, pitch, yaw, x, y, z, gripper] directions
        # Scale directions by actual speeds
        cartesian_directions = np.array([
            angular_direction[0] * (actual_angular_speed / self.angular_vel) if self.angular_vel > 0 else 0,  # roll
            angular_direction[1] * (actual_angular_speed / self.angular_vel) if self.angular_vel > 0 else 0,  # pitch  
            angular_direction[2] * (actual_angular_speed / self.angular_vel) if self.angular_vel > 0 else 0,  # yaw
            linear_direction[0] * (actual_linear_speed / self.linear_vel) if self.linear_vel > 0 else 0,      # x
            linear_direction[1] * (actual_linear_speed / self.linear_vel) if self.linear_vel > 0 else 0,      # y
            linear_direction[2] * (actual_linear_speed / self.linear_vel) if self.linear_vel > 0 else 0,      # z
            gripper_direction * gripper_speed  # gripper
        ])
        
        # Clamp directions to [-1, 1] range
        cartesian_directions = np.clip(cartesian_directions, -1.0, 1.0)
        
        return cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed
    
    
