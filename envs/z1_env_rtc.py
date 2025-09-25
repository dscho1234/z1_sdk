import sys
import os
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from typing import Tuple, Dict, Any, Optional, Callable
import gym
from gym import spaces

# Add the lib directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface


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
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        if self.is_initialized:
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
        
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        super().reset()
        
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
    
    def non_blocking_inference(self, inference_function: Callable[[], np.ndarray], inference_time: float = 0.15, global_step = None):
        start = time.time()
        """
        Start non-blocking inference that will return a future target pose sequence.
        
        Args:
            inference_function: Function that returns a sequence of actions (shape: [sequence_length, 8])
            inference_time: Expected inference time for timing calculations (seconds)
        """
        if self.inference_process and self.inference_process.is_alive():
            print("Warning: Inference is already running. Ignoring new inference request.")
            return
        
        # Create a new process for this inference (simpler approach)
        def _inference_worker(inference_function, action_queue, inference_ready):
            """Worker function that runs inference in a separate process."""
            try:
                # Call the actual inference function
                start_inference = time.time()
                new_sequence = inference_function()  # Should return [sequence_length, 8]
                real_inference_time = time.time() - start_inference
                print(f"Inference completed in {real_inference_time:.6f}s, generated sequence shape: {new_sequence.shape}")

                # Put result in queue
                action_queue.put(new_sequence)
                print(f"Inference completed, new sequence queued with shape: {new_sequence.shape}")
            except Exception as e:
                print(f"Error in inference function: {e}")
                # Put a dummy sequence to unblock the system
                action_queue.put(np.zeros((self.sequence_length, 8)))
            finally:
                inference_ready.set()
        
        # Start the inference process
        self.inference_process = Process(
            target=_inference_worker, 
            args=(inference_function, self.action_queue, self.inference_ready),
            daemon=True
        )
        self.inference_process.start()
        
        
    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using cartesianCtrlCmd.
        Simple version that just executes the given action for dt_ratio iterations.
        
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
    
    
