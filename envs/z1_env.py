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
from scipy.spatial.transform import Rotation as R

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
            # Wait for inference thread to finish if running
            if hasattr(self, 'inference_process') and self.inference_process and self.inference_process.is_alive():
                print("Waiting for inference process to finish...")
                self.inference_process.join(timeout=1.0)
                if self.inference_process.is_alive():
                    self.inference_process.terminate()
            
            self.arm.backToStart()
            self.arm.loopOff()
            self.is_initialized = False


class EEPoseCtrlWrapper(Z1BaseEnv):
    """
    End-effector pose control wrapper for Z1 robot using high-level MoveJ commands.
    This wrapper allows controlling the robot's end-effector pose directly using inverse kinematics.
    """
    
    def __init__(self, 
                 has_gripper: bool = True, 
                 control_frequency: float = 500.0,
                 position_tolerance: float = 0.01,
                 orientation_tolerance: float = 0.1,
                 move_speed: float = 1.0,
                 move_timeout: float = 5.0):
        """
        Initialize the end-effector pose control wrapper.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
            position_tolerance: Position tolerance for IK convergence
            orientation_tolerance: Orientation tolerance for IK convergence
            move_speed: Speed for MoveJ commands
            move_timeout: Timeout for MoveJ commands
        """
        super().__init__(has_gripper, control_frequency)
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.move_speed = move_speed
        self.move_timeout = move_timeout
        
        # Target pose (position + quaternion)
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
        
        # Set MoveJ to non-blocking mode for true non-blocking behavior
        self.arm.setWait(False)
        print("Set MoveJ to non-blocking mode (setWait=False)")
        
        # Set initial target to current end-effector pose
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        self.target_position = current_ee_pose[:3]
        self.target_orientation = current_ee_pose[3:7]
        self.target_gripper = current_ee_pose[7] if self.has_gripper else 0.0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using MoveJ.
        
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
        self.target_orientation = action[3:7]
        if self.has_gripper:
            self.target_gripper = action[7]
        
        # Normalize quaternion
        self.target_orientation = self._normalize_quaternion(self.target_orientation)
        
        # Convert target pose to transformation matrix
        target_T = self._pose_to_transformation_matrix(
            self.target_position, self.target_orientation
        )
        
        # print(f"action: {action}")
        # print(f"target_T: {target_T}")
        
        # Always send new MoveJ command immediately (interrupts any ongoing movement)
        # This is true non-blocking behavior - new commands always interrupt previous ones
        
        # Convert target transformation matrix to posture (roll, pitch, yaw, x, y, z)
        posture = unitree_arm_interface.homoToPosture(target_T)
        print(f"Starting MoveJ to posture: {posture}, target gripper: {self.target_gripper}, move speed: {self.move_speed}")
        success = self.arm.MoveJ(posture, self.target_gripper, self.move_speed)
        if success:
            print("MoveJ command sent successfully (non-blocking - interrupts previous command)")
            
            # Wait for move_timeout to ensure command is processed
            print(f"Waiting {self.move_timeout}s for MoveJ command to be processed...")
            time.sleep(self.move_timeout)
            
            # Check final FSM state
            final_fsm_state = self.arm.getCurrentState()
            if final_fsm_state == unitree_arm_interface.ArmFSMState.JOINTCTRL:
                print("MoveJ command completed successfully")
            else:
                print(f"MoveJ command still in progress (FSM state: {final_fsm_state})")
        else:
            print("Warning: MoveJ command failed")
    
        
        # Update state
        self._update_state()
        
        # Get observation, reward, and done status
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
            )
        }
        
        self.episode_step += 1
        
        return observation, reward, done, info
    
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
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize quaternion
        q = self._normalize_quaternion(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Convert to rotation matrix
        R_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R_matrix
    
    def _rotation_matrix_to_quaternion(self, R_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion."""
        # Shepperd's method for numerical stability
        trace = np.trace(R_matrix)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R_matrix[2, 1] - R_matrix[1, 2]) / s
            qy = (R_matrix[0, 2] - R_matrix[2, 0]) / s
            qz = (R_matrix[1, 0] - R_matrix[0, 1]) / s
        elif R_matrix[0, 0] > R_matrix[1, 1] and R_matrix[0, 0] > R_matrix[2, 2]:
            s = np.sqrt(1.0 + R_matrix[0, 0] - R_matrix[1, 1] - R_matrix[2, 2]) * 2  # s = 4 * qx
            qw = (R_matrix[2, 1] - R_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (R_matrix[0, 1] + R_matrix[1, 0]) / s
            qz = (R_matrix[0, 2] + R_matrix[2, 0]) / s
        elif R_matrix[1, 1] > R_matrix[2, 2]:
            s = np.sqrt(1.0 + R_matrix[1, 1] - R_matrix[0, 0] - R_matrix[2, 2]) * 2  # s = 4 * qy
            qw = (R_matrix[0, 2] - R_matrix[2, 0]) / s
            qx = (R_matrix[0, 1] + R_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (R_matrix[1, 2] + R_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R_matrix[2, 2] - R_matrix[0, 0] - R_matrix[1, 1]) * 2  # s = 4 * qz
            qw = (R_matrix[1, 0] - R_matrix[0, 1]) / s
            qx = (R_matrix[0, 2] + R_matrix[2, 0]) / s
            qy = (R_matrix[1, 2] + R_matrix[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def _pose_to_transformation_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self._quaternion_to_rotation_matrix(quaternion)
        T[:3, 3] = position
        return T
    
    
    
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


class EEPoseCtrlCartesianCmdWrapper(Z1BaseEnv):
    """
    End-effector pose control wrapper for Z1 robot using cartesianCtrlCmd.
    This wrapper allows controlling the robot's end-effector pose directly using cartesian velocity commands.
    """
    
    def __init__(self, 
                 has_gripper: bool = True, 
                 control_frequency: float = 500.0,
                 position_tolerance: float = 0.01,
                 orientation_tolerance: float = 0.1,
                 angular_vel: float = 0.3,
                 linear_vel: float = 0.3,
                 estimate_inference_time: float = 0.15,
                 ):
        """
        Initialize the end-effector pose control wrapper using cartesian commands.
        
        Args:
            has_gripper: Whether the robot has a gripper
            control_frequency: Control frequency in Hz
            position_tolerance: Position tolerance for convergence
            orientation_tolerance: Orientation tolerance for convergence
            angular_vel: Angular velocity for cartesian commands
            linear_vel: Linear velocity for cartesian commands
            estimate_inference_time: Estimated inference time for calculating movement during inference
            
        """
        super().__init__(has_gripper, control_frequency)
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.angular_vel = angular_vel
        self.linear_vel = linear_vel
        self.estimate_inference_time = estimate_inference_time
        # self.move_during_inference = move_during_inference
        # inference_time is not stored as self, it's passed per step
        
        # Store previous step's directions for continuous movement during inference
        self.previous_cartesian_directions = np.zeros(7)  # [roll, pitch, yaw, x, y, z, gripper]
        self.previous_angular_vel = angular_vel
        self.previous_linear_vel = linear_vel
        self.has_previous_directions = False
        
        # Non-blocking inference system using multiprocessing
        self.action_queue = Queue()
        self.inference_process = None
        self.inference_function = None
        self.inference_ready = Event()
        self.inference_result = None
        self.inference_exception = None
        
        # Target pose (position + quaternion)
        self.target_position = np.zeros(3)
        self.target_orientation = np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        self.target_gripper = 0.0
        
        # Previous target pose for inference movement
        self.previous_target_position = np.zeros(3)
        self.previous_target_orientation = np.array([1, 0, 0, 0])
        self.previous_target_gripper = 0.0
        
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
        
        # Reset previous directions flag for first step calculation
        self.has_previous_directions = False
        
        return self._get_observation()
    
    def non_blocking_inference(self, inference_function: Callable[[], np.ndarray], inference_time: float = 0.15, global_step = None):
        start = time.time()
        """
        Start non-blocking inference that will return an action.
        
        Args:
            inference_function: Function that returns an action (should handle its own timing internally)
            inference_time: Expected inference time for timing calculations (seconds)
        """
        if self.inference_process and self.inference_process.is_alive():
            print("Warning: Inference is already running. Ignoring new inference request.")
            return
        
        # Create a new process for this inference (simpler approach)
        def _inference_worker(inference_function, action_queue, inference_ready, inference_result, inference_exception):
            """Worker function that runs inference in a separate process."""
            try:
                # Call the actual inference function
                start_inference = time.time()
                new_action = inference_function()
                real_inference_time = time.time() - start_inference
                print(f"Inference completed in {real_inference_time:.6f}s")

                # Store result in shared memory
                action_bytes = new_action.tobytes()
                for i, byte in enumerate(action_bytes):
                    if i < len(inference_result):
                        inference_result[i] = byte
                # Clear exception
                for i in range(len(inference_exception)):
                    inference_exception[i] = b'\x00'
                action_queue.put(new_action)
                print(f"Inference completed, new action queued: {new_action}")
            except Exception as e:
                print(f"Error in inference function: {e}")
                error_bytes = str(e).encode()
                for i, byte in enumerate(error_bytes):
                    if i < len(inference_exception):
                        inference_exception[i] = byte
                # Clear result
                for i in range(len(inference_result)):
                    inference_result[i] = b'\x00'
                # Put a dummy action to unblock the system
                action_queue.put(np.zeros(8))
            finally:
                inference_ready.set()
        
        # Create shared memory for results
        self.inference_result = mp.Array('c', 1024)  # 1024 bytes buffer
        self.inference_exception = mp.Array('c', 1024)  # 1024 bytes buffer
        
        # Start the inference process
        self.inference_process = Process(
            target=_inference_worker, 
            args=(inference_function, self.action_queue, self.inference_ready, self.inference_result, self.inference_exception),
            daemon=True
        )
        self.inference_process.start()
        
        # Clear the shared memory arrays
        for i in range(len(self.inference_result)):
            self.inference_result[i] = b'\x00'
        for i in range(len(self.inference_exception)):
            self.inference_exception[i] = b'\x00'
        
        print(f"Started non-blocking inference (will complete in {inference_time:.3f}s, computed time is {time.time() - start:.6f}s)")
    
    def wait_for_inference_and_execute(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Wait for current inference to complete and execute the resulting action.
        This function blocks until inference is done and then executes the action.
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        if not (self.inference_process and self.inference_process.is_alive()):
            print("No inference running, returning current state")
            self._update_state()
            observation = self._get_observation()
            reward = self._get_reward()
            done = self._is_done()
            info = {
                'fsm_state': self.arm.getCurrentState(),
                'target_position': self.target_position.copy(),
                'target_orientation': self.target_orientation.copy(),
                'current_ee_pose': self._get_current_ee_pose(),
                'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
                'orientation_error': self._quaternion_distance(
                    self.target_orientation, self._get_current_ee_orientation()
                ),
                'cartesian_directions': self.previous_cartesian_directions.copy() if self.has_previous_directions else np.zeros(7),
                'actual_linear_speed': self.previous_linear_vel if self.has_previous_directions else 0.0,
                'actual_angular_speed': self.previous_angular_vel if self.has_previous_directions else 0.0,
                'gripper_speed': 0.0,
                'dt_ratio': int(self.dt / self.arm._ctrlComp.dt),
                'new_action_received_during_execution': False,
                'is_inference_running': False,
                'action_queue_size': self.action_queue.qsize()
            }
            return observation, reward, done, info
        
        print("Waiting for inference to complete...")
        # Wait for inference to complete using the event
        self.inference_ready.wait()
        
        # Check if there was an exception
        exception_bytes = bytes(self.inference_exception)
        if exception_bytes and exception_bytes != b'\x00' * len(self.inference_exception):
            print(f"Inference failed with exception: {exception_bytes.decode()}")
            # Fallback to current state
            self._update_state()
            observation = self._get_observation()
            reward = self._get_reward()
            done = self._is_done()
            info = {
                'fsm_state': self.arm.getCurrentState(),
                'target_position': self.target_position.copy(),
                'target_orientation': self.target_orientation.copy(),
                'current_ee_pose': self._get_current_ee_pose(),
                'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
                'orientation_error': self._quaternion_distance(
                    self.target_orientation, self._get_current_ee_orientation()
                ),
                'cartesian_directions': self.previous_cartesian_directions.copy() if self.has_previous_directions else np.zeros(7),
                'actual_linear_speed': self.previous_linear_vel if self.has_previous_directions else 0.0,
                'actual_angular_speed': self.previous_angular_vel if self.has_previous_directions else 0.0,
                'gripper_speed': 0.0,
                'dt_ratio': int(self.dt / self.arm._ctrlComp.dt),
                'new_action_received_during_execution': False,
                'is_inference_running': False,
                'action_queue_size': self.action_queue.qsize()
            }
            return observation, reward, done, info
        
        # Get the action from inference result
        result_bytes = bytes(self.inference_result)
        if result_bytes and result_bytes != b'\x00' * len(self.inference_result):
            new_action = np.frombuffer(result_bytes, dtype=np.float64)
            print(f"Got action from completed inference: {new_action}")
            
            # Update target pose from new action
            self.target_position = new_action[:3]
            self.target_orientation = self._normalize_quaternion(new_action[3:7])
            if self.has_gripper:
                self.target_gripper = new_action[7]
            
            # Get current state and calculate directions
            self._update_state()
            current_ee_pose = self._get_current_ee_pose()
            current_pos = current_ee_pose[:3]
            current_quat = current_ee_pose[3:7]
            current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
            
            # Calculate cartesian directions for full dt
            cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
                self._calculate_cartesian_directions(
                    self.target_position, self.target_orientation, self.target_gripper,
                    current_pos, current_quat, current_gripper_pos, self.dt
                )
            
            # Execute cartesianCtrlCmd for full dt
            dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
            for i in range(dt_ratio):
                self.arm.cartesianCtrlCmd(cartesian_directions, self.angular_vel, self.linear_vel)
                time.sleep(self.arm._ctrlComp.dt)
            
            # Store current directions for next step
            self.previous_cartesian_directions = cartesian_directions.copy()
            self.previous_angular_vel = self.angular_vel
            self.previous_linear_vel = self.linear_vel
            self.has_previous_directions = True
            
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
                'dt_ratio': dt_ratio,
                'new_action_received_during_execution': True,
                'is_inference_running': False,
                'action_queue_size': self.action_queue.qsize()
            }
            
            self.episode_step += 1
            return observation, reward, done, info
            
        else:
            print("No action available from inference result")
            # Fallback to current state
            self._update_state()
            observation = self._get_observation()
            reward = self._get_reward()
            done = self._is_done()
            info = {
                'fsm_state': self.arm.getCurrentState(),
                'target_position': self.target_position.copy(),
                'target_orientation': self.target_orientation.copy(),
                'current_ee_pose': self._get_current_ee_pose(),
                'position_error': np.linalg.norm(self.target_position - self._get_current_ee_position()),
                'orientation_error': self._quaternion_distance(
                    self.target_orientation, self._get_current_ee_orientation()
                ),
                'cartesian_directions': self.previous_cartesian_directions.copy() if self.has_previous_directions else np.zeros(7),
                'actual_linear_speed': self.previous_linear_vel if self.has_previous_directions else 0.0,
                'actual_angular_speed': self.previous_angular_vel if self.has_previous_directions else 0.0,
                'gripper_speed': 0.0,
                'dt_ratio': int(self.dt / self.arm._ctrlComp.dt),
                'new_action_received_during_execution': False,
                'is_inference_running': False,
                'action_queue_size': self.action_queue.qsize()
            }
            return observation, reward, done, info
    
    def step(self, global_step = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step with end-effector pose control using cartesianCtrlCmd.
        Now supports non-blocking inference system.
        
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Track if new action was received during execution
        new_action_received_during_execution = False
        
        # Get current state first
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # For first step after reset, maintain current pose as target
        if not self.has_previous_directions:
            
            
            self.target_position = current_pos.copy()
            self.target_orientation = current_quat.copy()
            self.target_gripper = current_gripper_pos
            # print("First step: maintaining current pose as target")
        
        
        # Use previous directions if available, otherwise calculate new ones
        if self.has_previous_directions:
            # Calculate directions to previous target pose for inference time
            cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
                self._calculate_cartesian_directions(
                    self.previous_target_position, self.previous_target_orientation, self.previous_target_gripper,
                    current_pos, current_quat, current_gripper_pos, self.estimate_inference_time
                )
            print("Using previous target pose for inference movement")
        else:
            # First step after reset - calculate initial directions
            cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
                self._calculate_cartesian_directions(
                    self.target_position, self.target_orientation, self.target_gripper,
                    current_pos, current_quat, current_gripper_pos, self.dt # actual_control_time
                )
            print("Calculated initial cartesian directions (first step)")
        
        # Calculate dt ratio for internal loop using actual control time
        dt_ratio = int(self.dt / self.arm._ctrlComp.dt)
        
        start = time.time()
        # Execute cartesianCtrlCmd for dt_ratio iterations with dynamic action checking
        for i in range(dt_ratio):
            # Check if there's a new action from non-blocking inference during execution
            try:
                new_action = self.action_queue.get_nowait()
                new_action_received_during_execution = True
                print(f"Got new action at iteration {i}")
                
                # Calculate remaining time for this step
                remaining_iterations = dt_ratio - i
                remaining_time = remaining_iterations * self.arm._ctrlComp.dt
                
                # Handle new action and update directions
                cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
                    self._handle_new_action_during_execution(new_action, remaining_time)
                
            except:
                # No new action available, continue with current directions
                pass
            # print('Global step: ', global_step, 'Iteration in inner control loop: ', i, 'angular vel: ', self.angular_vel, 'linear vel: ', self.linear_vel)
            
            # Measure cartesianCtrlCmd time
            cmd_start = time.time()
            self.arm.cartesianCtrlCmd(cartesian_directions, self.angular_vel, self.linear_vel)
            cmd_time = time.time() - cmd_start
            
            # Measure sleep time
            sleep_start = time.time()
            time.sleep(self.arm._ctrlComp.dt)
            sleep_time = time.time() - sleep_start
            
            if i % 1 == 0:  # Print every iteration
                process_alive = self.inference_process and self.inference_process.is_alive()
                print(f'Iteration {i}: dt={self.arm._ctrlComp.dt}, cmd={cmd_time*1000:.2f}ms, sleep={sleep_time*1000:.2f}ms, process_alive={process_alive}')
                print(f'Cartesian directions: {cartesian_directions}')
        
        print('Total time for inner control loop: ', time.time() - start)

        # Store current target pose for next step's inference movement
        self.previous_target_position = self.target_position.copy()
        self.previous_target_orientation = self.target_orientation.copy()
        self.previous_target_gripper = self.target_gripper
        
        # Store current directions for next step's inference movement
        self.previous_cartesian_directions = cartesian_directions.copy()
        self.previous_angular_vel = self.angular_vel
        self.previous_linear_vel = self.linear_vel
        self.has_previous_directions = True
        
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
            'dt_ratio': dt_ratio,
            # 'move_during_inference': self.move_during_inference,
            'new_action_received_during_execution': new_action_received_during_execution,
            'is_inference_running': self.inference_process and self.inference_process.is_alive(),
            'action_queue_size': self.action_queue.qsize()
        }
        
        self.episode_step += 1
        return observation, reward, done, info
    
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
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize quaternion
        q = self._normalize_quaternion(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Convert to rotation matrix
        R_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R_matrix
    
    def _rotation_matrix_to_quaternion(self, R_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion."""
        # Shepperd's method for numerical stability
        trace = np.trace(R_matrix)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R_matrix[2, 1] - R_matrix[1, 2]) / s
            qy = (R_matrix[0, 2] - R_matrix[2, 0]) / s
            qz = (R_matrix[1, 0] - R_matrix[0, 1]) / s
        elif R_matrix[0, 0] > R_matrix[1, 1] and R_matrix[0, 0] > R_matrix[2, 2]:
            s = np.sqrt(1.0 + R_matrix[0, 0] - R_matrix[1, 1] - R_matrix[2, 2]) * 2  # s = 4 * qx
            qw = (R_matrix[2, 1] - R_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (R_matrix[0, 1] + R_matrix[1, 0]) / s
            qz = (R_matrix[0, 2] + R_matrix[2, 0]) / s
        elif R_matrix[1, 1] > R_matrix[2, 2]:
            s = np.sqrt(1.0 + R_matrix[1, 1] - R_matrix[0, 0] - R_matrix[2, 2]) * 2  # s = 4 * qy
            qw = (R_matrix[0, 2] - R_matrix[2, 0]) / s
            qx = (R_matrix[0, 1] + R_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (R_matrix[1, 2] + R_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R_matrix[2, 2] - R_matrix[0, 0] - R_matrix[1, 1]) * 2  # s = 4 * qz
            qw = (R_matrix[1, 0] - R_matrix[0, 1]) / s
            qx = (R_matrix[0, 2] + R_matrix[2, 0]) / s
            qy = (R_matrix[1, 2] + R_matrix[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def _pose_to_transformation_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self._quaternion_to_rotation_matrix(quaternion)
        T[:3, 3] = position
        return T
    
    
    
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
    
    
    def _handle_new_action_during_execution(self, new_action, remaining_time):
        """
        Handle new action received during cartesianCtrlCmd execution.
        
        Args:
            new_action: New action received from inference
            remaining_time: Remaining time for this step
            
        Returns:
            cartesian_directions: Updated cartesian directions
            actual_linear_speed: Updated linear speed
            actual_angular_speed: Updated angular speed
            gripper_speed: Updated gripper speed
        """
        # print(f"New action received during execution: {new_action}")
        
        # Update target pose from new action
        self.target_position = new_action[:3]
        self.target_orientation = new_action[3:7]
        if self.has_gripper:
            self.target_gripper = new_action[7]
        
        # Normalize quaternion
        self.target_orientation = self._normalize_quaternion(self.target_orientation)
        
        # Update current state
        self._update_state()
        current_ee_pose = self._get_current_ee_pose()
        current_pos = current_ee_pose[:3]
        current_quat = current_ee_pose[3:7]
        current_gripper_pos = current_ee_pose[7] if self.has_gripper else 0.0
        
        # Recalculate directions for remaining time
        cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed = \
            self._calculate_cartesian_directions(
                self.target_position, self.target_orientation, self.target_gripper,
                current_pos, current_quat, current_gripper_pos, remaining_time
            )
        
        # print(f"Updated cartesian directions: {cartesian_directions}")
        return cartesian_directions, actual_linear_speed, actual_angular_speed, gripper_speed
