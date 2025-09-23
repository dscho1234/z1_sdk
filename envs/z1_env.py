import sys
import os
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
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
        try:
            if hasattr(self.arm, 'lowstate'):
                self.current_joint_pos = np.array(self.arm.lowstate.getQ())
                self.current_joint_vel = np.array(self.arm.lowstate.getQd())
                if self.has_gripper:
                    self.current_gripper_pos = self.arm.lowstate.getGripperQ()
                    self.current_gripper_vel = self.arm.gripperQd
        except Exception as e:
            print(f"Warning: Could not update state: {e}")
            # Keep previous state if update fails
    
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
        try:
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
        except Exception as e:
            print(f"Error sending MoveJ command: {e}")
        
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
