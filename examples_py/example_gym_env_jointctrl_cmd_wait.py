#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using joint commands
and non-blocking step execution.

This example demonstrates how to use the EEPoseCtrlJointCmdWrapper with the new
wait argument to control the Z1 robot arm's end-effector pose using jointCtrlCmd.
"""

import sys
import os
import numpy as np
import time
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env_jointctrl_wait_R import EEPoseCtrlJointCmdWrapper

# # custom (hand-made), 3x5 charuco reference marker: 1
T_B_M = np.array([[0.00000000, 0.00000000, 1.00000000, 0.86901688],
                [ 0.00000000, -1.00000000, 0.00000000,  0.0715125 ],
                [1.00000000, 0.00000000,  0.00000000,  0.57236933],
                [ 0.,          0.,          0.,          1.        ]])

# # # custom (hand-made)
# T_B_M = np.array([[-0.0211232, 0.00961882, -0.99973061, 0.86901688],
#                 [ 0.00934025, -0.99990818, -0.00981788,  0.0865125 ],
#                 [-0.99973325, -0.00954512,  0.02103141,  0.55736933],
#                 [ 0.,          0.,          0.,          1.        ]])

# original (accurate value when we make the robot to reach the [0,0,0.4] in marker coordinate) [0.5507966 , 0.12502528, 0.28048738]
# T_B_M = np.array([[-0.0211232, 0.00961882, -0.99973061, 0.87401688],
#                 [ 0.00934025, -0.99990818, -0.00981788,  0.0965125 ],
#                 [-0.99973325, -0.00954512,  0.02103141,  0.54736933],
#                 [ 0.,          0.,          0.,          1.        ]])

class DummyMLP(nn.Module):
    """Simple dummy MLP model for thread-torch compatibility testing."""
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=8):
        super(DummyMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def inference_function(observation, square_vertices, orientation_vertices, total_steps, current_step, dummy_mlp, sequence_length=16, inference_time=0.15, env=None):
    """
    Simulate inference function that takes observation and returns action sequence.
    This uses the same square sequence generation as the original example.
    
    Args:
        observation: Current observation from the environment
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        dummy_mlp: Dummy MLP model for thread-torch compatibility testing
        sequence_length: Length of target pose sequence to generate
        inference_time: Time to simulate inference (seconds)
        
    Returns:
        action_sequence: Generated action sequence [sequence_length, 8]
    """
    print(f"Starting inference with observation shape {observation.shape} (simulating {inference_time:.3f}s)...")
    
    # Dummy MLP inference for thread-torch compatibility testing
    with torch.no_grad():
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        dummy_output = dummy_mlp(obs_tensor)
        print(f"Dummy MLP inference completed: device: {dummy_output.device}, input shape {obs_tensor.shape}, output shape {dummy_output.shape}")
    
    start = time.time()
    for _ in range(10):
        env._get_current_ee_pose()
    print(f"In inference function, get_current_ee_pose time: {time.time() - start:.6f}s")

    # Simulate inference time
    time.sleep(inference_time)
    
    # Extract current position and orientation from observation
    current_pos = observation[12:15]  # End-effector position
    current_orient = observation[15:19]  # End-effector orientation (quaternion in [x,y,z,w] format)
    current_gripper = observation[19]  # Gripper position
    
    # Use the same square sequence generation as the original example
    action_sequence = generate_action_sequence(
        current_pos, current_orient, current_gripper, 
        square_vertices, orientation_vertices, 
        total_steps, current_step, sequence_length
    )
    
    print(f"Inference completed: generated sequence with shape {action_sequence.shape}")
    return action_sequence


def generate_action_sequence(base_position, base_orientation, base_gripper, square_vertices, orientation_vertices, total_steps, current_step, sequence_length=10):
    """
    Generate a target pose sequence that follows the square path.
    This function generates the sequence directly without simulation of inference time.
    
    Args:
        base_position: Base position to use as starting point
        base_orientation: Base orientation to use
        base_gripper: Base gripper position to use
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        sequence_length: Length of target pose sequence to generate
        
    Returns:
        target_sequence: Pre-computed target pose sequence [sequence_length, 8]
    """
    sequence = np.zeros((sequence_length, 8))
    
    # Calculate square path parameters based on total_steps
    total_edges = 4  # 4 edges of the square
    steps_per_edge = total_steps // total_edges
    
    for i in range(sequence_length):
        # Calculate the actual step number in the total path
        actual_step = current_step + i
        
        # Calculate which edge we're on and position along that edge
        edge_index = actual_step // steps_per_edge
        step_in_edge = actual_step % steps_per_edge
        
        # Ensure we don't go beyond the last vertex
        if edge_index >= 4:
            edge_index = 3
            step_in_edge = steps_per_edge - 1
        
        # Calculate current target position along the current edge
        start_vertex = square_vertices[edge_index]
        end_vertex = square_vertices[edge_index + 1]
        
        # Interpolate position along the edge
        t = step_in_edge / steps_per_edge
        target_position = start_vertex + t * (end_vertex - start_vertex)
        
        # Interpolate orientation along the edge
        start_orientation = orientation_vertices[edge_index]
        end_orientation = orientation_vertices[edge_index + 1]
        
        # Use SLERP (Spherical Linear Interpolation) for proper quaternion interpolation
        # start_orientation and end_orientation are in [x,y,z,w] format
        start_rot = R.from_quat(start_orientation)
        end_rot = R.from_quat(end_orientation)
        
        # Create a rotation that represents the interpolation
        # This uses scipy's built-in SLERP functionality
        new_rot = start_rot * (start_rot.inv() * end_rot) ** t
        
        # Convert back to quaternion (returns [x,y,z,w] format)
        new_orientation = new_rot.as_quat().squeeze()
        
        # Gripper: alternate between open and close for each edge
        # Edge 0: open (1), Edge 1: close (-1), Edge 2: open (1), Edge 3: close (-1)
        if edge_index % 2 == 0:
            new_gripper = -1.0  # Open
        else:
            new_gripper = 0.0  # Close
        
        # Combine into action: [x, y, z, qx, qy, qz, qw, gripper]
        # new_orientation is in [x,y,z,w] format, which matches the expected action format
        sequence[i] = np.concatenate([target_position, new_orientation, [new_gripper]])
    
    return sequence

def backproject_pixel(K, u, v, depth, dist_coeffs=None):
    """
    Backproject a 2D pixel coordinate to 3D point in camera coordinates.
    If dist_coeffs is provided, undistorts the pixel coordinate first.
    
    Args:
        K: (3,3) camera intrinsic matrix
        u, v: pixel coordinates
        depth: depth value (z coordinate in camera frame)
        dist_coeffs: (5,) or None, distortion coefficients
    
    Returns:
        point3d: (3,) 3D point in camera coordinates
    """
    if dist_coeffs is not None:
        # Undistort the pixel coordinate first
        pixel = np.array([[u, v]], dtype=np.float32)
        pixel_undist = cv2.undistortPoints(pixel, K, dist_coeffs, P=K)
        u_undist = pixel_undist[0, 0, 0]
        v_undist = pixel_undist[0, 0, 1]
    else:
        u_undist = u
        v_undist = v
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = depth
    x = (u_undist - cx) * z / fx
    y = (v_undist - cy) * z / fy
    return np.array([x, y, z])
    
def get_dift_point_base_frame_data_for_debug(camera_matrix=None, dist_coeffs=None):
    import zarr
    from im2flow2act.common.utility.zarr import parallel_reading
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    dift_point_tracking_sequence = data_buffer[f"episode_{episode_idx}/dift_point_tracking_sequence"][:, :, :3].copy().transpose(1, 0, 2) # [N, T, 4 -> 3] -> [T, N, 3] camera frame
    T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    dift_points = data_buffer[f"episode_{episode_idx}/dift_points"][:].copy() # [N, 2]
    assert dift_point_tracking_sequence.shape[0] == T_mc_transformation.shape[0]
    
    T, N, _ = dift_point_tracking_sequence.shape
    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    

    dift_points_custom_unprojected_list = []
    depth = parallel_reading(group=data_buffer[f"episode_{episode_idx}/camera_0"], array_name="depth")[0].astype(np.float32) / 1000.0
    for dift_x, dift_y in dift_points:
        dift_points_custom_unprojected = backproject_pixel(camera_matrix, dift_x, dift_y, depth[dift_y, dift_x], dist_coeffs=dist_coeffs)
        dift_points_custom_unprojected_list.append(dift_points_custom_unprojected)
    dift_points_custom_unprojected = np.tile(np.stack(dift_points_custom_unprojected_list), (T, 1, 1)) # [T, N, 3]
    dift_points_custom_unprojected_c_homo = np.concatenate([dift_points_custom_unprojected, np.ones((T, N, 1))], axis=2) # [T, N, 4]
    dift_points_custom_unprojected_c = dift_points_custom_unprojected.copy()
    # [T, 4, 4] @ [T, N, 4] -> [T, N, 4]
    dift_points_custom_unprojected_b = np.einsum('hij,hkj->hki', T_bc_transformation, dift_points_custom_unprojected_c_homo)[:, :, :3]


    
    
    # Convert to homogeneous coordinates: [T, N, 3] -> [T, N, 4]
    camera_points_homo = np.concatenate([
        dift_point_tracking_sequence, 
        np.ones((T, N, 1))
    ], axis=2)  # [T, N, 4]
    
    # Reshape for batch matrix multiplication: [T, N, 4] -> [T*N, 4]
    camera_points_homo_flat = camera_points_homo.reshape(-1, 4)  # [T*N, 4]
    
    # Expand transformation matrices: [T, 4, 4] -> [T*N, 4, 4]
    T_bc_expanded = np.repeat(T_bc_transformation, N, axis=0)  # [T*N, 4, 4]
    T_mc_expanded = np.repeat(T_mc_transformation, N, axis=0)  # [T*N, 4, 4]
    
    # Batch matrix multiplication: [T*N, 4, 4] @ [T*N, 4] -> [T*N, 4]
    base_points_homo_flat = np.einsum('ijk,ik->ij', T_bc_expanded, camera_points_homo_flat)  # [T*N, 4]
    marker_points_homo_flat = np.einsum('ijk,ik->ij', T_mc_expanded, camera_points_homo_flat)  # [T*N, 4]
    
    # Convert back to 3D coordinates and reshape: [T*N, 4] -> [T*N, 3] -> [T, N, 3]
    dift_point_base_frame = base_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]
    dift_point_marker_frame = marker_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]



    # point_in_front_of_the_marker_homo = np.array([0.0, 0.0, 0.4, 1.0])
    point_in_front_of_the_marker_homo = np.array([0.0, 0.0, -0.4, 1.0])
    point_in_front_of_the_marker_base_frame = np.einsum('ij,j->i', T_B_M, point_in_front_of_the_marker_homo)[:3]

    
    return dift_point_base_frame, dift_point_marker_frame, dift_point_tracking_sequence, point_in_front_of_the_marker_base_frame, dift_points_custom_unprojected_c, dift_points_custom_unprojected_b

def get_estimated_hand_pose_base_frame_data_for_debug():
    import zarr
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_dist_coeff"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    T = T_mc_transformation.shape[0]

    # original ver
    # proprioception = data_buffer[f"episode_{episode_idx}/proprioception"][:].copy() # [T, 7], camera frame

    # NOTE: for debug
    import pickle
    def load_data_dict(data_path):
        """Load the data dictionary from pickle file"""
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    data_dict = load_data_dict(data_buffer_path + f'/episode_{episode_idx}/data_dict.pkl')
    
    # use first obs
    R_cg_opt = data_dict['R_cg_opt_trajectory'][:] # [T, 3, 3]
    t_cg_opt_depth = data_dict['t_cg_opt_depth_trajectory'][:] # [T, 3]
    t_cg_opt = data_dict['t_cg_opt_trajectory'][:] # [T, 3]
    t_cg_closed = data_dict['t_cg_closed_trajectory'][:] # [T, 3]
    euler_cg_opt = R.from_matrix(R_cg_opt).as_euler('xyz') # [T, 3]
    temp_gripper = np.tile(np.array([0.0]), (T, 1)) # [T, 1]
    # dscho NOTE: depth-based one is much smoother when using accurate T_B_M. accuracy is slightly better.
    proprioception = np.concatenate([t_cg_opt_depth, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_opt, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_closed, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]

    # debugging (depth scaling)
    # proprioception[:, :3] = proprioception[:, :3] * 1.05 # 2.5% scaling (custom calibration)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@ apply scaling to the proprioception for debugging")
    # time.sleep(2)


    
    
    assert proprioception.shape[0] == T_mc_transformation.shape[0]
    
    T = proprioception.shape[0]
    proprioception_se3 = np.tile(np.eye(4), (T, 1, 1)) # [T, 4, 4]
    proprioception_se3[:, :3, :3] = R.from_euler('xyz', proprioception[:, 3:6]).as_matrix() # [T, 4, 4]
    proprioception_se3[:, :3, 3] = proprioception[:, :3] # [T, 4, 4]

    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    proprioception_se3_b = np.einsum('hij,hjk->hik', T_bc_transformation, proprioception_se3) # [T, 4, 4]
    
    pos = proprioception_se3_b[:, :3, 3] # [T, 3]
    euler = R.from_matrix(proprioception_se3_b[:, :3, :3]).as_euler('xyz')
    proprioception_b = np.concatenate([pos, euler, proprioception[:, 6:7]], axis=1) # [T, 7]
    return proprioception_b, proprioception

def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Non-blocking Step Execution")
    print("=" * 80)
    
    # Create dummy MLP model for thread-torch compatibility testing
    # Note: observation shape is 21, so we need to match that
    dummy_mlp = DummyMLP(input_dim=21, hidden_dim=4096, output_dim=8)
    dummy_mlp.eval()  # Set to evaluation mode
    print(f"Created dummy MLP model: {dummy_mlp}")
    
    # Create the environment with 2Hz control frequency
    control_frequency = 2 # 5
    sequence_length = 16   # Length of future target pose sequences
    step_interval = 1.0 / control_frequency  # Time between steps in seconds
    
    env = EEPoseCtrlJointCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 2Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        joint_speed=1.0,  # Joint speed limit
        sequence_length=sequence_length,  # Length of future sequences
        use_current_joint_pos_when_ik_fails = True,
        
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    try:
        # Reset the environment
        print("Resetting environment...")
        # joint_angle = np.array([1.0, 1.5, -1.0, -0.54, 0.0, 0.0])
        # joint_angle = np.array([-0.8, 2.572, -1.533, -0.609, 1.493, 1.004])
        joint_angle = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward
        # joint_angle = None
        obs = env.reset(joint_angle) # , option="lowcmd"
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial joint positions: {obs[:6]}")
        print(f"Initial end-effector position: {obs[12:15]}")
        print(f"Initial end-effector orientation (quaternion [x,y,z,w]): {obs[15:19]}")
        print(f"Initial gripper position: {obs[19]}")
        print()
    
        
        # Example: Non-blocking step execution with square movement
        print("Example: Non-blocking step execution with square movement")
        print("Using wait=False for non-blocking step execution")
        print("Gripper will alternate between open and close for each edge")
        print("Orientation will change roll: 0°, +90°, 0°, -90°, 0° for each edge")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Step interval: {step_interval:.3f}s")
        print(f"Joint speed limit: {env.joint_speed} rad/s")
        print(f"Sequence length: {sequence_length}")
        print()
        
        # Get current end-effector position and orientation from observation
        original_position = obs[12:15].copy()  # Store original position
        current_orientation = obs[15:19]  # Current end-effector orientation (quaternion in [x,y,z,w] format)
        target_gripper = 0 
        
        print(f"Original EE position: {original_position}")
        print(f"Current EE orientation (quaternion [x,y,z,w]): {current_orientation}")
        print()
        
        # dscho debug to specify the position and orientation
        original_position = np.array([0.41145274, -0.00121779, 0.40713578])
        # Convert from [w,x,y,z] to [x,y,z,w] format for scipy compatibility
        # current_orientation_wxyz = np.array([0.9998209, -0.0011671, -0.01868073, -0.00280654])  # [w,x,y,z]
        current_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        current_orientation = np.array([current_orientation_wxyz[1], current_orientation_wxyz[2], 
                                       current_orientation_wxyz[3], current_orientation_wxyz[0]])  # [x,y,z,w]

        # Define square vertices in YZ plane (0.1m x 0.1m square)
        square_size = 0.2  # 0.1m
        # square_vertices = [
        #     original_position + np.array([0.0, 0.0, 0.0]),           # Start point
        #     original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
        #     original_position + np.array([0.0, square_size, square_size]),  # Y+0.1, Z+0.1
        #     original_position + np.array([0.0, 0.0, square_size]),   # Z+0.1
        #     original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        # ]

        square_vertices = [
            original_position + np.array([0.0, 0.0, 0.0]),           # Start point
            original_position + np.array([0.0, 0.0, -square_size]),   # Z-0.1
            original_position + np.array([0.0, square_size, -square_size]),  # Y+0.1, Z-0.1
            original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
            original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        ]

        
        # D435 depth-based dift point (leftside bottle cap): array([0.592886  , 0.24988669, 0.1399751 ]), (on the socket): array([ 0.58816114, -0.0290327 , -0.00381193]), 
        # unidepth-based dift point (leftside bottle cap): array([0.62878956, 0.22977131, 0.13334631]), (on the socket): array([0.52487209, 0.02690133, 0.01839266]), 
        # NOTE
        dift_point_base_frame, dift_point_marker_frame, point_in_front_of_the_marker_base_frame = get_dift_point_base_frame_data_for_debug() # [T, N, 3]
        DEBUG_POINT = dift_point_base_frame[0,1] + np.array([0.0, 0.0, 0.1])
        # DEBUG_POINT = point_in_front_of_the_marker_base_frame

        # proprioception_base_frame = get_estimated_hand_pose_base_frame_data_for_debug() # [T, 7]
        # DEBUG_POINT = proprioception_base_frame[0, :3] # + np.array([0.0, 0.0, 0.1])
        
        
        assert DEBUG_ACTION[-1] == 0.0, "assume camera is attached, so the gripper should not be moved"
        DEBUG_ACTION = np.concatenate([DEBUG_POINT, current_orientation,  np.array([0.0])])
        print('@@@@@@@@@@@@@@@@@@@@@@@@@ Currently using DEBUG_ACTION :', DEBUG_ACTION)

        
        DEBUG_ROLL_ANGLES = [0, 0, 0, 0, 0]
        # Define orientation vertices with roll changes: 0°, +90°, 0°, -90°, 0°
        # roll_angles = [0, np.pi/4, 0, -np.pi/4, 0]  # 0°, +90°, 0°, -90°, 0°
        
        roll_angles = DEBUG_ROLL_ANGLES
        
        
        orientation_vertices = []
        
        for roll_angle in roll_angles:
            # Convert base orientation (quaternion in [x,y,z,w] format) to rotation matrix, apply roll, convert back
            base_rot = R.from_quat(current_orientation)  # current_orientation is in [x,y,z,w] format
            roll_rot = R.from_euler('x', roll_angle)
            new_rot = base_rot * roll_rot
            orientation_vertices.append(new_rot.as_quat())  # Returns [x,y,z,w] format
        
        print("Square movement plan:")
        for i, vertex in enumerate(square_vertices):
            gripper_state = "Open" if i % 2 == 0 else "Close"
            print(f"  Vertex {i}: {vertex} (Gripper: {gripper_state})")
        print()

        
        total_steps = 20
        inference_time = 0.15  # Inference time in seconds
        
        print(f"Running non-blocking control with overlapped inference for {total_steps} steps")
        print(f"Inference time: {inference_time}s")
        print(f"Step interval: {step_interval:.3f}s")
        print("Note: Get observation → Execute step immediately → Run inference while robot moves")
        print()
        
        # Track errors for average calculation
        position_errors = []
        orientation_errors = []
        
        # Sequence tracking variables
        current_action_sequence = None
        current_action_index = 0
        latest_inference_result = None
        
        # Generate initial action sequence from o_0 (before for loop)
        print("Generating initial action sequence from o_0...")
        current_action_sequence = inference_function(
            obs, square_vertices, orientation_vertices, total_steps, 0, dummy_mlp, sequence_length, inference_time, env
        ) # o_0 -> a_0, a_1, a_2, ...
        current_action_index = 0
        
        # Main execution loop for all steps
        for step in range(total_steps):
            print(f"\nStep {step}: Processing")
            print("-" * 40)
            
            # Select action for current step (use current sequence)
            if current_action_sequence is not None and current_action_index < len(current_action_sequence):
                # Use action from current sequence
                action = current_action_sequence[current_action_index] # a_t
                current_action_index += 1
                print(f"Step {step}: Using action a_{step} from sequence (index {current_action_index-1})")
            else:
                # Fallback: maintain current pose
                current_pos = obs[12:15]
                current_orient = obs[15:19]
                action = np.concatenate([current_pos, current_orient, [target_gripper]])
                print(f"Step {step}: No action available, maintaining current pose")
            
            # Print gripper command
            gripper_cmd = action[7] if len(action) > 7 else 0.0
            print(f"Step {step}: Gripper Command = {gripper_cmd:.3f} ({'Open' if gripper_cmd > 0 else 'Close' if gripper_cmd < 0 else 'Neutral'})")
            
            # Execute step with non-blocking execution
            print(f"Step {step}: Executing action a_{step} with non-blocking...")
            
            start = time.time()
            
            # env.step(action, wait=False) # a_t
            env.step(DEBUG_ACTION, wait=False) # a_t


            print(f"Step {step}: Started non-blocking execution in {time.time() - start:.6f}s")
            
            # Run inference in main process while robot is moving (step > 0 and not last step)
            if step > 0 and step < total_steps - 1:  # Don't run inference for step 0 or last step
                print(f"Step {step}: Running inference with o_{step} while robot moves...")
                inference_start_time = time.time()
                latest_inference_result = inference_function(
                    obs, square_vertices, orientation_vertices, total_steps, step, dummy_mlp, sequence_length, inference_time, env
                ) # o_t+1 -> a_t+1, a_t+2, a_t+3, ...
                inference_time_actual = time.time() - inference_start_time
                print(f"Step {step}: Inference completed in {inference_time_actual:.3f}s")
                
                # Update action sequence with latest inference result for next step
                if latest_inference_result is not None:
                    current_action_sequence = latest_inference_result
                    current_action_index = 1  # Reset index for new sequence, NOTE: we assume that inference is completed before env.step is finished
                    print(f"Step {step}: Updated action sequence for next steps")
            
            
            print(f"Step {step}: Waiting for o_{step+1}...")
            start = time.time()
            while not env.is_step_complete():
                time.sleep(0.001)  # Small sleep to avoid busy waiting
            print(f"Step {step}: Waited for {time.time() - start:.6f}s for o_{step+1}")

            # Get the result from background execution
            start = time.time()
            result = env.get_step_result()
            if result:
                obs, reward, done, info = result # o_t+1, r_t, etc
                print(f"Step {step}: Received o_{step+1}, time taken: {time.time() - start:.6f}s")
            else:
                print(f"Step {step}: Warning - No result from background step")
                # Use previous observation if no result
                pass
            
            # Print gripper state
            gripper_state = obs[19] if len(obs) > 19 else 0.0
            print(f"Step {step}: Gripper State = {gripper_state:.3f}")
            
            # Collect errors for average calculation
            position_errors.append(info['position_error'])
            orientation_errors.append(info['orientation_error'])
            
            if done:
                print(f"Episode finished at step {step}!")
                break
        
        
        
        # Print final status
        final_pos = obs[12:15]
        final_error = info['position_error']
        print(f"\nNon-blocking square movement completed!")
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Final position error: {final_error:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final joint directions: {info['joint_directions']}")
        print(f"Final speeds - Joint: {info['actual_joint_speed']:.3f}, Gripper: {info['gripper_speed']:.3f}")
        print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        print(f"Final sequence index: {env.sequence_index}")
        print(f"Final sequence length: {len(env.current_sequence) if env.current_sequence is not None else 0}")
        
        # Calculate and print average errors
        if position_errors:
            avg_position_error = np.mean(position_errors)
            avg_orientation_error = np.mean(orientation_errors)
            print(f"\nAverage Errors:")
            print(f"  Average position error: {avg_position_error:.4f} m")
            print(f"  Average orientation error: {avg_orientation_error:.4f} rad")
            print(f"  Total steps executed: {len(position_errors)}")
            print(f" last step error: {position_errors[-1]:.4f} m, {orientation_errors[-1]:.4f} rad")

        
        print()
        print("\nNon-blocking square movement with sequence handling completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Closing environment...")
        env.close()
        print("Environment closed.")


if __name__ == "__main__":    
    # Run main example
    main()
