#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using joint commands with IK
and non-blocking step execution.

This example demonstrates how to use the EEPoseCtrlJointCmdWrapper with the new
wait argument to control the Z1 robot arm's end-effector pose using joint-level commands with IK.
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
from envs.z1_env_low_cmd_wait_R import EEPoseCtrlJointCmdWrapper


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
    control_frequency = 2 # 2Hz control frequency
    sequence_length = 16   # Length of future target pose sequences
    step_interval = 1.0 / control_frequency  # Time between steps in seconds
    
    env = EEPoseCtrlJointCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 2Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        sequence_length=sequence_length,  # Length of future sequences
        
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    try:
        # Reset the environment
        print("Resetting environment...")
        # joint_angle = np.array([1.0, 1.5, -1.0, -0.54, 0.0, 0.0])
        joint_angle = np.array([-0.8, 2.572, -1.533, -0.609, 1.493, 1.004])
        joint_angle = None
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
        print(f"dt_ratio: {int(env.dt / env.arm._ctrlComp.dt)} control steps")
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
        current_orientation_wxyz = np.array([0.9998209, -0.0011671, -0.01868073, -0.00280654])  # [w,x,y,z]
        current_orientation = np.array([current_orientation_wxyz[1], current_orientation_wxyz[2], 
                                       current_orientation_wxyz[3], current_orientation_wxyz[0]])  # [x,y,z,w]

        # Define square vertices in YZ plane (0.1m x 0.1m square)
        square_size = 0.1  # 0.1m
        square_vertices = [
            original_position + np.array([0.0, 0.0, 0.0]),           # Start point
            original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
            original_position + np.array([0.0, square_size, square_size]),  # Y+0.1, Z+0.1
            original_position + np.array([0.0, 0.0, square_size]),   # Z+0.1
            original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        ]
        
        # Define orientation vertices with roll changes: 0°, +90°, 0°, -90°, 0°
        roll_angles = [0, np.pi/4, 0, -np.pi/4, 0]  # 0°, +90°, 0°, -90°, 0°
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

        
        total_steps = 30
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
            obs, square_vertices, orientation_vertices, total_steps, 0, dummy_mlp, sequence_length, 0.01, env
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
            env.step(action, wait=False) # a_t
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
        print(f"Final target joint angles: {info['target_joint_angles']}")
        print(f"Final current joint angles: {info['current_joint_angles']}")
        print(f"IK success: {info['ik_success']}, iterations: {info['ik_iterations']}, error: {info['ik_error']:.6f}")
        print(f"dt_ratio: {info['dt_ratio']} control steps")
        print(f"cmd_time: {info['cmd_time']:.6f}s")
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
        
        print()
        print("\nNon-blocking square movement with joint commands and IK completed successfully!")
        
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
