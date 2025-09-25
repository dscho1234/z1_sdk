#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using cartesian commands
and non-blocking inference system.

This example demonstrates how to use the EEPoseCtrlCartesianCmdWrapper with the new
non-blocking inference system to control the Z1 robot arm's end-effector pose.
"""

import sys
import os
import numpy as np
import time

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env_rtc import EEPoseCtrlCartesianCmdWrapper


def create_inference_function(target_sequence, inference_time=0.15):
    """
    Create an inference function that simulates neural network inference for RTC-style future sequences.
    This function simply returns the pre-computed target sequence after simulating inference time.
    
    Args:
        target_sequence: Pre-computed target pose sequence [sequence_length, 8]
        inference_time: Time to simulate inference
    """
    def inference_function():
        # Simulate neural network inference that generates a future target pose sequence
        # In real scenario, this would be your neural network model (e.g., diffusion/flow model)
        
        print(f"Starting RTC neural network inference (simulating {inference_time:.3f}s)...")
        
        # Simulate inference time - this is where your actual neural network would run
        time.sleep(inference_time)
        
        # Return the pre-computed target sequence
        print(f"RTC neural network inference completed: returning pre-computed sequence with shape {target_sequence.shape}")
        return target_sequence
    
    return inference_function


def generate_square_sequence(base_position, base_orientation, base_gripper, square_vertices, total_steps, current_step, sequence_length=10):
    """
    Generate a target pose sequence that follows the square path.
    The square path is divided into total_steps, and we return sequence_length steps starting from current_step.
    
    Args:
        base_position: Base position to use as starting point
        base_orientation: Base orientation to use
        base_gripper: Base gripper position to use
        square_vertices: List of square vertices defining the path
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
        
        # Orientation: keep base orientation
        new_orientation = base_orientation.copy()
        
        # Gripper: keep base gripper position
        new_gripper = base_gripper
        
        # Combine into action
        sequence[i] = np.concatenate([target_position, new_orientation, [new_gripper]])
    
    return sequence


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - RTC-style Future Sequence Handling")
    print("=" * 80)
    
    # Create the environment with 5Hz control frequency
    control_frequency = 10 # 5
    inference_time = 0.15  # Neural network inference time
    sequence_length = 16   # Length of future target pose sequences
    
    env = EEPoseCtrlCartesianCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        angular_vel=0.3,  # Angular velocity limit
        linear_vel=0.3,   # Linear velocity limit
        sequence_length=sequence_length,  # Length of future sequences from inference
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    try:
        # Reset the environment
        print("Resetting environment...")
        try:
            obs = env.reset()
            print(f"Initial observation shape: {obs.shape}")
            print(f"Initial joint positions: {obs[:6]}")
            print(f"Initial end-effector position: {obs[12:15]}")
            print(f"Initial end-effector orientation: {obs[15:19]}")
            print(f"Initial gripper position: {obs[19]}")
            print()
        except Exception as e:
            print(f"Error during reset: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Example: RTC-style future sequence handling with square movement
        print("Example: RTC-style future sequence handling with square movement")
        print("Using non_blocking_inference for future target pose sequences")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Angular velocity limit: {env.angular_vel} rad/s")
        print(f"Linear velocity limit: {env.linear_vel} m/s")
        print(f"Inference time: {inference_time}s")
        print(f"Sequence length: {sequence_length}")
        print()
        
        # Get current end-effector position and orientation from observation
        original_position = obs[12:15].copy()  # Store original position
        current_orientation = obs[15:19]  # Current end-effector orientation (quaternion)
        target_gripper = 0  # -1 open, 1 close
        
        print(f"Original EE position: {original_position}")
        print(f"Current EE orientation: {current_orientation}")
        print()
        
        # Define square vertices in YZ plane (0.1m x 0.1m square)
        square_size = 0.1  # 0.1m
        square_vertices = [
            original_position + np.array([0.0, 0.0, 0.0]),           # Start point
            original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
            original_position + np.array([0.0, square_size, square_size]),  # Y+0.1, Z+0.1
            original_position + np.array([0.0, 0.0, square_size]),   # Z+0.1
            original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        ]
        
        print("Square movement plan:")
        for i, vertex in enumerate(square_vertices):
            print(f"  Vertex {i}: {vertex}")
        print()

        # Run for a fixed number of steps to demonstrate RTC behavior
        total_steps = 30
        
        
        # Generate the first target sequence
        target_sequence = generate_square_sequence(
            original_position, current_orientation, target_gripper, square_vertices, total_steps, 0, sequence_length
        )
        print(f"Generated first target sequence with shape: {target_sequence.shape}")
        print()
        
        
        
        print(f"Running RTC-style control for {total_steps} steps")
        print(f"Each inference uses a pre-computed sequence of {sequence_length} future target poses")
        print(f"Inference time: {inference_time}s (simulated neural network processing)")
        print()
        
        # RTC-style execution: track inference delay and handle sequence switching
        current_sequence = target_sequence
        env.set_current_sequence(current_sequence, 0)
        
        # Track inference delay (steps passed during inference)
        inference_start_step = 0
        inference_running = False
        
        # Start first inference
        inference_func = create_inference_function(target_sequence, inference_time)
        env.non_blocking_inference(inference_func, inference_time)
        inference_running = True
        inference_start_step = 0
        
        for step in range(total_steps):
            # Check if inference is complete and get new sequence
            if inference_running and not (env.inference_process and env.inference_process.is_alive()):
                # Inference completed, get new sequence
                try:
                    new_sequence = env.action_queue.get_nowait()
                    inference_delay = step - inference_start_step
                    
                    print(f"RTC: Inference completed at step {step}, delay was {inference_delay} steps")
                    print(f"RTC: New sequence shape: {new_sequence.shape}")
                    
                    # RTC-style sequence switching: skip past poses
                    if inference_delay < len(new_sequence):
                        start_index = inference_delay
                        print(f"RTC: Starting from index {start_index} (skipped {inference_delay} past poses)")
                    else:
                        start_index = len(new_sequence) - 1
                        print(f"RTC: All poses are past, using last pose (index {start_index})")
                    
                    # Set new sequence with appropriate start index
                    env.set_current_sequence(new_sequence, start_index)
                    current_sequence = new_sequence
                    inference_running = False
                    
                    # Immediately start new inference after getting the result
                    current_pos = obs[12:15] if 'obs' in locals() else original_position
                    current_orient = obs[15:19] if 'obs' in locals() else current_orientation
                    
                    # Generate new target sequence starting from current step
                    new_target_sequence = generate_square_sequence(
                        current_pos, current_orient, target_gripper, square_vertices, total_steps, step, sequence_length
                    )
                    
                    # Start new inference immediately
                    inference_func = create_inference_function(new_target_sequence, inference_time)
                    env.non_blocking_inference(inference_func, inference_time)
                    inference_running = True
                    inference_start_step = step
                    print(f"RTC: Started new inference immediately at step {step}")
                    
                except:
                    print("RTC: No new sequence available from completed inference")
                    inference_running = False
            
            # Get next action from current sequence
            action = env.get_next_action_from_sequence()
            if action is None:
                print("This line should not be reached, since it means we used all action chunks and still do not have a new sequence.")
                print("RTC: No action available, maintaining current pose.")
                # Maintain current pose if no sequence available
                current_pos = obs[12:15] if 'obs' in locals() else original_position
                current_orient = obs[15:19] if 'obs' in locals() else current_orientation
                action = np.concatenate([current_pos, current_orient, [target_gripper]])
            
            # Execute step with the action
            obs, reward, done, info = env.step(action)
            
            # Print progress every 5 steps
            if step % 5 == 0:
                current_pos = obs[12:15]
                position_error = info['position_error']
                sequence_index = env.sequence_index
                sequence_length_current = len(env.current_sequence) if env.current_sequence is not None else 0
                
                print(f"Step {step:3d}: Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], "
                      f"Error: {position_error:.4f}")
                print(f"         Sequence index: {sequence_index}/{sequence_length_current-1}, "
                      f"Inference running: {inference_running}")
            
            if done:
                print(f"Episode finished at step {step}!")
                break
        
        # Print final status
        final_pos = obs[12:15]
        final_error = info['position_error']
        print(f"\nRTC-style square movement completed!")
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Final position error: {final_error:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final cartesian directions: {info['cartesian_directions']}")
        print(f"Final speeds - Linear: {info['actual_linear_speed']:.3f}, Angular: {info['actual_angular_speed']:.3f}, Gripper: {info['gripper_speed']:.3f}")
        print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        print(f"Final sequence index: {env.sequence_index}")
        print(f"Final sequence length: {len(env.current_sequence) if env.current_sequence is not None else 0}")
        print()
        
        print("\nRTC-style square movement with proper sequence handling completed successfully!")
        
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
