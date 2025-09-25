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
from envs.z1_env import EEPoseCtrlCartesianCmdWrapper


def create_inference_function(target_position, target_orientation, target_gripper, inference_time=0.15):
    """
    Create an inference function that simulates neural network inference.
    This function will be called in a separate thread and should handle its own timing.
    
    Args:
        target_position: Target position to use as base for inference
        target_orientation: Target orientation to use as base for inference
        target_gripper: Target gripper position to use as base for inference
        inference_time: Time to simulate inference
    """
    def inference_function():
        # Simulate neural network inference that generates a new action
        # In real scenario, this would be your neural network model
        
        # print(f"Starting neural network inference (simulating {inference_time:.3f}s)...")
        
        # Simulate inference time - this is where your actual neural network would run
        time.sleep(inference_time)
        
        # Generate a small random variation to simulate neural network output
        # noise = np.random.normal(0, 0.01, 3)  # Small random noise
        new_position = target_position # + noise
        
        # Use provided orientation and gripper
        new_orientation = target_orientation
        new_gripper = target_gripper
        
        # Combine into action
        new_action = np.concatenate([new_position, new_orientation, [new_gripper]])
        
        # print(f"Neural network inference completed: generated action {new_action}")
        return new_action
    
    return inference_function


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Non-blocking Inference System")
    print("=" * 80)
    
    # Create the environment with 5Hz control frequency
    control_frequency = 5
    inference_time = 0.15  # Neural network inference time
    
    env = EEPoseCtrlCartesianCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        angular_vel=0.3,  # Angular velocity limit
        linear_vel=0.3,   # Linear velocity limit
        estimate_inference_time=inference_time,  # Estimated inference time for movement calculation
        # move_during_inference=True,  # Enable continuous movement during inference
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
        
        # Example: Non-blocking inference with square movement
        print("Example: Non-blocking inference with square movement")
        print("Using non_blocking_inference for continuous movement")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Angular velocity limit: {env.angular_vel} rad/s")
        print(f"Linear velocity limit: {env.linear_vel} m/s")
        print(f"Inference time: {inference_time}s")
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
        
        # Move along square path in 20 steps
        total_steps = 20
        steps_per_edge = total_steps // 4  # 4 edges of the square
        
        print(f"Moving along square path in {total_steps} steps ({steps_per_edge} steps per edge)")
        print(f"Inference time: {inference_time}s (simulated neural network processing)")
        print()
        
        # Shared step counter for inference function
        step_count = [0]
        
        for step in range(total_steps):
            step_count[0] = step
            
            # Calculate which edge we're on and position along that edge
            edge_index = step // steps_per_edge
            step_in_edge = step % steps_per_edge
            
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
            
            # Start inference if current one is not running
            assert env.inference_function is None, "assume action inference process is not running"

            # Create inference function with the current target position
            inference_func = create_inference_function(
                target_position, current_orientation, target_gripper, inference_time
            )
            env.non_blocking_inference(inference_func, inference_time)
            
            # Send command - this will use new action from inference if available,
            # otherwise use previous directions
            obs, reward, done, info = env.step(step)
            
            # If inference is still running at the end of the loop, wait for it to complete
            if info['is_inference_running']:
                print("Waiting for final inference to complete...")
                obs, reward, done, info = env.wait_for_inference_and_execute()
            
            # Print progress every 10 steps
            if step % 10 == 0:
                current_pos = obs[12:15]
                position_error = info['position_error']
                new_action_received_during_execution = info.get('new_action_received_during_execution', False)
                is_inference_running = info.get('is_inference_running', False)
                action_queue_size = info.get('action_queue_size', 0)
                
                print(f"Step {step:3d}: Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], "
                      f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}], "
                      f"Error: {position_error:.4f}")
                # print(f"         New action received during execution: {new_action_received_during_execution}, "
                #       f"Inference running: {is_inference_running}, "
                #       f"Queue size: {action_queue_size}")
            
            if done:
                print(f"Episode finished at step {step}!")
                break
        
        # Print final status
        final_pos = obs[12:15]
        final_error = info['position_error']
        print(f"\nSquare movement completed!")
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Final position error: {final_error:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final cartesian directions: {info['cartesian_directions']}")
        print(f"Final speeds - Linear: {info['actual_linear_speed']:.3f}, Angular: {info['actual_angular_speed']:.3f}, Gripper: {info['gripper_speed']:.3f}")
        print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        print(f"New action received during execution: {info.get('new_action_received_during_execution', False)}")
        print(f"Is inference running: {info.get('is_inference_running', False)}")
        print(f"Action queue size: {info.get('action_queue_size', 0)}")
        print()
        
        print("\nAll examples completed successfully!")
        
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
