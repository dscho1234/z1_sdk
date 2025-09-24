#!/usr/bin/env python3
"""
Example script demonstrating the Z1 Gym Environment with Low-Level Commands.
This example shows how to use the EEPoseCtrlLowCmdWrapper for end-effector pose control
using inverse kinematics and low-level joint commands.
"""

import sys
import os
import numpy as np
import time

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))

from envs.z1_env_low_cmd import EEPoseCtrlLowCmdWrapper


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Low-Level Command Control")
    print("=" * 60)
    
    # Create the environment with low-level commands
    control_frequency = 5  # 5Hz for low-level control
    inference_time = 0.15  # Neural network inference time (can be changed each time)
    env = EEPoseCtrlLowCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        move_speed=None,  # Moderate speed for smooth movement
        move_timeout=float(1/control_frequency),  # 0.2 seconds timeout for each movement
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    # Reset the environment
    print("Resetting environment...")
    obs = env.reset()
    print()
    
    # Get initial pose
    original_position = obs[12:15]  # End-effector position from observation
    current_orientation = obs[15:19]  # End-effector orientation from observation
    target_gripper = 0.0
    
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
    
    # Move along square path in 25 steps
    total_steps = 25
    steps_per_edge = total_steps // 4  # 4 edges of the square
    
    print(f"Moving along square path in {total_steps} steps ({steps_per_edge} steps per edge)")
    print("Example: Low-level command movement using inverse kinematics")
    print("Using inverse kinematics and direct joint control")
    print(f"Timeout setting: {env.move_timeout} seconds")
    print(f"Inference time: {inference_time}s (simulated neural network processing)")
    print()
    
    for step in range(total_steps):
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
        
        # Create action
        action = np.concatenate([target_position, current_orientation, [target_gripper]])
        
        # Simulate neural network inference time
        time.sleep(inference_time)
        
        # Send command - step() function handles timing internally
        obs, reward, done, info = env.step(action, inference_time)
        
        # Print progress every 5 steps
        if step % 5 == 0:
            current_pos = obs[12:15]
            position_error = info['position_error']
            print(f"Step {step:3d}: Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], "
                  f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}], "
                  f"Error: {position_error:.4f}, Moving: {info['is_moving']}")
        
        if done:
            print(f"Episode finished at step {step}!")
            break
    
    # Print final status
    final_pos = obs[12:15]
    final_error = info['position_error']
    print(f"\nSquare movement completed!")
    print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"Final position error: {final_error:.4f}")
    print(f"Final DT ratio: {info['dt_ratio']}")
    print(f"Actual control time: {info['actual_control_time']:.3f}s, Inference time: {info['inference_time']:.3f}s")
    print()
    
    print("All examples completed successfully!")
    
    # Close the environment
    print("Closing environment...")
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
