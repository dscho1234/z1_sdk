#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using cartesian commands.

This example demonstrates how to use the EEPoseCtrlCartesianCmdWrapper to control
the Z1 robot arm's end-effector pose using cartesianCtrlCmd velocity commands.
"""

import sys
import os
import numpy as np
import time

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env import EEPoseCtrlCartesianCmdWrapper


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - End-Effector Pose Control with Cartesian Commands")
    print("=" * 80)
    
    # Create the environment with 5Hz control frequency
    control_frequency = 5
    inference_time = 0.01 # 15  # Neural network inference time (can be changed each time)
    '''
    How to address slow inference
    1. RTC
    2. super slow velocity (cannot reach the desired pose with the given time, but anyway we can do inference at that pose.)
    '''
    
    env = EEPoseCtrlCartesianCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        angular_vel=0.3,  # Angular velocity limit
        linear_vel=0.3,   # Linear velocity limit
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
        
        # Example: Cartesian velocity control for square movement in YZ plane
        print("Example: Cartesian velocity control for square movement in YZ plane")
        print("Using cartesianCtrlCmd for continuous velocity control")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Angular velocity limit: {env.angular_vel} rad/s")
        print(f"Linear velocity limit: {env.linear_vel} m/s")
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
        
        # Move along square path in 100 steps
        total_steps = 25
        steps_per_edge = total_steps // 4  # 4 edges of the square
        
        print(f"Moving along square path in {total_steps} steps ({steps_per_edge} steps per edge)")
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
            
            # Send command with inference time
            obs, reward, done, info = env.step(action, inference_time)
            
            # Print progress every 10 steps
            if step % 10 == 0:
                current_pos = obs[12:15]
                position_error = info['position_error']
                print(f"Step {step:3d}: Current pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}], "
                      f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}], "
                      f"Error: {position_error:.4f}")
            
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
        print(f"Actual control time: {info['actual_control_time']:.3f}s, Inference time: {info['inference_time']:.3f}s")
        print()
        
        # Additional example: Demonstrate gripper control
        print("\n" + "="*80)
        print("Additional Example: Gripper Control")
        print("="*80)
        
        # Use current state (don't reset)
        current_position = obs[12:15].copy()
        current_orientation = obs[15:19]
        
        # Test gripper opening and closing
        gripper_actions = [
            ("Open gripper", -1.0),
            ("Close gripper", 1.0),
            ("Half open", 0.0)
        ]
        
        for gripper_name, gripper_target in gripper_actions:
            print(f"=== {gripper_name} ===")
            
            action = np.concatenate([current_position, current_orientation, [gripper_target]])
            
            # Run for a few steps to see gripper movement
            for step in range(10):
                obs, reward, done, info = env.step(action, inference_time)
                
                if step % 3 == 0:
                    print(f"  Step {step}: Gripper position: {obs[19]:.3f}, Target: {gripper_target}")
                
                if done:
                    print(f"  Episode finished during {gripper_name}!")
                    break
            
            print(f"  {gripper_name} completed! Final gripper position: {obs[19]:.3f}")
            print()
            
            if done:
                break
            
            time.sleep(0.5)  # Short delay between gripper actions
        
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
