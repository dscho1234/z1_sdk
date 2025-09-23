#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control.

This example demonstrates how to use the EEPoseCtrlWrapper to control
the Z1 robot arm's end-effector pose using MoveJ commands.
"""

import sys
import os
import numpy as np
import time

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env import EEPoseCtrlWrapper


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - End-Effector Pose Control")
    print("=" * 60)
    
    # Create the environment with 5Hz control frequency
    control_frequency = 5
    env = EEPoseCtrlWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 5Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        move_speed=0.1,  # Slower speed to test timeout behavior
        move_timeout=float(1/control_frequency)  # 5 seconds timeout to allow actual movement
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
        
        # Example: Non-blocking movement in x, y, z directions
        print("Example: Non-blocking movement using setWait(False)")
        print("MoveJ commands will return immediately and FSM state will be monitored")
        print(f"Timeout setting: {env.move_timeout} seconds")
        print()
        
        # Get current end-effector position and orientation from observation
        original_position = obs[12:15].copy()  # Store original position
        current_orientation = obs[15:19]  # Current end-effector orientation (quaternion)
        target_gripper = 0  # -1 open, 1 close
        
        print(f"Original EE position: {original_position}")
        print(f"Current EE orientation: {current_orientation}")
        print()
        
        # Define movement sequence: x+0.1, return, y+0.1, return, z+0.1, return
        movements = [
            ("X+0.1", [0.1, 0.0, 0.0]),
            ("Return to original", [0.0, 0.0, 0.0]),
            ("Y+0.1", [0.0, 0.1, 0.0]),
            ("Return to original", [0.0, 0.0, 0.0]),
            ("Z+0.1", [0.0, 0.0, 0.1]),
            ("Return to original", [0.0, 0.0, 0.0])
        ]
        
        # Non-blocking approach: Send commands and wait for completion
        for i, (movement_name, offset) in enumerate(movements):
            print(f"=== Command {i+1}: {movement_name} ===")
            
            # Calculate target position
            if offset == [0.0, 0.0, 0.0]:  # Return to original
                target_position = original_position.copy()
            else:  # Move by offset from original
                target_position = original_position + np.array(offset)
            
            print(f"Target EE position: {target_position}")
            
            # Create action
            action = np.concatenate([target_position, current_orientation, [target_gripper]])
            
            # Send command - step() function now handles timing internally
            print(f"  Sending {movement_name} command...")
            obs, reward, done, info = env.step(action)
            
            # Print final status
            print(f"  {movement_name} completed! Final position: {obs[12:15]}")
            print(f"  Position error: {info['position_error']:.4f}, FSM state: {info['fsm_state']}")
            
            if done:
                print(f"  Episode finished during {movement_name}!")
                break
            
            print()
            
            # Small delay between commands
            time.sleep(1.0)  # 1 second delay between movements
        
        # # Additional example: Demonstrate timeout behavior
        # print("\n" + "="*60)
        # print("Additional Example: Demonstrating timeout behavior")
        # print("="*60)
        
        # # Use current state (don't reset)
        # current_position = obs[12:15].copy()
        # current_orientation = obs[15:19]
        
        # # Try to move to a very far position (likely to timeout)
        # far_position = current_position + np.array([0.5, 0.5, 0.5])  # Far target
        # print(f"Attempting to move to far position: {far_position}")
        # print(f"This will likely timeout after {env.move_timeout} seconds")
        
        # action = np.concatenate([far_position, current_orientation, [0]])
        
        # # Run for more steps to see timeout behavior
        # for step in range(100):
        #     obs, reward, done, info = env.step(action)
            
        #     if step % 20 == 0:
        #         print(f"Step {step}: Current position: {obs[12:15]}, Error: {info['position_error']:.4f}, FSM state: {info['fsm_state']}")
            
        #     # If FSM returned to JOINTCTRL (movement complete or timeout)
        #     if info['fsm_state'] == 2:  # 2 = JOINTCTRL
        #         if info['position_error'] > 0.1:
        #             print(f"Movement timed out at step {step}! FSM returned to JOINTCTRL")
        #         else:
        #             print(f"Movement completed at step {step}")
        #         break
            
        #     if done:
        #         print("Episode finished!")
        #         break
        
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


def test_environment_basic():
    """Basic test of the environment functionality."""
    print("Testing basic environment functionality...")
    
    env = EEPoseCtrlWrapper(has_gripper=True)
    
    # Test reset
    obs = env.reset()
    assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
    
    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape, "Obs shape mismatch after step"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(done, bool), "Done should be boolean"
    assert isinstance(info, dict), "Info should be dictionary"
    
    # Test close
    env.close()
    
    print("Basic tests passed!")


if __name__ == "__main__":
    # Run basic tests first
    # test_environment_basic()
    # print()
    
    # Run main example
    main()
