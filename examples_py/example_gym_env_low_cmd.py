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

from z1_env_low_cmd import EEPoseCtrlLowCmdWrapper


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Low-Level Command Control")
    print("=" * 60)
    
    # Create the environment with low-level commands
    control_frequency = 5  # 500Hz for low-level control
    env = EEPoseCtrlLowCmdWrapper(
        has_gripper=True,
        control_frequency=control_frequency,  # 500Hz control frequency
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        move_speed=None,  # Moderate speed for smooth movement
        move_timeout=float(1/control_frequency)  # 0.2 seconds timeout for each movement
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
    
    # Define movements: [name, [x_offset, y_offset, z_offset]]
    movements = [
        ("X+0.1", [0.1, 0.0, 0.0]),
        ("Return to original", [0.0, 0.0, 0.0]),
        ("Y+0.1", [0.0, 0.1, 0.0]),
        ("Return to original", [0.0, 0.0, 0.0]),
        ("Z+0.1", [0.0, 0.0, 0.1]),
        ("Return to original", [0.0, 0.0, 0.0]),
    ]
    
    print("Example: Low-level command movement using inverse kinematics")
    print("MoveJ commands will use inverse kinematics and direct joint control")
    print(f"Timeout setting: {env.move_timeout} seconds")
    print()
    
    # Execute movements
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
        
        for i in range(control_frequency): # 1 second
            obs, reward, done, info = env.step(action)
        
        # Print final status
        print(f"  {movement_name} completed! Final position: {obs[12:15]}")
        print(f"  Position error: {info['position_error']:.4f}, Is moving: {info['is_moving']}")
        
        # if done:
        #     print(f"  Episode finished during {movement_name}!")
        #     break
        
        print()
        
        # Small delay between commands
        time.sleep(0.5)  # 0.5 second delay between movements
    
    print("All examples completed successfully!")
    
    # Close the environment
    print("Closing environment...")
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
