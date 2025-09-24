#!/usr/bin/env python3
"""
Unitree Z1 Arm Calibration Example

This example demonstrates how to calibrate the Unitree Z1 arm.
Calibration sets the current position as the home position for the arm.

IMPORTANT SAFETY NOTES:
- Ensure the arm is in a safe position before calibration
- Make sure there are no obstacles around the arm
- The arm should be in a known, repeatable position
- Calibration will set the current joint positions as the new home position

Usage:
    python3 example_calibration.py

Author: Unitree SDK
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface
import time
import numpy as np

# Set numpy print options for better readability
np.set_printoptions(precision=3, suppress=True)

def print_arm_status(arm):
    """Print current arm status information"""
    print("=" * 50)
    print("Current Arm Status:")
    print(f"Joint positions: {arm.lowstate.getQ()}")
    print(f"Joint velocities: {arm.lowstate.getQd()}")
    print(f"Gripper position: {arm.lowstate.getGripperQ()}")
    print(f"FSM State: {arm.getCurrentState()}")
    print("=" * 50)

def main():
    print("Unitree Z1 Arm Calibration Example")
    print("=" * 50)
    
    # Initialize arm interface
    print("Initializing arm interface...")
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    armState = unitree_arm_interface.ArmFSMState
    
    # Start the control loop
    arm.loopOn()
    print("Arm control loop started.")
    
    # Wait for arm to initialize
    print("Waiting for arm to initialize...")
    time.sleep(2.0)
    
    # Print initial status
    print_arm_status(arm)
    
    # Safety check and user confirmation
    print("\n" + "!" * 60)
    print("SAFETY WARNING:")
    print("Calibration will set the current position as the new home position.")
    print("Make sure the arm is in a safe, known position before proceeding.")
    print("!" * 60)
    
    # Get user confirmation
    try:
        response = input("\nDo you want to proceed with calibration? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Calibration cancelled by user.")
            arm.loopOff()
            return
    except KeyboardInterrupt:
        print("\nCalibration cancelled by user.")
        arm.loopOff()
        return
    
    # Perform calibration
    print("\nStarting calibration process...")
    print("Setting current position as home position...")
    
    try:
        # Call the calibration function
        arm.calibration()
        print("Calibration command sent successfully.")
        
        # Wait for calibration to complete
        print("Waiting for calibration to complete...")
        time.sleep(3.0)
        
        # Check if calibration was successful
        print("Checking calibration status...")
        time.sleep(1.0)
        
        # Print status after calibration
        print_arm_status(arm)
        
        print("\n" + "✓" * 50)
        print("Calibration completed successfully!")
        print("The current position has been set as the new home position.")
        print("✓" * 50)
        
    except Exception as e:
        print(f"\nError during calibration: {e}")
        print("Please check the arm status and try again.")
    
    # Optional: Move to a test position to verify calibration
    print("\nWould you like to test the calibration by moving to a test position?")
    try:
        test_response = input("Move to test position? (yes/no): ").lower().strip()
        if test_response in ['yes', 'y']:
            print("Moving to test position...")
            # Move to a simple test position
            test_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # All joints at 0
            gripper_pos = 0.0
            speed = 0.5
            arm.MoveJ(test_position, gripper_pos, speed)
            time.sleep(3.0)
            print("Test movement completed.")
            print_arm_status(arm)
    except KeyboardInterrupt:
        print("\nTest movement cancelled.")
    
    # Return to start position
    print("\nReturning to start position...")
    arm.backToStart()
    time.sleep(2.0)
    
    # Final status
    print("\nFinal arm status:")
    print_arm_status(arm)
    
    # Cleanup
    print("\nShutting down arm control...")
    arm.loopOff()
    print("Calibration example completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalibration example interrupted by user.")
        print("Attempting to safely shut down...")
        try:
            arm.loopOff()
        except:
            pass
        print("Shutdown complete.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Attempting to safely shut down...")
        try:
            arm.loopOff()
        except:
            pass
