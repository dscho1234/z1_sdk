#!/usr/bin/env python3
"""
Example usage of the Unitree Z1 SDK package

This script demonstrates how to use the installed package in other projects.
"""

import numpy as np

def main():
    print("Unitree Z1 SDK Package Usage Example")
    print("=" * 40)
    
    # Import the main SDK package
    try:
        import unitree_arm_sdk
        print(f"✓ Successfully imported unitree_arm_sdk v{unitree_arm_sdk.get_version()}")
    except ImportError as e:
        print(f"✗ Failed to import unitree_arm_sdk: {e}")
        return
    
    # Import the environments package
    try:
        import envs
        print(f"✓ Successfully imported envs package")
        print(f"  Available environments: {envs.__all__}")
    except ImportError as e:
        print(f"✗ Failed to import envs: {e}")
        return
    
    # Try to import the compiled interface (this might fail if not built)
    try:
        from unitree_arm_sdk.unitree_arm_interface import ArmInterface, ArmFSMState
        print("✓ Successfully imported unitree_arm_interface")
        
        # Example of creating an arm interface (this would require actual hardware)
        # arm = ArmInterface(hasGripper=True)
        # print("✓ Created ArmInterface instance")
        
    except ImportError as e:
        print(f"⚠ unitree_arm_interface not available: {e}")
        print("  This is expected if the C++ extension hasn't been built yet.")
        print("  Run 'python setup.py build_ext' to build the extension.")
    
    # Example of using the environments
    try:
        from envs import Z1BaseEnv, EEPoseCtrlWrapper
        print("✓ Successfully imported environment classes")
        
        # Note: Creating actual environment instances would require the compiled interface
        # env = EEPoseCtrlWrapper(has_gripper=True)
        # print("✓ Created environment instance")
        
    except ImportError as e:
        print(f"⚠ Environment classes not fully available: {e}")
    
    print("\n" + "=" * 40)
    print("Package installation successful!")
    print("You can now use this package in other projects with:")
    print("  import unitree_arm_sdk")
    print("  import envs")
    print("\nFor full functionality, ensure the C++ extension is built.")

if __name__ == "__main__":
    main()
