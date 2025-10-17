#!/usr/bin/env python3
"""
Test unconstrained IK on real robot data with joint limits applied.
This script tests the UnconstrainedIKSolver which includes Z1 robot joint limits:
- J1: ±150° (±2.618 rad)
- J2: 0—180° (0—3.142 rad)  
- J3: -165°—0 (-2.879—0 rad)
- J4: ±80° (±1.396 rad)
- J5: ±85° (±1.484 rad)
- J6: ±160° (±2.793 rad)
"""

import pickle
import numpy as np
import sys
import os

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from unconstrained_ik import UnconstrainedIKSolver

def main():
    # Load data
    with open('free_drive_data/robot_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Initialize solver
    solver = UnconstrainedIKSolver()

    joint_angles = data['joint_angles']
    T_matrices = data['T_matrices']
    
    print(f"Testing unconstrained IK on {len(joint_angles)} samples...")
    
    # Test on the first few samples, especially the ones that failed with standard IK
    test_samples = [0, 1, 2, 3, 4]  # First 5 samples
    
    for i in test_samples:
        q_original = joint_angles[i]
        T_matrix = T_matrices[i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i}:")
        print(f"Original joint angles: {q_original}")
        print(f"Target position: {T_matrix[:3, 3]}")
        
        # Test unconstrained IK
        solver.test_methods(T_matrix, q_original)
    
    solver.cleanup()

if __name__ == "__main__":
    main()
