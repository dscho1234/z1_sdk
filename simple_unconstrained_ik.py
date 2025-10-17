#!/usr/bin/env python3
"""
Simple Unconstrained Inverse Kinematics Solver

This script provides a simple method to solve inverse kinematics without workspace constraints.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import unitree_arm_interface

def solve_ik_unconstrained(target_T, initial_guess=None, method='L-BFGS-B'):
    """
    Solve inverse kinematics with joint limits using optimization.
    
    Args:
        target_T: Target 4x4 transformation matrix
        initial_guess: Initial joint angle guess (optional)
        method: Optimization method ('BFGS', 'L-BFGS-B', 'SLSQP')
        
    Returns:
        tuple: (success, joint_angles, final_error)
    """
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    # Initialize arm model
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    
    # Z1 robot joint limits (in radians)
    joint_limits = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    def objective(q):
        """Objective function: pose error."""
        try:
            current_T = arm_model.forwardKinematics(q, 6)
            
            # Position error
            pos_error = np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])
            
            # Orientation error using Frobenius norm
            rot_error = np.linalg.norm(target_T[:3, :3] - current_T[:3, :3], 'fro')
            
            # Weighted combination (position is more important)
            total_error = pos_error + 0.1 * rot_error
            
            return total_error
        except:
            return 1e6  # Large error if FK fails
    
    # Use scipy optimization with joint limits
    result = minimize(
        objective, 
        initial_guess, 
        method=method,
        bounds=joint_limits if method in ['L-BFGS-B', 'SLSQP'] else None,
        options={'maxiter': 1000, 'gtol': 1e-8}
    )
    
    # Clean ups
    arm.loopOff()
    
    if result.success:
        return True, result.x, result.fun
    else:
        return False, result.x, result.fun

def solve_ik_multiple_attempts(target_T, num_attempts=5):
    """
    Solve IK with multiple random initial guesses within joint limits to find the best solution.
    
    Args:
        target_T: Target 4x4 transformation matrix
        num_attempts: Number of attempts with different initial guesses
        
    Returns:
        tuple: (success, best_joint_angles, best_error)
    """
    # Z1 robot joint limits (in radians)
    joint_limits = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    best_error = float('inf')
    best_q = None
    best_success = False
    
    for i in range(num_attempts):
        # Generate random initial guess within joint limits
        initial_guess = np.array([
            np.random.uniform(joint_limits[j][0], joint_limits[j][1]) 
            for j in range(6)
        ])
        
        # Try optimization
        success, q, error = solve_ik_unconstrained(target_T, initial_guess)
        
        if success and error < best_error:
            best_error = error
            best_q = q
            best_success = True
    
    return best_success, best_q, best_error

def test_on_real_data():
    """Test the unconstrained IK solver on real robot data."""
    import pickle
    
    # Load data
    with open('free_drive_data/robot_data.pkl', 'rb') as f:
        data = pickle.load(f)

    joint_angles = data['joint_angles']
    T_matrices = data['T_matrices']
    
    print(f"Testing unconstrained IK on {len(joint_angles)} samples...")
    
    # Test on first 5 samples
    for i in range(min(5, len(joint_angles))):
        # q_original = joint_angles[i]
        # dscho debug
        q_original = np.zeros(6)
        T_matrix = T_matrices[i]
        
        print(f"\n{'='*50}")
        print(f"Sample {i}:")
        print(f"Original joint angles: {q_original}")
        print(f"Target position: {T_matrix[:3, 3]}")
        
        # Test unconstrained IK
        success, q_result, error = solve_ik_unconstrained(T_matrix, q_original)
        
        print(f"Success: {success}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q_result}")
        
        if success:
            # Verify the result
            arm = unitree_arm_interface.ArmInterface(hasGripper=True)
            arm_model = arm._ctrlComp.armModel
            result_T = arm_model.forwardKinematics(q_result, 6)
            
            pos_error = np.linalg.norm(T_matrix[:3, 3] - result_T[:3, 3])
            rot_error = np.linalg.norm(T_matrix[:3, :3] - result_T[:3, :3], 'fro')
            
            print(f"Verification - Position error: {pos_error:.8f}")
            print(f"Verification - Rotation error: {rot_error:.8f}")
            
            arm.loopOff()

if __name__ == "__main__":
    test_on_real_data()
