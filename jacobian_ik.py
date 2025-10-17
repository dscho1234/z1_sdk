#!/usr/bin/env python3
"""
Standalone Jacobian-based Inverse Kinematics Solver

This script provides practical Jacobian-based IK methods that can be used
without workspace constraints.
"""

import sys
import os
import numpy as np

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import unitree_arm_interface

def solve_ik_jacobian_damped(target_T, initial_guess=None, max_iterations=100, tolerance=1e-6):
    """
    Solve IK using damped least squares Jacobian method with joint limits.
    
    Args:
        target_T: Target 4x4 transformation matrix
        initial_guess: Initial joint angle guess
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        tuple: (success, joint_angles, iterations, final_error)
    """
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    # Initialize arm model
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    
    # Z1 robot joint limits (in radians)
    JOINT_LIMITS = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    q = initial_guess.copy()
    damping = 0.01  # Damping factor
    
    for iteration in range(max_iterations):
        # Compute current forward kinematics
        current_T = arm_model.forwardKinematics(q, 6)
        
        # Compute error
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        rot_error = target_T[:3, :3] - current_T[:3, :3]
        
        # Combine errors (position is more important)
        error = np.concatenate([pos_error, rot_error.flatten()[:3]])
        error_norm = np.linalg.norm(error)
        
        # Check convergence
        if error_norm < tolerance:
            arm.loopOff()
            return True, q, iteration + 1, error_norm
        
        # Compute Jacobian
        J = arm_model.CalcJacobian(q)
        
        # Damped least squares: (J^T * J + λ^2 * I)^-1 * J^T * e
        try:
            J_damped = J.T @ J + damping**2 * np.eye(6)
            dq = np.linalg.solve(J_damped, J.T @ error)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            dq = np.linalg.pinv(J) @ error
        
        # Update joint angles
        q += dq
        
        # Apply joint limits
        q = np.clip(q, [limit[0] for limit in JOINT_LIMITS], [limit[1] for limit in JOINT_LIMITS])
    
    # Final error check
    current_T = arm_model.forwardKinematics(q, 6)
    pos_error = target_T[:3, 3] - current_T[:3, 3]
    rot_error = target_T[:3, :3] - current_T[:3, :3]
    final_error = np.linalg.norm(np.concatenate([pos_error, rot_error.flatten()[:3]]))
    
    arm.loopOff()
    return final_error < tolerance, q, max_iterations, final_error

def solve_ik_jacobian_gradient(target_T, initial_guess=None, max_iterations=100, tolerance=1e-6, step_size=0.01):
    """
    Solve IK using Jacobian transpose (gradient descent) method with joint limits.
    
    Args:
        target_T: Target 4x4 transformation matrix
        initial_guess: Initial joint angle guess
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        step_size: Learning rate
        
    Returns:
        tuple: (success, joint_angles, iterations, final_error)
    """
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    # Initialize arm model
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    
    # Z1 robot joint limits (in radians)
    JOINT_LIMITS = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    q = initial_guess.copy()
    
    for iteration in range(max_iterations):
        # Compute current forward kinematics
        current_T = arm_model.forwardKinematics(q, 6)
        
        # Compute error
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        rot_error = target_T[:3, :3] - current_T[:3, :3]
        
        # Combine errors
        error = np.concatenate([pos_error, rot_error.flatten()[:3]])
        error_norm = np.linalg.norm(error)
        
        # Check convergence
        if error_norm < tolerance:
            arm.loopOff()
            return True, q, iteration + 1, error_norm
        
        # Compute Jacobian
        J = arm_model.CalcJacobian(q)
        
        # Gradient descent: dq = -α * J^T * e
        dq = -step_size * J.T @ error
        
        # Update joint angles
        q += dq
        
        # Apply joint limits
        q = np.clip(q, [limit[0] for limit in JOINT_LIMITS], [limit[1] for limit in JOINT_LIMITS])
    
    # Final error check
    current_T = arm_model.forwardKinematics(q, 6)
    pos_error = target_T[:3, 3] - current_T[:3, 3]
    rot_error = target_T[:3, :3] - current_T[:3, :3]
    final_error = np.linalg.norm(np.concatenate([pos_error, rot_error.flatten()[:3]]))
    
    arm.loopOff()
    return final_error < tolerance, q, max_iterations, final_error

def solve_ik_jacobian_adaptive(target_T, initial_guess=None, max_iterations=100, tolerance=1e-6):
    """
    Solve IK using adaptive Jacobian method with line search and joint limits.
    
    Args:
        target_T: Target 4x4 transformation matrix
        initial_guess: Initial joint angle guess
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        tuple: (success, joint_angles, iterations, final_error)
    """
    if initial_guess is None:
        initial_guess = np.zeros(6)
    
    # Initialize arm model
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    
    # Z1 robot joint limits (in radians)
    JOINT_LIMITS = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    q = initial_guess.copy()
    step_size = 0.1
    min_step_size = 1e-6
    
    for iteration in range(max_iterations):
        # Compute current forward kinematics
        current_T = arm_model.forwardKinematics(q, 6)
        
        # Compute error
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        rot_error = target_T[:3, :3] - current_T[:3, :3]
        error = np.concatenate([pos_error, rot_error.flatten()[:3]])
        error_norm = np.linalg.norm(error)
        
        # Check convergence
        if error_norm < tolerance:
            arm.loopOff()
            return True, q, iteration + 1, error_norm
        
        # Compute Jacobian
        J = arm_model.CalcJacobian(q)
        
        # Compute search direction using damped least squares
        try:
            damping = max(0.001, min(0.1, error_norm * 0.01))
            J_damped = J.T @ J + damping**2 * np.eye(6)
            dq = np.linalg.solve(J_damped, J.T @ error)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(J) @ error
        
        # Line search for optimal step size
        for step_attempt in range(10):
            q_new = q + step_size * dq
            
            # Apply joint limits
            q_new = np.clip(q_new, [limit[0] for limit in JOINT_LIMITS], [limit[1] for limit in JOINT_LIMITS])
            
            try:
                new_T = arm_model.forwardKinematics(q_new, 6)
                new_pos_error = target_T[:3, 3] - new_T[:3, 3]
                new_rot_error = target_T[:3, :3] - new_T[:3, :3]
                new_error = np.concatenate([new_pos_error, new_rot_error.flatten()[:3]])
                new_error_norm = np.linalg.norm(new_error)
                
                if new_error_norm < error_norm:
                    q = q_new
                    step_size = min(step_size * 1.2, 1.0)
                    break
                else:
                    step_size *= 0.5
            except:
                step_size *= 0.5
            
            if step_size < min_step_size:
                break
    
    # Final error check
    current_T = arm_model.forwardKinematics(q, 6)
    pos_error = target_T[:3, 3] - current_T[:3, 3]
    rot_error = target_T[:3, :3] - current_T[:3, :3]
    final_error = np.linalg.norm(np.concatenate([pos_error, rot_error.flatten()[:3]]))
    
    arm.loopOff()
    return final_error < tolerance, q, max_iterations, final_error

def test_jacobian_methods():
    """Test all Jacobian methods on real robot data."""
    import pickle
    
    # Load data
    with open('free_drive_data/robot_data.pkl', 'rb') as f:
        data = pickle.load(f)

    joint_angles = data['joint_angles']
    T_matrices = data['T_matrices']
    
    print(f"Testing Jacobian IK methods on {len(joint_angles)} samples...")
    
    # Test on first 3 samples
    for i in range(min(3, len(joint_angles))):
        # q_original = joint_angles[i]
        # dscho debug
        q_original = np.zeros(6)
        T_matrix = T_matrices[i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i}:")
        print(f"Original joint angles: {q_original}")
        print(f"Target position: {T_matrix[:3, 3]}")
        
        # Test Method 1: Damped Least Squares
        print(f"\n--- Method 1: Damped Least Squares ---")
        success, q, iterations, error = solve_ik_jacobian_damped(T_matrix, q_original, max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Test Method 2: Gradient Descent
        print(f"\n--- Method 2: Gradient Descent ---")
        success, q, iterations, error = solve_ik_jacobian_gradient(T_matrix, q_original, max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Test Method 3: Adaptive with Line Search
        print(f"\n--- Method 3: Adaptive with Line Search ---")
        success, q, iterations, error = solve_ik_jacobian_adaptive(T_matrix, q_original, max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")

if __name__ == "__main__":
    test_jacobian_methods()
