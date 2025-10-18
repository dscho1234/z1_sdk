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

def _compute_pose_error(target_T, current_T):
    """
    Compute pose error between target and current transformation matrices.
    
    Args:
        target_T: Target 4x4 transformation matrix
        current_T: Current 4x4 transformation matrix
        
    Returns:
        np.array: 6D error vector [position_error(3), orientation_error(3)]
    """
    # Position error
    pos_error = target_T[:3, 3] - current_T[:3, 3]
    
    # Orientation error using axis-angle representation
    R_rel = target_T[:3, :3].T @ current_T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    
    if angle > 1e-6:
        axis = np.array([R_rel[2,1] - R_rel[1,2], 
                        R_rel[0,2] - R_rel[2,0], 
                        R_rel[1,0] - R_rel[0,1]]) / (2 * np.sin(angle))
        rot_error = angle * axis
    else:
        rot_error = np.zeros(3)
    
    # Combine position and rotation errors
    error = np.concatenate([pos_error, rot_error])
    
    return error

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
        
        # Compute error using proper pose error calculation
        error = _compute_pose_error(target_T, current_T)
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
    final_error = np.linalg.norm(_compute_pose_error(target_T, current_T))
    
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
        
        # Compute error using proper pose error calculation
        error = _compute_pose_error(target_T, current_T)
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
    final_error = np.linalg.norm(_compute_pose_error(target_T, current_T))
    
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
        
        # Compute error using proper pose error calculation
        error = _compute_pose_error(target_T, current_T)
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
                new_error = _compute_pose_error(target_T, new_T)
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
    final_error = np.linalg.norm(_compute_pose_error(target_T, current_T))
    
    arm.loopOff()
    return final_error < tolerance, q, max_iterations, final_error

def solve_ik_null_space(target_T, initial_guess=None, max_iterations=100, tolerance=1e-2, tolerance_null=1e-5, epsilon=1e-6):
    """
    Solve IK using pseudo-inverse with null-space approach.
    
    Args:
        target_T: Target 4x4 transformation matrix
        initial_guess: Initial joint angle guess
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for position error
        tolerance_null: Convergence tolerance for null objective
        epsilon: Small value for numerical differentiation
        desired_rotation: 3x3 rotation matrix for null objective (default: identity)
        
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
    
    # SO3 constraint null objective function
    class SO3Constraint:
        def __init__(self, SO3_des=None):
            if SO3_des is None:
                # Default to identity matrix (no rotation preference)
                self.SO3_des = np.eye(3)
            else:
                self.SO3_des = SO3_des.copy()
        
        def evaluate(self, SO3):
            # SO3 error metric: 0.5 * (3 - trace(R * R_des^T))
            # This measures the deviation from desired rotation
            so3_err = 0.5 * (3 - np.trace(SO3 @ self.SO3_des.T))
            return so3_err
    
    # Initialize null objective with desired rotation (default: identity)
    target_SO3 = target_T[:3, :3]
    null_obj = SO3Constraint(target_SO3)
    
    iter_taken = 0

    def jacobian_position(q):
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        T = arm_model.forwardKinematics(q, 6)
        p = T[:3, 3]
        jac = np.zeros([3, 6])
        for i in range(6):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            T_ = arm_model.forwardKinematics(q_, 6)
            p_ = T_[:3, 3]
            jac[:, i] = (p_ - p)*epsilon_inv
        return jac
    
    while True:
        # Compute current forward kinematics
        current_T = arm_model.forwardKinematics(q, 6)
        
        # Compute position error only (like in the original code)
        pos_error = target_T[:3, 3] - current_T[:3, 3]
        err = np.linalg.norm(pos_error)
        
        # Compute null objective value
        current_SO3 = current_T[:3, :3]
        null_obj_val = null_obj.evaluate(current_SO3)

        # Compute Jacobian
        # J = arm_model.CalcJacobian(q)
        J = jacobian_position(q)
        
        # Check convergence: both position error and null objective must be satisfied
        if (err < tolerance and null_obj_val < tolerance_null) or iter_taken >= max_iterations:
            break
        else:
            iter_taken += 1
        
        
        # Pseudo-inverse approach
        
        J_dagger = np.linalg.pinv(J)
        J_null = np.eye(6) - J_dagger @ J  # null space of Jacobian
        
        # Compute null objective gradient using numerical differentiation
        phi = np.zeros(6)
        
        for i in range(6):
            q_perturb = q.copy()
            q_perturb[i] += epsilon
            # Apply joint limits to perturbed configuration
            q_perturb = np.clip(q_perturb, [limit[0] for limit in JOINT_LIMITS], [limit[1] for limit in JOINT_LIMITS])
            
            
            perturb_T = arm_model.forwardKinematics(q_perturb, 6)
            perturb_SO3 = perturb_T[:3, :3]
            null_obj_val_perturb = null_obj.evaluate(perturb_SO3)
            phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon
        
        
        # Update using pseudo-inverse + null-space approach
        # delta_x = ee_pos - x (position error)
        delta_x = pos_error
        delta_q = J_dagger @ delta_x - J_null @ phi
        q = q + delta_q
        
        
        # Apply joint limits
        q = np.clip(q, [limit[0] for limit in JOINT_LIMITS], [limit[1] for limit in JOINT_LIMITS])
    
    # Final error check (position error only, like in original code)
    current_T = arm_model.forwardKinematics(q, 6)
    # final_pos_error = target_T[:3, 3] - current_T[:3, 3]
    final_error = err # np.linalg.norm(final_pos_error)
    
    # Check if both conditions are satisfied for success
    success = (final_error < tolerance and null_obj_val < tolerance_null)
    
    arm.loopOff()
    return success, q, iter_taken, final_error, null_obj_val, current_T

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
        q_original = joint_angles[i] 
        q_init = q_original.copy() + np.random.normal(0, 0.1, 6)
        # dscho debug
        # q_original = np.zeros(6)
        T_matrix = T_matrices[i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i}:")
        print(f"Original joint angles: {q_original.round(4)}")
        print(f"Target matrix: {T_matrix}")
        
        # Test Method 1: Damped Least Squares
        print(f"\n--- Method 1: Damped Least Squares ---")
        success, q, iterations, error = solve_ik_jacobian_damped(T_matrix.copy(), q_init.copy(), max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Test Method 2: Gradient Descent
        print(f"\n--- Method 2: Gradient Descent ---")
        success, q, iterations, error = solve_ik_jacobian_gradient(T_matrix.copy(), q_init.copy(), max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Test Method 3: Adaptive with Line Search
        print(f"\n--- Method 3: Adaptive with Line Search ---")
        success, q, iterations, error = solve_ik_jacobian_adaptive(T_matrix.copy(), q_init.copy(), max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Test Method 4: Null-Space Approach
        print(f"\n--- Method 4: Null-Space Approach ---")
        success, q, iterations, error, null_obj_val, current_T = solve_ik_null_space(T_matrix.copy(), q_init.copy(), max_iterations=1000)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Null objective value: {null_obj_val:.8f}")
        print(f"Result joint angles: {q}")
        print(f"Current transformation matrix: {current_T}")

if __name__ == "__main__":
    test_jacobian_methods()
