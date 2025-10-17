#!/usr/bin/env python3
"""
Unconstrained Inverse Kinematics Solver

This script provides methods to solve inverse kinematics without workspace constraints,
focusing only on achieving the target pose regardless of singularities or workspace limits.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize
import time

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import unitree_arm_interface

class UnconstrainedIKSolver:
    """Solver for inverse kinematics without workspace constraints."""
    
    # Z1 robot joint limits (in radians)
    JOINT_LIMITS = [
        (-2.618, 2.618),   # J1: ±150°
        (0, 3.142),        # J2: 0—180°
        (-2.879, 0),       # J3: -165°—0
        (-1.396, 1.396),   # J4: ±80°
        (-1.484, 1.484),   # J5: ±85°
        (-2.793, 2.793)    # J6: ±160°
    ]
    
    def __init__(self):
        """Initialize the IK solver."""
        print("Initializing Z1 robot arm interface for unconstrained IK...")
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.arm_model = self.arm._ctrlComp.armModel
        print("Robot arm interface initialized successfully.")
        
    def solve_ik_jacobian_iterative(self, target_T, initial_guess=None, max_iterations=100, tolerance=1e-6):
        """
        Solve IK using Jacobian-based iterative method with improved convergence.
        
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
        
        q = initial_guess.copy()
        best_error = float('inf')
        best_q = q.copy()
        
        # Adaptive step size
        step_size = 0.1
        min_step_size = 1e-6
        
        for iteration in range(max_iterations):
            # Compute current forward kinematics
            current_T = self.arm_model.forwardKinematics(q, 6)
            
            # Compute error
            error = self._compute_pose_error(target_T, current_T)
            error_norm = np.linalg.norm(error)
            
            # Keep track of best solution
            if error_norm < best_error:
                best_error = error_norm
                best_q = q.copy()
            
            # Check convergence
            if error_norm < tolerance:
                return True, q, iteration + 1, error_norm
            
            # Compute Jacobian
            J = self.arm_model.CalcJacobian(q)
            
            # Solve for joint velocity using damped least squares
            try:
                # Adaptive damping based on error magnitude
                damping = max(0.001, min(0.1, error_norm * 0.01))
                
                # Damped least squares: (J^T * J + λ^2 * I)^-1 * J^T * e
                J_damped = J.T @ J + damping**2 * np.eye(6)
                dq = np.linalg.solve(J_damped, J.T @ error)
                
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse with regularization
                try:
                    dq = np.linalg.pinv(J) @ error
                except:
                    # If all else fails, use gradient descent
                    dq = J.T @ error * 0.01
            
            # Adaptive step size with line search
            original_q = q.copy()
            for step_attempt in range(10):
                # Try step
                q_new = q + step_size * dq
                
                # Apply joint limits
                q_new = np.clip(q_new, [limit[0] for limit in self.JOINT_LIMITS], [limit[1] for limit in self.JOINT_LIMITS])
                
                # Check if new position is better
                try:
                    new_T = self.arm_model.forwardKinematics(q_new, 6)
                    new_error = self._compute_pose_error(target_T, new_T)
                    new_error_norm = np.linalg.norm(new_error)
                    
                    if new_error_norm < error_norm:
                        q = q_new
                        step_size = min(step_size * 1.2, 1.0)  # Increase step size
                        break
                    else:
                        step_size *= 0.5  # Decrease step size
                        
                except:
                    step_size *= 0.5  # Decrease step size if FK fails
                
                if step_size < min_step_size:
                    break
            
            # If no improvement found, use best solution so far
            if step_size < min_step_size:
                q = best_q
                break
        
        # Final error check
        current_T = self.arm_model.forwardKinematics(q, 6)
        final_error = np.linalg.norm(self._compute_pose_error(target_T, current_T))
        
        return final_error < tolerance, q, max_iterations, final_error
    
    def solve_ik_jacobian_simple(self, target_T, initial_guess=None, max_iterations=50, tolerance=1e-6):
        """
        Simple Jacobian-based IK solver with fixed step size.
        
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
        
        q = initial_guess.copy()
        step_size = 0.01  # Small fixed step size
        
        for iteration in range(max_iterations):
            # Compute current forward kinematics
            current_T = self.arm_model.forwardKinematics(q, 6)
            
            # Compute error
            error = self._compute_pose_error(target_T, current_T)
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < tolerance:
                return True, q, iteration + 1, error_norm
            
            # Compute Jacobian
            J = self.arm_model.CalcJacobian(q)
            
            # Simple gradient descent: dq = -α * J^T * e
            dq = -step_size * J.T @ error
            
            # Update joint angles
            q += dq
            
            # Apply joint limits
            q = np.clip(q, [limit[0] for limit in self.JOINT_LIMITS], [limit[1] for limit in self.JOINT_LIMITS])
        
        # Final error check
        current_T = self.arm_model.forwardKinematics(q, 6)
        final_error = np.linalg.norm(self._compute_pose_error(target_T, current_T))
        
        return final_error < tolerance, q, max_iterations, final_error
    
    def solve_ik_optimization(self, target_T, initial_guess=None):
        """
        Solve IK using optimization-based method with joint limits.
        
        Args:
            target_T: Target 4x4 transformation matrix
            initial_guess: Initial joint angle guess
            
        Returns:
            tuple: (success, joint_angles, final_error)
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        # Use class joint limits
        
        def objective(q):
            """Objective function: weighted pose error (position prioritized over rotation)."""
            try:
                current_T = self.arm_model.forwardKinematics(q, 6)
                error = self._compute_pose_error(target_T, current_T)
                
                # Separate position and rotation errors
                pos_error = error[:3]  # First 3 elements: position error
                rot_error = error[3:]  # Last 3 elements: rotation error
                
                # Apply different weights: position weight = 1.0, rotation weight = 0.1
                pos_weight = 1.0
                rot_weight = 1.0
                
                # Weighted error: prioritize position accuracy over rotation
                weighted_error = np.concatenate([
                    pos_weight * pos_error,
                    rot_weight * rot_error
                ])
                
                return np.sum(weighted_error**2)
            except:
                return 1e6  # Large error if FK fails
        
        # Use scipy optimization with bounds
        result = minimize(
            objective, 
            initial_guess, 
            method='L-BFGS-B',
            bounds=self.JOINT_LIMITS,
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        
        if result.success:
            final_error = np.sqrt(result.fun)
            return True, result.x, final_error
        else:
            return False, result.x, np.sqrt(result.fun)
    
    def solve_ik_multiple_guesses(self, target_T, initial_guess=None, num_guesses=10, noise_std=0.1):
        """
        Solve IK using multiple initial guesses within joint limits.
        
        Args:
            target_T: Target 4x4 transformation matrix
            initial_guess: Initial joint angle guess (if None, use random guesses)
            num_guesses: Number of initial guesses to try
            noise_std: Standard deviation for noise around initial_guess
            
        Returns:
            tuple: (success, best_joint_angles, best_error)
        """
        best_error = float('inf')
        best_q = None
        best_success = False
        
        for i in range(num_guesses):
            if initial_guess is not None and i == 0:
                # First guess: use the provided initial_guess
                guess = initial_guess.copy()
            elif initial_guess is not None:
                # Subsequent guesses: add noise around initial_guess
                noise = np.random.normal(0, noise_std, 6)
                guess = initial_guess + noise
                
                # Ensure the guess is within joint limits
                guess = np.clip(guess, 
                              [limit[0] for limit in self.JOINT_LIMITS], 
                              [limit[1] for limit in self.JOINT_LIMITS])
            else:
                # No initial_guess provided: generate random guess within joint limits
                guess = np.array([
                    np.random.uniform(self.JOINT_LIMITS[j][0], self.JOINT_LIMITS[j][1]) 
                    for j in range(6)
                ])
            
            # Try optimization method
            success, q, error = self.solve_ik_optimization(target_T, guess)
            
            if success and error < best_error:
                best_error = error
                best_q = q
                best_success = True
        
        return best_success, best_q, best_error
    
    def _compute_pose_error(self, target_T, current_T):
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
    
    def test_methods(self, target_T, original_q=None):
        """
        Test all IK methods on a given target pose.
        
        Args:
            target_T: Target 4x4 transformation matrix
            original_q: Original joint angles (for comparison)
        """
        print(f"\nTesting IK methods for target pose:")
        print(f"Target position: {target_T[:3, 3]}")
        print(f"Target rotation matrix:\n{target_T[:3, :3]}")
        
        if original_q is not None:
            print(f"Original joint angles: {original_q}")
        
        # Method 1: Jacobian iterative (advanced)
        print(f"\n--- Method 1: Jacobian Iterative (Advanced) ---")
        success, q, iterations, error = self.solve_ik_jacobian_iterative(target_T)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Method 2: Jacobian simple
        print(f"\n--- Method 2: Jacobian Simple ---")
        success, q, iterations, error = self.solve_ik_jacobian_simple(target_T)
        print(f"Success: {success}")
        print(f"Iterations: {iterations}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Method 3: Optimization
        print(f"\n--- Method 3: Optimization ---")
        success, q, error = self.solve_ik_optimization(target_T)
        print(f"Success: {success}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Method 4: Multiple guesses
        print(f"\n--- Method 4: Multiple Random Guesses ---")
        start = time.time()
        success, q, error = self.solve_ik_multiple_guesses(target_T, original_q, num_guesses=10)
        end = time.time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Success: {success}")
        print(f"Final error: {error:.8f}")
        print(f"Result joint angles: {q}")
        
        # Verify results
        if success:
            print(f"\n--- Verification ---")
            result_T = self.arm_model.forwardKinematics(q, 6)
            pos_error = np.linalg.norm(target_T[:3, 3] - result_T[:3, 3])
            rot_error = np.linalg.norm(target_T[:3, :3] - result_T[:3, :3], 'fro')
            print(f"Position error: {pos_error:.8f}")
            print(f"Rotation error: {rot_error:.8f}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.arm.loopOff()
            print("Robot arm interface closed.")
        except:
            pass

def main():
    """Test the unconstrained IK solver."""
    solver = UnconstrainedIKSolver()
    
    try:
        
        # Test on real robot data
        print(f"\n" + "="*60)
        print("Testing on real robot data...")
        
        # Load and test real data
        import pickle
        with open('free_drive_data/robot_data.pkl', 'rb') as f:
        # with open('free_drive_data_20251004_161305/robot_data.pkl', 'rb') as f:
            data = pickle.load(f)

        joint_angles = data['joint_angles']
        T_matrices = data['T_matrices']
        
        print(f"Testing on {len(joint_angles)} samples from robot_data.pkl...")
        
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
            
            # Test unconstrained IK
            solver.test_methods(T_matrix, q_original)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        solver.cleanup()

if __name__ == "__main__":
    main()
