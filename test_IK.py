#!/usr/bin/env python3
"""
Inverse Kinematics Test Script for Z1 Robot Arm

This script loads robot_data.pkl files from different directories, performs inverse kinematics
using the stored T_matrices, and compares the results with the original joint_angles to
evaluate IK accuracy.

Usage:
    python test_IK.py

Author: Generated for Z1 SDK
"""

import sys
import os
import pickle
import numpy as np
import glob
from pathlib import Path

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import unitree_arm_interface

# Set numpy print options
np.set_printoptions(precision=6, suppress=True)

class IKTestSuite:
    """Test suite for evaluating inverse kinematics accuracy."""
    
    def __init__(self):
        """Initialize the IK test suite."""
        print("Initializing Z1 robot arm interface for IK testing...")
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.arm_model = self.arm._ctrlComp.armModel
        print("Robot arm interface initialized successfully.")
        
        # Statistics storage
        self.all_errors = []
        self.all_ik_success_rates = []
        self.dataset_results = {}
        
    def find_robot_data_files(self, base_dir="."):
        """Find all robot_data.pkl files in subdirectories."""
        pattern = os.path.join(base_dir, "**/robot_data.pkl")
        files = glob.glob(pattern, recursive=True)
        return files
    
    def load_robot_data(self, file_path):
        """Load robot data from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def perform_inverse_kinematics(self, T_matrix, initial_guess=None):
        """
        Perform inverse kinematics for a given transformation matrix.
        
        Args:
            T_matrix: 4x4 transformation matrix
            initial_guess: Initial joint angle guess (optional)
            
        Returns:
            tuple: (success, joint_angles, fk_error_info)
        """
        try:
            # Use zero initial guess if not provided
            if initial_guess is None:
                initial_guess = np.zeros(6)
            
            # Perform inverse kinematics
            # The third parameter (True) means we don't require the result to be near the initial guess
            has_ik, joint_angles = self.arm_model.inverseKinematics(T_matrix, initial_guess, True)
            
            # If IK fails, calculate forward kinematics error for the initial guess
            fk_error_info = None
            if not has_ik:
                fk_error_info = self.calculate_fk_error(T_matrix, initial_guess)
            
            return has_ik, joint_angles, fk_error_info
            
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            return False, None, None
    
    def calculate_fk_error(self, target_T, joint_angles):
        """
        Calculate forward kinematics error when IK fails.
        
        Args:
            target_T: Target transformation matrix
            joint_angles: Joint angles to test
            
        Returns:
            dict: Error metrics between target and computed T matrices
        """
        try:
            # Compute forward kinematics
            computed_T = self.arm_model.forwardKinematics(joint_angles, 6)
            
            # Calculate position error
            target_pos = target_T[:3, 3]
            computed_pos = computed_T[:3, 3]
            position_error = np.linalg.norm(target_pos - computed_pos)
            
            # Calculate rotation error (Frobenius norm of rotation matrix difference)
            target_rot = target_T[:3, :3]
            computed_rot = computed_T[:3, :3]
            rotation_error = np.linalg.norm(target_rot - computed_rot, 'fro')
            
            # Calculate individual position errors
            pos_errors = np.abs(target_pos - computed_pos)
            
            # Calculate angle between rotation matrices (in degrees)
            # Using the trace formula: angle = arccos((trace(R1^T * R2) - 1) / 2)
            trace = np.trace(target_rot.T @ computed_rot)
            angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
            
            return {
                'position_error': position_error,
                'rotation_error': rotation_error,
                'angle_error_deg': angle_error,
                'pos_errors': pos_errors,
                'target_position': target_pos,
                'computed_position': computed_pos,
                'target_rotation': target_rot,
                'computed_rotation': computed_rot
            }
            
        except Exception as e:
            print(f"Error calculating FK error: {e}")
            return None
    
    def calculate_joint_angle_error(self, q_original, q_ik):
        """
        Calculate error between original and IK joint angles.
        
        Args:
            q_original: Original joint angles
            q_ik: IK computed joint angles
            
        Returns:
            dict: Various error metrics
        """
        if q_ik is None:
            return None
            
        # Absolute error for each joint
        abs_errors = np.abs(q_original - q_ik)
        
        # Root mean square error
        rmse = np.sqrt(np.mean((q_original - q_ik) ** 2))
        
        # Mean absolute error
        mae = np.mean(abs_errors)
        
        # Maximum absolute error
        max_error = np.max(abs_errors)
        
        # Error for each joint
        joint_errors = {
            f'joint_{i+1}': abs_errors[i] for i in range(6)
        }
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'joint_errors': joint_errors,
            'abs_errors': abs_errors
        }
    
    def test_single_dataset(self, data, dataset_name):
        """
        Test IK accuracy for a single dataset.
        
        Args:
            data: Robot data dictionary
            dataset_name: Name of the dataset
            
        Returns:
            dict: Test results for this dataset
        """
        print(f"\n=== Testing Dataset: {dataset_name} ===")
        
        joint_angles = data['joint_angles']
        T_matrices = data['T_matrices']
        num_samples = len(joint_angles)
        
        print(f"Number of samples: {num_samples}")
        
        # Storage for results
        ik_successes = 0
        errors = []
        joint_wise_errors = [[] for _ in range(6)]
        fk_errors = []  # Store FK errors when IK fails
        
        # Test each sample
        for i in range(num_samples):
            T_matrix = T_matrices[i]
            q_original = joint_angles[i]
            
            # Perform inverse kinematics
            has_ik, q_ik, fk_error_info = self.perform_inverse_kinematics(T_matrix, q_original)
            
            if has_ik:
                ik_successes += 1
                
                # Calculate errors
                error_metrics = self.calculate_joint_angle_error(q_original, q_ik)
                if error_metrics:
                    errors.append(error_metrics)
                    
                    # Store joint-wise errors
                    for j in range(6):
                        joint_wise_errors[j].append(error_metrics['abs_errors'][j])
            else:
                print(f"  Sample {i}: IK failed")
                if fk_error_info:
                    fk_errors.append(fk_error_info)
        
        # Calculate statistics
        ik_success_rate = ik_successes / num_samples * 100
        
        if errors:
            # Overall statistics
            rmse_values = [e['rmse'] for e in errors]
            mae_values = [e['mae'] for e in errors]
            max_error_values = [e['max_error'] for e in errors]
            
            overall_stats = {
                'ik_success_rate': ik_success_rate,
                'num_samples': num_samples,
                'successful_ik': ik_successes,
                'failed_ik': num_samples - ik_successes,
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'rmse_min': np.min(rmse_values),
                'rmse_max': np.max(rmse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'mae_min': np.min(mae_values),
                'mae_max': np.max(mae_values),
                'max_error_mean': np.mean(max_error_values),
                'max_error_std': np.std(max_error_values),
                'max_error_min': np.min(max_error_values),
                'max_error_max': np.max(max_error_values)
            }
        else:
            overall_stats = {
                'ik_success_rate': ik_success_rate,
                'num_samples': num_samples,
                'successful_ik': ik_successes,
                'failed_ik': num_samples - ik_successes
            }
        
        # Calculate FK error statistics when IK fails
        if fk_errors:
            pos_errors = [e['position_error'] for e in fk_errors]
            rot_errors = [e['rotation_error'] for e in fk_errors]
            angle_errors = [e['angle_error_deg'] for e in fk_errors]
            
            # Position error statistics (x, y, z components)
            pos_x_errors = [e['pos_errors'][0] for e in fk_errors]
            pos_y_errors = [e['pos_errors'][1] for e in fk_errors]
            pos_z_errors = [e['pos_errors'][2] for e in fk_errors]
            
            fk_stats = {
                'position_error_mean': np.mean(pos_errors),
                'position_error_std': np.std(pos_errors),
                'position_error_min': np.min(pos_errors),
                'position_error_max': np.max(pos_errors),
                'rotation_error_mean': np.mean(rot_errors),
                'rotation_error_std': np.std(rot_errors),
                'rotation_error_min': np.min(rot_errors),
                'rotation_error_max': np.max(rot_errors),
                'angle_error_mean': np.mean(angle_errors),
                'angle_error_std': np.std(angle_errors),
                'angle_error_min': np.min(angle_errors),
                'angle_error_max': np.max(angle_errors),
                'pos_x_error_mean': np.mean(pos_x_errors),
                'pos_x_error_std': np.std(pos_x_errors),
                'pos_y_error_mean': np.mean(pos_y_errors),
                'pos_y_error_std': np.std(pos_y_errors),
                'pos_z_error_mean': np.mean(pos_z_errors),
                'pos_z_error_std': np.std(pos_z_errors)
            }
            
            overall_stats['fk_error_stats'] = fk_stats
            
            # Joint-wise statistics
            joint_stats = {}
            for j in range(6):
                if joint_wise_errors[j]:
                    joint_errors = np.array(joint_wise_errors[j])
                    joint_stats[f'joint_{j+1}'] = {
                        'mean_error': np.mean(joint_errors),
                        'std_error': np.std(joint_errors),
                        'min_error': np.min(joint_errors),
                        'max_error': np.max(joint_errors)
                    }
            
            overall_stats['joint_stats'] = joint_stats
            
        else:
            overall_stats = {
                'ik_success_rate': ik_success_rate,
                'num_samples': num_samples,
                'successful_ik': ik_successes,
                'error': 'No successful IK solutions'
            }
        
        # Print results for this dataset
        self.print_dataset_results(dataset_name, overall_stats)
        
        return overall_stats
    
    def print_dataset_results(self, dataset_name, stats):
        """Print results for a single dataset."""
        print(f"\n--- Results for {dataset_name} ---")
        print(f"IK Success Rate: {stats['ik_success_rate']:.2f}% ({stats['successful_ik']}/{stats['num_samples']})")
        
        if 'failed_ik' in stats:
            print(f"IK Failures: {stats['failed_ik']}")
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return
        
        if 'rmse_mean' in stats:
            print(f"RMSE - Mean: {stats['rmse_mean']:.6f}, Std: {stats['rmse_std']:.6f}, Range: [{stats['rmse_min']:.6f}, {stats['rmse_max']:.6f}]")
            print(f"MAE  - Mean: {stats['mae_mean']:.6f}, Std: {stats['mae_std']:.6f}, Range: [{stats['mae_min']:.6f}, {stats['mae_max']:.6f}]")
            print(f"Max Error - Mean: {stats['max_error_mean']:.6f}, Std: {stats['max_error_std']:.6f}, Range: [{stats['max_error_min']:.6f}, {stats['max_error_max']:.6f}]")
            
            print("\nJoint-wise Error Statistics:")
            for joint_name, joint_stat in stats['joint_stats'].items():
                print(f"  {joint_name}: Mean={joint_stat['mean_error']:.6f}, Std={joint_stat['std_error']:.6f}, Range=[{joint_stat['min_error']:.6f}, {joint_stat['max_error']:.6f}]")
        
        # Print FK error statistics when IK fails
        if 'fk_error_stats' in stats:
            fk_stats = stats['fk_error_stats']
            print(f"\n--- Forward Kinematics Error Statistics (IK Failures) ---")
            print(f"Position Error - Mean: {fk_stats['position_error_mean']:.6f}, Std: {fk_stats['position_error_std']:.6f}, Range: [{fk_stats['position_error_min']:.6f}, {fk_stats['position_error_max']:.6f}]")
            print(f"Rotation Error - Mean: {fk_stats['rotation_error_mean']:.6f}, Std: {fk_stats['rotation_error_std']:.6f}, Range: [{fk_stats['rotation_error_min']:.6f}, {fk_stats['rotation_error_max']:.6f}]")
            print(f"Angle Error - Mean: {fk_stats['angle_error_mean']:.2f}°, Std: {fk_stats['angle_error_std']:.2f}°, Range: [{fk_stats['angle_error_min']:.2f}°, {fk_stats['angle_error_max']:.2f}°]")
            print(f"Position X Error - Mean: {fk_stats['pos_x_error_mean']:.6f}, Std: {fk_stats['pos_x_error_std']:.6f}")
            print(f"Position Y Error - Mean: {fk_stats['pos_y_error_mean']:.6f}, Std: {fk_stats['pos_y_error_std']:.6f}")
            print(f"Position Z Error - Mean: {fk_stats['pos_z_error_mean']:.6f}, Std: {fk_stats['pos_z_error_std']:.6f}")
    
    def print_overall_statistics(self):
        """Print overall statistics across all datasets."""
        if not self.dataset_results:
            print("No dataset results available.")
            return
            
        print("\n" + "="*80)
        print("OVERALL STATISTICS ACROSS ALL DATASETS")
        print("="*80)
        
        # Collect all success rates
        success_rates = [stats['ik_success_rate'] for stats in self.dataset_results.values() if 'ik_success_rate' in stats]
        
        if success_rates:
            print(f"\nIK Success Rates:")
            print(f"  Mean: {np.mean(success_rates):.2f}%")
            print(f"  Std:  {np.std(success_rates):.2f}%")
            print(f"  Min:  {np.min(success_rates):.2f}%")
            print(f"  Max:  {np.max(success_rates):.2f}%")
        
        # Collect all RMSE values
        rmse_values = []
        mae_values = []
        max_error_values = []
        
        for stats in self.dataset_results.values():
            if 'rmse_mean' in stats:
                rmse_values.append(stats['rmse_mean'])
                mae_values.append(stats['mae_mean'])
                max_error_values.append(stats['max_error_mean'])
        
        if rmse_values:
            print(f"\nRMSE Statistics:")
            print(f"  Mean: {np.mean(rmse_values):.6f}")
            print(f"  Std:  {np.std(rmse_values):.6f}")
            print(f"  Min:  {np.min(rmse_values):.6f}")
            print(f"  Max:  {np.max(rmse_values):.6f}")
            
            print(f"\nMAE Statistics:")
            print(f"  Mean: {np.mean(mae_values):.6f}")
            print(f"  Std:  {np.std(mae_values):.6f}")
            print(f"  Min:  {np.min(mae_values):.6f}")
            print(f"  Max:  {np.max(mae_values):.6f}")
            
            print(f"\nMax Error Statistics:")
            print(f"  Mean: {np.mean(max_error_values):.6f}")
            print(f"  Std:  {np.std(max_error_values):.6f}")
            print(f"  Min:  {np.min(max_error_values):.6f}")
            print(f"  Max:  {np.max(max_error_values):.6f}")
        
        # Joint-wise overall statistics
        joint_overall_stats = {}
        for j in range(6):
            joint_errors = []
            for stats in self.dataset_results.values():
                if 'joint_stats' in stats and f'joint_{j+1}' in stats['joint_stats']:
                    joint_errors.append(stats['joint_stats'][f'joint_{j+1}']['mean_error'])
            
            if joint_errors:
                joint_overall_stats[f'joint_{j+1}'] = {
                    'mean': np.mean(joint_errors),
                    'std': np.std(joint_errors),
                    'min': np.min(joint_errors),
                    'max': np.max(joint_errors)
                }
        
        if joint_overall_stats:
            print(f"\nJoint-wise Overall Error Statistics:")
            for joint_name, joint_stat in joint_overall_stats.items():
                print(f"  {joint_name}: Mean={joint_stat['mean']:.6f}, Std={joint_stat['std']:.6f}, Range=[{joint_stat['min']:.6f}, {joint_stat['max']:.6f}]")
        
        # Overall FK error statistics
        fk_pos_errors = []
        fk_rot_errors = []
        fk_angle_errors = []
        fk_pos_x_errors = []
        fk_pos_y_errors = []
        fk_pos_z_errors = []
        
        for stats in self.dataset_results.values():
            if 'fk_error_stats' in stats:
                fk_stats = stats['fk_error_stats']
                fk_pos_errors.append(fk_stats['position_error_mean'])
                fk_rot_errors.append(fk_stats['rotation_error_mean'])
                fk_angle_errors.append(fk_stats['angle_error_mean'])
                fk_pos_x_errors.append(fk_stats['pos_x_error_mean'])
                fk_pos_y_errors.append(fk_stats['pos_y_error_mean'])
                fk_pos_z_errors.append(fk_stats['pos_z_error_mean'])
        
        if fk_pos_errors:
            print(f"\n--- Overall Forward Kinematics Error Statistics (IK Failures) ---")
            print(f"Position Error - Mean: {np.mean(fk_pos_errors):.6f}, Std: {np.std(fk_pos_errors):.6f}, Range: [{np.min(fk_pos_errors):.6f}, {np.max(fk_pos_errors):.6f}]")
            print(f"Rotation Error - Mean: {np.mean(fk_rot_errors):.6f}, Std: {np.std(fk_rot_errors):.6f}, Range: [{np.min(fk_rot_errors):.6f}, {np.max(fk_rot_errors):.6f}]")
            print(f"Angle Error - Mean: {np.mean(fk_angle_errors):.2f}°, Std: {np.std(fk_angle_errors):.2f}°, Range: [{np.min(fk_angle_errors):.2f}°, {np.max(fk_angle_errors):.2f}°]")
            print(f"Position X Error - Mean: {np.mean(fk_pos_x_errors):.6f}, Std: {np.std(fk_pos_x_errors):.6f}")
            print(f"Position Y Error - Mean: {np.mean(fk_pos_y_errors):.6f}, Std: {np.std(fk_pos_y_errors):.6f}")
            print(f"Position Z Error - Mean: {np.mean(fk_pos_z_errors):.6f}, Std: {np.std(fk_pos_z_errors):.6f}")
    
    def run_all_tests(self):
        """Run IK tests on all available datasets."""
        print("Z1 Robot Arm Inverse Kinematics Test Suite")
        print("="*50)
        
        # Find all robot_data.pkl files
        data_files = self.find_robot_data_files()
        
        if not data_files:
            print("No robot_data.pkl files found!")
            return
        
        print(f"Found {len(data_files)} robot_data.pkl files:")
        for file_path in data_files:
            print(f"  - {file_path}")
        
        # Test each dataset
        for file_path in data_files:
            # Extract dataset name from path
            dataset_name = os.path.basename(os.path.dirname(file_path))
            if not dataset_name:
                dataset_name = "root"
            
            # Load data
            data = self.load_robot_data(file_path)
            if data is None:
                continue
            
            # Test this dataset
            results = self.test_single_dataset(data, dataset_name)
            self.dataset_results[dataset_name] = results
        
        # Print overall statistics
        self.print_overall_statistics()
        
        print(f"\nTest completed! Processed {len(self.dataset_results)} datasets.")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.arm.loopOff()
            print("Robot arm interface closed.")
        except:
            pass

def main():
    """Main function."""
    test_suite = IKTestSuite()
    
    try:
        test_suite.run_all_tests()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    main()
