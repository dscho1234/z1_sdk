#!/usr/bin/env python3
"""
Free Drive Data Collection Example for Z1 Robot Arm with RealSense D435 Camera

This script enables free drive mode for the Z1 robot arm, allowing manual manipulation
while collecting data from a RealSense D435 camera mounted on the end-effector.
The script collects images, joint angles, and end-effector poses for 1000 steps 
and saves them as PNG images and PKL data files.

Requirements:
- Intel RealSense D435 camera
- pyrealsense2, opencv-python, numpy, pickle

Usage:
    python example_free_drive.py

Author: Generated for Z1 SDK
"""

import sys
import os
import time
import pickle
import numpy as np
import cv2

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    print("Warning: pyrealsense2 not available. Camera functionality will be disabled.")
    REALSENSE_AVAILABLE = False

# Set numpy print options
np.set_printoptions(precision=3, suppress=True)

class FreeDriveDataCollector:
    """Data collector for free drive mode with camera and robot data collection."""
    
    def __init__(self, output_dir="free_drive_data", has_gripper=True):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            has_gripper: Whether the robot has a gripper
        """
        self.output_dir = output_dir
        self.has_gripper = has_gripper
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        
        # Initialize robot arm
        print("Initializing Z1 robot arm...")
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel
        
        # Initialize camera if available
        self.camera_available = False
        if REALSENSE_AVAILABLE:
            self._init_camera()
        
        # Data collection parameters
        self.collection_steps = 10
        self.step_delay = 0.1  # 100ms between steps
        self.warmup_steps = 30  # Warmup steps (data collected but not saved)
        
        # Initialize data storage arrays
        self.joint_angles_data = []
        self.joint_velocities_data = []
        self.gripper_positions_data = []
        self.gripper_velocities_data = []
        self.ee_positions_data = []
        self.ee_quaternions_data = []
        self.ee_rotation_matrices_data = []
        self.T_matrices_data = []
        self.timestamps_data = []
        
        print(f"Data collector initialized. Output directory: {output_dir}")
        print(f"Camera available: {self.camera_available}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Collection steps: {self.collection_steps}")
    
    def _init_camera(self):
        """Initialize RealSense D435 camera."""
        try:
            print("Initializing RealSense D435 camera...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure color stream
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            self.pipeline.start(config)
            self.camera_available = True
            print("RealSense D435 camera initialized successfully.")
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera_available = False
    
    
    def _capture_image(self):
        """Capture image from RealSense camera."""
        if not self.camera_available:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            # Convert to numpy array
            image = np.asanyarray(color_frame.get_data())
            return image
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    
    def _get_robot_state(self):
        """Get current robot state including joint angles and end-effector pose."""
        try:
            # Get joint angles
            joint_angles = np.array(self.arm.lowstate.getQ())
            joint_velocities = np.array(self.arm.lowstate.getQd())
            
            # Get gripper state
            gripper_pos = self.arm.lowstate.getGripperQ() if self.has_gripper else 0.0
            gripper_vel = self.arm.gripperQd if self.has_gripper else 0.0
            
            # Calculate end-effector pose using forward kinematics
            T = self.arm_model.forwardKinematics(joint_angles, 6)
            
            # Extract position and orientation
            ee_position = T[:3, 3]
            ee_rotation_matrix = T[:3, :3]
            
            # Convert rotation matrix to quaternion
            ee_quaternion = self._rotation_matrix_to_quaternion(ee_rotation_matrix)
            
            return {
                'joint_angles': joint_angles,
                'joint_velocities': joint_velocities,
                'gripper_position': gripper_pos,
                'gripper_velocity': gripper_vel,
                'ee_position': ee_position,
                'ee_quaternion': ee_quaternion,
                'ee_rotation_matrix': ee_rotation_matrix,
                'T' : T,
            }
            
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return None
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def _save_image(self, step, image):
        """Save image for one step."""
        try:
            if image is not None:
                image_path = os.path.join(self.output_dir, "images", f"image_{step:04d}.png")
                cv2.imwrite(image_path, image)
                return True
            return False
        except Exception as e:
            print(f"Error saving image for step {step}: {e}")
            return False
    
    def _store_robot_data(self, robot_state):
        """Store robot data in arrays for batch saving."""
        try:
            if robot_state is not None:
                self.joint_angles_data.append(robot_state['joint_angles'])
                self.joint_velocities_data.append(robot_state['joint_velocities'])
                self.gripper_positions_data.append(robot_state['gripper_position'])
                self.gripper_velocities_data.append(robot_state['gripper_velocity'])
                self.ee_positions_data.append(robot_state['ee_position'])
                self.ee_quaternions_data.append(robot_state['ee_quaternion'])
                self.ee_rotation_matrices_data.append(robot_state['ee_rotation_matrix'])
                self.T_matrices_data.append(robot_state['T'])
                self.timestamps_data.append(time.time())
                return True
            return False
        except Exception as e:
            print(f"Error storing robot data: {e}")
            return False
    
    def _save_all_data(self):
        """Save all collected robot data as numpy arrays in a single pkl file."""
        try:
            # Convert lists to numpy arrays
            data_dict = {
                'joint_angles': np.array(self.joint_angles_data),
                'joint_velocities': np.array(self.joint_velocities_data),
                'gripper_positions': np.array(self.gripper_positions_data),
                'gripper_velocities': np.array(self.gripper_velocities_data),
                'ee_positions': np.array(self.ee_positions_data),
                'ee_quaternions': np.array(self.ee_quaternions_data),
                'ee_rotation_matrices': np.array(self.ee_rotation_matrices_data),
                'T_matrices': np.array(self.T_matrices_data),
                'timestamps': np.array(self.timestamps_data),
                'collection_steps': len(self.joint_angles_data)
            }
            
            # Save to single pkl file
            data_path = os.path.join(self.output_dir, "robot_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
            print(f"Robot data saved to: {data_path}")
            print(f"Data shapes:")
            for key, value in data_dict.items():
                if key != 'collection_steps':
                    print(f"  {key}: {value.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error saving all robot data: {e}")
            return False
    
    def enable_free_drive_mode(self):
        """Enable free drive mode by setting low-level commands with zero torque."""
        print("Enabling free drive mode...")
        
        # Set FSM to low-level command mode
        self.arm.setFsmLowcmd()
        
        # Get current joint positions
        current_q = np.array(self.arm.lowstate.getQ())
        current_qd = np.zeros(6)  # Zero velocity
        current_tau = np.zeros(6)  # Zero torque for free drive
        
        # Set gripper commands
        if self.has_gripper:
            current_gripper_q = self.arm.lowstate.getGripperQ()
            current_gripper_qd = 0.0
            current_gripper_tau = 0.0
        else:
            current_gripper_q = 0.0
            current_gripper_qd = 0.0
            current_gripper_tau = 0.0
        
        # Send commands
        self.arm.setArmCmd(current_q, current_qd, current_tau)
        if self.has_gripper:
            self.arm.setGripperCmd(current_gripper_q, current_gripper_qd, current_gripper_tau)
        self.arm.sendRecv()
        
        print("Free drive mode enabled. You can now manually move the robot arm.")
        print("The robot will maintain its current position with zero torque.")
    
    def disable_free_drive_mode(self):
        """Disable free drive mode and return to normal control."""
        print("Disabling free drive mode...")
        
        # Return to normal control mode
        self.arm.loopOn()
        self.arm.backToStart()
        self.arm.loopOff()
        
        print("Free drive mode disabled.")
    
    def collect_data(self):
        """Main data collection loop."""
        total_steps = self.warmup_steps + self.collection_steps
        print(f"\nStarting data collection...")
        print(f"Warmup steps: {self.warmup_steps} (data collected but not saved)")
        print(f"Collection steps: {self.collection_steps} (data saved)")
        print(f"Total steps: {total_steps}")
        print("Move the robot arm manually to different positions to collect diverse data.")
        print("Press Ctrl+C to stop collection early.\n")
        
        successful_steps = 0
        
        try:
            for step in range(total_steps):
                is_warmup = step < self.warmup_steps
                step_type = "Warmup" if is_warmup else "Collection"
                step_num = step + 1
                
                if is_warmup:
                    print(f"{step_type} Step {step_num}/{self.warmup_steps}", end=" ")
                else:
                    collection_step = step_num - self.warmup_steps
                    print(f"{step_type} Step {collection_step}/{self.collection_steps}", end=" ")
                
                # Capture image
                image = self._capture_image()
                
                # Get robot state
                robot_state = self._get_robot_state()
                
                if is_warmup:
                    # During warmup, collect data but don't save
                    print(f"- Data collected (warmup, not saved)")
                else:
                    # During collection, save data
                    collection_step = step - self.warmup_steps
                    
                    # Save image
                    image_saved = self._save_image(collection_step, image)
                    
                    # Store robot data
                    data_stored = self._store_robot_data(robot_state)
                    
                    if image_saved and data_stored:
                        successful_steps += 1
                        print(f"- Image and data saved successfully")
                    elif image_saved:
                        print(f"- Image saved, data storage failed")
                    elif data_stored:
                        print(f"- Data stored, image save failed")
                    else:
                        print(f"- Failed to save image and data")
                
                # Wait before next step
                time.sleep(self.step_delay)
                
        except KeyboardInterrupt:
            print(f"\nData collection interrupted by user.")
        
        print(f"\nData collection completed!")
        print(f"Warmup steps completed: {self.warmup_steps}")
        print(f"Successfully collected {successful_steps}/{self.collection_steps} steps")
        
        # Save all robot data as numpy arrays in a single pkl file
        if successful_steps > 0:
            print("Saving all robot data...")
            self._save_all_data()
            print(f"Data saved in: {self.output_dir}")
        else:
            print("No data was collected, skipping save.")
        
        return successful_steps
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera_available:
            try:
                self.pipeline.stop()
                print("Camera pipeline stopped.")
            except:
                pass
        
        try:
            self.arm.loopOff()
            print("Robot arm interface closed.")
        except:
            pass


def main():
    """Main function."""
    print("Z1 Robot Arm Free Drive Data Collection")
    print("=" * 50)
    
    # Create data collector
    collector = FreeDriveDataCollector()
    
    try:
        # Enable free drive mode
        collector.enable_free_drive_mode()
        
        # Wait for user to position robot
        input("\nPosition the robot arm and ensure camera view is clear.")
        input("Press Enter when ready to start data collection...")
        
        # Collect data
        successful_steps = collector.collect_data()
        
        # Disable free drive mode
        collector.disable_free_drive_mode()
        
        print(f"\nData collection summary:")
        print(f"- Warmup steps completed: {collector.warmup_steps}")
        print(f"- Collection steps attempted: {collector.collection_steps}")
        print(f"- Successful collection steps: {successful_steps}")
        print(f"- Success rate: {successful_steps/collector.collection_steps*100:.1f}%")
        print(f"- Output directory: {collector.output_dir}")
        if successful_steps > 0:
            print(f"- Images saved: {collector.output_dir}/images/")
            print(f"- Robot data saved: {collector.output_dir}/robot_data.pkl")
        else:
            print("- No data was saved (only warmup completed)")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        collector.cleanup()
        print("\nCleanup completed.")


if __name__ == "__main__":
    main()
