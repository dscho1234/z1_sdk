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
from datetime import datetime
from scipy.spatial.transform import Rotation as R
# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface


import pyrealsense2 as rs


# Set numpy print options
np.set_printoptions(precision=3, suppress=True)

class FreeDriveDataCollector:
    """Data collector for free drive mode with camera and robot data collection."""
    
    def __init__(self, output_dir="free_drive_data", has_gripper=True, resolution="HD"):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            has_gripper: Whether the robot has a gripper
            resolution: Camera resolution ("VGA", "HD", "FHD")
        """
        self.output_dir = output_dir
        self.has_gripper = has_gripper
        self.resolution = resolution
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "depth_images"), exist_ok=True)
        
        
        # Initialize robot arm
        print("Initializing Z1 robot arm...")
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel
        
        # Initialize camera if available
        self.camera_available = False
        
        self._init_camera()
        
        # Data collection parameters
        
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
        self.images_data = []  # Store images in memory for batch saving
        self.depth_images_data = []  # Store depth images in memory for batch saving
        
        print(f"Data collector initialized. Output directory: {output_dir}")
        print(f"Camera available: {self.camera_available}")
        print(f"Camera resolution: {self.resolution}")
        print(f"Warmup steps: {self.warmup_steps}")
        
    def _init_camera(self):
        """Initialize RealSense D435 camera."""
        try:
            print("Initializing RealSense D435 camera...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Get resolution settings based on resolution parameter
            width, height, fps = self._get_resolution_settings()
            
            # Configure color stream
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            # 정렬 필터 설정 (depth와 color 프레임 정렬)
            self.align = rs.align(rs.stream.color)
            
            # Start streaming
            self.pipeline.start(config)
            self.camera_available = True
            print(f"RealSense D435 camera initialized successfully with {width}x{height} @ {fps}fps.")
            
            # Configure camera settings (disable auto exposure and auto focus)
            self._configure_camera_settings()
            
            # Get camera intrinsic parameters
            self._print_camera_intrinsics()
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera_available = False
    
    def _configure_camera_settings(self):
        """Configure camera settings to disable auto exposure and auto focus."""
        try:
            # Get the active profile and device
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            
            # Get color sensor (usually the second sensor)
            sensors = device.query_sensors()
            color_sensor = None
            for sensor in sensors:
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                    color_sensor = sensor
                    break
            
            if color_sensor is not None:
                print("Configuring camera settings...")
                
                # Disable auto exposure
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    print("  - Auto exposure disabled")
                
                # Set manual exposure (adjust this value based on your lighting conditions)
                if color_sensor.supports(rs.option.exposure):
                    exposure_value = 166  # Default exposure in milliseconds
                    color_sensor.set_option(rs.option.exposure, exposure_value)
                    print(f"  - Manual exposure set to {exposure_value}ms")
                
                # Disable auto white balance
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    print("  - Auto white balance disabled")
                
                # Set manual white balance (adjust based on your lighting)
                if color_sensor.supports(rs.option.white_balance):
                    white_balance_value = 3000  # Default white balance (Kelvin)
                    color_sensor.set_option(rs.option.white_balance, white_balance_value)
                    print(f"  - Manual white balance set to {white_balance_value}K")
                
                # Set brightness
                if color_sensor.supports(rs.option.brightness):
                    brightness_value = 0  # Default brightness
                    color_sensor.set_option(rs.option.brightness, brightness_value)
                    print(f"  - Brightness set to {brightness_value}")
                
                # Set contrast
                if color_sensor.supports(rs.option.contrast):
                    contrast_value = 50  # Default contrast
                    color_sensor.set_option(rs.option.contrast, contrast_value)
                    print(f"  - Contrast set to {contrast_value}")
                
                print("Camera settings configured successfully.")
            else:
                print("Warning: Could not find RGB camera sensor for configuration.")
                
        except Exception as e:
            print(f"Warning: Failed to configure camera settings: {e}")
            print("Camera will use default auto settings.")
    
    def _get_resolution_settings(self):
        """Get camera resolution settings based on resolution parameter."""
        resolution_settings = {
            "VGA": (640, 480, 30),    # VGA resolution
            "HD": (1280, 720, 30),    # HD resolution  
            "FHD": (1920, 1080, 30)   # Full HD resolution
        }
        
        if self.resolution in resolution_settings:
            return resolution_settings[self.resolution]
        else:
            print(f"Unknown resolution '{self.resolution}', using HD as default.")
            return resolution_settings["HD"]
    
    def _print_camera_intrinsics(self):
        """Print camera intrinsic parameters."""
        try:
            # Get the active profile and color stream
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            print("\n=== Camera Intrinsic Parameters ===")
            print(f"Width: {intrinsics.width}")
            print(f"Height: {intrinsics.height}")
            print(f"Focal Length (fx, fy): ({intrinsics.fx:.6f}, {intrinsics.fy:.6f})")
            print(f"Principal Point (cx, cy): ({intrinsics.ppx:.6f}, {intrinsics.ppy:.6f})")
            print(f"Distortion Model: {intrinsics.model}")
            print(f"Distortion Coefficients: {intrinsics.coeffs}")
            print("=====================================\n")
            
        except Exception as e:
            print(f"Failed to get camera intrinsics: {e}")
    
    
    def _capture_image(self):
        """Capture image from RealSense camera."""
        if not self.camera_available:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            
            if not color_frame:
                return None
            
            # Convert to numpy array
            image = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()) # uint16
            return image, depth
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def _show_streaming_window(self):
        """Show real-time camera streaming window."""
        if not self.camera_available:
            print("Camera not available for streaming.")
            return
        
        print("Starting camera streaming...")
        print("Press 'q' to quit streaming and continue with data collection.")
        print("Press 'c' to capture current frame for data collection.")
        print("Make sure the camera window is focused for keyboard input to work.")
        
        # Create a named window first
        cv2.namedWindow('Camera Stream - Press q to quit, c to capture', cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # Capture image
                image, depth = self._capture_image()
                if image is not None:
                    # Get robot state for real-time display
                    robot_state = self._get_robot_state()
                    
                    # Add text overlay
                    display_image = image.copy()
                    cv2.putText(display_image, "Press 'q' to quit, 'c' to capture", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Images collected: {len(self.images_data)}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add real-time robot state info to image
                    if robot_state is not None:
                        joint_angles = robot_state['joint_angles']
                        ee_pos = robot_state['ee_position']
                        ee_quat = robot_state['ee_quaternion']
                        ee_euler = robot_state['ee_euler']
                        se3_transform = robot_state['T']
                        
                        # Display joint angles
                        cv2.putText(display_image, f"Joint: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}]", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display_image, f"      [{joint_angles[3]:.3f}, {joint_angles[4]:.3f}, {joint_angles[5]:.3f}]", 
                                  (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Display end effector position
                        cv2.putText(display_image, f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]", 
                                  (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Display end effector quaternion
                        cv2.putText(display_image, f"EE Quat: [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]", 
                                  (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Display end effector Euler angles (radian)
                        cv2.putText(display_image, f"EE Euler: [{ee_euler[0]:.3f}, {ee_euler[1]:.3f}, {ee_euler[2]:.3f}] (rad)", 
                                  (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Display SE3 transform
                        cv2.putText(display_image, f"SE3 Transform: [{se3_transform[0, 0]:.3f}, {se3_transform[0, 1]:.3f}, {se3_transform[0, 2]:.3f}, {se3_transform[0, 3]:.3f}]", 
                                  (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display_image, f"               [{se3_transform[1, 0]:.3f}, {se3_transform[1, 1]:.3f}, {se3_transform[1, 2]:.3f}, {se3_transform[1, 3]:.3f}]", 
                                  (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display_image, f"               [{se3_transform[2, 0]:.3f}, {se3_transform[2, 1]:.3f}, {se3_transform[2, 2]:.3f}, {se3_transform[2, 3]:.3f}]", 
                                  (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display_image, f"               [{se3_transform[3, 0]:.3f}, {se3_transform[3, 1]:.3f}, {se3_transform[3, 2]:.3f}, {se3_transform[3, 3]:.3f}]", 
                                  (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    else:
                        cv2.putText(display_image, "Robot state unavailable", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Show color image
                    cv2.imshow('Camera Stream - Press q to quit, c to capture', display_image)
                    
                    
                    
                    # Check for key press with longer wait time
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC key
                        print("\nQuitting streaming window...")
                        break
                    elif key == ord('c'):
                        # Capture current frame
                        print(f"\nCapturing frame {len(self.images_data) + 1}...")
                        if robot_state is not None:
                            image_stored = self._store_image(image)
                            depth_stored = self._store_depth_image(depth)
                            data_stored = self._store_robot_data(robot_state)
                            
                            if image_stored and depth_stored and data_stored:
                                print(f"✓ Frame captured successfully! Total: {len(self.images_data)}")
                                # Print robot state info
                                joint_angles = robot_state['joint_angles']
                                ee_pos = robot_state['ee_position']
                                print(f"  Joint Angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}, {joint_angles[3]:.3f}, {joint_angles[4]:.3f}, {joint_angles[5]:.3f}]")
                                print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                            else:
                                print("✗ Failed to capture frame")
                                if not image_stored:
                                    print("  - Color image storage failed")
                                if not depth_stored:
                                    print("  - Depth image storage failed")
                                if not data_stored:
                                    print("  - Robot data storage failed")
                        else:
                            print("✗ Failed to get robot state")
                else:
                    print("Failed to capture image")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStreaming interrupted by user.")
        finally:
            cv2.destroyAllWindows()
            print("Streaming window closed.")
    
    
    def _get_robot_state(self):
        """Get current robot state including joint angles and end-effector pose."""
        try:
            # Simply read the current state without sending any commands
            # This should work if the robot is in free drive mode (loopOff)
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
            
            # Convert rotation matrix to quaternion and Euler angles
            rotation = R.from_matrix(ee_rotation_matrix)
            ee_quaternion = rotation.as_quat()
            ee_euler = rotation.as_euler('xyz')  # Euler angles in radians (roll, pitch, yaw)
            
            return {
                'joint_angles': joint_angles,
                'joint_velocities': joint_velocities,
                'gripper_position': gripper_pos,
                'gripper_velocity': gripper_vel,
                'ee_position': ee_position,
                'ee_quaternion': ee_quaternion,
                'ee_euler': ee_euler,
                'ee_rotation_matrix': ee_rotation_matrix,
                'T' : T,
            }
            
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return None
    
    
    
    def _store_image(self, image):
        """Store image in memory for batch saving."""
        try:
            if image is not None:
                # Check if image has valid dimensions
                if len(image.shape) == 3 and image.shape[2] == 3:
                    self.images_data.append(image.copy())
                    print(f"Image stored successfully. Shape: {image.shape}")
                    return True
                else:
                    print(f"Invalid image shape: {image.shape}")
                    return False
            else:
                print("Image is None, cannot store")
                return False
        except Exception as e:
            print(f"Error storing image: {e}")
            return False
    
    def _store_depth_image(self, depth_image):
        """Store depth image in memory for batch saving."""
        try:
            if depth_image is not None:
                # Check if depth image has valid dimensions
                if len(depth_image.shape) == 2:
                    self.depth_images_data.append(depth_image.copy())
                    print(f"Depth image stored successfully. Shape: {depth_image.shape}")
                    return True
                else:
                    print(f"Invalid depth image shape: {depth_image.shape}")
                    return False
            else:
                print("Depth image is None, cannot store")
                return False
        except Exception as e:
            print(f"Error storing depth image: {e}")
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
        """Save all collected robot data and images."""
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
            
            # Save robot data to pkl file
            data_path = os.path.join(self.output_dir, "robot_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
            print(f"Robot data saved to: {data_path}")
            print(f"Data shapes:")
            for key, value in data_dict.items():
                if key != 'collection_steps':
                    print(f"  {key}: {value.shape}")
            
            # Save all images at once
            if self.images_data:
                print("Saving color images...")
                images_saved = 0
                for i, image in enumerate(self.images_data):
                    if image is not None:
                        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
                        image_path = os.path.join(self.output_dir, "images", f"image_{i:04d}.png")
                        success = cv2.imwrite(image_path, image)
                        if success:
                            images_saved += 1
                        else:
                            print(f"Warning: Failed to save image {i}")
                    else:
                        print(f"Warning: Image {i} is None, skipping...")
                print(f"Saved {images_saved}/{len(self.images_data)} color images to: {os.path.join(self.output_dir, 'images')}")
            else:
                print("No color images to save.")
            
            # Save all depth images at once
            if self.depth_images_data:
                print("Saving depth images...")
                depth_images_saved = 0
                for i, depth_image in enumerate(self.depth_images_data):
                    if depth_image is not None:
                        os.makedirs(os.path.join(self.output_dir, "depth_images"), exist_ok=True)
                        depth_image_path = os.path.join(self.output_dir, "depth_images", f"depth_{i:04d}.png")
                        success = cv2.imwrite(depth_image_path, depth_image)
                        if success:
                            depth_images_saved += 1
                        else:
                            print(f"Warning: Failed to save depth image {i}")
                    else:
                        print(f"Warning: Depth image {i} is None, skipping...")
                print(f"Saved {depth_images_saved}/{len(self.depth_images_data)} depth images to: {os.path.join(self.output_dir, 'depth_images')}")
            else:
                print("No depth images to save.")
            
            return True
            
        except Exception as e:
            print(f"Error saving all data: {e}")
            return False
    
    def enable_free_drive_mode(self):
        """Enable free drive mode by completely turning off robot control."""
        print("Enabling free drive mode...")
        
        # Turn off the control loop completely
        self.arm.loopOn()
        
        print("Free drive mode enabled. You can now manually move the robot arm.")
        print("Robot control is completely disabled - no resistance to manual movement.")
    
    
    def disable_free_drive_mode(self):
        """Disable free drive mode and return to normal control."""
        print("Disabling free drive mode...")
        
        # Return to normal control mode
        # self.arm.loopOn()
        self.arm.backToStart()
        self.arm.loopOff()
        
        print("Free drive mode disabled.")
    
    def collect_data(self):
        """Main data collection loop."""
        print(f"\nStarting data collection...")
        print(f"Warmup steps: {self.warmup_steps} (data collected but not saved)")
        print("Move the robot arm manually to different positions to collect diverse data.")
        print("Press Ctrl+C to stop collection early.\n")
        
        successful_steps = 0
        
        try:
            # Warmup phase - continuous data collection without saving
            print("=== WARMUP PHASE ===")
            for step in range(self.warmup_steps):
                step_num = step + 1
                print(f"Warmup Step {step_num}/{self.warmup_steps}", end=" ")
                
                # Capture image
                image = self._capture_image()
                
                # Get robot state (no commands sent during data collection)
                robot_state = self._get_robot_state()

                # Print robot state for every step
                if robot_state is not None:
                    joint_angles = robot_state['joint_angles']
                    ee_pos = robot_state['ee_position']
                    ee_quat = robot_state['ee_quaternion']
                    ee_euler = robot_state['ee_euler']
                    se3_transform = robot_state['T']
                    print(f"- Joint Angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}, {joint_angles[3]:.3f}, {joint_angles[4]:.3f}, {joint_angles[5]:.3f}]")
                    print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                    print(f"  EE Quaternion: [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]")
                    print(f"  EE Euler: [{ee_euler[0]:.3f}, {ee_euler[1]:.3f}, {ee_euler[2]:.3f}] (rad)")
                    
                
                print(f"- Data collected (warmup, not saved)")
                
                # Wait before next step
                time.sleep(self.step_delay)
            
            print(f"\nWarmup completed! Starting collection phase...")
            print("=== COLLECTION PHASE ===")
            print("Camera streaming window will open. Use it to preview and capture images.")
            print("Press 'c' in the streaming window to capture current frame.")
            print("Press 'q' in the streaming window when you're done collecting data.\n")
            
            # Collection phase - use streaming window for interactive data collection
            if self.camera_available:
                self._show_streaming_window()
                successful_steps = len(self.images_data)
            else:
                print("Camera not available. Cannot proceed with collection phase.")
                return 0
                
        except KeyboardInterrupt:
            print(f"\nData collection interrupted by user.")
        
        print(f"\nData collection completed!")
        print(f"Warmup steps completed: {self.warmup_steps}")
        print(f"Successfully collected {successful_steps} images and poses")
        
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
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"free_drive_data_{timestamp}"
    
    # Camera resolution options: "VGA" (640x480), "HD" (1280x720), "FHD" (1920x1080)
    resolution = "VGA" # "FHD"  # Change this to "VGA", "HD", or "FHD" as needed
    
    # Create data collector
    collector = FreeDriveDataCollector(output_dir=output_dir, resolution=resolution)
    
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
        print(f"- Successful collection steps: {successful_steps}")
        print(f"- Output directory: {collector.output_dir}")
        if successful_steps > 0:
            print(f"- Color images saved: {collector.output_dir}/images/")
            print(f"- Depth images saved: {collector.output_dir}/depth_images/")
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
