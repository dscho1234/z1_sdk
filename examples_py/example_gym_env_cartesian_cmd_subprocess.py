#!/usr/bin/env python3
"""
Z1 Gym Environment Example - Non-blocking Step Execution with Subprocess
================================================================================

This example demonstrates non-blocking step execution using subprocess.Popen
to run the environment in a separate Python process. This approach completely
avoids pickle issues with ArmInterface objects.

The main process:
1. Starts a subprocess running env_worker.py
2. Sends actions via JSON to the subprocess
3. Receives results back from the subprocess
4. Runs inference while the subprocess executes steps

The subprocess:
1. Initializes the environment with ArmInterface
2. Executes steps in the background
3. Returns results to the main process
"""

import sys
import os
import time
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))


class DummyMLP(nn.Module):
    """Simple dummy MLP model for thread-torch compatibility testing."""
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=8):
        super(DummyMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class SubprocessEnvironmentManager:
    """Manages environment operations through a subprocess."""
    
    def __init__(self, config):
        """
        Initialize the subprocess environment manager.
        
        Args:
            config: Environment configuration dictionary
        """
        self.config = config
        self.process = None
        self.is_initialized = False
        
    def start_worker(self):
        """Start the environment worker subprocess."""
        try:
            # Start the worker subprocess
            worker_script = os.path.join(os.path.dirname(__file__), "..", "env_worker.py")
            self.process = subprocess.Popen(
                [sys.executable, worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            
            print("SubprocessEnvironmentManager: Started worker subprocess")
            
            # Initialize the environment in the subprocess
            init_command = {
                "type": "init",
                "config": self.config
            }
            
            response = self._send_command(init_command)
            if response["status"] == "success":
                self.is_initialized = True
                print("SubprocessEnvironmentManager: Environment initialized in subprocess")
                return True
            else:
                print(f"SubprocessEnvironmentManager: Failed to initialize: {response['message']}")
                return False
                
        except Exception as e:
            print(f"SubprocessEnvironmentManager: Failed to start worker: {str(e)}")
            return False
    
    def _send_command(self, command):
        """Send a command to the worker subprocess and get response."""
        try:
            # Send command
            command_json = json.dumps(command) + "\n"
            print(f"SubprocessEnvironmentManager: Sending command: {command_json.strip()}")
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                return {"status": "error", "message": "No response from worker"}
            
            print(f"SubprocessEnvironmentManager: Received response: {response_line.strip()}")
            response = json.loads(response_line.strip())
            return response
            
        except Exception as e:
            return {"status": "error", "message": f"Communication error: {str(e)}"}
    
    def reset(self):
        """Reset the environment."""
        if not self.is_initialized:
            return None
        
        command = {"type": "reset"}
        response = self._send_command(command)
        
        if response["status"] == "success":
            return {
                "observation": np.array(response["observation"]),
                "action_space": response["action_space"],
                "observation_space": response["observation_space"]
            }
        else:
            print(f"SubprocessEnvironmentManager: Reset failed: {response['message']}")
            return None
    
    def step(self, action):
        """Execute a step in the environment."""
        if not self.is_initialized:
            return None
        
        command = {
            "type": "step",
            "action": action.tolist() if isinstance(action, np.ndarray) else action
        }
        
        response = self._send_command(command)
        
        if response["status"] == "success":
            return {
                "observation": np.array(response["observation"]),
                "reward": response["reward"],
                "done": response["done"],
                "info": response["info"]
            }
        else:
            print(f"SubprocessEnvironmentManager: Step failed: {response['message']}")
            return None
    
    def close(self):
        """Close the environment and terminate the subprocess."""
        if self.process is not None:
            try:
                # Send close command
                command = {"type": "close"}
                self._send_command(command)
                
                # Wait for process to terminate
                self.process.wait(timeout=5.0)
                print("SubprocessEnvironmentManager: Worker subprocess terminated")
                
            except subprocess.TimeoutExpired:
                print("SubprocessEnvironmentManager: Force killing worker subprocess")
                self.process.kill()
                self.process.wait()
                
            except Exception as e:
                print(f"SubprocessEnvironmentManager: Error closing worker: {str(e)}")
            
            finally:
                self.process = None
                self.is_initialized = False


def inference_function(observation, square_vertices, orientation_vertices, total_steps, current_step, dummy_mlp, sequence_length=16, inference_time=0.15):
    """
    Simulate inference function that takes observation and returns action sequence.
    This uses the same square sequence generation as the original example.
    
    Args:
        observation: Current observation from the environment
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        dummy_mlp: Dummy MLP model for thread-torch compatibility testing
        sequence_length: Length of target pose sequence to generate
        inference_time: Simulated inference time in seconds
        
    Returns:
        action_sequence: Array of shape (sequence_length, 8) containing future actions
    """
    # Simulate inference time
    time.sleep(inference_time)
    
    # Generate action sequence based on current step and square vertices
    action_sequence = []
    
    for i in range(sequence_length):
        # Calculate which vertex we should be targeting
        target_step = current_step + i
        vertex_index = target_step % len(square_vertices)
        
        # Get target position and orientation
        target_pos = square_vertices[vertex_index]
        target_quat = orientation_vertices[vertex_index]
        
        # Create action: [x, y, z, qw, qx, qy, qz, gripper]
        action = np.array([
            target_pos[0], target_pos[1], target_pos[2],  # position
            target_quat[0], target_quat[1], target_quat[2], target_quat[3],  # quaternion
            1.0 if vertex_index % 2 == 0 else -1.0  # gripper (alternate open/close)
        ])
        
        action_sequence.append(action)
    
    return np.array(action_sequence)


def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Non-blocking Step Execution with Subprocess")
    print("=" * 80)
    
    # Create dummy MLP model
    dummy_mlp = DummyMLP(input_dim=21, hidden_dim=4096, output_dim=8)
    print(f"Created dummy MLP model: {dummy_mlp}")
    
    # Environment configuration
    control_frequency = 2.0  # Hz
    sequence_length = 16
    step_interval = 1.0 / control_frequency  # Time between steps in seconds
    
    env_config = {
        'has_gripper': True,
        'control_frequency': control_frequency,
        'position_tolerance': 0.01,
        'orientation_tolerance': 0.1,
        'angular_vel': 1.0,
        'linear_vel': 0.3,
        'sequence_length': sequence_length,
    }
    
    # Create subprocess environment manager
    env_manager = SubprocessEnvironmentManager(env_config)
    
    try:
        # Start the worker subprocess
        if not env_manager.start_worker():
            print("Failed to start environment worker")
            return
        
        # Reset environment
        print("\nResetting environment...")
        reset_result = env_manager.reset()
        if reset_result is None:
            print("Failed to reset environment")
            return
        
        obs = reset_result["observation"]
        action_space = reset_result["action_space"]
        observation_space = reset_result["observation_space"]
        
        print(f"Action space: {action_space}")
        print(f"Observation space: {observation_space}")
        print(f"Initial observation shape: {obs.shape}")
        
        # Define square movement pattern
        original_pos = obs[:3]  # First 3 elements are position
        square_size = 0.1  # 10cm square
        
        square_vertices = [
            original_pos + np.array([0, 0, 0]),                    # Vertex 0
            original_pos + np.array([0, square_size, 0]),          # Vertex 1
            original_pos + np.array([0, square_size, square_size]), # Vertex 2
            original_pos + np.array([0, 0, square_size]),          # Vertex 3
            original_pos + np.array([0, 0, 0]),                    # Back to start
        ]
        
        # Define orientations (roll changes for each vertex)
        base_quat = np.array([1, 0, 0, 0])  # Identity quaternion
        orientation_vertices = [
            base_quat,  # 0°
            R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),   # +90°
            base_quat,  # 0°
            R.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),  # -90°
            base_quat,  # 0°
        ]
        
        print(f"\nOriginal EE position: {original_pos}")
        print(f"Current EE orientation: {obs[3:7]}")
        
        print(f"\nSquare movement plan:")
        for i, (pos, quat) in enumerate(zip(square_vertices, orientation_vertices)):
            gripper_state = "Open" if i % 2 == 0 else "Close"
            print(f"  Vertex {i}: {pos} (Gripper: {gripper_state})")
        
        # Run non-blocking control with overlapped inference
        total_steps = 30
        inference_time = 0.15  # seconds
        
        print(f"\nRunning non-blocking control with overlapped inference for {total_steps} steps")
        print(f"Inference time: {inference_time}s")
        print(f"Step interval: {step_interval:.3f}s")
        print("Note: Get observation → Execute step immediately → Run inference while robot moves")
        
        # Generate initial action sequence
        print(f"\nGenerating initial action sequence from o_0...")
        action_sequence = inference_function(obs, square_vertices, orientation_vertices, total_steps, 0, dummy_mlp, sequence_length, inference_time)
        print(f"Inference completed: generated sequence with shape {action_sequence.shape}")
        
        position_errors = []
        orientation_errors = []
        
        for step in range(total_steps):
            print(f"\nStep {step}: Processing")
            print("-" * 40)
            
            # Get current action from sequence
            action_index = step % len(action_sequence)
            action = action_sequence[action_index]
            
            gripper_cmd = action[7]
            print(f"Step {step}: Using action a_{step} from sequence (index {action_index})")
            print(f"Step {step}: Gripper Command = {gripper_cmd:.3f} ({'Open' if gripper_cmd > 0 else 'Close' if gripper_cmd < 0 else 'Neutral'})")
            
            # Execute step with non-blocking execution
            print(f"Step {step}: Executing action a_{step} with non-blocking subprocess...")
            
            start = time.time()
            step_result = env_manager.step(action)
            if step_result is None:
                print(f"Step {step}: Failed to execute step")
                continue
            
            obs, reward, done, info = step_result["observation"], step_result["reward"], step_result["done"], step_result["info"]
            print(f"Step {step}: Started non-blocking execution in {time.time() - start:.6f}s")
            
            # Run inference in main process while robot is moving (step > 0 and not last step)
            if step > 0 and step < total_steps - 1:
                print(f"Step {step}: Running inference with o_{step} while robot moves...")
                start = time.time()
                action_sequence = inference_function(obs, square_vertices, orientation_vertices, total_steps, step, dummy_mlp, sequence_length, inference_time)
                print(f"Step {step}: Inference completed in {time.time() - start:.3f}s")
                print(f"Step {step}: Updated action sequence for next steps")
            
            # Print gripper state
            gripper_state = obs[19] if len(obs) > 19 else 0.0
            print(f"Step {step}: Gripper State = {gripper_state:.3f}")
            
            # Collect errors for average calculation
            position_errors.append(info['position_error'])
            orientation_errors.append(info['orientation_error'])
            
            if done:
                print(f"Step {step}: Episode done!")
                break
        
        # Print final results
        print(f"\nNon-blocking square movement completed!")
        print(f"Final position: {obs[:3]}")
        print(f"Final position error: {info['position_error']:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final cartesian directions: {info['cartesian_directions']}")
        print(f"Final speeds - Linear: {info['actual_linear_speed']:.3f}, Angular: {info['actual_angular_speed']:.3f}, Gripper: {info['gripper_speed']:.3f}")
        print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        print(f"Final sequence index: {action_index}")
        print(f"Final sequence length: {len(action_sequence)}")
        
        print(f"\nAverage Errors:")
        print(f"  Average position error: {np.mean(position_errors):.4f} m")
        print(f"  Average orientation error: {np.mean(orientation_errors):.4f} rad")
        print(f"  Total steps executed: {len(position_errors)}")
        
        print(f"\nNon-blocking square movement with sequence handling completed successfully!")
        
    finally:
        # Clean up
        print("\nCleaning up...")
        print("Closing environment...")
        env_manager.close()


if __name__ == "__main__":
    # Run main example
    main()
