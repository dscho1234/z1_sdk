#!/usr/bin/env python3
"""
Example usage of the Z1 Gym environment with end-effector pose control using joint commands
and non-blocking step execution.

This example demonstrates how to use the EEPoseCtrlJointCmdWrapper with the new
wait argument to control the Z1 robot arm's end-effector pose using jointCtrlCmd.
"""

import sys
import os
import numpy as np
import time
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "envs"))
from envs.z1_env_jointctrl_wait import EEPoseCtrlJointCmdWrapper

# # charuco, view policy setting (from RHWE calibration, fk_debug fix)
# T_B_M = np.array([[ 0.02165318, 0.69297017, 0.72064103, 0.90887787],
#                 [-0.05427247, -0.71893243, 0.69295791, 0.35263668],
#                 [ 0.99829136, -0.05411571, 0.02204203, 0.55767073],
#                 [ 0.        , 0.        , 0.        , 1.        ]])

# charuco, view policy setting (from RHWE calibration, fk_debug fix, filtered data)
T_B_M = np.array([[ 0.02191792,  0.69378431, 0.71984924, 0.91460907],
                [-0.05328004, -0.71818842, 0.6938059, 0.35592764],
                [ 0.99833904, -0.05356038, 0.02122366, 0.55818676],
                [ 0.        , 0.        , 0.        , 1.        ]])

# charuco, view policy setting (from RHWE calibration, fk_debug fix)
T_E_C = np.array([[ 0.00101469, -0.09831225,  0.9951551,   0.02260249],
                [-0.99794707,  0.06362638,  0.00730324,  0.03172527],
                [-0.06403612, -0.99311952, -0.09804586,  0.05429471],
                [ 0.,          0.,          0.,          1.        ]])

import multiprocessing
import cloudpickle
import pickle
import threading
import traceback


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    
    # Dictionary to store active step threads: cmd_id -> thread
    active_step_threads = {}
    step_results = {}  # cmd_id -> (observation, reward, done, info)
    step_lock = threading.Lock()
    
    def _step_thread(cmd_id, action, wait):
        """Execute step in a separate thread."""
        try:
            observation, reward, done, info = env.step(action, wait)
            with step_lock:
                step_results[cmd_id] = (observation, reward, done, info)
                # Send result back to main process
                remote.send((cmd_id, 'step', (observation, reward, done, info)))
        except Exception as e:
            # Send error result with full traceback to main process
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            try:
                with step_lock:
                    step_results[cmd_id] = None
                    remote.send((cmd_id, 'step_error', error_msg))
            except Exception as send_error:
                # If we can't send the error, at least print it to stderr
                print(f"ERROR in worker process (step_thread): {error_msg}", file=sys.stderr, flush=True)
                print(f"ERROR sending error message: {send_error}", file=sys.stderr, flush=True)
        finally:
            with step_lock:
                if cmd_id in active_step_threads:
                    del active_step_threads[cmd_id]
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                cmd_id, action, wait = data
                # Start step in a separate thread so we can continue processing other commands
                thread = threading.Thread(target=_step_thread, args=(cmd_id, action, wait), daemon=True)
                thread.start()
                with step_lock:
                    active_step_threads[cmd_id] = thread
                # Don't wait for step to complete - return immediately
                # Result will be sent when step completes
            elif cmd == 'reset':
                cmd_id, kwargs = data
                try:
                    observation = env.reset(**kwargs)
                    remote.send((cmd_id, 'reset', observation))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    try:
                        remote.send((cmd_id, 'reset_error', error_msg))
                    except Exception as send_error:
                        print(f"ERROR in worker process (reset): {error_msg}", file=sys.stderr, flush=True)
                        print(f"ERROR sending error message: {send_error}", file=sys.stderr, flush=True)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                try:
                    remote.send((None, 'get_spaces', (env.observation_space, env.action_space)))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    print(f"ERROR in worker process (get_spaces): {error_msg}", file=sys.stderr, flush=True)
            elif cmd == 'env_method':
                cmd_id, method_name, method_args, method_kwargs = data
                try:
                    method = getattr(env, method_name)
                    result = method(*method_args, **method_kwargs)
                    remote.send((cmd_id, 'env_method', result))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    try:
                        remote.send((cmd_id, 'env_method_error', error_msg))
                    except Exception as send_error:
                        print(f"ERROR in worker process (env_method): {error_msg}", file=sys.stderr, flush=True)
                        print(f"ERROR sending error message: {send_error}", file=sys.stderr, flush=True)
            elif cmd == 'get_attr':
                cmd_id, attr_name = data
                try:
                    remote.send((cmd_id, 'get_attr', getattr(env, attr_name)))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    try:
                        remote.send((cmd_id, 'get_attr_error', error_msg))
                    except Exception as send_error:
                        print(f"ERROR in worker process (get_attr): {error_msg}", file=sys.stderr, flush=True)
                        print(f"ERROR sending error message: {send_error}", file=sys.stderr, flush=True)
            elif cmd == 'set_attr':
                cmd_id, attr_name, value = data
                try:
                    remote.send((cmd_id, 'set_attr', setattr(env, attr_name, value)))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    try:
                        remote.send((cmd_id, 'set_attr_error', error_msg))
                    except Exception as send_error:
                        print(f"ERROR in worker process (set_attr): {error_msg}", file=sys.stderr, flush=True)
                        print(f"ERROR sending error message: {send_error}", file=sys.stderr, flush=True)
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        except EOFError:
            break
        except Exception as e:
            # Catch any unexpected errors in the main loop and send to main process
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"ERROR in worker process (main loop): {error_msg}", file=sys.stderr, flush=True)
            try:
                # Try to send error to main process if we have a way to identify it
                remote.send(('worker_error', error_msg))
            except Exception as send_error:
                print(f"ERROR sending error message to main process: {send_error}", file=sys.stderr, flush=True)
            # Continue the loop to avoid crashing the worker process
            # The main process should detect the error and handle it appropriately


class PipeEnv:
    """
    Single environment wrapper using multiprocessing pipe for communication.
    
    This class solves the multiprocessing send/recv ordering problem by using
    command IDs to match requests with their corresponding responses.
    
    Usage example:
        # Start a step
        cmd_id = env.step(action)
        
        # Call other methods while step is running
        result = env.env_method('some_method', arg1, arg2)
        
        # Wait for the specific step to complete
        obs, reward, done, info = env.step_wait(cmd_id)
    """
    
    def __init__(self, env_fn, start_method=None):
        self.waiting = False
        self.closed = False
        self.cmd_counter = 0
        self.pending_results = {}  # cmd_id -> result
        
        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remote, self.work_remote = ctx.Pipe(duplex=True)
        args = (self.work_remote, self.remote, CloudpickleWrapper(env_fn))
        # daemon=True: if the main process crashes, we should not cause things to hang
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.work_remote.close()

        # self.remote.send(('get_spaces', None))
        # self.observation_space, self.action_space = self.remote.recv()

    def step_async(self, action, wait=False):
        self.cmd_counter += 1
        cmd_id = self.cmd_counter
        self.remote.send(('step', (cmd_id, action, wait)))
        self.waiting = True
        return cmd_id

    # def step_wait(self, cmd_id=None):
    #     if cmd_id is None:
    #         # Legacy mode: just receive the next result
    #         result = self.remote.recv()
    #         self.waiting = False
    #         if isinstance(result, tuple) and len(result) == 3:
    #             # New format: (cmd_id, cmd_type, data)
    #             cmd_id, cmd_type, data = result
    #             if cmd_type == 'step':
    #                 observation, reward, done, info = data
    #                 return observation, reward, done, info
    #         else:
    #             # Old format: (observation, reward, done, info)
    #             observation, reward, done, info = result
    #             return observation, reward, done, info
    #     else:
    #         # New mode: wait for specific command ID
    #         while cmd_id not in self.pending_results:
    #             result = self.remote.recv()
    #             if isinstance(result, tuple) and len(result) == 3:
    #                 recv_cmd_id, cmd_type, data = result
    #                 self.pending_results[recv_cmd_id] = (cmd_type, data)
    #             else:
    #                 # Handle legacy format
    #                 self.pending_results[cmd_id] = ('step', result)
    #                 break
            
    #         cmd_type, data = self.pending_results.pop(cmd_id)
    #         self.waiting = False
    #         if cmd_type == 'step':
    #             observation, reward, done, info = data
    #             return observation, reward, done, info
    #         else:
    #             raise ValueError(f"Expected step result, got {cmd_type}")
    #     return observation, reward, done, info

    def step_wait(self, cmd_id=None):
        """
        Wait for step result to be available.
        
        Args:
            cmd_id: Command ID from step() call. If None, waits for any pending step.
            
        Returns:
            observation, reward, done, info tuple
        """
        if cmd_id is None:
            # Legacy mode: just receive the next result
            result = self.remote.recv()
            self.waiting = False
            if isinstance(result, tuple) and len(result) == 3:
                # New format: (cmd_id, cmd_type, data)
                recv_cmd_id, cmd_type, data = result
                if cmd_type == 'step':
                    observation, reward, done, info = data
                    return observation, reward, done, info
                elif cmd_type == 'step_error':
                    raise RuntimeError(f"Step failed in worker process:\n{data}")
                elif cmd_type == 'worker_error':
                    raise RuntimeError(f"Worker process error:\n{data}")
            else:
                # Old format: (observation, reward, done, info)
                observation, reward, done, info = result
                return observation, reward, done, info
        else:
            # New mode: wait for specific command ID
            while cmd_id not in self.pending_results:
                result = self.remote.recv()
                if isinstance(result, tuple) and len(result) == 3:
                    recv_cmd_id, cmd_type, data = result
                    # Handle worker_error immediately
                    if cmd_type == 'worker_error':
                        raise RuntimeError(f"Worker process error:\n{data}")
                    self.pending_results[recv_cmd_id] = (cmd_type, data)
                else:
                    # Handle legacy format
                    self.pending_results[cmd_id] = ('step', result)
                    break
            
            cmd_type, data = self.pending_results.pop(cmd_id)
            self.waiting = False
            if cmd_type == 'step':
                observation, reward, done, info = data
                return observation, reward, done, info
            elif cmd_type == 'step_error':
                raise RuntimeError(f"Step failed in worker process:\n{data}")
            else:
                raise ValueError(f"Expected step result, got {cmd_type}")

    def step(self, action, wait=False):
        """
        Step the environment with the given action (non-blocking)
        This starts the step in background and returns immediately.
        Use step_wait() to get the results.

        :param action: ([int] or [float]) the action
        :param wait: If True, wait for step to complete before returning. If False, return cmd_id immediately.
        :return: If wait=False, returns cmd_id (use step_wait(cmd_id) to get results). If wait=True, returns (observation, reward, done, info).
        """
        cmd_id = self.step_async(action, wait)
        if wait:
            return self.step_wait(cmd_id)
        return cmd_id

    def reset(self, **kwargs):
        self.cmd_counter += 1
        cmd_id = self.cmd_counter
        self.remote.send(('reset', (cmd_id, kwargs)))
        
        # Wait for the specific result
        while cmd_id not in self.pending_results:
            result = self.remote.recv()
            if isinstance(result, tuple) and len(result) == 3:
                recv_cmd_id, cmd_type, data = result
                # Handle worker_error immediately
                if cmd_type == 'worker_error':
                    raise RuntimeError(f"Worker process error:\n{data}")
                self.pending_results[recv_cmd_id] = (cmd_type, data)
            else:
                # Handle legacy format - assume it's for this command
                self.pending_results[cmd_id] = ('reset', result)
                break
        
        cmd_type, data = self.pending_results.pop(cmd_id)
        if cmd_type == 'reset':
            return data
        elif cmd_type == 'reset_error':
            raise RuntimeError(f"Reset failed in worker process:\n{data}")
        else:
            raise ValueError(f"Expected reset result, got {cmd_type}")

    def close(self):
        if self.closed:
            return
        try:
            if self.waiting:
                self.remote.recv()
            self.remote.send(('close', None))
        except (EOFError, BrokenPipeError, ConnectionResetError):
            # Worker process already terminated, skip communication
            pass
        except Exception as e:
            print(f"Warning: Error during close communication: {e}")
        
        try:
            self.process.join(timeout=5.0)  # Add timeout to prevent hanging
        except Exception as e:
            print(f"Warning: Error joining process: {e}")
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
                if self.process.is_alive():
                    self.process.kill()
        
        self.closed = True
    
    def env_method(self, method_name, *method_args, **method_kwargs):
        """Call instance methods of environment."""
        self.cmd_counter += 1
        cmd_id = self.cmd_counter
        self.remote.send(('env_method', (cmd_id, method_name, method_args, method_kwargs)))
        
        # Wait for the specific result
        while cmd_id not in self.pending_results:
            result = self.remote.recv()
            if isinstance(result, tuple) and len(result) == 3:
                recv_cmd_id, cmd_type, data = result
                # Handle worker_error immediately
                if cmd_type == 'worker_error':
                    raise RuntimeError(f"Worker process error:\n{data}")
                self.pending_results[recv_cmd_id] = (cmd_type, data)
            else:
                # Handle legacy format - assume it's for this command
                self.pending_results[cmd_id] = ('env_method', result)
                break
        
        cmd_type, data = self.pending_results.pop(cmd_id)
        if cmd_type == 'env_method':
            return data
        elif cmd_type == 'env_method_error':
            raise RuntimeError(f"env_method '{method_name}' failed in worker process:\n{data}")
        else:
            raise ValueError(f"Expected env_method result, got {cmd_type}")

    def get_attr(self, attr_name):
        """Return attribute from environment."""
        self.cmd_counter += 1
        cmd_id = self.cmd_counter
        self.remote.send(('get_attr', (cmd_id, attr_name)))
        
        # Wait for the specific result
        while cmd_id not in self.pending_results:
            result = self.remote.recv()
            if isinstance(result, tuple) and len(result) == 3:
                recv_cmd_id, cmd_type, data = result
                # Handle worker_error immediately
                if cmd_type == 'worker_error':
                    raise RuntimeError(f"Worker process error:\n{data}")
                self.pending_results[recv_cmd_id] = (cmd_type, data)
            else:
                # Handle legacy format - assume it's for this command
                self.pending_results[cmd_id] = ('get_attr', result)
                break
        
        cmd_type, data = self.pending_results.pop(cmd_id)
        if cmd_type == 'get_attr':
            return data
        elif cmd_type == 'get_attr_error':
            raise RuntimeError(f"get_attr '{attr_name}' failed in worker process:\n{data}")
        else:
            raise ValueError(f"Expected get_attr result, got {cmd_type}")

    def set_attr(self, attr_name, value):
        """Set attribute inside environment."""
        self.cmd_counter += 1
        cmd_id = self.cmd_counter
        self.remote.send(('set_attr', (cmd_id, attr_name, value)))
        
        # Wait for the specific result
        while cmd_id not in self.pending_results:
            result = self.remote.recv()
            if isinstance(result, tuple) and len(result) == 3:
                recv_cmd_id, cmd_type, data = result
                # Handle worker_error immediately
                if cmd_type == 'worker_error':
                    raise RuntimeError(f"Worker process error:\n{data}")
                self.pending_results[recv_cmd_id] = (cmd_type, data)
            else:
                # Handle legacy format - assume it's for this command
                self.pending_results[cmd_id] = ('set_attr', result)
                break
        
        cmd_type, data = self.pending_results.pop(cmd_id)
        if cmd_type == 'set_attr':
            return data
        elif cmd_type == 'set_attr_error':
            raise RuntimeError(f"set_attr '{attr_name}' failed in worker process:\n{data}")
        else:
            raise ValueError(f"Expected set_attr result, got {cmd_type}")



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


def inference_function(observation, square_vertices, orientation_vertices, total_steps, current_step, dummy_mlp, sequence_length=16, inference_time=0.15, env=None):
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
        inference_time: Time to simulate inference (seconds)
        
    Returns:
        action_sequence: Generated action sequence [sequence_length, 8]
    """
    print(f"Starting inference with observation shape {observation.shape} (simulating {inference_time:.3f}s)...")
    
    # Dummy MLP inference for thread-torch compatibility testing
    with torch.no_grad():
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        dummy_output = dummy_mlp(obs_tensor)
        print(f"Dummy MLP inference completed: device: {dummy_output.device}, input shape {obs_tensor.shape}, output shape {dummy_output.shape}")
    
    start = time.time()
    for _ in range(10):
        env.env_method('_get_current_ee_pose')
    print(f"In inference function, get_current_ee_pose time: {time.time() - start:.6f}s")

    # Simulate inference time
    time.sleep(inference_time)
    
    # Extract current position and orientation from observation
    current_pos = observation[12:15]  # End-effector position
    current_orient = observation[15:19]  # End-effector orientation (quaternion in [x,y,z,w] format)
    current_gripper = observation[19]  # Gripper position
    
    # Use the same square sequence generation as the original example
    action_sequence = generate_action_sequence(
        current_pos, current_orient, current_gripper, 
        square_vertices, orientation_vertices, 
        total_steps, current_step, sequence_length
    )
    
    print(f"Inference completed: generated sequence with shape {action_sequence.shape}")
    return action_sequence


def generate_action_sequence(base_position, base_orientation, base_gripper, square_vertices, orientation_vertices, total_steps, current_step, sequence_length=10):
    """
    Generate a target pose sequence that follows the square path.
    This function generates the sequence directly without simulation of inference time.
    
    Args:
        base_position: Base position to use as starting point
        base_orientation: Base orientation to use
        base_gripper: Base gripper position to use
        square_vertices: List of square vertices defining the path
        orientation_vertices: List of orientations for each vertex
        total_steps: Total number of steps for the entire square path
        current_step: Current step to start the sequence from
        sequence_length: Length of target pose sequence to generate
        
    Returns:
        target_sequence: Pre-computed target pose sequence [sequence_length, 8]
    """
    sequence = np.zeros((sequence_length, 8))
    
    # Calculate square path parameters based on total_steps
    total_edges = 4  # 4 edges of the square
    steps_per_edge = total_steps // total_edges
    
    for i in range(sequence_length):
        # Calculate the actual step number in the total path
        actual_step = current_step + i
        
        # Calculate which edge we're on and position along that edge
        edge_index = actual_step // steps_per_edge
        step_in_edge = actual_step % steps_per_edge
        
        # Ensure we don't go beyond the last vertex
        if edge_index >= 4:
            edge_index = 3
            step_in_edge = steps_per_edge - 1
        
        # Calculate current target position along the current edge
        start_vertex = square_vertices[edge_index]
        end_vertex = square_vertices[edge_index + 1]
        
        # Interpolate position along the edge
        t = step_in_edge / steps_per_edge
        target_position = start_vertex + t * (end_vertex - start_vertex)
        
        # Interpolate orientation along the edge
        start_orientation = orientation_vertices[edge_index]
        end_orientation = orientation_vertices[edge_index + 1]
        
        # Use SLERP (Spherical Linear Interpolation) for proper quaternion interpolation
        # start_orientation and end_orientation are in [x,y,z,w] format
        start_rot = R.from_quat(start_orientation)
        end_rot = R.from_quat(end_orientation)
        
        # Create a rotation that represents the interpolation
        # This uses scipy's built-in SLERP functionality
        new_rot = start_rot * (start_rot.inv() * end_rot) ** t
        
        # Convert back to quaternion (returns [x,y,z,w] format)
        new_orientation = new_rot.as_quat().squeeze()
        
        # Gripper: alternate between open and close for each edge
        # Edge 0: open (1), Edge 1: close (-1), Edge 2: open (1), Edge 3: close (-1)
        if edge_index % 2 == 0:
            new_gripper = -1.0  # Open
        else:
            new_gripper = 0.0  # Close
        
        # Combine into action: [x, y, z, qx, qy, qz, qw, gripper]
        # new_orientation is in [x,y,z,w] format, which matches the expected action format
        sequence[i] = np.concatenate([target_position, new_orientation, [new_gripper]])
    
    return sequence

def backproject_pixel(K, u, v, depth, dist_coeffs=None):
    """
    Backproject a 2D pixel coordinate to 3D point in camera coordinates.
    If dist_coeffs is provided, undistorts the pixel coordinate first.
    
    Args:
        K: (3,3) camera intrinsic matrix
        u, v: pixel coordinates
        depth: depth value (z coordinate in camera frame)
        dist_coeffs: (5,) or None, distortion coefficients
    
    Returns:
        point3d: (3,) 3D point in camera coordinates
    """
    if dist_coeffs is not None:
        # Undistort the pixel coordinate first
        pixel = np.array([[u, v]], dtype=np.float32)
        pixel_undist = cv2.undistortPoints(pixel, K, dist_coeffs, P=K)
        u_undist = pixel_undist[0, 0, 0]
        v_undist = pixel_undist[0, 0, 1]
    else:
        u_undist = u
        v_undist = v
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = depth
    x = (u_undist - cx) * z / fx
    y = (v_undist - cy) * z / fy
    return np.array([x, y, z])

def get_dift_point_base_frame_data_for_debug(camera_matrix=None, dist_coeffs=None):
    import zarr
    from im2flow2act.common.utility.zarr import parallel_reading
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    dift_point_tracking_sequence = data_buffer[f"episode_{episode_idx}/dift_point_tracking_sequence"][:, :, :3].copy().transpose(1, 0, 2) # [N, T, 4 -> 3] -> [T, N, 3] camera frame
    T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    dift_points = data_buffer[f"episode_{episode_idx}/dift_points"][:].copy() # [N, 2]
    assert dift_point_tracking_sequence.shape[0] == T_mc_transformation.shape[0]
    
    T, N, _ = dift_point_tracking_sequence.shape
    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    

    dift_points_custom_unprojected_list = []
    depth = parallel_reading(group=data_buffer[f"episode_{episode_idx}/camera_0"], array_name="depth")[0].astype(np.float32) / 1000.0
    for dift_x, dift_y in dift_points:
        dift_points_custom_unprojected = backproject_pixel(camera_matrix, dift_x, dift_y, depth[dift_y, dift_x], dist_coeffs=dist_coeffs)
        dift_points_custom_unprojected_list.append(dift_points_custom_unprojected)
    dift_points_custom_unprojected = np.tile(np.stack(dift_points_custom_unprojected_list), (T, 1, 1)) # [T, N, 3]
    dift_points_custom_unprojected_c_homo = np.concatenate([dift_points_custom_unprojected, np.ones((T, N, 1))], axis=2) # [T, N, 4]
    dift_points_custom_unprojected_c = dift_points_custom_unprojected.copy()
    # [T, 4, 4] @ [T, N, 4] -> [T, N, 4]
    dift_points_custom_unprojected_b = np.einsum('hij,hkj->hki', T_bc_transformation, dift_points_custom_unprojected_c_homo)[:, :, :3]


    
    
    # Convert to homogeneous coordinates: [T, N, 3] -> [T, N, 4]
    camera_points_homo = np.concatenate([
        dift_point_tracking_sequence, 
        np.ones((T, N, 1))
    ], axis=2)  # [T, N, 4]
    
    # Reshape for batch matrix multiplication: [T, N, 4] -> [T*N, 4]
    camera_points_homo_flat = camera_points_homo.reshape(-1, 4)  # [T*N, 4]
    
    # Expand transformation matrices: [T, 4, 4] -> [T*N, 4, 4]
    T_bc_expanded = np.repeat(T_bc_transformation, N, axis=0)  # [T*N, 4, 4]
    T_mc_expanded = np.repeat(T_mc_transformation, N, axis=0)  # [T*N, 4, 4]
    
    # Batch matrix multiplication: [T*N, 4, 4] @ [T*N, 4] -> [T*N, 4]
    base_points_homo_flat = np.einsum('ijk,ik->ij', T_bc_expanded, camera_points_homo_flat)  # [T*N, 4]
    marker_points_homo_flat = np.einsum('ijk,ik->ij', T_mc_expanded, camera_points_homo_flat)  # [T*N, 4]
    
    # Convert back to 3D coordinates and reshape: [T*N, 4] -> [T*N, 3] -> [T, N, 3]
    dift_point_base_frame = base_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]
    dift_point_marker_frame = marker_points_homo_flat[:, :3].reshape(T, N, 3)  # [T, N, 3]



    # point_in_front_of_the_marker_homo = np.array([0.0, 0.0, 0.4, 1.0])
    point_in_front_of_the_marker_homo = np.array([0.0, 0.0, -0.6, 1.0])
    point_in_front_of_the_marker_base_frame = np.einsum('ij,j->i', T_B_M, point_in_front_of_the_marker_homo)[:3]

    se3_in_front_of_the_marker = np.eye(4)
    se3_in_front_of_the_marker[:3, :3] = np.array([[0.0, 0.0, 1.0], 
                                                    [0.0, -1.0, 0.0], 
                                                    [1.0, 0.0, 0.0]])

    se3_in_front_of_the_marker[:3, 3] = point_in_front_of_the_marker_base_frame
    se3_in_front_of_the_marker_base_frame = np.einsum('ij,jk->ik', T_B_M, se3_in_front_of_the_marker) # [4, 4]


    
    return dift_point_base_frame, dift_point_marker_frame, dift_point_tracking_sequence, point_in_front_of_the_marker_base_frame, dift_points_custom_unprojected_c, dift_points_custom_unprojected_b, se3_in_front_of_the_marker_base_frame

def get_estimated_hand_pose_base_frame_data_for_debug():
    import zarr
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_eye_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_dist_coeff"
    # data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/multi_marker_test_for_hand_pose_calib_debug_w_wrist_depth_scale"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_test_for_hand_pose_calib_debug"
    data_buffer = zarr.open(data_buffer_path, mode="a")
    episode_idx = 0
    T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt"][:].copy() # [T, 4, 4]
    T = T_mc_transformation.shape[0]

    # original ver
    # proprioception = data_buffer[f"episode_{episode_idx}/proprioception"][:].copy() # [T, 7], camera frame

    # NOTE: for debug
    import pickle
    def load_data_dict(data_path):
        """Load the data dictionary from pickle file"""
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    data_dict = load_data_dict(data_buffer_path + f'/episode_{episode_idx}/data_dict.pkl')
    
    # use first obs
    R_cg_opt = data_dict['R_cg_opt_trajectory'][:] # [T, 3, 3]
    t_cg_opt_depth = data_dict['t_cg_opt_depth_trajectory'][:] # [T, 3]
    t_cg_opt = data_dict['t_cg_opt_trajectory'][:] # [T, 3]
    t_cg_closed = data_dict['t_cg_closed_trajectory'][:] # [T, 3]
    euler_cg_opt = R.from_matrix(R_cg_opt).as_euler('xyz') # [T, 3]
    temp_gripper = np.tile(np.array([0.0]), (T, 1)) # [T, 1]
    # dscho NOTE: depth-based one is much smoother when using accurate T_B_M. accuracy is slightly better.
    proprioception = np.concatenate([t_cg_opt_depth, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_opt, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]
    # proprioception = np.concatenate([t_cg_closed, euler_cg_opt, temp_gripper], axis=-1) # [T, 7]

    # debugging (depth scaling)
    # proprioception[:, :3] = proprioception[:, :3] * 1.05 # 2.5% scaling (custom calibration)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@ apply scaling to the proprioception for debugging")
    # time.sleep(2)


    
    
    assert proprioception.shape[0] == T_mc_transformation.shape[0]
    
    T = proprioception.shape[0]
    proprioception_se3 = np.tile(np.eye(4), (T, 1, 1)) # [T, 4, 4]
    proprioception_se3[:, :3, :3] = R.from_euler('xyz', proprioception[:, 3:6]).as_matrix() # [T, 4, 4]
    proprioception_se3[:, :3, 3] = proprioception[:, :3] # [T, 4, 4]

    T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M, T_mc_transformation) # [T, 4, 4]
    proprioception_se3_b = np.einsum('hij,hjk->hik', T_bc_transformation, proprioception_se3) # [T, 4, 4]
    
    pos = proprioception_se3_b[:, :3, 3] # [T, 3]
    euler = R.from_matrix(proprioception_se3_b[:, :3, :3]).as_euler('xyz')
    proprioception_b = np.concatenate([pos, euler, proprioception[:, 6:7]], axis=1) # [T, 7]
    return proprioception_b, proprioception


def make_view_env(control_frequency=5, sequence_length=16, position_tolerance=0.01, 
                orientation_tolerance=0.1, has_gripper=False, T_E_C_view=None, view_urdf_path=None):
    """
    Utility function for multiprocessed view env (dummy).

    :param control_frequency: (int) control frequency in Hz
    :param sequence_length: (int) length of future target pose sequences
    :param position_tolerance: (float) position tolerance
    :param orientation_tolerance: (float) orientation tolerance
    :param angular_vel: (float) angular velocity limit
    :param linear_vel: (float) linear velocity limit
    :param has_gripper: (bool) whether to include gripper
    """
    def _init():
        
        # z1 jointCtrlCmd
        env = EEPoseCtrlJointCmdWrapper(
            has_gripper=has_gripper,
            control_frequency=control_frequency,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            sequence_length=sequence_length,
            joint_speed=0.5,
            use_current_joint_pos_when_ik_fails=True,
            T_E_C = T_E_C_view,
            urdf_path = view_urdf_path,
            fk_debug=True,
        )

        return env
    
    return _init



def main():
    """Main example function."""
    print("Z1 Gym Environment Example - Non-blocking Step Execution")
    print("=" * 80)
    
    # Create dummy MLP model for thread-torch compatibility testing
    # Note: observation shape is 21, so we need to match that
    dummy_mlp = DummyMLP(input_dim=21, hidden_dim=4096, output_dim=8)
    dummy_mlp.eval()  # Set to evaluation mode
    print(f"Created dummy MLP model: {dummy_mlp}")
    
    # Create the environment with 2Hz control frequency
    control_frequency = 2 # 5
    sequence_length = 16   # Length of future target pose sequences
    step_interval = 1.0 / control_frequency  # Time between steps in seconds


    view_env_fn = make_view_env(
        control_frequency=control_frequency,
        sequence_length=sequence_length,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        has_gripper=False,  # 카메라는 gripper가 없음
        T_E_C_view=T_E_C,
        view_urdf_path = "/home/dcho302/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf"
    )
    
    # Create multiprocess environments
    env = PipeEnv(view_env_fn)
    

    
    # env = EEPoseCtrlJointCmdWrapper(
    #     has_gripper=False, #True,
    #     control_frequency=control_frequency,  # 2Hz control frequency
    #     position_tolerance=0.005,
    #     orientation_tolerance=0.1,
    #     joint_speed=1.0,  # Joint speed limit
    #     sequence_length=sequence_length,  # Length of future sequences
    #     use_current_joint_pos_when_ik_fails = True,
        
    # )
    
    
    try:
        # Reset the environment
        print("Resetting environment...")
        # joint_angle = np.array([1.0, 1.5, -1.0, -0.54, 0.0, 0.0])
        # joint_angle = np.array([-0.8, 2.572, -1.533, -0.609, 1.493, 1.004])
        joint_angle = np.array([0.0, 0.5, -1.0, 0.54, 0.0, 0.0]) # forward
        # joint_angle = np.array([0.0, 0.5, -0.5, -0.54, 0.0, 0.0]) #forward
        
        # joint_angle = None
        obs = env.reset(joint_angle=joint_angle) # , option="lowcmd"
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial joint positions: {obs[:6]}")
        print(f"Initial end-effector position: {obs[12:15]}")
        print(f"Initial end-effector orientation (quaternion [x,y,z,w]): {obs[15:19]}")
        print(f"Initial gripper position: {obs[19]}")
        print()
    
        
        # Example: Non-blocking step execution with square movement
        print("Example: Non-blocking step execution with square movement")
        print("Using wait=False for non-blocking step execution")
        print("Gripper will alternate between open and close for each edge")
        print("Orientation will change roll: 0°, +90°, 0°, -90°, 0° for each edge")
        print(f"Control frequency: {control_frequency} Hz")
        print(f"Step interval: {step_interval:.3f}s")
        print(f"Sequence length: {sequence_length}")
        print()
        
        # Get current end-effector position and orientation from observation
        original_position = obs[12:15].copy()  # Store original position
        current_orientation = obs[15:19]  # Current end-effector orientation (quaternion in [x,y,z,w] format)
        target_gripper = 0 
        
        print(f"Original EE position: {original_position}")
        print(f"Current EE orientation (quaternion [x,y,z,w]): {current_orientation}")
        print()
        
        # dscho debug to specify the position and orientation
        # original_position = np.array([0.41145274, -0.00121779, 0.40713578])
        # Convert from [w,x,y,z] to [x,y,z,w] format for scipy compatibility
        # current_orientation_wxyz = np.array([0.9998209, -0.0011671, -0.01868073, -0.00280654])  # [w,x,y,z]
        current_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        current_orientation = np.array([current_orientation_wxyz[1], current_orientation_wxyz[2], 
                                       current_orientation_wxyz[3], current_orientation_wxyz[0]])  # [x,y,z,w]

        # action should be T_B_C, not T_B_E
        if T_E_C is not None:
            T_B_E = np.eye(4)
            T_B_E[:3, :3] = R.from_quat(current_orientation).as_matrix()
            T_B_E[:3, 3] = original_position
            T_B_C = T_B_E @ T_E_C

            current_orientation = T_B_C[:3, :3].copy()
            current_orientation = R.from_matrix(current_orientation).as_quat()
            original_position = T_B_C[:3, 3].copy()




        # Define square vertices in YZ plane (0.1m x 0.1m square)
        square_size = 0.1  # 0.1m
        # square_vertices = [
        #     original_position + np.array([0.0, 0.0, 0.0]),           # Start point
        #     original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
        #     original_position + np.array([0.0, square_size, square_size]),  # Y+0.1, Z+0.1
        #     original_position + np.array([0.0, 0.0, square_size]),   # Z+0.1
        #     original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        # ]

        square_vertices = [
            original_position + np.array([0.0, 0.0, 0.0]),           # Start point
            original_position + np.array([0.0, 0.0, -square_size]),   # Z-0.1
            original_position + np.array([0.0, square_size, -square_size]),  # Y+0.1, Z-0.1
            original_position + np.array([0.0, square_size, 0.0]),   # Y+0.1
            original_position + np.array([0.0, 0.0, 0.0])            # Back to start
        ]

        # (640, 480), charuco marker, halab D435, with dist=0 calib
        camera_matrix = np.array([[589.42356484, 0.0, 324.66477806],
                        [0.0, 589.20431488, 246.12765546],
                        [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = None
        
        # D435 depth-based dift point (leftside bottle cap): array([0.592886  , 0.24988669, 0.1399751 ]), (on the socket): array([ 0.58816114, -0.0290327 , -0.00381193]), 
        # unidepth-based dift point (leftside bottle cap): array([0.62878956, 0.22977131, 0.13334631]), (on the socket): array([0.52487209, 0.02690133, 0.01839266]), 
        # NOTE
        dift_point_base_frame, dift_point_marker_frame, dift_point_tracking_sequence, point_in_front_of_the_marker_base_frame, dift_points_custom_unprojected_c, dift_points_custom_unprojected_b, se3_in_front_of_the_marker_base_frame = get_dift_point_base_frame_data_for_debug(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs) # [T, N, 3]
        # DEBUG_POINT = dift_point_base_frame[0,1] + np.array([0.0, 0.0, 0.1])
        # # DEBUG_POINT = point_in_front_of_the_marker_base_frame

        # # proprioception_base_frame = get_estimated_hand_pose_base_frame_data_for_debug() # [T, 7]
        # # DEBUG_POINT = proprioception_base_frame[0, :3] # + np.array([0.0, 0.0, 0.1])
        
        
        # assert DEBUG_ACTION[-1] == 0.0, "assume camera is attached, so the gripper should not be moved"
        # DEBUG_ACTION = np.concatenate([DEBUG_POINT, current_orientation,  np.array([0.0])])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@ Currently using DEBUG_ACTION :', DEBUG_ACTION)

        
        DEBUG_ROLL_ANGLES = [0, 0, 0, 0, 0]
        # Define orientation vertices with roll changes: 0°, +90°, 0°, -90°, 0°
        # roll_angles = [0, np.pi/4, 0, -np.pi/4, 0]  # 0°, +90°, 0°, -90°, 0°
        
        roll_angles = DEBUG_ROLL_ANGLES
        
        
        orientation_vertices = []
        
        for roll_angle in roll_angles:
            # Convert base orientation (quaternion in [x,y,z,w] format) to rotation matrix, apply roll, convert back
            base_rot = R.from_quat(current_orientation)  # current_orientation is in [x,y,z,w] format
            roll_rot = R.from_euler('x', roll_angle)
            new_rot = base_rot * roll_rot
            orientation_vertices.append(new_rot.as_quat())  # Returns [x,y,z,w] format
        
        print("Square movement plan:")
        for i, vertex in enumerate(square_vertices):
            gripper_state = "Open" if i % 2 == 0 else "Close"
            print(f"  Vertex {i}: {vertex} (Gripper: {gripper_state})")
        print()

        
        total_steps = 30
        inference_time = 0.15  # Inference time in seconds
        
        # Action chunk configuration
        use_action_chunk = True # False  # Set to True to use chunk execution with interpolator
        chunk_size = 8  # Number of actions to execute in a single chunk
        
        print(f"Running non-blocking control with overlapped inference for {total_steps} steps")
        print(f"Inference time: {inference_time}s")
        print(f"Step interval: {step_interval:.3f}s")
        print(f"Use action chunk: {use_action_chunk} (chunk_size: {chunk_size})")
        print("Note: Get observation → Execute step immediately → Run inference while robot moves")
        print()
        
        # Track errors for average calculation
        position_errors = []
        orientation_errors = []
        
        # Sequence tracking variables
        current_action_sequence = None
        current_action_index = 0
        latest_inference_result = None
        
        # Generate initial action sequence from o_0 (before for loop)
        print("Generating initial action sequence from o_0...")
        current_action_sequence = inference_function(
            obs, square_vertices, orientation_vertices, total_steps, 0, dummy_mlp, sequence_length, inference_time, env
        ) # o_0 -> a_0, a_1, a_2, ...
        current_action_index = 0
        
        # Main execution loop for all steps
        step = 0
        while step < total_steps:
            print(f"\nStep {step}: Processing")
            print("-" * 40)
            
            # Select action for current step (use current sequence)
            if use_action_chunk:
                # Chunk mode: select chunk_size actions from sequence
                if current_action_sequence is not None and current_action_index < len(current_action_sequence):
                    # Get chunk_size actions from current sequence
                    end_index = min(current_action_index + chunk_size, len(current_action_sequence))
                    action_chunk = current_action_sequence[current_action_index:end_index]  # [chunk_size, 8]
                    current_action_index = end_index
                    print(f"Step {step}: Using action chunk from sequence (indices {current_action_index - len(action_chunk)} to {current_action_index - 1}, chunk_size: {len(action_chunk)})")
                    print(f"Step {step}: Action chunk: {action_chunk}")
                    # If chunk is smaller than chunk_size, pad with last action
                    if len(action_chunk) < chunk_size:
                        last_action = action_chunk[-1] if len(action_chunk) > 0 else np.concatenate([obs[12:15], obs[15:19], [target_gripper]])
                        padding = np.tile(last_action, (chunk_size - len(action_chunk), 1))
                        action_chunk = np.vstack([action_chunk, padding])
                        print(f"Step {step}: Padded chunk to size {chunk_size}")
                else:
                    # Fallback: maintain current pose for chunk_size steps
                    current_pos = obs[12:15]
                    current_orient = obs[15:19]
                    single_action = np.concatenate([current_pos, current_orient, [target_gripper]])
                    action_chunk = np.tile(single_action, (chunk_size, 1))
                    print(f"Step {step}: No action available, maintaining current pose for chunk")
                
                # Print gripper command (from first action in chunk)
                gripper_cmd = action_chunk[0, 7] if action_chunk.shape[1] > 7 else 0.0
                print(f"Step {step}: Gripper Command (first in chunk) = {gripper_cmd:.3f} ({'Open' if gripper_cmd > 0 else 'Close' if gripper_cmd < 0 else 'Neutral'})")
                
                # Execute step with non-blocking execution (chunk mode)
                print(f"Step {step}: Executing action chunk with non-blocking...")
                start = time.time()
                env.step(action_chunk, wait=False)  # [chunk_size, 8]
                print(f"Step {step}: Started non-blocking chunk execution in {time.time() - start:.6f}s")
                
                # Collect intermediate results for each action in chunk
                chunk_intermediate_results = []
                for h in range(chunk_size):
                    # Wait for this action in chunk to complete
                    print(f"Step {step}: Waiting for action {h} in chunk to complete...")
                    wait_start = time.time()
                    while not env.env_method('is_action_in_chunk_complete', h):
                        time.sleep(0.001)  # Small sleep to avoid busy waiting
                    print(f"Step {step}: Action {h} in chunk completed (waited {time.time() - wait_start:.6f}s)")
                    
                    # Get intermediate result for this action
                    intermediate_result = env.env_method('get_action_in_chunk_intermediate_result', h)
                    if intermediate_result:
                        chunk_intermediate_results.append(intermediate_result)
                        print(f"Step {step}: Got intermediate result for action {h} in chunk")
                    else:
                        print(f"Step {step}: Warning - No intermediate result for action {h} in chunk")
                
                # All actions in chunk are complete, get final observation from last action's result
                if chunk_intermediate_results:
                    last_intermediate_result = chunk_intermediate_results[-1]
                    # Get final observation using env method (state should be updated after last action)
                    obs = env.env_method('_get_observation')
                    reward = 0.0  # Reward not available in intermediate results
                    done = env.env_method('_is_done')
                    
                    # Collect errors from intermediate results
                    for h, intermediate_result in enumerate(chunk_intermediate_results):
                        if intermediate_result:
                            position_errors.append(intermediate_result.get('position_error', 0.0))
                            orientation_errors.append(intermediate_result.get('orientation_error', 0.0))
                            print(f"Step {step}: Action {h} in chunk - position_error: {intermediate_result.get('position_error', 0.0):.4f}, orientation_error: {intermediate_result.get('orientation_error', 0.0):.4f}")
                    
                    # Print gripper state from final observation
                    gripper_state = obs[19] if len(obs) > 19 else 0.0
                    print(f"Step {step}: Final Gripper State = {gripper_state:.3f}")
                else:
                    print(f"Step {step}: Warning - No intermediate results collected")
                    # Fallback: get observation directly
                    obs = env.env_method('_get_observation')
                    reward = 0.0
                    done = env.env_method('_is_done')
                
                # Increment step by chunk_size since we executed chunk_size actions
                step += chunk_size  # while loop이므로 -1 필요 없음

                # Run inference in main process while robot is moving (step > 0 and not last step)
                # Check if we need to run inference for the next chunk
                if step < total_steps:  # If there are more steps to execute, run inference
                    print(f"Step {step}: Running inference with o_{step} while robot moves...")
                    inference_start_time = time.time()
                    latest_inference_result = inference_function(
                        obs, square_vertices, orientation_vertices, total_steps, step, dummy_mlp, sequence_length, inference_time, env
                    ) # o_t+1 -> a_t+1, a_t+2, a_t+3, ...
                    inference_time_actual = time.time() - inference_start_time
                    print(f"Step {step}: Inference completed in {inference_time_actual:.3f}s")
                    
                    # Update action sequence with latest inference result for next step
                    if latest_inference_result is not None:
                        current_action_sequence = latest_inference_result
                        current_action_index = 0  # Reset index for new sequence
                        print(f"Step {step}: Updated action sequence for next steps")
                else:
                    time.sleep(0.1)
                    
                    # get result at the last iteration (just for logging)
                    
                    print(f"Step {step}: Waiting for o_{step+1}...")
                    start = time.time()
                    while not env.env_method('is_step_complete'):
                        time.sleep(0.001)  # Small sleep to avoid busy waiting
                    print(f"Step {step}: Waited for {time.time() - start:.6f}s for o_{step+1}")
                    
                    start = time.time()
                    result = env.env_method('get_step_result')
                    if result:
                        obs, reward, done, info = result # o_t+1, r_t, etc
                        print(f"Step {step}: Received o_{step+1}, time taken: {time.time() - start:.6f}s")
                    else:
                        print(f"Step {step}: Warning - No result from background step")
                        # Use previous observation if no result
                        pass
                        
                
                
                
            else:
                # Single action mode: use current code
                if current_action_sequence is not None and current_action_index < len(current_action_sequence):
                    # Use action from current sequence
                    action = current_action_sequence[current_action_index] # a_t
                    current_action_index += 1
                    print(f"Step {step}: Using action a_{step} from sequence (index {current_action_index-1})")
                else:
                    # Fallback: maintain current pose
                    current_pos = obs[12:15]
                    current_orient = obs[15:19]
                    action = np.concatenate([current_pos, current_orient, [target_gripper]])
                    print(f"Step {step}: No action available, maintaining current pose")
                
                # Print gripper command
                gripper_cmd = action[7] if len(action) > 7 else 0.0
                print(f"Step {step}: Gripper Command = {gripper_cmd:.3f} ({'Open' if gripper_cmd > 0 else 'Close' if gripper_cmd < 0 else 'Neutral'})")
                
                # Execute step with non-blocking execution
                print(f"Step {step}: Executing action a_{step} with non-blocking...")
                
                se3_in_front_of_the_marker_pos = se3_in_front_of_the_marker_base_frame[:3, 3]
                se3_in_front_of_the_marker_quat = R.from_matrix(se3_in_front_of_the_marker_base_frame[:3, :3]).as_quat()
                DEBUG_ACTION = np.concatenate([se3_in_front_of_the_marker_pos, se3_in_front_of_the_marker_quat]) # np.array([0.0])
                
                start = time.time()
                env.step(action, wait=False) # a_t
                # env.step(DEBUG_ACTION, wait=False) # a_t
                step +=1

                print(f"Step {step}: Started non-blocking execution in {time.time() - start:.6f}s")
                
                # Run inference in main process while robot is moving (step > 0 and not last step)
                if step > 0 and step < total_steps - 1:  # Don't run inference for step 0 or last step
                    print(f"Step {step}: Running inference with o_{step} while robot moves...")
                    inference_start_time = time.time()
                    latest_inference_result = inference_function(
                        obs, square_vertices, orientation_vertices, total_steps, step, dummy_mlp, sequence_length, inference_time, env
                    ) # o_t+1 -> a_t+1, a_t+2, a_t+3, ...
                    inference_time_actual = time.time() - inference_start_time
                    print(f"Step {step}: Inference completed in {inference_time_actual:.3f}s")
                    
                    # Update action sequence with latest inference result for next step
                    if latest_inference_result is not None:
                        current_action_sequence = latest_inference_result
                        current_action_index = 1  # Reset index for new sequence, NOTE: we assume that inference is completed before env.step is finished
                        print(f"Step {step}: Updated action sequence for next steps")
                else:
                    time.sleep(0.1)
                
                print(f"Step {step}: Waiting for o_{step+1}...")
                start = time.time()
                while not env.env_method('is_step_complete'):
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                print(f"Step {step}: Waited for {time.time() - start:.6f}s for o_{step+1}")

                # Get the result from background execution
                start = time.time()
                result = env.env_method('get_step_result')
                if result:
                    obs, reward, done, info = result # o_t+1, r_t, etc
                    print(f"Step {step}: Received o_{step+1}, time taken: {time.time() - start:.6f}s")
                else:
                    print(f"Step {step}: Warning - No result from background step")
                    # Use previous observation if no result
                    pass
                
                # Print gripper state
                gripper_state = obs[19] if len(obs) > 19 else 0.0
                print(f"Step {step}: Gripper State = {gripper_state:.3f}")
                
                # Collect errors for average calculation
                position_errors.append(info['position_error'])
                orientation_errors.append(info['orientation_error'])
            
            if done:
                print(f"Episode finished at step {step}!")
                break
        
        
        
        # Print final status
        final_pos = obs[12:15]
        
        if use_action_chunk:
            # Chunk mode: info contains arrays of length horizon, use mean values
            horizon = info.get('horizon', chunk_size)
            final_error = np.mean(info['position_error']) if isinstance(info['position_error'], np.ndarray) else float(info['position_error'])
            final_orientation_error = np.mean(info['orientation_error']) if isinstance(info['orientation_error'], np.ndarray) else float(info['orientation_error'])
            joint_speed = np.mean(info['actual_joint_speed']) if isinstance(info['actual_joint_speed'], np.ndarray) else float(info['actual_joint_speed'])
            gripper_speed = np.mean(info['gripper_speed']) if isinstance(info['gripper_speed'], np.ndarray) else float(info['gripper_speed'])
            joint_directions = info['joint_directions']
            if isinstance(joint_directions, np.ndarray) and len(joint_directions.shape) > 1:
                # If 2D array, use mean across horizon
                joint_directions = np.mean(joint_directions, axis=0)
        else:
            # Single action mode: info contains single values
            final_error = float(info['position_error'])
            final_orientation_error = float(info['orientation_error']) if 'orientation_error' in info else 0.0
            joint_speed = float(info['actual_joint_speed'])
            gripper_speed = float(info['gripper_speed'])
            joint_directions = info['joint_directions']
        
        print(f"\nNon-blocking square movement completed!")
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        if use_action_chunk:
            print(f"Final position error (mean over {horizon} actions): {final_error:.4f}")
            print(f"Final orientation error (mean over {horizon} actions): {final_orientation_error:.4f}")
        else:
            print(f"Final position error: {final_error:.4f}")
        print(f"Final FSM state: {info['fsm_state']}")
        print(f"Final joint directions: {joint_directions}")
        print(f"Final speeds - Joint: {joint_speed:.3f}, Gripper: {gripper_speed:.3f}")
        # print(f"DT ratio (actual_control_time/arm_dt): {info['dt_ratio']}")
        
        # Calculate and print average errors
        if position_errors:
            avg_position_error = np.mean(position_errors)
            avg_orientation_error = np.mean(orientation_errors)
            print(f"\nAverage Errors:")
            print(f"  Average position error: {avg_position_error:.4f} m")
            print(f"  Average orientation error: {avg_orientation_error:.4f} rad")
            print(f"  Total steps executed: {len(position_errors)}")
            print(f" last step error: {position_errors[-1]:.4f} m, {orientation_errors[-1]:.4f} rad")

        
        print()
        print("\nNon-blocking square movement with sequence handling completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Closing environment...")
        env.reset(joint_angle=np.array([0.0, 0.01, -0.01, 0.0, 0.0, 0.0]), reset_for_end=True)
        env.close()
        print("Environment closed.")


if __name__ == "__main__":    
    # Run main example
    main()
