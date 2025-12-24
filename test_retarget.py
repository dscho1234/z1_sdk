import numpy as np
import sys
import os
import zarr
import time
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
import plotly.graph_objects as go

# Add the envs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from envs.z1_env_jointctrl_wait import EEPoseCtrlJointCmdWrapper


def compute_se3_errors(T_target: np.ndarray, T_retargeted: np.ndarray):
    """
    Compute position and orientation errors between target and retargeted SE(3) trajectories.
    
    Args:
        T_target: Target SE(3) trajectory [N, 4, 4]
        T_retargeted: Retargeted SE(3) trajectory [N, 4, 4]
    
    Returns:
        position_errors: Position errors in meters [N]
        orientation_errors: Orientation errors in radians [N]
    """
    N = T_target.shape[0]
    position_errors = np.zeros(N, dtype=np.float64)
    orientation_errors = np.zeros(N, dtype=np.float64)
    
    for t in range(N):
        # Position error: ||p_target - p_retargeted||
        p_target = T_target[t, :3, 3]
        p_retargeted = T_retargeted[t, :3, 3]
        position_errors[t] = np.linalg.norm(p_target - p_retargeted)
        
        # Orientation error: ||log(R_target^T @ R_retargeted)||
        R_target = T_target[t, :3, :3]
        R_retargeted = T_retargeted[t, :3, :3]
        R_error = R_target.T @ R_retargeted
        # Convert to rotation vector and compute norm
        rot_error = R.from_matrix(R_error)
        rotvec = rot_error.as_rotvec()
        orientation_errors[t] = np.linalg.norm(rotvec)
    
    return position_errors, orientation_errors


def visualize_retargeting(T_target: np.ndarray, T_retargeted: np.ndarray, episode_idx: int, 
                          arrow_length: float = 0.02, sample_step: int = 1):
    """
    Visualize SE(3) coordinate frames before and after retargeting.
    
    Args:
        T_target: Target SE(3) trajectory [N, 4, 4] - RGB colors (Red, Green, Blue for x, y, z)
        T_retargeted: Retargeted SE(3) trajectory [N, 4, 4] - Magenta, Cyan, Yellow for x, y, z
        episode_idx: Episode index for filename
        arrow_length: Length of coordinate frame arrows in meters
        sample_step: Step size for sampling frames (1 = all frames, 2 = every other frame, etc.)
    """
    N = T_target.shape[0]
    
    # Sample frames to avoid overcrowding
    sample_indices = list(range(0, N, sample_step))
    if sample_indices[-1] != N - 1:
        sample_indices.append(N - 1)  # Always include last frame
    
    fig = go.Figure()
    
    # Color definitions
    # Target (before retargeting): RGB
    target_colors = {
        'x': 'rgb(255, 0, 0)',      # Red
        'y': 'rgb(0, 255, 0)',      # Green
        'z': 'rgb(0, 0, 255)',      # Blue
    }
    
    # Retargeted (after retargeting): Magenta, Cyan, Yellow
    retargeted_colors = {
        'x': 'rgb(255, 0, 255)',    # Magenta
        'y': 'rgb(0, 255, 255)',    # Cyan
        'z': 'rgb(255, 255, 0)',    # Yellow
    }
    
    # Plot target trajectory (before retargeting) with time-based transparency
    for t in sample_indices:
        T = T_target[t]
        origin = T[:3, 3]
        x_axis = T[:3, 0] * arrow_length
        y_axis = T[:3, 1] * arrow_length
        z_axis = T[:3, 2] * arrow_length
        
        # Calculate transparency based on time
        alpha = 0.3 + 0.7 * (t / max(1, N - 1))
        
        # X-axis (Red)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + x_axis[0]],
            y=[origin[1], origin[1] + x_axis[1]],
            z=[origin[2], origin[2] + x_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(255, 0, 0, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(255, 0, 0, {alpha})'),
            name=f'Target X-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='target',
        ))
        
        # Y-axis (Green)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + y_axis[0]],
            y=[origin[1], origin[1] + y_axis[1]],
            z=[origin[2], origin[2] + y_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(0, 255, 0, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(0, 255, 0, {alpha})'),
            name=f'Target Y-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='target',
        ))
        
        # Z-axis (Blue)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + z_axis[0]],
            y=[origin[1], origin[1] + z_axis[1]],
            z=[origin[2], origin[2] + z_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(0, 0, 255, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(0, 0, 255, {alpha})'),
            name=f'Target Z-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='target',
        ))
    
    # Plot retargeted trajectory (after retargeting) with time-based transparency
    for t in sample_indices:
        T = T_retargeted[t]
        origin = T[:3, 3]
        x_axis = T[:3, 0] * arrow_length
        y_axis = T[:3, 1] * arrow_length
        z_axis = T[:3, 2] * arrow_length
        
        # Calculate transparency based on time
        alpha = 0.3 + 0.7 * (t / max(1, N - 1))
        
        # X-axis (Magenta)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + x_axis[0]],
            y=[origin[1], origin[1] + x_axis[1]],
            z=[origin[2], origin[2] + x_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(255, 0, 255, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(255, 0, 255, {alpha})'),
            name=f'Retargeted X-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='retargeted',
        ))
        
        # Y-axis (Cyan)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + y_axis[0]],
            y=[origin[1], origin[1] + y_axis[1]],
            z=[origin[2], origin[2] + y_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(0, 255, 255, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(0, 255, 255, {alpha})'),
            name=f'Retargeted Y-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='retargeted',
        ))
        
        # Z-axis (Yellow)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + z_axis[0]],
            y=[origin[1], origin[1] + z_axis[1]],
            z=[origin[2], origin[2] + z_axis[2]],
            mode='lines+markers',
            line=dict(color=f'rgba(255, 255, 0, {alpha})', width=4),
            marker=dict(size=3, color=f'rgba(255, 255, 0, {alpha})'),
            name=f'Retargeted Z-axis (t={t})' if t == sample_indices[0] else '',
            showlegend=(t == sample_indices[0]),
            legendgroup='retargeted',
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Retargeting Visualization (Episode {episode_idx})<br>'
              f'<span style="color:red">Red</span>, <span style="color:green">Green</span>, '
              f'<span style="color:blue">Blue</span> = Target (Before)<br>'
              f'<span style="color:magenta">Magenta</span>, <span style="color:cyan">Cyan</span>, '
              f'<span style="color:yellow">Yellow</span> = Retargeted (After)<br>'
              f'Transparency increases with time',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        width=1200,
        height=800,
    )
    
    # Save to HTML
    output_filename = f'retargeting_visualization_episode_{episode_idx}.html'
    fig.write_html(output_filename)
    print(f"Visualization saved to: {output_filename}")



import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def _se3_residual(T_fk: np.ndarray, T_tgt: np.ndarray) -> np.ndarray:
    """
    SE(3) residual as 6D vector:
    - translation error: p_fk - p_tgt   (in meters, assuming)
    - rotation error:    log(R_tgt^T R_fk) as rotvec (in radians)
    """
    assert T_fk.shape == (4, 4)
    assert T_tgt.shape == (4, 4)

    p_fk = T_fk[:3, 3]
    p_tgt = T_tgt[:3, 3]

    R_fk = T_fk[:3, :3]
    R_tgt = T_tgt[:3, :3]

    # rotation error: identity when fk == tgt
    R_err = R_tgt.T @ R_fk
    rotvec = R.from_matrix(R_err).as_rotvec()  # 3-vector (axis * angle)

    return np.concatenate([p_fk - p_tgt, rotvec], axis=0)


def retarget_se3_trajectory(
    env,
    target_T: np.ndarray,          # [T,4,4]
    q0: np.ndarray,                # [DoF]
    *,
    w_pos: float = 1.0,            # weight for position residual
    w_rot: float = 1.0,            # weight for rotation residual
    lambda_smooth: float = 1e-2,   # smoothness weight (q_t - q_{t-1})
    bounds=None,                   # (lower, upper) each can be scalar or [DoF]
    max_nfev: int = 50,
    tol: float = 1e-6,
    verbose: int = 0,
):
    """
    Retarget SE(3) target trajectory into joint trajectory via per-timestep IK:
        q_t = argmin_q  || e_se3(fk(q), T_target[t]) ||^2
                        + lambda_smooth * || q - q_{t-1} ||^2

    Returns:
        q_traj: [T, DoF]
        info:   dict with per-step cost and solver status
    """
    target_T = np.asarray(target_T)
    q0 = np.asarray(q0).astype(float)

    assert target_T.ndim == 3 and target_T.shape[1:] == (4, 4), "target_T must be [T,4,4]"
    T = target_T.shape[0]
    dof = q0.shape[0]

    if bounds is None:
        lb = -np.inf * np.ones(dof)
        ub = +np.inf * np.ones(dof)
    else:
        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (dof,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (dof,)).copy()

    # sqrt weights for least_squares residual scaling
    s_pos = np.sqrt(w_pos)
    s_rot = np.sqrt(w_rot)
    s_sm  = np.sqrt(lambda_smooth)

    q_prev = q0.copy()
    q_traj = np.zeros((T, dof), dtype=float)

    costs = []
    statuses = []
    nfev_list = []  # Track number of function evaluations per timestep

    for t in range(T):
        T_tgt = target_T[t]

        def residual(q):
            T_fk = env.compute_forward_kinematics(q)  # must return (4,4)
            e6 = _se3_residual(T_fk, T_tgt)   # [6] = [pos(3), rot(3)]
            # scale residuals
            e6_scaled = np.concatenate([s_pos * e6[:3], s_rot * e6[3:]], axis=0)
            smooth = s_sm * (q - q_prev)      # [DoF]
            return np.concatenate([e6_scaled, smooth], axis=0)

        # First timestep may need more iterations if starting from poor initial guess
        # Use more iterations for first timestep, fewer for subsequent (warm start)
        current_max_nfev = max_nfev * 2 if t == 0 else max_nfev

        res = least_squares(
            residual,
            x0=q_prev,             # warm start
            bounds=(lb, ub),
            max_nfev=current_max_nfev,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            verbose=verbose,
        )

        q_prev = res.x
        q_traj[t] = q_prev

        # (optional) store diagnostics
        costs.append(res.cost)      # 0.5 * sum(residual^2)
        statuses.append({"success": bool(res.success), "status": int(res.status), "message": res.message})
        nfev_list.append(res.nfev)  # Number of function evaluations for this timestep

    info = {
        "costs": np.asarray(costs, dtype=float),
        "statuses": statuses,
        "weights": {"w_pos": w_pos, "w_rot": w_rot, "lambda_smooth": lambda_smooth},
        "nfev_list": np.asarray(nfev_list, dtype=int),  # Function evaluations per timestep
        "total_nfev": sum(nfev_list),  # Total function evaluations
    }
    return q_traj, info

def se3_inv(T: np.ndarray) -> np.ndarray:
    """Fast inverse of SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ri = R.T
    Ti[:3, :3] = Ri
    Ti[:3, 3] = -Ri @ t
    return Ti

def retargeting_test_from_gpt():



    # charuco, view policy setting (from RHWE calibration, fk_debug fix) (After move to the edge)
    T_B_M_view = np.array([[-0.01867103, -0.00241674,  0.99982276,  0.74527762],
                            [-0.02852406, -0.99958876, -0.00294884, -0.63116821], #  original value -0.63116821
                            [ 0.99941872, -0.02857406,  0.01859442,  0.55722613], # normal height : 0.55722613
                            [ 0.,          0.,          0.,          1.        ]])

    # charuco, view policy setting (from RHWE calibration, fk_debug fix) (After move to the edge)
    T_E_C_view = np.array([[ 0.00523694, -0.09772001,  0.99520017,  0.02812511],
                            [-0.998772,    0.04851893,  0.01001987,  0.03098309],
                            [-0.04926519, -0.99403054, -0.09734592,  0.05319822],   
                            [ 0.,          0.,          0.,          1.        ]])
                            
    downsample_ratio = 2
    backward_episode_len = 350
    
    

    # Create environment
    env = EEPoseCtrlJointCmdWrapper(
            has_gripper=True,
            control_frequency=2.0,
            position_tolerance=0.01,
            orientation_tolerance=0.1,
            sequence_length=16,
            joint_speed=0.5,
            use_current_joint_pos_when_ik_fails=True,
            T_E_C = T_E_C_view,
            urdf_path = "/home/dcho302/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf",
            fk_debug=True,
        )
    
    # Get joint limits from environment
    joint_limits = env.joint_limits
    q_min = np.array([limit[0] for limit in joint_limits])
    q_max = np.array([limit[1] for limit in joint_limits])
    
    
    # Get T_ee_cam from environment (T_E_C)
    T_C_E = se3_inv(env.T_E_C.copy())
    
    # Load data
    # dataset_name = 'charuco_marker_bottle_under_table'
    dataset_name = 'charuco_marker_bottle_under_table_invisible'
    data_buffer_path = f"/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/{dataset_name}"
    data_buffer = zarr.open(data_buffer_path, mode="r")
    
    # Find all episodes
    episode_keys = [key for key in data_buffer.keys() if key.startswith("episode_")]
    episode_indices = sorted([int(key.split("_")[1]) for key in episode_keys])
    
    # for debug
    episode_indices = [0]
    # Configuration for evaluation
    start_timestep = 0  # Start timestep (0-indexed, after downsampling)
    N = 24  # Number of steps to evaluate
    
    print(f"Found {len(episode_indices)} episodes: {episode_indices}")
    
    
    # Process each episode
    for episode_idx in episode_indices:
        print(f"\n{'='*80}")
        print(f"Retargeting optimization for episode {episode_idx}")
        print(f"{'='*80}")
        
        # Get target trajectory: T_mc_transformation (SE(3) trajectory)
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt_droid"][-backward_episode_len:][::downsample_ratio].copy() # [T, 4, 4]
        
        T = T_mc_transformation.shape[0]
        print(f"Episode {episode_idx} has {T} timesteps (after downsampling)")
        
        # Convert T_mc to T_bc using T_B_M (target camera trajectory in base frame)
        # T_bc_transformation = T_B_M @ T_mc_transformation for each timestep
        T_bc_target = np.einsum('ij,hjk->hik', T_B_M_view, T_mc_transformation) # [T, 4, 4]
        
        # Check if start_timestep and N are valid
        if start_timestep < 0 or start_timestep >= T:
            print(f"Warning: start_timestep {start_timestep} is out of range [0, {T-1}], using 0")
            actual_start_timestep = 0
        else:
            actual_start_timestep = start_timestep
        
        end_timestep = min(actual_start_timestep + N, T)
        actual_N = end_timestep - actual_start_timestep
        
        if actual_N < N:
            print(f"Warning: Requested {N} steps but only {actual_N} steps available from timestep {actual_start_timestep}")
        
        # Extract target camera trajectory from start_timestep to start_timestep + N
        T_B_C_target = T_bc_target[actual_start_timestep:end_timestep].copy()  # [actual_N, 4, 4]
        print(f"Using timesteps {actual_start_timestep} to {end_timestep-1} ({actual_N} steps) for retargeting optimization")
        
        # Use actual_N for the rest of the code
        N = actual_N
            
        # Convert target camera trajectory to end-effector trajectory
        T_B_E_target = np.einsum('hij,jk->hik', T_B_C_target, T_C_E)  # [N, 4, 4]

        # Try to solve IK for first timestep to get better initial guess
        print(f"\nSolving IK for first timestep to get better initial guess...")
        env.current_joint_pos = np.zeros(6)
        T_B_C_first = T_B_C_target[0]
        target_position = T_B_C_first[:3, 3]
        target_orientation = R.from_matrix(T_B_C_first[:3, :3]).as_quat()  # [x, y, z, w]
        

        if 'charuco_marker_static_bottle_v2' in dataset_name:
            # after move to the edge
            pass
        elif 'charuco_marker_bottle_under_table' in dataset_name:
            # after move to the edge
            view_joint_angle = np.array([-0.802, 1.214, -1.531, 0.976, -0.096, -0.021]) # side
            # view_joint_angle = np.array([-1.552, 1.772, -2.033, 1.220, 0.685, -0.808]) # front (fk fail)

        elif 'charuco_marker_box_under_drawer' in dataset_name:
            # after move to the edge
            pass
        else:
            raise NotImplementedError

        env.current_joint_pos = view_joint_angle.copy()
        
        q0, ik_success = env.compute_target_pos(
            target_position,
            target_orientation,
            return_success=True,
            ik_type='null_space'
        )
        if ik_success:
            print(f"  IK solved successfully for first timestep")
        else:
            print(f"  Warning: IK did not converge")
        
        

        # Check IK solvability for each timestep in target trajectory
        print(f"\nChecking IK solvability for target SE(3) trajectory...")
        ik_solvable_list = []
        env.current_joint_pos = view_joint_angle.copy()
        
        for t in range(N):
            # Use T_B_C (base to camera) instead of T_B_E for compute_target_pos
            T_B_C_t = T_B_C_target[t]
            target_position = T_B_C_t[:3, 3]
            target_orientation = R.from_matrix(T_B_C_t[:3, :3]).as_quat()  # [x, y, z, w]
            
            
            _, ik_success = env.compute_target_pos(
                target_position,
                target_orientation,
                return_success=True,
                ik_type='jacobian' # 'null_space'
            )
            ik_solvable_list.append(ik_success)
            
        
        ik_solvable_count = sum(ik_solvable_list)
        print(f"  IK solvable: {ik_solvable_count}/{N} timesteps")
        
        # Measure optimization time
        print(f"\nStarting retargeting optimization with q0: {q0} ")
        opt_start_time = time.time()
        
        q_traj, info = retarget_se3_trajectory(
            env,
            T_B_E_target,
            q0,
            w_pos=1.0,
            w_rot=0.5,
            lambda_smooth=1e-3,
            bounds=(q_min, q_max),
            max_nfev=100,
        )
        
        opt_end_time = time.time()
        opt_elapsed_time = opt_end_time - opt_start_time

        print(f"\n{'='*80}")
        print(f"Retargeting optimization results:")
        print(f"{'='*80}")
        print(f"q_traj shape: {q_traj.shape}")
        print(f"Final cost: {info['costs'][-1]:.6f}")
        print(f"Total function evaluations: {info['total_nfev']}")
        print(f"Average function evaluations per timestep: {np.mean(info['nfev_list']):.2f}")
        print(f"Optimization time: {opt_elapsed_time:.3f} seconds ({opt_elapsed_time/60:.2f} minutes)")
        print(f"Number of timesteps optimized: {len(q_traj)}")
        
        # Compute optimized end-effector trajectory using FK
        print(f"\nComputing optimized end-effector trajectory using FK...")
        T_be_opt = np.empty((N, 4, 4), dtype=np.float64)
        for t in range(N):
            T_be_opt[t] = env.compute_forward_kinematics(q_traj[t])
        
        # Check IK solvability for retargeted SE(3) trajectory
        print(f"\nChecking IK solvability for retargeted SE(3) trajectory...")
        ik_solvable_retargeted_list = []
        
        # Get T_E_C from environment (end-effector to camera transform)
        T_E_C = env.T_E_C.copy()
        
        for t in range(N):
            # Convert T_B_E to T_B_C: T_B_C = T_B_E @ T_E_C
            T_B_E_retargeted = T_be_opt[t]
            T_B_C_retargeted = T_B_E_retargeted @ T_E_C
            
            target_position = T_B_C_retargeted[:3, 3]
            target_orientation = R.from_matrix(T_B_C_retargeted[:3, :3]).as_quat()  # [x, y, z, w]
            
            # Use the known q_traj[t] as initial guess (this is the joint configuration that produces T_be_opt[t])
            env.current_joint_pos = q_traj[t].copy()
            
            
            _, ik_success = env.compute_target_pos(
                target_position,
                target_orientation,
                return_success=True,
                ik_type='null_space'
            )
            ik_solvable_retargeted_list.append(ik_success)
            
        ik_solvable_retargeted_count = sum(ik_solvable_retargeted_list)
        print(f"  IK solvable: {ik_solvable_retargeted_count}/{N} timesteps")
        
        # Compute and print position and orientation errors
        print(f"\n{'='*80}")
        print(f"Retargeting Errors (Position and Orientation)")
        print(f"{'='*80}")
        position_errors, orientation_errors = compute_se3_errors(T_B_E_target, T_be_opt)
        
        print(f"\n{'Timestep':<10} {'Target IK':<15} {'Retarget IK':<15} {'Position Error (m)':<20} {'Orientation Error (rad)':<25} {'Orientation Error (deg)':<25}")
        print(f"{'-'*120}")
        for t in range(N):
            target_ik_status = "Yes" if ik_solvable_list[t] else "No"
            retarget_ik_status = "Yes" if ik_solvable_retargeted_list[t] else "No"
            print(f"{t:<10} {target_ik_status:<15} {retarget_ik_status:<15} {position_errors[t]:<20.6f} {orientation_errors[t]:<25.6f} {np.rad2deg(orientation_errors[t]):<25.6f}")
        
        print(f"\n{'Summary Statistics':<80}")
        print(f"{'-'*80}")
        print(f"Position Error - Mean: {np.mean(position_errors):.6f} m, Std: {np.std(position_errors):.6f} m")
        print(f"Position Error - Min: {np.min(position_errors):.6f} m, Max: {np.max(position_errors):.6f} m")
        print(f"Orientation Error - Mean: {np.mean(orientation_errors):.6f} rad ({np.rad2deg(np.mean(orientation_errors)):.6f} deg)")
        print(f"Orientation Error - Std: {np.std(orientation_errors):.6f} rad ({np.rad2deg(np.std(orientation_errors)):.6f} deg)")
        print(f"Orientation Error - Min: {np.min(orientation_errors):.6f} rad ({np.rad2deg(np.min(orientation_errors)):.6f} deg)")
        print(f"Orientation Error - Max: {np.max(orientation_errors):.6f} rad ({np.rad2deg(np.max(orientation_errors)):.6f} deg)")
        print(f"{'='*80}\n")
        
        # Visualize retargeting results (before and after)
        print(f"\nVisualizing retargeting results...")
        visualize_retargeting(T_B_E_target, T_be_opt, episode_idx)
        
        


if __name__ == "__main__":
    retargeting_test_from_gpt()