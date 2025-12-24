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



def compute_target_pos_test():
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
    dataset_name = 'charuco_marker_bottle_under_table'

    env = EEPoseCtrlJointCmdWrapper(
                has_gripper=True,
                control_frequency=2.0,
                position_tolerance=0.01,
                orientation_tolerance=0.1,
                sequence_length=16,
                joint_speed=0.5,
                use_current_joint_pos_when_ik_fails=True,
                T_E_C = T_E_C,
                urdf_path = "/home/dcho302/Workspace/unitree_ros/robots/z1_description/xacro/z1.urdf",
                fk_debug=True,
            )


    if dataset_name == 'charuco_marker_static_bottle_v2':
        view_joint_angle = np.array([0.0, 0.01, -0.5, 0.5, 0.0, 0.0])
    elif dataset_name == 'charuco_marker_bottle_under_table':
        view_joint_angle = np.array([0.903, 1.613, -2.319, 1.660, -0.469, 0.748])
    elif dataset_name == 'charuco_marker_box_under_drawer':
        view_joint_angle = np.array([0.797, 1.917, -2.222, 1.670, -0.768, 1.393])
    else:
        raise NotImplementedError

    # view_initial_T_marker = T_mc_opt_from_data.copy()[0] # [4, 4], marker coordinate
    # view_initial_T = T_B_M_view @ view_initial_T_marker # [4, 4], base coordinate

    view_initial_T_be = env.compute_forward_kinematics(view_joint_angle)
    view_initial_T_bc = view_initial_T_be @ T_E_C

    # env.current_joint_pos = view_joint_angle - 0.1
    env.current_joint_pos = np.zeros(6)

    view_joint_angle = env.compute_target_pos(view_initial_T_bc[:3, 3], R.from_matrix(view_initial_T_bc[:3, :3]).as_quat(), ik_type='jacobian')
    view_joint_angle = env.compute_target_pos(view_initial_T_bc[:3, 3], R.from_matrix(view_initial_T_bc[:3, :3]).as_quat(), ik_type='null_space')
    print(view_joint_angle)


def T_mc_reachable_test():
    # charuco, view policy setting (from RHWE calibration, fk_debug fix) (After data collection)
    T_B_M_view = np.array([[ 2.05324101e-02,  6.98917300e-01,  7.14907705e-01,  9.20522136e-01],
                        [-2.23645068e-02, -7.14558449e-01,  6.99218172e-01,  3.67244651e-01],
                        [ 9.99539018e-01, -3.03451925e-02,  9.59333627e-04,  7.15193435e-01], # normal height : 5.51193435e-01
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # # for debug
    # T_B_M_view = np.array([[ 2.05324101e-02,  6.98917300e-01,  7.14907705e-01,  9.20522136e-01],
    #                     [-2.23645068e-02, -7.14558449e-01,  6.99218172e-01,  3.67244651e-01],
    #                     [ 9.99539018e-01, -3.03451925e-02,  9.59333627e-04,  8.15193435e-01], # normal height : 5.51193435e-01
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # charuco, view policy setting (from RHWE calibration, fk_debug fix) (After data collection)
    T_E_C_view = np.array([[ 0.00516172, -0.09810991,  0.9951622,   0.03981211],
                            [-0.99924287,  0.03787068,  0.00891643,  0.03112069],
                            [-0.03856226, -0.99445475, -0.09784015,  0.05397355],
                            [ 0.,          0.,          0.,          1.        ]])
    import zarr
    import matplotlib.pyplot as plt
    
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
    
    # Load data - using the same path as in example_gym_env_jointctrl_cmd_wait.py
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_bottle_under_table"
    data_buffer_path = "/home/dcho302/slow_storage/dscho/im2flow2act/data/realworld_human_demonstration_custom/object_first/charuco_marker_box_under_drawer"
    data_buffer = zarr.open(data_buffer_path, mode="r")
    
    # Find all episodes
    episode_keys = [key for key in data_buffer.keys() if key.startswith("episode_")]
    episode_indices = sorted([int(key.split("_")[1]) for key in episode_keys])
    
    # # NOTE: for debug
    # episode_indices = list(np.arange(100, 147))
    episode_indices = [1]
    
    print(f"Found {len(episode_indices)} episodes: {episode_indices}")
    
    # Store results: (episode_idx, timestep, is_reachable)
    results = []
    all_reachable = []
    all_unreachable = []
    
    # Process each episode
    for episode_idx in episode_indices:
        
        T_mc_transformation = data_buffer[f"episode_{episode_idx}/T_mc_opt_droid"][-backward_episode_len:][::downsample_ratio].copy() # [T, 4, 4]
        
        
        T = T_mc_transformation.shape[0]
        print(f"Processing episode {episode_idx} with {T} timesteps...")
        
        # Convert T_mc to T_bc using T_B_M
        # T_bc_transformation = T_B_M @ T_mc_transformation for each timestep
        T_bc_transformation = np.einsum('ij,hjk->hik', T_B_M_view, T_mc_transformation) # [T, 4, 4]
        
        
        if 'charuco_marker_bottle_under_table' in data_buffer_path:
            joint_angle = np.array([-0.044, 2.349, 0.610, 1.5, -0.348, 0.045])
        elif 'charuco_marker_box_under_drawer' in data_buffer_path:
            joint_angle = np.array([-0.686, 1.322, 0.982, -0.793, -0.205, -0.576])
        # Initialize joint position for this episode (start from zeros)
        env.current_joint_pos = joint_angle # np.zeros(6)
        prev_joint_angle = joint_angle # np.zeros(6)
        
        # Check IK for each timestep
        episode_reachable = []
        for t in range(T):
            T_bc = T_bc_transformation[t]  # [4, 4]
            
            # Extract position and orientation from T_bc
            target_position = T_bc[:3, 3]
            target_orientation = R.from_matrix(T_bc[:3, :3]).as_quat()  # [x, y, z, w]
            
            # Set current joint position to previous joint angle before IK check
            env.current_joint_pos = prev_joint_angle.copy()
            
            # Check if IK exists using compute_target_pos with return_success=True
            try:
                target_joint_pos, ik_success = env.compute_target_pos(
                    target_position,
                    target_orientation,
                    return_success=True,
                    ik_type='null_space'
                )
                
                # Update prev_joint_angle for next iteration
                prev_joint_angle = target_joint_pos.copy()
                
                # Store result
                results.append((episode_idx, t, ik_success))
                episode_reachable.append(ik_success)
                
                if ik_success:
                    all_reachable.append((episode_idx, t))
                else:
                    all_unreachable.append((episode_idx, t))
                    
            except Exception as e:
                print(f"Error at episode {episode_idx}, timestep {t}: {e}")
                results.append((episode_idx, t, False))
                episode_reachable.append(False)
                all_unreachable.append((episode_idx, t))
                # Keep previous joint angle even on error
        
        # Print episode statistics
        num_reachable = sum(episode_reachable)
        num_unreachable = len(episode_reachable) - num_reachable
        reachability_rate = num_reachable / len(episode_reachable) * 100 if len(episode_reachable) > 0 else 0
        print(f"Episode {episode_idx}: {num_reachable}/{len(episode_reachable)} reachable ({reachability_rate:.2f}%)")
    
    # Overall statistics
    total_poses = len(results)
    total_reachable = sum([r[2] for r in results])
    total_unreachable = total_poses - total_reachable
    overall_reachability_rate = (total_reachable / total_poses * 100) if total_poses > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total episodes processed: {len(episode_indices)}")
    print(f"Total poses checked: {total_poses}")
    print(f"Reachable poses: {total_reachable} ({overall_reachability_rate:.2f}%)")
    print(f"Unreachable poses: {total_unreachable} ({100 - overall_reachability_rate:.2f}%)")
    print("="*80)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Reachability over time for each episode
    ax1 = axes[0, 0]
    for episode_idx in episode_indices:
        episode_results = [r for r in results if r[0] == episode_idx]
        if len(episode_results) > 0:
            timesteps = [r[1] for r in episode_results]
            reachable = [1 if r[2] else 0 for r in episode_results]
            ax1.plot(timesteps, reachable, label=f'Episode {episode_idx}', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reachable (1) / Unreachable (0)')
    ax1.set_title('Reachability Over Time by Episode')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reachability rate per episode
    ax2 = axes[0, 1]
    episode_stats = {}
    for episode_idx in episode_indices:
        episode_results = [r for r in results if r[0] == episode_idx]
        if len(episode_results) > 0:
            num_reachable = sum([1 for r in episode_results if r[2]])
            reachability_rate = num_reachable / len(episode_results) * 100
            episode_stats[episode_idx] = reachability_rate
    
    episodes = sorted(episode_stats.keys())
    rates = [episode_stats[ep] for ep in episodes]
    ax2.bar(episodes, rates, alpha=0.7)
    ax2.set_xlabel('Episode Index')
    ax2.set_ylabel('Reachability Rate (%)')
    ax2.set_title('Reachability Rate per Episode')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative reachability over all timesteps
    ax3 = axes[1, 0]
    # Sort results by (episode, timestep)
    sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
    cumulative_reachable = []
    cumulative_total = []
    count = 0
    for _, _, is_reachable in sorted_results:
        count += 1
        cumulative_total.append(count)
        cumulative_reachable.append(sum([1 for r in sorted_results[:count] if r[2]]))
    
    ax3.plot(cumulative_total, [r/t*100 if t > 0 else 0 for r, t in zip(cumulative_reachable, cumulative_total)], 
             linewidth=2, label='Cumulative Reachability Rate')
    ax3.axhline(y=overall_reachability_rate, color='r', linestyle='--', 
                label=f'Overall Rate: {overall_reachability_rate:.2f}%')
    ax3.set_xlabel('Cumulative Timestep')
    ax3.set_ylabel('Cumulative Reachability Rate (%)')
    ax3.set_title('Cumulative Reachability Rate Over All Timesteps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of reachable vs unreachable
    ax4 = axes[1, 1]
    categories = ['Reachable', 'Unreachable']
    counts = [total_reachable, total_unreachable]
    colors = ['green', 'red']
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Overall Reachability Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total_poses*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'T_mc_reachability_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close()
    
    return results, {
        'total_episodes': len(episode_indices),
        'total_poses': total_poses,
        'total_reachable': total_reachable,
        'total_unreachable': total_unreachable,
        'overall_reachability_rate': overall_reachability_rate,
        'episode_stats': episode_stats
    }
    




if __name__ == '__main__':
    # compute_target_pos_test()
    T_mc_reachable_test()
    