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
