import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

print("Press ctrl+\ to quit process.")

arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState
arm.loopOn()

# # 1. highcmd_basic : armCtrlInJointCtrl
arm.labelRun("forward")
arm.startTrack(armState.JOINTCTRL)
jnt_speed = 1.0
for i in range(0, 1000):
    # dp = directions * speed; include 7 joints
    arm.jointCtrlCmd([0,0,0,-1,0,0,-1], jnt_speed)
    time.sleep(arm._ctrlComp.dt)

# # 2. highcmd_basic : armCtrlByFSM
# arm.labelRun("forward")
# gripper_pos = 0.0
# jnt_speed = 1.0
# arm.MoveJ([0.5,0.1,0.1,0.5,-0.2,0.5], gripper_pos, jnt_speed)
# # arm.MoveJ([0.1,0.1,0.1,0.1,0.1,0.1], gripper_pos, jnt_speed)


# gripper_pos = -1.0
# cartesian_speed = 0.5
# arm.MoveL([0,0,0,0.45,-0.2,0.2], gripper_pos, cartesian_speed)
# gripper_pos = 0.0
# arm.MoveC([0,0,0,0.45,0,0.4], [0,0,0,0.45,0.2,0.2], gripper_pos, cartesian_speed)





# # 3. highcmd_basic : armCtrlInCartesian
# arm.labelRun("forward")
# arm.startTrack(armState.CARTESIAN)
# angular_vel = 0.05
# linear_vel = 0.05


# custom_dt = 0.2 # 5 Hz
# dt_ratio = int(custom_dt / arm._ctrlComp.dt)
# q_current = arm.lowstate.getQ()
# T_current = arm._ctrlComp.armModel.forwardKinematics(q_current, 6)
# print('current position: ', T_current[:3, 3])

# original_iter = 1000 # 500 hz *2 seconds

# modified_iter = int(original_iter / dt_ratio)

# start = time.time()
# for k in range(modified_iter): # 2 seconds
#     if k < int(modified_iter / 2):
#         direction = [0,0,0,0,0,-1,-1]
#     else:
#         direction = [0,0,0,0,0,1,-1]
#     # direction = [0,0,0,0,0,-1,-1] # original code
#     for i in range(dt_ratio): # 0.2 seconds for 1 step 
#         arm.cartesianCtrlCmd(direction, angular_vel, linear_vel)
#         time.sleep(arm._ctrlComp.dt)

#     # time.sleep(custom_dt)
# print(f'dt: {arm._ctrlComp.dt}')
# time_taken = time.time() - start
# print(f"Time taken: {time_taken} seconds")
# q_after = arm.lowstate.getQ()
# T_after = arm._ctrlComp.armModel.forwardKinematics(q_after, 6)
# print(f'after position: ', T_after[:3, 3])
# print(f'velocity computed by cartesianCtrlCmd: {(T_after[:3, 3] - T_current[:3, 3]) / time_taken}')


arm.backToStart()
arm.loopOff()

