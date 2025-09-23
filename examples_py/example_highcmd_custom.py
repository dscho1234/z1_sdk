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


arm.labelRun("forward")
# time.sleep(3)

gripper_pos = 0.0
jnt_speed = 1.0

current_joint_pos = arm.lowstate.getQ()
T_forward = arm._ctrlComp.armModel.forwardKinematics(current_joint_pos, 6)
print(f"current_joint_pos: {current_joint_pos}")
print(f"current T_forward: {T_forward}")
T_forward[2,3] += 0.1 # z-direction forward
print(f"target T_forward: {T_forward}")

posture = unitree_arm_interface.homoToPosture(T_forward)
arm.MoveJ(posture, gripper_pos, jnt_speed)
# time.sleep(3)

arm.backToStart()
arm.loopOff()

