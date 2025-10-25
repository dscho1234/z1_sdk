import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import unitree_arm_interface
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

print("Press ctrl+\ to quit process.")
dt = 0.2 # 너무 짧으면 작동이 잘 안됨. 감가속 하느라 시간안에 목표한 곳에 못감
arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState
arm.loopOn()


arm.labelRun("forward")
# time.sleep(3)
wait = False # False: non-blocking, True: blocking
arm.setWait(wait)


gripper_pos = 0.0
jnt_speed = 1.0

current_joint_pos = arm.lowstate.getQ()
T_forward = arm._ctrlComp.armModel.forwardKinematics(current_joint_pos, 6)
print(f"current_joint_pos: {current_joint_pos}")
print(f"current T_forward: {T_forward}")
posture = unitree_arm_interface.homoToPosture(T_forward)

# warm up (maintaining current pose)
for _ in range(5):
    arm.MoveJ(posture, gripper_pos, jnt_speed)
    time.sleep(dt)




start = time.time()
for _ in range(10):
    T_forward[2,3] += 0.02 # z-direction forward
    print(f"target T_forward: {T_forward}")
    posture = unitree_arm_interface.homoToPosture(T_forward)
    target_pos = T_forward[:3, 3].copy()
    arm.MoveJ(posture, gripper_pos, jnt_speed)
    time.sleep(dt)
    current_joint_pos = arm.lowstate.getQ()
    T_current = arm._ctrlComp.armModel.forwardKinematics(current_joint_pos, 6)
    current_pos = T_current[:3, 3]
    print('error : ', np.linalg.norm(current_pos - target_pos))

end = time.time()
print(f"MoveJ with setWait({wait}) time taken: {end - start}")
current_joint_pos = arm.lowstate.getQ()
T_forward = arm._ctrlComp.armModel.forwardKinematics(current_joint_pos, 6)
print(f"final T_forward: {T_forward}")
# start = time.time()
# end = time.time()
# print(f"time.sleep({dt}) time taken: {end - start}")

arm.backToStart()
arm.loopOff()

