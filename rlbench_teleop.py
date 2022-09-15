import time
import numpy as np
from human_feedback import correct_action
from utils import KeyboardObserver, loop_sleep
from custom_env import CustomEnv
from pyquaternion import Quaternion
import argparse

config = {
    'task': "CloseMicrowave",  #
    'static_env': False,  #
    'headless_env': False,  #
    'episodes': 3,  #
    'sequence_len': 150,  #
    'obs_type': "LowDimension"  # LowDimension WristCameraRGB
}

state_list = []

env = CustomEnv(config)
keyboard_obs = KeyboardObserver()
env.reset()
gripper_open = 0.9
time.sleep(5)
print("Go!")
episodes_count = 0
first_flag = 0
while episodes_count < config['episodes']:
    start_time = time.time()
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])

    if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
        action = correct_action(keyboard_obs, action)
        gripper_open = action[-1]
    next_obs, _, _, reward, done = env.step(action)  # Step

    # for testing and modelling reward function
    ee_pose = np.array([getattr(next_obs, 'gripper_pose')[:3]])
    target_pose = np.array([getattr(next_obs, 'task_low_dim_state')])
    touch_force = env.task._task.robot.gripper.get_touch_sensor_forces()

    distance = np.sqrt((target_pose[0, 0] - ee_pose[0, 0]) ** 2 + (target_pose[0, 1] - ee_pose[0, 1]) ** 2 + (
                target_pose[0, 2] - ee_pose[0, 2]) ** 2)
    x, y, z, qx, qy, qz, qw = env.task._robot.arm.get_tip().get_pose() # end effector
    wp_x, wp_y, wp_z = env.task._task.get_low_dim_state()[21:24] # waypoint
    arm_rot = Quaternion(qw, qx, qy, qz)

    print(reward)



    if keyboard_obs.reset_button:
        env.reset()
        gripper_open = 0.9
        keyboard_obs.reset()
    # elif done:
    #     env.reset()
    #     gripper_open = 0.9
    #     episodes_count += 1
    #     first_flag = 0
    #     keyboard_obs.reset()
    #     done = False
    else:
        loop_sleep(start_time)

