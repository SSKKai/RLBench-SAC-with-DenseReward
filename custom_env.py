import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pyrep.const import RenderMode
from pyrep.errors import IKError
from rlbench.environment import Environment
from rlbench.task_environment import InvalidActionError
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from utils import task_switch, euler_to_quaternion
from pyquaternion import Quaternion
import math

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


class CustomEnv:
    def __init__(self, config):

        # obs_config = ObservationConfig(
        #     left_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
        #     right_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
        #     front_camera=CameraConfig(rgb=False, depth=False, mask=False),
        #     wrist_camera=CameraConfig(
        #         rgb=True, depth=False, mask=False, render_mode=RenderMode.OPENGL
        #     ),
        #     joint_positions=True,
        #     joint_velocities=True,
        #     joint_forces=False,
        #     gripper_pose=False,
        #     task_low_dim_state=False,
        # )

        self.obs_type = config['obs_type']

        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.joint_velocities = True
        obs_config.joint_positions = True
        obs_config.gripper_pose = True
        obs_config.wrist_camera.rgb = True
        obs_config.task_low_dim_state = True



        action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
        self.env = Environment(
            action_mode,
            obs_config=obs_config,
            static_positions=config["static_env"],
            headless=config["headless_env"],
        )
        self.env.launch()
        self.task_name = config["task"]
        self.task = self.env.get_task(task_switch[config["task"]])
        self.gripper_open = 0.9 #0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        self.first_flag = 0
        self.dm_init = 0
        self.z_init = 0
        self.reward_section = 1
        self.init_state = {}
        return

    def reset(self):
        self.gripper_open = 0.9 #0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        descriptions, obs = self.task.reset()
        task_obs, state_obs = self.obs_split(obs)
        self.first_flag = 0
        self.reward_section = 1
        return obs, task_obs, state_obs

    def step(self, action):
        action_delayed = self.postprocess_action(action)
        try:
            next_obs, reward, done = self.task.step(action_delayed)
        except (IKError and InvalidActionError):
            zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, action_delayed[-1]]
            next_obs, reward, done = self.task.step(zero_action)

        if self.task_name == "CloseMicrowave":
            if done:
                if self.task._task.get_low_dim_state()[49] > 0.1:
                    done = False

        task_obs, state_obs = self.obs_split(next_obs)
        reward = self.get_reward()
        return next_obs, task_obs, state_obs, reward, done

    def get_reward(self):
        if self.task_name == "ReachTarget":
            x, y, z, qx, qy, qz, qw = self.task._robot.arm.get_tip().get_pose()
            tar_x, tar_y, tar_z, _, _, _, _ = self.task._task.target.get_pose()
            reward = -(np.abs(x - tar_x) + np.abs(y - tar_y) + np.abs(z - tar_z))

        #############################################################
        #############################################################
        elif self.task_name == "PushButton":
            ############################################################################
            # Push Button Horizontally
            x, y, z, qx, qy, qz, qw = self.task._robot.arm.get_tip().get_pose()
            tar_x, tar_y, tar_z, _, _, _, _ = self.task._task.target_button.get_pose()
            tar_rot = Quaternion(0,0,1,0)
            arm_rot = Quaternion(qw, qx, qy, qz)*Quaternion(0.7071,0,-0.7071,0)
            trans_rot = arm_rot.inverse * tar_rot
            std_vec = [0, 0, 1]
            arm_vec = trans_rot.rotate(std_vec)
            rot_distance = np.arccos(np.clip(np.dot(arm_vec, std_vec), -1.0, 1.0))
            distance = np.sqrt((x - tar_x) ** 2 + (y - tar_y) ** 2 + (z - tar_z) ** 2)

            if self.first_flag == 0:

                self.init_state['distance'] = distance
                self.init_state['rot_distance'] = rot_distance
                self.first_flag = 1

            reward_distance = np.clip(-1 * distance/self.init_state['distance'], -2, 0)

            reward_rot = np.clip(-0.5 * rot_distance/(self.init_state['rot_distance']), -2, 0)

            reward = reward_distance + reward_rot
            #############################################################################################
            # Push Button Vertically
            # x, y, z, qx, qy, qz, qw = self.task._robot.arm.get_tip().get_pose()
            # tar_x, tar_y, tar_z, _, _, _, _ = self.task._task.target_button.get_pose()
            # tar_rot = Quaternion(0,0,1,0)
            # arm_rot = Quaternion(qw, qx, qy, qz)
            # trans_rot = arm_rot.inverse * tar_rot
            # std_vec = [0, 0, 1]
            # arm_vec = trans_rot.rotate(std_vec)
            # rot_distance = np.arccos(np.clip(np.dot(arm_vec, std_vec), -1.0, 1.0))
            #
            # if self.first_flag == 0:
            #
            #     self.init_state['z_tar_distance'] = z - tar_z
            #     self.init_state['xy_distance'] = np.sqrt((x - tar_x) ** 2 + (y - tar_y) ** 2)
            #     self.init_state['z_distance'] = z - 0.8430
            #     self.init_state['final_distance'] = 0.8430 - tar_z
            #     self.init_state['final_dis'] = np.sqrt((0.8430 - tar_z)**2)
            #     self.init_state['rot_distance'] = rot_distance
            #     self.first_flag = 1
            #
            # xy_distance = np.sqrt((x - tar_x) ** 2 + (y - tar_y) ** 2)
            # z_tar_distance = z - tar_z
            # reward_1 = np.clip(-0.5*xy_distance/self.init_state['xy_distance'], -1, 0) \
            #            + np.clip(-0.2*z_tar_distance/self.init_state['z_tar_distance'], -0.2, 0)
            #
            # # reward_1 = np.clip(-1 * xy_distance/self.init_state['xy_distance'], -1, 0)
            #
            # z_distance = z - 0.8430
            # reward_2 = np.clip(-0.5 * z_distance/self.init_state['z_distance'], -1, 0)
            #
            # final_distance = np.sqrt((x - tar_x) ** 2 + (y - tar_y) ** 2 + (z - tar_z) ** 2)
            # reward_3 = np.clip(-0.5 * final_distance/self.init_state['final_distance'], -1, 0)
            #
            # reward_rot = np.clip(-0.25 * rot_distance/(self.init_state['rot_distance']+0.2), -0.5, 0)
            #
            # if self.reward_section == 1:
            #     reward_2 = -1
            #     reward_3 = -1
            #     if xy_distance < 0.2:  # if abs(xy_distance/self.init_state['xy_distance']) < 0.1
            #         self.reward_section = 2
            #         self.init_state['z_distance'] = z - 0.8430
            # elif self.reward_section == 2:
            #     reward_3 = -1
            #     if abs(reward_2) < 0.1:
            #         self.reward_section = 3
            #
            # reward = reward_1 + reward_2 + reward_3 + reward_rot

        #####################################################
        #####################################################
        elif self.task_name == "CloseMicrowave":
            x, y, z, qx, qy, qz, qw  = self.task._robot.arm.get_tip().get_pose()
            door_x, door_y, door_z, door_qx, door_qy, door_qz, door_qw = self.task._task.get_low_dim_state()[50:57]
            wp_x, wp_y, wp_z = self.task._task.get_low_dim_state()[21:24]
            door_open = self.task._task.get_low_dim_state()[49]

            xy_distance = np.sqrt((x-wp_x)**2 + (y-wp_y)**2)
            z_distance = np.sqrt((z-wp_z)**2)

            rot_trans = Quaternion(0.0, -0.0, -0.7071067811865475, -0.7071067811865475)
            arm_rot = Quaternion(qw, qx, qy, qz)
            tar_rot = Quaternion(door_qw, door_qx, door_qy, door_qz)
            rot_distance = Quaternion.absolute_distance(arm_rot, tar_rot * rot_trans)

            dx, dy, dz = tar_rot.rotate([1, 0, 0])
            vx, vy, vz = self.task._robot.gripper.get_velocity()[0]
            angle = np.arccos(
                np.clip(np.dot([dx, dy], [vx / np.sqrt(vx ** 2 + vy ** 2), vy / np.sqrt(vx ** 2 + vy ** 2)]), -1.0, 1.0))
            velocity = np.sqrt(vx ** 2 + vy ** 2) * math.cos(angle)

            if self.first_flag == 0:

                self.init_state['door_open'] = door_open
                self.init_state['rot_distance'] = rot_distance
                self.init_state['xy_distance'] = xy_distance
                self.init_state['z_distance'] = z_distance
                self.first_flag = 1

            main_reward = np.clip(-0.5 * door_open/self.init_state['door_open'], -1, 0)

            rot_reward = np.clip(-0.5 * rot_distance/self.init_state['rot_distance'], -1, 0)
            xy_reward = np.clip(-0.5*xy_distance/self.init_state['xy_distance'], -1, 0)
            z_reward = np.clip(-0.25*z_distance/self.init_state['z_distance'], -0.5, 0)
            vel_reward = np.clip(velocity, -0.2, 0.2)

            if self.reward_section == 1:
                vel_reward = 0
                if xy_distance < 0.2 and z_distance < 0.2: # z_distance < 0.2
                    self.reward_section = 2
            elif self.reward_section == 2:
                xy_reward = 0

            reward = main_reward + rot_reward + xy_reward + 1.2*z_reward + vel_reward #

            # print(reward, main_reward, rot_reward, vel_reward, self.reward_section)

        else:
            reward = 0

        return reward

    def render(self):
        return

    def close(self):
        self.env.shutdown()
        return

    def postprocess_action(self, action):
        delta_position = action[:3] * 0.01
        delta_angle_quat = euler_to_quaternion(action[3:6] * 0.04)
        gripper_delayed = self.delay_gripper(action[-1])
        action_post = np.concatenate(
            (delta_position, delta_angle_quat, [gripper_delayed])
        )
        return action_post

    def delay_gripper(self, gripper_action):
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9
        self.gripper_deque.append(gripper_action)

        self.gripper_open = 0
        return self.gripper_open

    def obs_split(self, obs):
        if self.obs_type == "WristCameraRGB":
            task_obs = obs.wrist_rgb.transpose(
                (2, 0, 1)
            )  # Transpose it into torch order (CHW)

        elif self.obs_type == "LowDimension":
            if self.task_name == "ReachTarget":
                task_obs = obs.task_low_dim_state

            elif self.task_name == "PushButton":
                task_obs = obs.task_low_dim_state[0:3]

            elif self.task_name == "CloseMicrowave":
                # micro_pos = obs.task_low_dim_state[7:10]
                door_pos = obs.task_low_dim_state[50:57]
                door_micro = obs.task_low_dim_state[49]
                wp_pos = obs.task_low_dim_state[21:24]
                task_obs = np.append(door_micro, wp_pos)
                task_obs = np.append(task_obs, door_pos)

            else:
                task_obs = obs.task_low_dim_state

        state_obs = np.append(obs.joint_positions, obs.gripper_open)
        return task_obs, state_obs  # task_obs, state_obs

    def randn_action(self):
        action_size = self.env.action_size
        # arm = np.random.normal(0.0, 0.1, size=(action_size - 1,))
        arm = -1+2*np.random.random_sample(action_size - 1)
        gripper = [0.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)



