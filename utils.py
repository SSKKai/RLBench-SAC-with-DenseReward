import math
import random
import numpy as np
import time
import torch
from pynput import keyboard
from functools import partial
from rlbench.tasks import (
    CloseMicrowave,
    PushButton,
    TakeLidOffSaucepan,
    UnplugCharger,
    ReachTarget,
    PickAndLift
)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


task_switch = {
    "CloseMicrowave": CloseMicrowave,
    "PushButton": PushButton,
    "TakeLidOffSaucepan": TakeLidOffSaucepan,
    "UnplugCharger": UnplugCharger,
    "ReachTarget": ReachTarget,
    "PickAndLift": PickAndLift
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class KeyboardObserver:
    def __init__(self):
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": self.reset_episode,
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, 1),  # left
            "d": (1, -1),  # right
            "s": (0, 1),  # backward
            "w": (0, -1),  # forward
            "q": (2, 1),  # down
            "e": (2, -1),  # up
            "j": (3, -1),  # look left
            "l": (3, 1),  # look right
            "i": (4, -1),  # look up
            "k": (4, 1),  # look down
            "u": (5, -1),  # rotate left
            "o": (5, 1),  # rotate right
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        print("gripper set to: ", value)
        return

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            pass
        return

    def reset_direction(self, key):
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            pass
        return

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):
        return self.get_gripper() is not None

    def get_ee_action(self):
        return self.direction * 0.9

    def reset_episode(self):
        self.reset_button = True
        return

    def reset(self):
        self.set_label(1)
        self.set_gripper(None)
        self.reset_button = False
        return


def downsample_traj(traj, target_len):
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        return traj + [traj[-1]] * (target_len - len(traj))
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return np.array([traj[i] for i in indeces])


def loop_sleep(start_time):
    dt = 0.05
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return


def euler_to_quaternion(euler_angle):
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
