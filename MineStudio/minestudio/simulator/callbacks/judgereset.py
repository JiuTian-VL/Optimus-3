from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI, GUIConstants
from minestudio.simulator.utils.gui import PointDrawCall

import time
from typing import Dict, Literal, Optional, Callable
import cv2


class JudgeResetCallback(MinecraftCallback):
    def __init__(self, time_limit: int = 600):
        super().__init__()
        self.time_limit = time_limit
        self.time_step = 0

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.time_step += 1
        if terminated or self.time_step > self.time_limit - 1:
            print(f"Time limit reached, resetting the environment:", self.time_step)
            self.time_step = 0
            terminated = True
        return obs, reward, terminated, truncated, info
