"""
Date: 2024-11-11 15:59:38
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-17 21:43:39
FilePath: /Minestudio/minestudio/simulator/callbacks/speed_test.py
"""

import time

from minestudio.simulator.callbacks.callback import MinecraftCallback


class SpeedTestCallback(MinecraftCallback):
    def __init__(self, interval: int = 100):
        super().__init__()
        self.interval = interval
        self.num_steps = 0
        self.total_times = 0

    def before_step(self, sim, action):
        self.start_time = time.time()
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        end_time = time.time()
        self.num_steps += 1
        self.total_times += end_time - self.start_time
        if self.num_steps % self.interval == 0:
            print(
                f"Speed Test Status: \n"
                f"Average Time: {self.total_times / self.num_steps:.2f} \n"
                f"Average FPS: {self.num_steps / self.total_times:.2f} \n"
                f"Total Steps: {self.num_steps} \n"
            )
        return obs, reward, terminated, truncated, info
