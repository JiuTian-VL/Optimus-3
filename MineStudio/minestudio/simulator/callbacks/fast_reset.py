"""
Date: 2024-11-11 16:15:32
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-11 19:57:42
FilePath: /MineStudio/minestudio/simulator/minerl/callbacks/fast_reset.py
"""

import random

import numpy as np

from minestudio.simulator.callbacks.callback import MinecraftCallback


class FastResetCallback(MinecraftCallback):
    def __init__(self, biomes, random_tp_range, start_time=0, start_weather="clear"):
        super().__init__()
        self.biomes = biomes
        self.random_tp_range = random_tp_range
        self.start_time = start_time
        self.start_weather = start_weather

    def before_reset(self, sim, reset_flag):
        if not sim.already_reset:
            return reset_flag
        biome = random.choice(self.biomes)
        x = np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2)
        z = np.random.randint(-self.random_tp_range // 2, self.random_tp_range // 2)
        fast_reset_commands = [
            "/kill",
            f"/time set {self.start_time}",
            f"/weather {self.start_weather}",
            "/kill @e[type=!player]",
            "/kill @e[type=item]",
            f"/teleportbiome @a {biome} {x} ~0 {z}",
        ]
        for command in fast_reset_commands:
            obs, _, done, info = sim.env.execute_cmd(command)
        return False
