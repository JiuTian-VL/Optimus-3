"""
Date: 2024-11-12 11:53:55
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-29 14:32:28
FilePath: /MineStudio/minestudio/tutorials/simulator/test_sim.py
"""

import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback,
    RecordCallback,
    SummonMobsCallback,
    MaskActionsCallback,
    RewardsCallback,
    CommandsCallback,
    TaskCallback,
    FastResetCallback,
)

if __name__ == "__main__":
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            SpeedTestCallback(50),
            SummonMobsCallback([{"name": "cow", "number": 10, "range_x": [-5, 5], "range_z": [-5, 5]}]),
            MaskActionsCallback(inventory=0, camera=np.array([0.0, 0.0])),
            RecordCallback(record_path="./output", fps=30),
            RewardsCallback(
                [
                    {
                        "event": "kill_entity",
                        "objects": ["cow", "sheep"],
                        "reward": 1.0,
                        "identity": "kill sheep or cow",
                        "max_reward_times": 5,
                    }
                ]
            ),
            CommandsCallback(
                commands=[
                    "/give @p minecraft:iron_sword 1",
                    "/give @p minecraft:diamond 64",
                ]
            ),
            FastResetCallback(
                biomes=["mountains"],
                random_tp_range=1000,
            ),
            TaskCallback(
                [
                    {"name": "chop", "text": "mine the oak logs"},
                    {"name": "diamond", "text": "mine the diamond ore"},
                ]
            ),
        ],
    )
    obs, info = sim.reset()
    import ipdb

    ipdb.set_trace()
    print(sim.action_space)
    for i in range(100):
        action = sim.action_space.sample()
        # action = sim.noop_action()
        obs, reward, terminated, truncated, info = sim.step(action)
        # if (i+1) % 150 == 0:
        #     obs, info = sim.reset()
    sim.close()
