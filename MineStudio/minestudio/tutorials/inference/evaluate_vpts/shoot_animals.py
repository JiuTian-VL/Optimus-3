"""
Date: 2024-12-13 14:31:12
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-04 13:54:09
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_vpts/shoot_animals.py
"""

from functools import partial

import ray
from minestudio.inference import EpisodePipeline, InfoBaseFilter, MineGenerator
from minestudio.models import VPTPolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, FastResetCallback, SpeedTestCallback, SummonMobsCallback
from rich import print


if __name__ == "__main__":
    ray.init()
    env_generator = partial(
        MinecraftSim,
        obs_size=(128, 128),
        preferred_spawn_biome="plains",
        callbacks=[
            SpeedTestCallback(50),
            SummonMobsCallback([{"name": "cow", "number": 10, "range_x": [-5, 5], "range_z": [-5, 5]}]),
            FastResetCallback(
                biomes=["plains"],
                random_tp_range=1000,
            ),
            CommandsCallback(
                commands=[
                    "/give @p minecraft:bow 1",
                    "/give @p minecraft:arrow 64",
                ]
            ),
        ],
    )
    agent_generator = lambda: VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x")
    worker_kwargs = dict(
        env_generator=env_generator,
        agent_generator=agent_generator,
        num_max_steps=600,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )
    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=2,
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs,
        ),
        episode_filter=InfoBaseFilter(
            key="kill_entity",
            val="cow",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)
