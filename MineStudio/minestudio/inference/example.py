"""
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 13:36:22
FilePath: /MineStudio/minestudio/inference/example.py
"""

from minestudio.models import load_vpt_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback


if __name__ == "__main__":
    policy = load_vpt_policy(
        model_path="/data7/Users/xyq/developer/MinecraftOptimus/models/vpt/2x.model",
        weights_path="/data7/Users/xyq/developer/MinecraftOptimus/models/vpt/bc-early-game-2x.weights",
    ).to("cuda")

    env = MinecraftSim(
        obs_size=(128, 128),
        preferred_spawn_biome="forest",
        callbacks=[
            RecordCallback(record_path="./output", fps=30, frame_type="pov"),
            SpeedTestCallback(50),
        ],
    )
    memory = None
    obs, info = env.reset()
    pass

    for i in range(600):
        action, memory = policy.get_action(obs, memory, input_shape="*")
        obs, reward, terminated, truncated, info = env.step(action)
    env.reset()
    print("Resetting the environment")
    for i in range(600):
        action, memory = policy.get_action(obs, memory, input_shape="*")
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
