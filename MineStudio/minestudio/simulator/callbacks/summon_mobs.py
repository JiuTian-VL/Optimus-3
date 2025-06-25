"""
Date: 2024-11-11 17:26:22
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 00:12:08
FilePath: /MineStudio/minestudio/simulator/callbacks/summon_mobs.py
"""

from minestudio.simulator.callbacks.callback import MinecraftCallback


class SummonMobsCallback(MinecraftCallback):
    def __init__(self, mobs) -> None:
        self.mobs = mobs
        """
        Examples:
            mobs = [{
                'name': 'cow',
                'number': 10,
                'range_x': [-5, 5],
                'range_z': [-5, 5],
            }]
        """

    def after_reset(self, sim, obs, info):
        chats = []
        for mob in self.mobs:
            for _ in range(mob["number"]):
                name = mob["name"]
                x = sim.np_random.uniform(*mob["range_x"])
                z = sim.np_random.uniform(*mob["range_z"])
                chat = f"/execute as @p at @p run summon minecraft:{name} ~{x} ~ ~{z} {{Age:0}}"
                chats.append(chat)
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
