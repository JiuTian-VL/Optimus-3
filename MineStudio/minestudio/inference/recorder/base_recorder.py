"""
Date: 2024-11-25 07:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:55:03
FilePath: /MineStudio/minestudio/inference/recorder/base_recorder.py
"""

from abc import abstractmethod
from typing import List, Dict, Union, Generator


class EpisodeRecorder:
    def __init__(self):
        pass

    def record(self, episode_generator: Generator) -> Union[Dict, str]:
        num_yes = 0
        num_episodes = 0
        for episode in episode_generator:
            num_episodes += 1
            if episode.get("status") == "yes":
                num_yes += 1
        return {
            "num_yes": num_yes,
            "num_episodes": num_episodes,
            "yes_rate": f"{num_yes / num_episodes * 100:.2f}%",
        }
