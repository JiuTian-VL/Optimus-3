"""
Date: 2024-11-25 12:39:01
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 08:19:00
FilePath: /MineStudio/minestudio/inference/filter/info_base_filter.py
"""

import re
import pickle
from minestudio.inference.filter.base_filter import EpisodeFilter


class InfoBaseFilter(EpisodeFilter):
    def __init__(self, key: str, regex: str, num: int, label: str = "status"):
        self.key = key
        self.regex = regex
        self.num = num
        self.label = label

    def filter(self, episode_generator):
        for episode in episode_generator:
            info = pickle.loads(open(episode["info_path"], "rb").read())
            total = 0
            last_info = info[-1][self.key]
            for event in last_info:
                if re.match(self.regex, event):
                    total += last_info.get(event, 0)
            if total >= self.num:
                episode[self.label] = "yes"
            yield episode
