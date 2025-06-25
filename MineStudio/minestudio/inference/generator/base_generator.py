"""
Date: 2024-11-25 07:36:18
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-25 12:06:21
FilePath: /MineStudio/minestudio/inference/generator/base_generator.py
"""

from abc import abstractmethod
from typing import List, Dict, Any, Tuple, Generator


class EpisodeGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate(self) -> Generator:
        pass


class AgentInterface:
    @abstractmethod
    def get_action(self, input: Dict, state: Any, **kwargs) -> Tuple[Any, Any]:
        pass
