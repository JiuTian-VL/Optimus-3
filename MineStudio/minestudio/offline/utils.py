"""
Date: 2024-11-26 06:26:26
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-26 06:28:27
FilePath: /MineStudio/minestudio/train/utils.py
"""

from typing import Dict, List

from omegaconf import DictConfig, ListConfig


def convert_to_normal(obj):
    if isinstance(obj, DictConfig) or isinstance(obj, Dict):
        return {key: convert_to_normal(value) for key, value in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, List):
        return [convert_to_normal(item) for item in obj]
    else:
        return obj
