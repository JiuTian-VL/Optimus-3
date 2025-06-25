"""
Date: 2025-01-06 17:32:04
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-06 17:52:28
FilePath: /MineStudio/minestudio/simulator/callbacks/callback.py
"""

import os
import yaml
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal


class MinecraftCallback:
    def load_data_from_conf(source: Union[str, Dict]) -> Dict:
        """
        source can be a yaml file or a dict.
        """
        if isinstance(source, Dict):
            data = source
        else:
            assert os.path.exists(source), f"File {source} not exists."
            with open(source, "r") as f:
                data = yaml.safe_load(f)
        return data

    def create_from_conf(yaml_file: Union[str, Dict]):
        return None

    def before_step(self, sim, action):
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        return obs, reward, terminated, truncated, info

    def before_reset(self, sim, reset_flag: bool) -> bool:  # whether need to call env reset
        return reset_flag

    def after_reset(self, sim, obs, info):
        return obs, info

    def before_close(self, sim):
        return

    def after_close(self, sim):
        return

    def before_render(self, sim):
        return

    def after_render(self, sim):
        return
