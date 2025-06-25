"""
Date: 2024-11-11 07:53:19
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 03:18:54
FilePath: /MineStudio/minestudio/simulator/callbacks/__init__.py
"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.callbacks.commands import CommandsCallback
from minestudio.simulator.callbacks.demonstration import DemonstrationCallback
from minestudio.simulator.callbacks.fast_reset import FastResetCallback
from minestudio.simulator.callbacks.init_inventory import InitInventoryCallback
from minestudio.simulator.callbacks.judgereset import JudgeResetCallback
from minestudio.simulator.callbacks.mask_actions import MaskActionsCallback
from minestudio.simulator.callbacks.play import PlayCallback
from minestudio.simulator.callbacks.point import PlaySegmentCallback, PointCallback
from minestudio.simulator.callbacks.record import RecordCallback
from minestudio.simulator.callbacks.reward_gate import GateRewardsCallback
from minestudio.simulator.callbacks.rewards import RewardsCallback
from minestudio.simulator.callbacks.speed_test import SpeedTestCallback
from minestudio.simulator.callbacks.summon_mobs import SummonMobsCallback
from minestudio.simulator.callbacks.task import TaskCallback
from minestudio.simulator.callbacks.voxels import VoxelsCallback


def load_callbacks_from_config(config: Union[str, Dict]):
    from minestudio.utils.register import Registers

    callbacks = []
    for key in Registers.simulator_callback.keys():
        callback = Registers.simulator_callback[key].create_from_conf(config)
        if callback is not None:
            callbacks.append(callback)
    return callbacks
