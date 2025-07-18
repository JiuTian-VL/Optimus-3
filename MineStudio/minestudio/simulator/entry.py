"""
Date: 2024-11-11 05:20:17
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-02-25 16:04:19
FilePath: /HierarchicalAgent/scratch/muzhancun/MineStudio/minestudio/simulator/entry.py
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minestudio.utils import get_mine_studio_dir
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer


@dataclass
class CameraConfig:
    camera_binsize: int = 2
    camera_maxval: int = 10
    camera_mu: float = 10.0
    camera_quantization_scheme: str = "mu_law"

    def __post_init__(self):
        if self.camera_quantization_scheme not in ["mu_law", "linear"]:
            raise ValueError("camera_quantization_scheme must be 'mu_law' or 'linear'")

    @property
    def n_camera_bins(self):
        """The bin number of the setting."""
        return 2 * self.camera_maxval // self.camera_binsize + 1

    @property
    def action_transformer_kwargs(self):
        """Dictionary of camera settings used by an action transformer."""
        return {
            "camera_binsize": self.camera_binsize,
            "camera_maxval": self.camera_maxval,
            "camera_mu": self.camera_mu,
            "camera_quantization_scheme": self.camera_quantization_scheme,
        }


def random_ore(env, ORE_MAP, ypos: float, thresold: float = 0.9):
    prob = random.random()
    if prob <= thresold:
        return
    dy = random.randint(-5, -3)
    new_pos = int(ypos + dy)
    ypos = int(ypos)
    if 45 <= ypos <= 50:  # max: 6
        # coal_ore
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 45:
            ORE_MAP[new_pos] = "coal_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:coal_ore".format(dy))
            print(f"coal ore at {new_pos}")
    elif 26 <= ypos <= 43:  # max: 17
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 26:
            ORE_MAP[new_pos] = "iron_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:iron_ore".format(dy))
            print(f"iron ore at {new_pos}")

    elif 14 < ypos <= 26:
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 17:  # max: 10
            ORE_MAP[new_pos] = "gold_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:gold_ore".format(dy))
            print(f"gold ore at {new_pos}")
        elif ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos <= 16:  # max:12
            ORE_MAP[new_pos] = "redstone_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:redstone_ore".format(dy))
            print(f"redstone ore at {new_pos}")
    elif ypos <= 14 and ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 1:  # max: 14
        ORE_MAP[new_pos] = "diamond_ore"
        ORE_MAP[ypos] = 1
        env.execute_cmd("/setblock ~ ~{} ~ minecraft:diamond_ore".format(dy))
        print(f"diamond ore at {new_pos}")


def download_engine():
    import zipfile

    import huggingface_hub

    local_dir = get_mine_studio_dir()
    print(f"Downloading simulator engine to {local_dir}")
    huggingface_hub.hf_hub_download(repo_id="CraftJarvis/SimulatorEngine", filename="engine.zip", local_dir=local_dir)
    with zipfile.ZipFile(os.path.join(local_dir, "engine.zip"), "r") as zip_ref:
        zip_ref.extractall(local_dir)
    os.remove(os.path.join(local_dir, "engine.zip"))


def check_engine(skip_confirmation=False):
    if not os.path.exists(os.path.join(get_mine_studio_dir(), "engine", "build", "libs", "mcprec-6.13.jar")):
        if skip_confirmation:
            download_engine()
        else:
            response = input(
                "Detecting missing simulator engine, do you want to download it from huggingface (Y/N)?\n"
            )
            if response == "Y" or response == "y":
                download_engine()
            else:
                exit(0)


class MinecraftSim(gymnasium.Env):
    def __init__(
        self,
        action_type: Literal["env", "agent"] = "agent",  # the style of the action space
        obs_size: Tuple[int, int] = (224, 224),  # the resolution of the observation (cv2 resize)
        render_size: Tuple[int, int] = (640, 360),  # the original resolution of the game is 640x360
        seed: int = 0,  # the seed of the minecraft world
        inventory: Dict = {},  # the initial inventory of the agent
        preferred_spawn_biome: Optional[str] = None,  # the preferred spawn biome when call reset
        num_empty_frames: int = 20,  # the number of empty frames to skip when calling reset
        callbacks: List[MinecraftCallback] = [],  # the callbacks to be called before and after each basic calling
        camera_config: CameraConfig = None,  # the configuration for camera quantization and binning settings
        **kwargs,
    ) -> Any:
        super().__init__()
        check_engine()
        self.obs_size = obs_size
        self.action_type = action_type
        self.render_size = render_size
        self.seed = seed
        self.num_empty_frames = num_empty_frames
        self.callbacks = callbacks
        self.callback_messages = set()  # record messages from callbacks, for example the help messages

        self.env = HumanSurvival(
            fov_range=[70, 70],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0],
            frameskip=1,
            resolution=render_size,
            inventory=inventory,
            preferred_spawn_biome=preferred_spawn_biome,
        ).make()

        self.env.seed(seed)
        self.already_reset = False
        self._only_once = False

        if camera_config is None:
            camera_config = CameraConfig()

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)

    def agent_action_to_env_action(self, action: Dict[str, Any]):
        #! This is quite important step (for some reason).
        #! For the sake of your sanity, remember to do this step (manual conversion to numpy)
        #! before proceeding. Otherwise, your agent might be a little derp.
        if "attack" in action:
            return action
        if isinstance(action, tuple):
            action = {
                "buttons": action[0],
                "camera": action[1],
            }
        # Second, convert the action to the type of numpy
        if isinstance(action["buttons"], torch.Tensor):
            action = {"buttons": action["buttons"].cpu().numpy(), "camera": action["camera"].cpu().numpy()}
        action = self.action_mapper.to_factored(action)
        action = self.action_transformer.policy2env(action)
        return action

    def env_action_to_agent_action(self, action: Dict[str, Any]):
        action = self.action_transformer.env2policy(action)
        action = self.action_mapper.from_factored(action)
        return action

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.action_type == "agent":
            action = self.agent_action_to_env_action(action)
        for callback in self.callbacks:
            action = callback.before_step(self, action)
        obs, reward, done, info = self.env.step(action.copy())
        self.obs, self.info = obs, info
        if self._only_once:
            random_ore(self.env, self.ORE_MAP, info["location_stats"]["ypos"].item(), thresold=0.2)
            self._only_once = False

        terminated, truncated = done, done
        obs, info = self._wrap_obs_info(obs, info)
        for callback in self.callbacks:
            obs, reward, terminated, truncated, info = callback.after_step(
                self, obs, reward, terminated, truncated, info
            )
            self.obs, self.info = obs, info
        return obs, reward, terminated, truncated, info

    def reset(self) -> Tuple[np.ndarray, Dict]:
        reset_flag = True
        self.ORE_MAP = {}
        self._only_once = False
        for callback in self.callbacks:
            reset_flag = callback.before_reset(self, reset_flag)
        if reset_flag:  # hard reset
            self.env.reset()
            self.already_reset = True
        for _ in range(self.num_empty_frames):  # skip the frames to avoid the initial black screen
            action = self.env.action_space.no_op()
            obs, reward, done, info = self.env.step(action)
        obs, info = self._wrap_obs_info(obs, info)
        for callback in self.callbacks:
            obs, info = callback.after_reset(self, obs, info)
            self.obs, self.info = obs, info
        return obs, info

    def _wrap_obs_info(self, obs: Dict, info: Dict) -> Dict:
        _info = info.copy()
        _info.update(obs)
        _obs = {"image": cv2.resize(obs["pov"], dsize=self.obs_size, interpolation=cv2.INTER_LINEAR)}
        if getattr(self, "info", None) is None:
            self.info = {}
        for key, value in _info.items():
            self.info[key] = value
        _info = self.info.copy()
        return _obs, _info

    def noop_action(self) -> Dict[str, Any]:
        if self.action_type == "agent":
            return {
                "buttons": np.array([[0]]),
                "camera": np.array([[60]]),
            }
        else:
            return self.env.action_space.no_op()

    def close(self) -> None:
        for callback in self.callbacks:
            callback.before_close(self)
        close_status = self.env.close()
        for callback in self.callbacks:
            callback.after_close(self)
        return close_status

    def render(self) -> None:
        for callback in self.callbacks:
            callback.before_render(self)
        #! core logic
        for callback in self.callbacks:
            callback.after_render(self)

    @property
    def action_space(self) -> spaces.Dict:
        if self.action_type == "agent":
            return gymnasium.spaces.Dict({
                "buttons": gymnasium.spaces.MultiDiscrete([8641]),
                "camera": gymnasium.spaces.MultiDiscrete([121]),
            })
        elif self.action_type == "env":
            return gymnasium.spaces.Dict({
                "attack": gymnasium.spaces.Discrete(2),
                "back": gymnasium.spaces.Discrete(2),
                "forward": gymnasium.spaces.Discrete(2),
                "jump": gymnasium.spaces.Discrete(2),
                "left": gymnasium.spaces.Discrete(2),
                "right": gymnasium.spaces.Discrete(2),
                "sneak": gymnasium.spaces.Discrete(2),
                "sprint": gymnasium.spaces.Discrete(2),
                "use": gymnasium.spaces.Discrete(2),
                "hotbar.1": gymnasium.spaces.Discrete(2),
                "hotbar.2": gymnasium.spaces.Discrete(2),
                "hotbar.3": gymnasium.spaces.Discrete(2),
                "hotbar.4": gymnasium.spaces.Discrete(2),
                "hotbar.5": gymnasium.spaces.Discrete(2),
                "hotbar.6": gymnasium.spaces.Discrete(2),
                "hotbar.7": gymnasium.spaces.Discrete(2),
                "hotbar.8": gymnasium.spaces.Discrete(2),
                "hotbar.9": gymnasium.spaces.Discrete(2),
                "inventory": gymnasium.spaces.Discrete(2),
                "camera": gymnasium.spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
            })
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

    @property
    def observation_space(self) -> spaces.Dict:
        height, width = self.obs_size
        return gymnasium.spaces.Dict({
            "image": gymnasium.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        })

    def _find_in_inventory(self, item: str):
        inventory = self.info["inventory"]
        for slot, it in inventory.items():
            if it["type"] == item:
                return slot
        return None

    def find_best_pickaxe(self):
        # find pickaxe
        inventory_id_diamond = self._find_in_inventory("diamond_pickaxe")
        inventory_id_iron = self._find_in_inventory("iron_pickaxe")
        inventory_id_stone = self._find_in_inventory("stone_pickaxe")
        inventory_id_wooden = self._find_in_inventory("wooden_pickaxe")
        if inventory_id_diamond is not None:
            return "diamond_pickaxe"
        if inventory_id_iron is not None:
            return "iron_pickaxe"
        if inventory_id_stone is not None:
            return "stone_pickaxe"
        if inventory_id_wooden is not None:
            return "wooden_pickaxe"
        return None


if __name__ == "__main__":
    # test if the simulator works
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation", default=False)
    args = parser.parse_args()

    if args.yes:
        check_engine(skip_confirmation=True)

    from minestudio.simulator.callbacks import SpeedTestCallback

    sim = MinecraftSim(action_type="env", callbacks=[SpeedTestCallback(50)])
    obs, info = sim.reset()
    pass
    for i in range(100):
        action = sim.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()
