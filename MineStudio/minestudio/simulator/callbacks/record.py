"""
Date: 2024-11-11 16:40:57
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-20 00:30:51
FilePath: /MineStudio/minestudio/simulator/callbacks/record.py
"""

import json
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
from gymnasium import spaces
from rich import print

from minestudio.simulator.callbacks.callback import MinecraftCallback


class RecordCallback(MinecraftCallback):
    def __init__(
        self,
        record_path: str,
        fps: int = 20,
        frame_type: Literal["pov", "obs"] = "pov",
        recording: bool = True,
        show_actions=False,
        record_actions=False,
        record_infos=False,
        record_origin_observation=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.record_path = Path(record_path)
        self.record_path.mkdir(parents=True, exist_ok=True)
        self.recording = recording
        self.record_actions = record_actions
        self.show_actions = show_actions
        self.record_infos = record_infos
        self.record_origin_observation = record_origin_observation
        if recording:
            print(f"[green]Recording enabled, saving episodes to {self.record_path}[/green]")
        self.fps = fps
        self.frame_type = frame_type
        self.episode_id = 0
        self.frames = []
        self.infos = []
        self.actions = []

    def _get_message(self, info):
        message = info.get("message", {})
        message["RecordCallback"] = (
            f"Recording: {'On' if self.recording else 'Off'}, Recording Time: {len(self.frames)}"
        )
        return message

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.recording:
            self._save_episode()
            self.episode_id += 1
        return reset_flag

    def after_reset(self, sim, obs, info):
        sim.callback_messages.add("Press 'R' to start/stop recording.")
        # this message would be displayed in the GUI when command mode is on
        info["message"] = self._get_message(info)
        if self.recording:
            if self.frame_type == "obs":
                self.frames.append(obs["image"])
            elif self.frame_type == "pov":
                self.frames.append(info["pov"])
            else:
                raise ValueError(f"Invalid frame_type: {self.frame_type}")
            if self.record_actions or self.show_actions:
                self.actions.append({})  # empty for reset
            if self.record_infos:
                self.infos.append(info)

        return obs, info

    def before_step(self, sim, action):
        if self.recording and (self.record_actions or self.show_actions):
            self.actions.append(action)
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.recording and not info.get("R", True):
            self.recording = False
            print("[red]Recording stopped[/red]")
            self._save_episode()
            self.episode_id += 1

        if not self.recording and info.get("R", False):
            self.recording = True
            print("[green]Start recording[/green]")

        if self.recording:
            if self.frame_type == "obs":
                self.frames.append(obs["image"])
            elif self.frame_type == "pov":
                self.frames.append(info["pov"])
            else:
                raise ValueError(f"Invalid frame_type: {self.frame_type}")
            if self.record_infos:
                self.infos.append(info)

        info["message"] = self._get_message(info)
        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        if self.recording:
            if getattr(sim, "status", None) is not None:
                self._save_episode(sim.status)
            else:
                self._save_episode()

    def _save_episode(self, status: str | None = None):
        if len(self.frames) == 0:
            return
        uid = str(uuid.uuid4())
        if status is not None:
            uid += f"_{status}"
        output_path = self.record_path / f"{uid}_episode_{self.episode_id}.mp4"
        with av.open(str(output_path), mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=self.fps)
            stream.width = self.frames[0].shape[1]
            stream.height = self.frames[0].shape[0]
            for idx, frame in enumerate(self.frames):
                # print actions on the frames
                if self.show_actions:
                    for row, (k, v) in enumerate(self.actions[idx].items()):
                        if k in {"chat", "mobs", "voxels"}:
                            continue
                        if k in {"camera"}:
                            color = (234, 53, 70) if (v != 0).any() else (249, 200, 14)
                        else:
                            color = (234, 53, 70) if v != 0 else (249, 200, 14)
                        if k == "camera":
                            v = "[{:.2f}, {:.2f}]".format(v[0], v[1])
                        frame = frame.copy()
                        cv2.putText(frame, f"{k}: {v}", (10, 25 + row * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if self.record_origin_observation:
            output_origin_path = self.record_path / f"episode_{self.episode_id}.npy"
            all_frames = np.array(self.frames)
            np.save(output_origin_path, all_frames)
        print(f"[green]Episode {self.episode_id} saved at {output_path}[/green]")
        self.frames = []

        if self.record_actions:  # assert self.actions>0 sense self.frame > 0
            output_action_path = self.record_path / f"{uid}_episode_{self.episode_id}_action.json"
            record_actions = [self._process_action(action) for action in self.actions]
            with open(output_action_path, "w", encoding="utf-8") as file:
                json.dump(record_actions, file)
            self.actions = []

        if self.record_infos:  # assert self.actions>0 sense self.frame > 0
            output_info_path = self.record_path / f"{uid}_episode_{self.episode_id}_info.json"
            record_infos = [self._process_info(info) for info in self.infos]
            with open(output_info_path, "w", encoding="utf-8") as file:
                json.dump(record_infos, file)
            self.infos = []

    def _process_info(self, info: dict):
        record_info = deepcopy(info)
        if self.frame_type == "pov":
            del record_info["pov"]
        record_info = self._convert_data(record_info)
        return record_info

    def _process_action(self, action: spaces.Dict):
        record_action = dict(deepcopy(action))
        record_action = self._convert_data(record_action)
        return record_action

    def _convert_data(self, data):
        if isinstance(data, dict):
            # Iterate over items and apply conversion recursively
            return {key: self._convert_data(value) for key, value in data.items()}
        elif isinstance(data, defaultdict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, spaces.Dict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
