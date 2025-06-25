from multiprocessing.connection import Connection
from torch.multiprocessing import Process
from typing import Dict, Callable, Optional, Tuple, Union
import cv2
import torch, rich
import traceback
import av
import time

# import tracemalloc
from omegaconf import DictConfig
import uuid
from threading import Thread
from queue import Queue
import numpy as np
from pathlib import Path, PosixPath
import random
import logging
import os

# from minestudio.simulator.entry import MinecraftSim
import copy
from minestudio.simulator import MinecraftSim
import subprocess


class VideoWriter(Thread):
    def __init__(self, video_fps: int, queue_size: int = 200):
        super().__init__()
        self.cmd_queue = Queue(queue_size)
        self.video_container = None
        self.video_stream = None
        self.video_fps = video_fps
        self.openess = 0

    def open_video(self, path: PosixPath):
        assert self.openess == 0
        self.cmd_queue.put(("open", path))
        self.openess = 1

    def close_video(self):
        self.cmd_queue.put(("close",))
        self.openess = 0

    def write_frame(self, frame: np.ndarray):
        assert self.openess == 1
        self.cmd_queue.put(("write", frame))

    def run(self):
        while True:
            cmd, *args = self.cmd_queue.get()
            if cmd == "open":
                (path,) = args
                if not path.parent.exists():
                    try:
                        path.parent.mkdir(parents=True)
                    except FileExistsError:
                        pass
                assert self.video_container is None
                self._next_path = path  # lazy open
            elif cmd == "write":
                if self.video_container is None:
                    self.video_container = av.open(str(self._next_path), mode="w", format="mp4")
                    self.video_stream = self.video_container.add_stream("h264", rate=self.video_fps)
                    height, width, _ = args[0].shape
                    self.video_stream.width = width
                    self.video_stream.height = height
                    self.video_stream.pix_fmt = "yuv420p"
                frame = av.VideoFrame.from_ndarray(args[0], format="rgb24")
                for packet in self.video_stream.encode(frame):
                    self.video_container.mux(packet)
            elif cmd == "close":
                assert self.video_container is not None
                for packet in self.video_stream.encode():
                    self.video_container.mux(packet)
                self.video_stream.close()
                self.video_container.close()
                self.video_container = None
                self.video_stream = None


def draw_vpred(img: np.ndarray, vpred: float, additional_text: Optional[str] = ""):
    img = img.copy()
    h, w, c = img.shape
    if c == 1:
        img = img.repeat(3, axis=2)
    ref_text = "vpred: -1000.000"
    text = f"vpred: %0.3f" % vpred + additional_text
    ref_font_scale = 1
    ref_thickness = 4
    (ref_text_width, ref_text_height), baseline = cv2.getTextSize(
        ref_text, cv2.FONT_HERSHEY_SIMPLEX, ref_font_scale, ref_thickness
    )  # type: ignore
    desired_width = 0.4 * w
    desired_height = 0.2 * h
    scale = min(desired_width / ref_text_width, desired_height / ref_text_height)
    text_height = int(ref_text_height * scale)
    offset = int(min(w, h) * 0.05)
    text_org = (offset, text_height + offset)
    img = cv2.putText(
        img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, scale * ref_font_scale, (0, 255, 0), int(ref_thickness * scale)
    )  # type: ignore
    return img


class EnvWorker(Process):
    def __init__(
        self,
        env_generator: Callable[[], MinecraftSim],
        conn: Connection,
        video_output_dir: str,
        video_fps: int,
        restart_interval: Optional[int] = None,
        max_fast_reset: int = 10000,
        env_id: int = 0,
        rollout_worker_id: int = 0,
    ):
        super().__init__()
        self.max_fast_reset = max_fast_reset
        self.env_generator = copy.deepcopy(env_generator)
        self.env_id = env_id
        self.conn = conn
        self.restart_interval = restart_interval
        self.video_output_dir = Path(video_output_dir)
        self.rollout_worker_id = rollout_worker_id
        if not self.video_output_dir.exists():
            try:
                self.video_output_dir.mkdir(parents=True)
            except FileExistsError:
                pass

        self.video_fps = video_fps

    def step_agent(
        self, obs: dict, last_reward: float, last_terminated: bool, last_truncated: bool, episode_uuid: str
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        self.conn.send(("step_agent", obs, last_reward, last_terminated, last_truncated, episode_uuid))
        action, vpred = self.conn.recv()
        return action, vpred

    def reset_state(self) -> Dict[str, torch.Tensor]:
        self.conn.send(("reset_state", None))
        return self.conn.recv()

    def report_rewards(self, rewards: np.ndarray):
        self.conn.send(("report_rewards", rewards))
        return self.conn.recv()

    def run(self) -> None:
        video_writer = VideoWriter(video_fps=self.video_fps)
        video_writer.start()
        record = False
        self.env = self.env_generator()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        episode_info = None
        while True:
            time.sleep(random.randint(0, 20))
            self.env.close()
            self.env = self.env_generator()
            try:
                start_time = time.time()
                reward = 0.0
                terminated, truncated = (
                    True,
                    False,
                )  # HACK: make sure the 'first' flag of the first step in the first episode is True
                while True:
                    if self.restart_interval is not None and time.time() - start_time >= self.restart_interval:
                        raise Exception("Restart interval reached")
                    self.reset_state()
                    obs, _ = self.env.reset()
                    reward_list = []
                    step = 0
                    v_preds = []
                    obs_imgs = []
                    episode_uuid = str(uuid.uuid4())
                    while True:
                        if step % 100 == 1:
                            logging.getLogger("ray").info(
                                f"working..., max_fast_reset: {self.max_fast_reset}, env_id: {self.env_id}, rollout_worker_id: {self.rollout_worker_id}, step: {step}"
                            )
                        step += 1
                        action, vpred = self.step_agent(
                            obs,
                            last_reward=float(reward),
                            last_terminated=terminated,
                            last_truncated=truncated,
                            episode_uuid=episode_uuid,
                        )

                        v_preds.append(vpred)
                        if record:
                            obs_imgs.append(torch.from_numpy(obs["image"]).unsqueeze(0))
                        obs, reward, terminated, truncated, _ = self.env.step(action)

                        reward_list.append(reward)
                        if terminated or truncated:
                            break

                    if record:
                        record = False
                        obs_img = torch.cat(obs_imgs, dim=0).numpy()
                        imgs = obs_img.astype(np.uint8)
                        for jdx in range(imgs.shape[0]):
                            img = draw_vpred(imgs[jdx], v_preds[jdx], additional_text="")
                            video_writer.write_frame(img)
                        video_writer.close_video()
                    # _result = self.report_rewards(np.array(reward_list))

                    _result, episode_info = self.report_rewards(np.array(reward_list))
                    obs["online_info"] = episode_info

                    if _result is not None:
                        record = True
                        video_step = _result
                        vidoe_uuid = str(uuid.uuid4())

                        save_video_name = f"{timestamp}/" + f"{video_step} - {vidoe_uuid}.mp4".replace(
                            "/", "_"
                        ).replace("\\", "_")
                        video_path = self.video_output_dir / save_video_name
                        os.makedirs(video_path.parent, exist_ok=True)
                        if video_writer.openess == 1:
                            video_writer.close_video()
                        video_writer.open_video(video_path)

            except Exception as e:
                traceback.print_exc()
                rich.print(f"[bold red]An error occurred in EnvWorker: {e}[/bold red]")
