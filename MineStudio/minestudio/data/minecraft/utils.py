"""
Date: 2024-11-10 10:06:28
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-30 21:15:52
FilePath: /MineStudio/var/minestudio/data/minecraft/utils.py
"""

import os
import pickle
import random
import shutil
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import av
import cv2
import numpy as np
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from rich import print
from torch.utils.data import Sampler
from tqdm import tqdm


def get_repo_total_size(repo_id, repo_type="dataset", branch="main"):
    def fetch_file_list(path=""):
        url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}/{path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def calculate_size(path=""):
        files = fetch_file_list(path)
        total_size = 0
        for file in files:
            if file["type"] == "file":
                total_size += file["size"]
            elif file["type"] == "directory":
                total_size += calculate_size(file["path"])
        return total_size

    total_size_bytes = calculate_size()
    total_size_gb = total_size_bytes / (1024**3)
    return total_size_bytes, total_size_gb


def download_dataset_from_huggingface(
    name: Literal["6xx", "7xx", "8xx", "9xx", "10xx"], base_dir: Optional[str] = None
):
    if base_dir is None:
        from minestudio.utils import get_mine_studio_dir

        base_dir = get_mine_studio_dir()

    total, used, free = shutil.disk_usage(base_dir)
    repo_id = f"CraftJarvis/minestudio-data-{name}"
    total_size, _ = get_repo_total_size(repo_id)
    print(
        f"""
        [bold]Download Dataset[/bold]
        Dataset: {name}
        Base Dir: {base_dir}
        Total Size: {total_size / 1024 / 1024 / 1024:.2f} GB
        Free Space: {free / 1024 / 1024 / 1024:.2f} GB
        """
    )
    if total_size > free:
        raise ValueError(f"Insufficient space for downloading {name}. ")
    dataset_dir = os.path.join(get_mine_studio_dir(), "contractors", f"dataset_{name}")
    local_dataset_dir = snapshot_download(repo_id, repo_type="dataset", local_dir=dataset_dir)
    return local_dataset_dir


def pull_datasets_from_remote(dataset_dirs: List[str]) -> List[str]:
    new_dataset_dirs = []
    for path in dataset_dirs:
        if path in ["6xx", "7xx", "8xx", "9xx", "10xx"]:
            return_path = download_dataset_from_huggingface(path)
            new_dataset_dirs.append(return_path)
        else:
            new_dataset_dirs.append(path)
    return new_dataset_dirs


def write_video(
    file_name: str, frames: Sequence[np.ndarray], width: int = 640, height: int = 360, fps: int = 20
) -> None:
    """Write video frames to video files."""
    with av.open(file_name, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            assert frame.shape[1] == width and frame.shape[0] == height, (
                f"frame shape {frame.shape} not match {width}x{height}"
            )
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def batchify(batch_in: Sequence[Dict[str, Any]]) -> Any:
    example = batch_in[0]
    if isinstance(example, Dict):
        batch_out = {
            k: batchify([item[k] for item in batch_in])
            for k in example.keys()
            if k not in ["ESC", "swapHands", "pickItem"]
        }
    elif isinstance(example, torch.Tensor):
        batch_out = torch.stack(batch_in, dim=0)
    elif isinstance(example, int):
        batch_out = torch.tensor(batch_in, dtype=torch.int32)
    elif isinstance(example, float):
        batch_out = torch.tensor(batch_in, dtype=torch.float32)
    else:
        batch_out = batch_in
    return batch_out


class MineDistributedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,  # num_replicas is the number of processes participating in the training
        rank=None,  # rank is the rank of the current process within num_replicas
        shuffle=False,
        drop_last=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                rank = dist.get_rank()
            except:
                rank = 0
        assert shuffle is False, "shuffle must be False in sampler."
        assert drop_last is True, "drop_last must be True in sampler."
        # print(f"{rank = }, {num_replicas = }")
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_total_samples = len(self.dataset)
        self.num_samples_per_replica = self.num_total_samples // num_replicas
        replica_range = (
            self.num_samples_per_replica * rank,
            self.num_samples_per_replica * (rank + 1),
        )  # [start, end)

        num_past_samples = 0
        episodes_within_replica = []  # (episode, epsode_start_idx, episode_end_idx, item_bias)
        self.episodes_with_items = self.dataset.episodes_with_items
        for episode, length, item_bias in self.episodes_with_items:
            if num_past_samples + length > replica_range[0] and num_past_samples < replica_range[1]:
                episode_start_idx = max(0, replica_range[0] - num_past_samples)
                episode_end_idx = min(length, replica_range[1] - num_past_samples)
                episodes_within_replica.append((episode, episode_start_idx, episode_end_idx, item_bias))
            num_past_samples += length
        self.episodes_within_replica = episodes_within_replica

    def __iter__(self):
        """
        Build batch of episodes, each batch is consisted of `self.batch_size` episodes.
        Only if one episodes runs out of samples, the batch is filled with the next episode.
        """
        next_episode_idx = 0
        reading_episodes = [None for _ in range(self.batch_size)]
        while True:
            batch = [None for _ in range(self.batch_size)]
            # feed `reading_episodes` with the next episode
            for i in range(self.batch_size):
                if reading_episodes[i] is None:
                    if next_episode_idx >= len(self.episodes_within_replica):
                        break
                    reading_episodes[i] = self.episodes_within_replica[next_episode_idx]
                    next_episode_idx += 1
            # use while loop to build batch
            while any([x is None for x in batch]):
                record_batch_length = sum([x is not None for x in batch])
                # get the position that needs to be filled
                for cur in range(self.batch_size):
                    if batch[cur] is None:
                        break
                # get the episode that has the next sample
                if reading_episodes[cur] is not None:
                    use_eps_idx = cur
                else:
                    for use_eps_idx in range(self.batch_size):
                        if reading_episodes[use_eps_idx] is not None:
                            break
                # if all episodes are None, then stop iteration
                if reading_episodes[use_eps_idx] is None:
                    return None
                # fill the batch with the next sample
                episode, start_idx, end_idx, item_bias = reading_episodes[use_eps_idx]
                batch[cur] = item_bias + start_idx
                if start_idx + 1 < end_idx:
                    reading_episodes[use_eps_idx] = (episode, start_idx + 1, end_idx, item_bias)
                else:
                    reading_episodes[use_eps_idx] = None
            yield batch

    def __len__(self):
        return self.num_samples_per_replica // self.batch_size


def visualize_dataloader(
    dataloader,
    num_samples: int = 1,
    resolution: Tuple[int, int] = (320, 180),
    legend: bool = False,
    save_fps: int = 20,
    output_dir: str = "./",
    **kwargs,
) -> None:
    frames = []
    for idx, data in enumerate(tqdm(dataloader)):
        # continue
        if idx > num_samples:
            break
        action = data["env_action"]
        prev_action = data.get("env_prev_action", None)
        image = data["image"].numpy()
        text = data["text"]

        color = (255, 0, 0)
        for bidx, (tframes, txt) in enumerate(zip(image, text)):
            cache_frames = []
            for tidx, frame in enumerate(tframes):
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
                if "segment" in data:
                    COLORS = [
                        (255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                        (255, 255, 0),
                        (255, 0, 255),
                        (0, 255, 255),
                        (255, 255, 255),
                        (0, 0, 0),
                        (128, 128, 128),
                        (128, 0, 0),
                        (128, 128, 0),
                        (0, 128, 0),
                        (128, 0, 128),
                        (0, 128, 128),
                        (0, 0, 128),
                    ]
                    obj_id = data["segment"]["obj_id"][bidx][tidx].item()
                    if obj_id != -1:
                        segment_mask = data["segment"]["obj_mask"][bidx][tidx]
                        if isinstance(segment_mask, torch.Tensor):
                            segment_mask = segment_mask.numpy()
                        colors = np.array(COLORS[obj_id]).reshape(1, 1, 3)
                        segment_mask = (segment_mask[..., None] * colors).astype(np.uint8)
                        segment_mask = segment_mask[:, :, ::-1]  # bgr -> rgb
                        frame = cv2.addWeighted(frame, 1.0, segment_mask, 0.5, 0.0)

                if "timestamp" in data:
                    timestamp = data["timestamp"][bidx][tidx]
                    cv2.putText(
                        frame, f"timestamp: {timestamp}", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 55), 2
                    )

                if legend:
                    cv2.putText(frame, f"frame: {tidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    cv2.putText(frame, txt, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                    if "contractor_info" in data:
                        try:
                            pitch = data["contractor_info"]["pitch"][bidx][tidx]
                            yaw = data["contractor_info"]["yaw"][bidx][tidx]
                            cursor_x = data["contractor_info"]["cursor_x"][bidx][tidx]
                            cursor_y = data["contractor_info"]["cursor_y"][bidx][tidx]
                            isGuiInventory = data["contractor_info"]["isGuiInventory"][bidx][tidx]
                            isGuiOpen = data["contractor_info"]["isGuiOpen"][bidx][tidx]
                            cv2.putText(
                                frame,
                                f"Pitch: {pitch:.2f}",
                                (150, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                frame, f"Yaw: {yaw:.2f}", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                            )
                            cv2.putText(
                                frame,
                                f"isGuiOpen: {isGuiOpen}",
                                (150, 130),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"isGuiInventory: {isGuiInventory}",
                                (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"CursorX: {cursor_x:.2f}",
                                (150, 170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                frame,
                                f"CursorY: {cursor_y:.2f}",
                                (150, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                        except:
                            cv2.putText(
                                frame,
                                "No Contractor Info",
                                (150, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )

                    act = {k: v[bidx][tidx].numpy() for k, v in action.items()}
                    if prev_action is not None:
                        pre_act = {k: v[bidx][tidx].numpy() for k, v in prev_action.items()}
                    for row, ((k, v), (_, pv)) in enumerate(zip(act.items(), pre_act.items())):
                        if k != "camera":
                            v = int(v.item())
                            pv = int(pv.item())
                        cv2.putText(
                            frame,
                            f"{k}: {v}({pv})",
                            (10, 45 + row * 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                cache_frames.append(frame.astype(np.uint8))

            frames = frames + cache_frames

    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    file_name = f"save_{timestamp}.mp4"
    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_video(file_path, frames, fps=save_fps, width=resolution[0], height=resolution[1])


def dump_trajectories(dataloader, num_samples: int = 1, save_fps: int = 20, **kwargs) -> None:
    def un_batchify_actions(actions_in: Dict[str, torch.Tensor]) -> List[Dict]:
        actions_out = []
        for bidx in range(len(actions_in["attack"])):
            action = {}
            for k, v in actions_in.items():
                action[k] = v[bidx].numpy()
            actions_out.append(action)
        return actions_out

    traj_dir = Path("./traj_dir")
    video_dir = traj_dir / "videos"
    action_dir = traj_dir / "actions"
    video_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)
    for idx, data in enumerate(tqdm(dataloader)):
        if idx > num_samples:
            break
        image = data["img"]
        action = data["action"]
        action = un_batchify_actions(action)
        B, T = image.shape[:2]
        for i in range(B):
            vid = "".join(random.choices(string.ascii_letters + string.digits, k=11))
            write_video(
                file_name=str(video_dir / f"{vid}.mp4"),
                frames=image[i].numpy().astype(np.uint8),
            )
            with open(action_dir / f"{vid}.pkl", "wb") as f:
                pickle.dump(action[i], f)
