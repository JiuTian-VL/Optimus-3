"""
Date: 2024-11-08 04:17:36
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-31 23:46:58
FilePath: /MineStudio/minestudio/data/minecraft/core.py
"""

import hashlib
import io
import pickle
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import albumentations as A
import av
import cv2
import lmdb
import numpy as np
import torch
from rich import print
from rich.console import Console
from torch.utils.data import Dataset

from minestudio.data.minecraft.utils import pull_datasets_from_remote
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

SEG_RE_MAP = {0: 0, 1: 3, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def rle_encode(binary_mask):
    """
    binary_mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = binary_mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


def merge_segment_chunks(
    chunks: Sequence[bytes], width: int = 128, height: int = 128, **kwargs
) -> Mapping[str, Sequence[Any]]:
    """Decode bytes to segmentation masks.
    Suppose the segment info is a dict, and the dict is a sequence of dict.
    """
    all_obj_masks = []
    for bytes in chunks:
        chunk = pickle.loads(bytes)
        all_obj_masks += chunk
    # resize the segmentation mask into fixed size
    for i in range(len(all_obj_masks)):
        for obj_id, obj_mask in all_obj_masks[i].items():
            assert isinstance(obj_mask, str), f"obj_mask should be rle-format (str), but got {type(obj_mask)}"
            obj_mask = rle_decode(obj_mask, (360, 640))
            all_obj_masks[i][obj_id] = cv2.resize(obj_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # random choose one object mask as the target
    nb_frames = len(all_obj_masks)
    target_obj_mask = {
        "obj_id": [-1 for _ in range(nb_frames)],
        "obj_mask": [np.zeros((height, width), dtype=np.uint8) for _ in range(nb_frames)],
    }
    last_obj_id = None
    for frame_idx in range(len(all_obj_masks) - 1, -1, -1):
        frame_obj_masks = all_obj_masks[frame_idx]
        if len(frame_obj_masks) == 0:
            last_obj_id = None
            continue
        candidates = list(frame_obj_masks.keys())
        if last_obj_id not in candidates:
            last_obj_id = random.choice(candidates)
        choice = last_obj_id
        target_obj_mask["obj_id"][frame_idx] = SEG_RE_MAP[choice]
        target_obj_mask["obj_mask"][frame_idx] = frame_obj_masks[choice]
        # if random.random() < 0.25: #! random drop 1-p object masks
        #     target_obj_mask["obj_id"][frame_idx] = SEG_RE_MAP[choice]
        #     target_obj_mask["obj_mask"][frame_idx] = frame_obj_masks[choice]

    target_obj_mask["obj_id"] = np.array(target_obj_mask["obj_id"], dtype=np.int32)
    target_obj_mask["obj_mask"] = np.array(target_obj_mask["obj_mask"])
    return target_obj_mask


def extract_segment_chunks(continuous_chunks: Sequence[Any], start: int, end: int, skip_frame: int) -> Sequence[Any]:
    """Extract segmentation mask slice."""
    return {k: v[start:end:skip_frame] for k, v in continuous_chunks.items()}


def merge_action_chunks(chunks: Sequence[bytes], **kwargs) -> Mapping[str, Sequence[np.ndarray]]:
    """Decode byte sequence to actions and merge them into action dict."""
    chunks = [pickle.loads(bytes) for bytes in chunks]
    continuous_chunks = {}
    for chunk in chunks:
        for k, v in chunk.items():
            if k not in continuous_chunks:
                continuous_chunks[k] = []
            continuous_chunks[k].append(v)
    continuous_chunks = {k: np.concatenate(v, axis=0) for k, v in continuous_chunks.items()}
    return continuous_chunks


def extract_action_chunks(
    continuous_chunks: Mapping[str, Sequence[np.ndarray]],
    start: int,
    end: int,
    skip_frame: int,
) -> Mapping[str, Sequence[np.ndarray]]:
    """Extract specified slice from action sequence."""
    result = {k: v[start:end:skip_frame] for k, v in continuous_chunks.items()}
    return result


def decode_video_chunk(chunk: bytes, width: int = 128, height: int = 128) -> np.ndarray:
    """Decode bytes to video frames."""

    def convert_and_resize(frame, width, height):
        frame = frame.to_ndarray(format="rgb24")
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    future_frames = []
    with io.BytesIO(chunk) as input:
        with ThreadPoolExecutor(max_workers=5) as executor:
            container = av.open(input, "r")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            packet_generator = container.demux(stream)
            for packet in packet_generator:
                for av_frame in packet.decode():
                    future = executor.submit(convert_and_resize, av_frame, width, height)
                    future_frames.append(future)
            frames = [future.result() for future in future_frames]
            stream.close()
            container.close()

    frames = np.array(frames)

    return frames


def merge_video_chunks(
    chunks: Sequence[bytes],
    width: int = 128,
    height: int = 128,
    **kwargs,
) -> Sequence[np.ndarray]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = list(executor.map(partial(decode_video_chunk, width=width, height=height), chunks))
    continuous_chunks = np.concatenate(frames, axis=0)
    return continuous_chunks


def extract_video_chunks(
    continuous_chunks: Sequence[np.ndarray], start: int, end: int, skip_frame: int
) -> Sequence[np.ndarray]:
    """Extract video frame slice."""
    return continuous_chunks[start:end:skip_frame]


def merge_contractor_info_chunks(chunks: Sequence[bytes], **kwargs) -> Mapping[str, Sequence[Any]]:
    """Decode bytes to contractor info.
    Suppose the contractor info is a dict, and the dict is a sequence of dict.
    The key includes 'delta_inventory', 'hotbar', 'yaw', 'pitch'.
    """
    continuous_chunks = {}
    for bytes in chunks:
        chunk = pickle.loads(bytes)
        for frame_info in chunk:
            for k, v in frame_info.items():
                if k not in continuous_chunks:
                    continuous_chunks[k] = []
                continuous_chunks[k].append(v)
    return continuous_chunks


def extract_contractor_info_chunks(
    continuous_chunks: Sequence[Any], start: int, end: int, skip_frame: int
) -> Sequence[Any]:
    """Extract contractor info slice."""
    return {k: v[start:end:skip_frame] for k, v in continuous_chunks.items()}


def padding_video(frames: np.ndarray, win_len: int) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    """Padding video frames into fixed length and return mask."""
    traj_len = frames.shape[0]
    dims = frames.shape[1:]
    padded_frames = np.concatenate([frames, np.zeros((win_len - traj_len, *dims), dtype=np.uint8)], axis=0)
    mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len - traj_len, dtype=np.uint8)], axis=0)
    return padded_frames, mask


def padding_action(actions: Dict, win_len: int) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    """Padding actions into fixed length and return mask."""
    result = dict()
    traj_len = 0
    for key, val in actions.items():
        traj_len = val.shape[0]
        dims = val.shape[1:]
        padded_val = np.concatenate([val, np.zeros((win_len - traj_len, *dims), dtype=np.uint8)], axis=0)
        result[key] = padded_val
    mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len - traj_len, dtype=np.uint8)], axis=0)
    return result, mask


def padding_contractor_info(
    contractor_infos: Mapping[str, Sequence[Any]], win_len: int
) -> Tuple[Sequence[Any], np.ndarray]:
    """Padding contractor info into fixed length and return mask."""
    padded_contractor_infos = dict()
    traj_len = 0
    for key, seq in contractor_infos.items():
        traj_len = len(seq)
        if isinstance(seq, np.ndarray):
            padded_contractor_infos[key] = np.concatenate(
                [np.array(seq), np.zeros(win_len - traj_len, dtype=seq.dtype)], axis=0
            )
        else:
            padded_contractor_infos[key] = seq + [None] * (win_len - len(seq))
    mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len - traj_len, dtype=np.uint8)], axis=0)
    return padded_contractor_infos, mask


def padding_segment(segments: Dict[str, np.ndarray], win_len: int) -> Tuple[Sequence[Any]]:
    """Padding segments info into fixed length and return mask."""
    traj_len = len(segments["obj_id"])
    padded_segments = dict()
    padded_segments["obj_id"] = np.concatenate(
        [segments["obj_id"], np.zeros(win_len - traj_len, dtype=np.int32)], axis=0
    )
    padded_segments["obj_mask"] = np.concatenate(
        [segments["obj_mask"], np.zeros((win_len - traj_len, *segments["obj_mask"].shape[1:]), dtype=np.uint8)], axis=0
    )
    mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len - traj_len, dtype=np.uint8)], axis=0)
    return padded_segments, mask


MERGE_FUNCTIONS = {
    "video": merge_video_chunks,
    "action": merge_action_chunks,
    "contractor_info": merge_contractor_info_chunks,
    "segment": merge_segment_chunks,
}

EXTRACT_FUNCTIONS = {
    "video": extract_video_chunks,
    "action": extract_action_chunks,
    "contractor_info": extract_contractor_info_chunks,
    "segment": extract_segment_chunks,
}

PADDING_FUNCTIONS = {
    "video": padding_video,
    "action": padding_action,
    "contractor_info": padding_contractor_info,
    "segment": padding_segment,
}


class LMDBDriver(object):
    SHORT_NAME_LENGTH = 8

    def __init__(
        self, source_dirs: List[str], short_name: bool = False, resolution: Tuple[int, int] = (128, 128)
    ) -> None:
        source_dirs = [Path(source_dir) for source_dir in sorted(source_dirs)]
        self.resol = resolution
        self.episode_infos = []
        self.num_episodes = 0
        self.num_total_frames = 0
        self.chunk_size = None
        # merge all lmdb files into one single view
        for source_dir in source_dirs:
            for lmdb_path in source_dir.iterdir():
                stream = lmdb.open(str(lmdb_path), max_readers=128, lock=False, readonly=True)
                # self.lmdb_streams.append(stream)
                with stream.begin() as txn:
                    # read meta infos from each lmdb file
                    __chunk_size__ = pickle.loads(txn.get("__chunk_size__".encode()))
                    __chunk_infos__ = pickle.loads(txn.get("__chunk_infos__".encode()))
                    __num_episodes__ = pickle.loads(txn.get("__num_episodes__".encode()))
                    __num_total_frames__ = pickle.loads(txn.get("__num_total_frames__".encode()))
                    # merge meta infos to a single view
                    for chunk_info in __chunk_infos__:
                        chunk_info["lmdb_stream"] = stream
                        if short_name:
                            chunk_info["episode"] = hashlib.md5(chunk_info["episode"].encode()).hexdigest()[
                                : self.SHORT_NAME_LENGTH
                            ]
                    self.episode_infos += __chunk_infos__
                    self.num_episodes += __num_episodes__
                    self.num_total_frames += __num_total_frames__
                    self.chunk_size = __chunk_size__
        # create a episode to index mapping
        self.eps_idx_mapping = {info["episode"]: idx for idx, info in enumerate(self.episode_infos)}

    def read_chunks(self, eps: str, start: int, end: int) -> List[bytes]:
        """
        Given episode name and required interval, return the corresponding chunks.
        [start, end] refer to frame-level indexes, which % self.chunk_size == 0.
        """
        assert start % self.chunk_size == 0 and end % self.chunk_size == 0
        meta_info = self.episode_infos[self.eps_idx_mapping[eps]]
        read_chunks = []
        for chunk_id in range(start, end + self.chunk_size, self.chunk_size):
            with meta_info["lmdb_stream"].begin() as txn:
                key = str((meta_info["episode_idx"], chunk_id)).encode()
                chunk_bytes = txn.get(key)
                read_chunks.append(chunk_bytes)

        return read_chunks

    def read_frames(
        self,
        eps: str,
        start: int,
        win_len: int,
        skip_frame: int,
        merge_fn: Callable,
        extract_fn: Callable,
        padding_fn: Callable,
    ) -> Tuple[Sequence[np.ndarray], np.ndarray]:
        """
        Given episode name and required interval, return the corresponding frames.
        [start, end] refer to a frame-level index, 0 <= start <= end < num_frames
        """
        meta_info = self.episode_infos[self.eps_idx_mapping[eps]]
        end = min(start + win_len * skip_frame - 1, meta_info["num_frames"] - 1)  # include
        chunks = self.read_chunks(
            eps, start // self.chunk_size * self.chunk_size, end // self.chunk_size * self.chunk_size
        )
        # 1. merge chunks into continuous frames
        frames = merge_fn(chunks, width=self.resol[0], height=self.resol[1])
        # 2. extract frames according to skip_frame
        bias = (start // self.chunk_size) * self.chunk_size
        frames = extract_fn(frames, start - bias, end - bias + 1, skip_frame)
        # 3. padding frames and get masks
        frames, mask = padding_fn(frames, win_len)
        return frames, mask

    def get_episode_list(self) -> List[str]:
        return [info["episode"] for info in self.episode_infos]

    def get_num_frames(self, episodes: Optional[List[str]] = None):
        if episodes is None:
            episodes = self.eps_idx_mapping.keys()

        num_frames = 0
        for eps in episodes:
            info_idx = self.eps_idx_mapping[eps]
            num_frames += self.episode_infos[info_idx]["num_frames"]

        return num_frames


class Kernel:
    def __init__(
        self,
        dataset_dirs: List[str],
        enable_video: bool = True,
        enable_action: bool = True,
        enable_contractor_info: bool = False,
        enable_segment: bool = False,
        frame_width: int = 128,
        frame_height: int = 128,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """The kernel class for managing datasets. It provides a unified interface for
        accessing demonstrations information such as video, action, and contractor_info.
        Args:
            dataset_dirs (List[str]): A list of paths to dataset directories.
                It is supposed the dataset directory contains the following subdirectories:
                video, action, contractor_info, ...
            enable_video (bool): indicate whether the output should contain video frames.
            enable_action (bool): indicate whether the output should contain actions.
            frame_width (int, optional): width of output video frames. Defaults to 128.
            frame_height (int, optional): height of output video frames. Defaults to 128.
        """
        dataset_dirs = pull_datasets_from_remote(dataset_dirs)
        self.dataset_dirs = [Path(dataset_dir) for dataset_dir in sorted(dataset_dirs)]

        enable_sources = {
            "video": enable_video,
            "action": enable_action,
            "contractor_info": enable_contractor_info,
            "segment": enable_segment,
        }
        self.enable_sources = [k for k, v in enable_sources.items() if v]

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.verbose = verbose
        self.load_drivers()

    def load_drivers(self):
        self.drivers = dict()
        episodes = None
        for source_type in self.enable_sources:
            dataset_dirs = [str(dataset_dir / source_type) for dataset_dir in self.dataset_dirs]
            driver = LMDBDriver(dataset_dirs, short_name=False, resolution=(self.frame_width, self.frame_height))
            self.drivers[source_type] = driver
            part_episodes = set(driver.get_episode_list())
            if self.verbose:
                Console().log(f"[Kernel] Driver {source_type} load {len(part_episodes)} episodes. ")
            episodes = episodes.intersection(part_episodes) if episodes is not None else part_episodes

        self.num_frames = 0
        self.episodes_with_length = OrderedDict()
        for episode in sorted(list(episodes)):
            num_list = [driver.get_num_frames([episode]) for driver in self.drivers.values()]
            if len(set(num_list)) != 1:
                continue
            self.num_frames += num_list[0]
            self.episodes_with_length[episode] = num_list[0]

        if self.verbose:
            Console().log(f"[Kernel] episodes: {len(self.episodes_with_length)}, frames: {self.num_frames}. ")

    def read_frames(
        self, eps: str, start: int, win_len: int, skip_frame: int, source_type: str
    ) -> Tuple[Sequence[np.ndarray], np.ndarray]:
        """Read frames from lmdb files."""
        driver = self.drivers[source_type]
        merge_fn = MERGE_FUNCTIONS[source_type]
        extract_fn = EXTRACT_FUNCTIONS[source_type]
        padding_fn = PADDING_FUNCTIONS[source_type]
        return driver.read_frames(eps, start, win_len, skip_frame, merge_fn, extract_fn, padding_fn)

    def read(self, eps: str, start: int, win_len: int, skip_frame: int):
        """Read all avaliable source data from lmdb files."""
        result = {}
        for source_type in self.enable_sources:
            frames, mask = self.read_frames(eps, start, win_len, skip_frame, source_type)
            result[source_type] = frames
            result[f"{source_type}_mask"] = mask
            if source_type == "action":
                prev_frames, prev_mask = self.read_frames(
                    eps, start - 1, win_len, skip_frame, source_type
                )  # start must > 0
                result[f"prev_{source_type}"] = prev_frames
                result[f"prev_{source_type}_mask"] = prev_mask
        return result

    def get_num_frames(self):
        return self.num_frames

    def get_episodes_with_length(self):
        return self.episodes_with_length


def write_video(
    file_name: str, frames: Sequence[np.ndarray], width: int = 640, height: int = 360, fps: int = 20
) -> None:
    """Write video frames to video files."""
    with av.open(file_name, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class VideoAugmentation:
    def __init__(self, frame_width: int = 224, frame_height: int = 224):
        self.transform = A.ReplayCompose(
            [
                A.Sequential(
                    [
                        A.ColorJitter(
                            hue=(-0.1, 0.1), saturation=(0.8, 1.2), brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=1.0
                        ),
                        A.Affine(rotate=(-4, 2), scale=(0.98, 1.02), shear=2, p=1.0),
                        # A.OneOf([
                        #     A.CropAndPad(px=(0, 30), keep_size=True, p=1.0),
                        #     A.RandomResizedCrop(scale=(0.9, 0.9), ratio=(1.0, 1.0), width=frame_width, height=frame_height, p=1.0),
                        # ], p=1.0),
                    ],
                    p=1.0,
                ),
            ]
        )
        # self.transform = A.OneOf([
        #     A.ImageCompression(quality_lower=5, quality_upper=50, p=1.0),
        #     A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=1.0),
        #     A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, mode='fast', p=1.0),
        #     A.PixelDropout(dropout_prob=0.05, p=1.0),
        # ], p=1.0)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        data = self.transform(image=video[0])
        future_images = []
        with ThreadPoolExecutor() as executor:
            for image in video:
                future_images += [executor.submit(partial(A.ReplayCompose.replay, data["replay"], image=image))]
        video = [future.result()["image"] for future in future_images]
        aug_video = np.array(video).astype(np.uint8)
        return aug_video


class BaseDataset(Dataset):
    def __init__(self, verbose: bool = False, enable_augmentation: bool = False, **kernel_kwargs) -> Any:
        """Base class for all datasets. It provides basic functions for loading data.
        All the datasets should inherit this class.
        Args:
            kernel_kwargs: Arguments for Kernel.
        """
        super(BaseDataset, self).__init__()
        self.verbose = verbose
        self.kernel_kwargs = kernel_kwargs
        self.kernel = Kernel(verbose=self.verbose, **self.kernel_kwargs)
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            self.augmentor = VideoAugmentation(
                frame_width=kernel_kwargs.get("frame_width", 224), frame_height=kernel_kwargs.get("frame_height", 224)
            )

    def build_items(self) -> None:
        """Define how to load data from lmdb."""
        raise NotImplementedError("You have to implement this method.")

    def to_tensor(self, item: Union[np.ndarray, List, Dict]) -> Union[np.ndarray, List, Dict]:
        """Convert numpy array to torch tensor."""
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        elif isinstance(item, List):
            return [self.to_tensor(val) for val in item]
        elif isinstance(item, Dict):
            return {key: self.to_tensor(val) for key, val in item.items()}
        else:
            return item

    def postprocess(self, item: Dict) -> Dict:
        # rename the keys
        if "action" in item:
            action = item.pop("action")
            item["env_action"] = action
            item["agent_action"] = action_mapper.from_factored(action_transformer.env2policy(action))
        if "prev_action" in item:
            prev_action = item.pop("prev_action")
            item["env_prev_action"] = prev_action
            item["agent_prev_action"] = action_mapper.from_factored(action_transformer.env2policy(action))
        if "video" in item:
            item["image"] = item.pop("video")

        masks = []
        for key in list(item.keys()):
            if key.endswith("mask"):
                masks.append(item.pop(key))
        item["mask"] = masks[0]

        if self.enable_augmentation:
            item["image"] = self.augmentor(item["image"])

        item = self.to_tensor(item)

        return item


if __name__ == "__main__":
    demo_eps = "shabby-viridian-beaver-cfb07e8db714-20220421-015533"
    # driver = LMDBDriver(['./tools/tmp/video'], short_name=False)
    # frames, mask = driver.read_frames(eps, 2, 128, 1, merge_video_chunks, extract_video_chunks, padding_video)

    kernel = Kernel(dataset_dirs=["./tools/tmp"], enable_contractor_info=True)

    start = 11
    win_len = 128

    result = kernel.read(demo_eps, start, win_len, 1)
    frames = kernel.get_num_frames()
    print(frames)

    videos, video_mask = kernel.read_frames(demo_eps, start, win_len, 1, "video")
    actions, action_mask = kernel.read_frames(demo_eps, start, win_len, 1, "action")
    contractor_infos, contractor_info_mask = kernel.read_frames(demo_eps, start, win_len, 1, "contractor_info")
