"""
Date: 2024-11-10 10:25:38
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-12 11:34:15
FilePath: /MineStudio/minestudio/data/minecraft/dataset.py
"""

from typing import Dict, List, Literal, Optional

import gymnasium
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor

from minestudio.data.minecraft.part_event import EventDataset
from minestudio.data.minecraft.part_raw import RawDataset


class MinecraftDataset(Dataset):
    def __init__(
        self,
        mode: Literal["raw", "event"] = "raw",
        split: Literal["train", "val"] = "train",
        # below are parameters for kernel
        dataset_dirs: List[str] = [],
        enable_video: bool = True,
        enable_action: bool = True,
        enable_contractor_info: bool = False,
        enable_segment: bool = False,
        enable_augmentation: bool = False,
        frame_width: int = 128,
        frame_height: int = 128,
        enable_resize: bool = True,
        win_len: int = 128,
        skip_frame: int = 1,
        split_ratio: float = 0.9,
        shuffle: bool = False,  # episode-level shuffle
        # below are parameters for event dataset
        bias: int = 0,
        event_regex: str = "",
        min_nearby: Optional[int] = None,  # the minimum interval between two events
        max_within: Optional[int] = None,  # the maximum numbers of samples within each event
        processor: AutoProcessor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.split = split
        self.common_kwargs = dict(
            dataset_dirs=dataset_dirs,
            enable_video=enable_video,
            enable_action=enable_action,
            enable_contractor_info=enable_contractor_info,
            enable_segment=enable_segment,
            enable_augmentation=enable_augmentation,
            frame_width=frame_width,
            frame_height=frame_height,
            enable_resize=enable_resize,
            shuffle=shuffle,
            task_description_file=kwargs.get("task_description_file", None),
            filter_label=kwargs.get("filter_label", None),
        )
        self.raw_dataset_kwargs = dict(
            win_len=win_len,
            skip_frame=skip_frame,
            split_ratio=split_ratio,
        )
        self.event_dataset_kwargs = dict(
            win_len=win_len,
            skip_frame=skip_frame,
            split_ratio=split_ratio,
            event_regex=event_regex,
            bias=bias,
            min_nearby=min_nearby,
            max_within=max_within,
        )
        self.processor = processor

        if mode == "event":
            self.dataset_kwargs = self.event_dataset_kwargs
        elif mode == "raw":
            self.dataset_kwargs = self.raw_dataset_kwargs
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        if self.mode == "event":
            self.dataset = EventDataset(**self.dataset_kwargs, split=self.split, **self.common_kwargs)
        elif self.mode == "raw":
            self.dataset = RawDataset(**self.dataset_kwargs, split=self.split, **self.common_kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        # todo: image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(item["image"][0].numpy().astype(np.uint8)),
                    },
                    {"type": "text", "text": item["task"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": item["label"],
                    },
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs["labels"] = inputs["input_ids"].clone()
        condition = (inputs["labels"] < 151665) | (inputs["labels"] > 151674)
        inputs["labels"][condition] = -100
        inputs["tasks"] = torch.tensor(3)
        inputs["prompt"] = item["task"]

        item["mllm"] = inputs

        return item

    def __len__(self):
        return len(self.dataset)

    @property
    def episodes_with_items(self):
        return self.dataset.episodes_with_items

    @classmethod
    def build_state_space(cls, params: Dict):
        width = params.get("width", 224)
        height = params.get("height", 224)
        return {"image": gymnasium.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)}

    @classmethod
    def build_action_space(cls, params: Dict):
        """
        Using the action space defined by OpenAI VPT.
        """
        return gymnasium.spaces.Dict({
            "buttons": gymnasium.spaces.MultiDiscrete([8641]),
            "camera": gymnasium.spaces.MultiDiscrete([121]),
        })


def load_dataset(**kwargs):
    return MinecraftDataset(**kwargs)
