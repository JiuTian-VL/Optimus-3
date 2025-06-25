from numpy import roll
from omegaconf import OmegaConf
from omegaconf import DictConfig
import ray
import wandb
import uuid
import torch


@ray.remote
class TrainingSession:
    def __init__(self, logger_config: DictConfig, hyperparams: DictConfig):
        self.session_id = str(uuid.uuid4())
        hyperparams_dict = OmegaConf.to_container(hyperparams, resolve=True)
        wandb.init(config=hyperparams_dict, **logger_config)  # type: ignore

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def define_metric(self, *args, **kwargs):
        wandb.define_metric(*args, **kwargs)

    def log_video(self, data: dict, video_key: str, fps: int):
        data[video_key] = wandb.Video(data[video_key], fps=fps, format="mp4")
        wandb.log(data)

    def get_session_id(self):
        return self.session_id
