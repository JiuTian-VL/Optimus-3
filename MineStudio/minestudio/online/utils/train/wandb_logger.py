from typing import Dict, Any
import ray
import logging
from minestudio.online.utils.train import get_current_session

logger = logging.getLogger("ray")


def log(*args, **kwargs):
    if (training_session := get_current_session()) is not None:
        try:
            ray.get(training_session.log.remote(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")


def define_metric(*args, **kwargs):
    assert (training_session := get_current_session()) is not None
    try:
        ray.get(training_session.define_metric.remote(*args, **kwargs))
    except Exception as e:
        logger.error(f"Error defining metric to wandb: {e}")


def log_video(data: Dict[str, Any], video_key: str, fps: int):
    if (training_session := get_current_session()) is not None:
        try:
            ray.get(training_session.log_video.remote(data, video_key, fps))
        except Exception as e:
            logger.error(f"Error logging video to wandb: {e}")
