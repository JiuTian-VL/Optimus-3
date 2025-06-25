"""
Date: 2024-11-25 08:35:59
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-02-25 16:12:36
FilePath: /HierarchicalAgent/scratch/muzhancun/MineStudio/minestudio/inference/generator/mine_generator.py
"""

import os
import ray
from typing import Callable, Optional, List, Dict, Tuple, Literal, Generator
from minestudio.inference.generator.base_generator import EpisodeGenerator, AgentInterface


class Worker:
    def __init__(
        self,
        env_generator: Callable,
        agent_generator: Callable,
        num_max_steps: int,
        num_episodes: int,
        agent_type: str = "policy",
        tmpdir: Optional[str] = None,
        image_media: Literal["h264", "jpeg"] = "h264",
        **unused_kwargs,
    ):
        self.num_max_steps = num_max_steps
        self.num_episodes = num_episodes
        self.env = env_generator()
        self.agent = agent_generator().to("cuda")
        self.agent.eval()
        self.image_media = image_media
        self.tmpdir = tmpdir
        self.agent_type = agent_type

        self.generator = self._run()
        os.makedirs(self.tmpdir, exist_ok=True)

    def append_image_and_info(self, info: Dict, images: List, infos: List):
        info = info.copy()
        image = info.pop("pov")
        for key, val in info.items():  # use clean dict type
            if hasattr(info[key], "values"):
                info[key] = dict(info[key])
        images.append(image)
        infos.append(info)

    def save_to_file(self, images: List, actions: List, infos: List):
        import av, pickle, uuid
        from PIL import Image

        episode_id = str(uuid.uuid4())
        episode = {}
        episode["info_path"] = f"{self.tmpdir}/info_{episode_id}.pkl"
        with open(episode["info_path"], "wb") as f:
            pickle.dump(infos, f)
        episode["action_path"] = f"{self.tmpdir}/action_{episode_id}.pkl"
        with open(episode["action_path"], "wb") as f:
            pickle.dump(actions, f)
        if self.image_media == "h264":
            episode["video_path"] = f"{self.tmpdir}/video_{episode_id}.mp4"
            with av.open(episode["video_path"], mode="w", format="mp4") as container:
                stream = container.add_stream("h264", rate=30)
                stream.width = images[0].shape[1]
                stream.height = images[0].shape[0]
                for image in images:
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        elif self.image_media == "jpeg":
            episode["base_image_path"] = f"{self.tmpdir}/images_{episode_id}"
            os.makedirs(episode["base_image_path"], exist_ok=True)
            for i, image in enumerate(images):
                image = Image.fromarray(image)
                image.save(f"{episode['base_image_path']}/{i}.jpeg")
        else:
            raise ValueError(f"Invalid image_media: {self.image_media}")
        return episode

    def _run(self) -> List:
        for eps_id in range(self.num_episodes):
            memory = None
            actions = []
            images = []
            infos = []
            obs, info = self.env.reset()
            self.append_image_and_info(info, images, infos)
            for step in range(self.num_max_steps):
                if self.agent_type == "policy":
                    action, memory = self.agent.get_action(obs, memory, input_shape="*")
                elif self.agent_type == "agent":
                    action = self.agent.get_action(obs, info)
                actions.append(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.append_image_and_info(info, images, infos)
            yield self.save_to_file(images, actions, infos)
        self.env.close()

    def get_next(self):
        if self.generator is None:
            raise ValueError("Generator is not initialized. Call init_generator first.")
        try:
            return next(self.generator)
        except StopIteration:
            return None


class MineGenerator(EpisodeGenerator):
    def __init__(
        self,
        num_workers: int = 1,
        num_gpus: float = 0.5,
        max_restarts: int = 3,
        **worker_kwargs,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.workers = []
        for worker_id in range(num_workers):
            self.workers.append(
                ray.remote(
                    num_gpus=num_gpus,
                    max_restarts=max_restarts,
                )(Worker).remote(**worker_kwargs)
            )

    def generate(self) -> Generator:
        pools = {worker.get_next.remote(): worker for worker in self.workers}
        while pools:
            done, _ = ray.wait(list(pools.keys()))
            for task in done:
                worker = pools.pop(task)
                episode = ray.get(task)
                if episode is not None:
                    yield episode
                    pools[worker.get_next.remote()] = worker
