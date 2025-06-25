import ray
from omegaconf import DictConfig
from minestudio.online.rollout.replay_buffer.fragment_store import FragmentStore
from minestudio.online.rollout.replay_buffer.actor import ReplayBufferActor
from minestudio.online.utils.rollout.datatypes import FragmentMetadata
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment
from typing import List, Optional, Tuple


class ReplayBufferInterface:
    def __init__(self, config: Optional[DictConfig] = None):
        existing_actor = None
        try:
            existing_actor = ray.get_actor("replay_buffer")
        except ValueError:
            pass

        if config is not None:
            if existing_actor is not None:
                raise ValueError("Replay buffer already initialized")
            self.actor = ReplayBufferActor.options(name="replay_buffer").remote(**config)  # type: ignore
        else:
            if existing_actor is None:
                raise ValueError("Replay buffer not initialized")
            self.actor = existing_actor

        self.database_config = ray.get(self.actor.get_database_config.remote())
        self.store = FragmentStore(**self.database_config)

    def update_training_session(self):
        return ray.get(self.actor.update_training_session.remote())

    def add_fragment(self, fragment: SampleFragment, metadata: FragmentMetadata):
        fragment_id = self.store.add_fragment(fragment)

        ray.get(
            self.actor.add_fragment.remote(
                fragment_id=fragment_id,
                metadata=metadata,
            )
        )

    def load_fragment(self, fragment_id: str) -> SampleFragment:
        return self.store.get_fragment(fragment_id)

    def fetch_fragments(self, num_fragments: int) -> List[Tuple[FragmentIndex, str]]:
        return ray.get(self.actor.fetch_fragments.remote(num_fragments=num_fragments))

    def prepared_fragments(self) -> List[Tuple[FragmentIndex, str]]:
        return ray.get(self.actor.prepared_fragments.remote())

    def update_model_version(self, session_id: str, model_version: int):
        return ray.get(self.actor.update_model_version.remote(session_id=session_id, model_version=model_version))
