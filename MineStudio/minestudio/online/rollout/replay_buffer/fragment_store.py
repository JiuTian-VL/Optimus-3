import ray
import logging
from diskcache import FanoutCache
from uuid import uuid4
from minestudio.online.utils.rollout.datatypes import SampleFragment


class LocalFragmentStoreImpl:
    def __init__(
        self,
        path: str,
        num_shards: int,
    ):
        self.cache = FanoutCache(path, shards=num_shards, eviction_policy="none")

    def add_fragment(self, fragment: SampleFragment):
        fragment_uuid = str(uuid4())
        self.cache[fragment_uuid] = fragment
        return fragment_uuid

    def get_fragment(self, fragment_uuid: str):
        return self.cache[fragment_uuid]

    def delete_fragment(self, fragment_uuid: str):
        del self.cache[fragment_uuid]

    def clear(self):
        self.cache.clear()

    def get_disk_space(self):
        return self.cache.volume()


@ray.remote(resources={"database": 0.0001})
class RemoteFragmentStoreImpl:
    def __init__(self, **kwargs):
        self.local_impl = LocalFragmentStoreImpl(**kwargs)

    def add_fragment(self, fragment: SampleFragment):
        return self.local_impl.add_fragment(fragment)

    def get_fragment(self, fragment_uuid: str):
        return self.local_impl.get_fragment(fragment_uuid)

    def delete_fragment(self, fragment_uuid: str):
        return self.local_impl.delete_fragment(fragment_uuid)

    def clear(self):
        return self.local_impl.clear()

    def get_disk_space(self):
        return self.local_impl.get_disk_space()


class FragmentStore:
    def __init__(self, **kwargs):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local = None
        for node in ray.nodes():
            if node["NodeID"] == self.node_id:
                resources = node["Resources"]
                if resources.get("database", 0) > 0:
                    self.local = True
                else:
                    logging.warn("Remote fragment store has not been tested yet")
                    self.local = False
                break

        assert self.local is not None

        if not self.local:
            self.remote_impl = RemoteFragmentStoreImpl.remote(**kwargs)
        else:
            self.local_impl = LocalFragmentStoreImpl(**kwargs)

    def add_fragment(self, fragment: SampleFragment):
        if self.local:
            return self.local_impl.add_fragment(fragment)
        else:
            return ray.get(self.remote_impl.add_fragment.remote(fragment))  # type: ignore

    def get_fragment(self, fragment_uuid: str) -> SampleFragment:
        if self.local:
            return self.local_impl.get_fragment(fragment_uuid)  # type: ignore
        else:
            return ray.get(self.remote_impl.get_fragment.remote(fragment_uuid))  # type: ignore

    def delete_fragment(self, fragment_uuid: str):
        if self.local:
            return self.local_impl.delete_fragment(fragment_uuid)
        else:
            return ray.get(self.remote_impl.delete_fragment.remote(fragment_uuid))  # type: ignore

    def clear(self):
        if self.local:
            return self.local_impl.clear()
        else:
            return ray.get(self.remote_impl.clear.remote())  # type: ignore

    def get_disk_space(self):
        if self.local:
            return self.local_impl.get_disk_space()
        else:
            return ray.get(self.remote_impl.get_disk_space.remote())  # type: ignore
