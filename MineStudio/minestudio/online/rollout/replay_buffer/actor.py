import asyncio
from collections import defaultdict, deque
from omegaconf import DictConfig
import ray
from typing import Deque, List, Dict, Any, Tuple, cast
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment
from minestudio.online.rollout.replay_buffer.fragment_store import FragmentStore
from minestudio.online.utils.rollout.datatypes import FragmentMetadata
import minestudio.online.utils.train.wandb_logger as wandb_logger
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import logging
import torchmetrics

logger = logging.getLogger("ray")


class FragmentRecord:
    def __init__(
        self,
        fragment_id: str,
        metadata: FragmentMetadata,
        manager,
    ):
        self.fragment_id = fragment_id
        self.metadata = metadata
        self.manager = manager
        self.manager.ref_count[fragment_id] += 1

    def __del__(self):
        self.manager.ref_count[self.fragment_id] -= 1
        if self.manager.ref_count[self.fragment_id] == 0:
            self.manager.clean(self.fragment_id)


class FragmentManager:
    def __init__(self, fragment_store: FragmentStore):
        self.fragment_store = fragment_store
        self.ref_count = defaultdict(int)

    def create_fragment_record(self, fragment_id: str, metadata: FragmentMetadata):
        return FragmentRecord(fragment_id=fragment_id, metadata=metadata, manager=self)

    def clean(self, fragment_id: str):
        del self.ref_count[fragment_id]
        self.fragment_store.delete_fragment(fragment_id)


@dataclass
class ChunkRecord:
    fragment_records: List[FragmentRecord]
    model_version: int
    session_id: str
    use_count: int


@ray.remote
class ReplayBufferActor:
    def __init__(
        self,
        max_chunks: int,
        max_staleness: int,
        max_reuse: int,
        database_config: DictConfig,
        fragments_per_chunk: int,
        fragments_per_report: Optional[int] = None,
    ):
        self.fragments_per_report = fragments_per_report

        self.fragment_added = 0
        self._last_report_time = time.time()

        self.max_reuse = max_reuse
        self.current_model_version = -1
        self.current_session_id = ""

        self.max_staleness = max_staleness
        self.fragments_per_chunk = fragments_per_chunk

        self.database_config = database_config
        self.fragment_store = FragmentStore(**database_config)  # type: ignore
        self.fragment_store.clear()

        self.fragment_manager = FragmentManager(self.fragment_store)

        self.max_chunks = max_chunks
        self.chunks: Deque[ChunkRecord] = deque()

        self.recv_buffer: Dict[str, Deque[FragmentRecord]] = {}
        self.fragments_to_return: List[FragmentRecord] = []

        self.fetch_reqs: List[Tuple[int, asyncio.Event]] = []

        self.reuse_metric = torchmetrics.MeanMetric()
        self.staleness_metric = torchmetrics.MeanMetric()

    def update_training_session(self):
        pass

    def pop_chunk(self) -> None:
        self.reuse_metric.update(self.chunks[0].use_count, self.fragments_per_chunk)
        self.chunks.popleft()

    def add_chunk(self, chunk_record: ChunkRecord) -> None:
        while len(self.chunks) >= self.max_chunks:
            self.pop_chunk()

        if (
            self.current_model_version - chunk_record.model_version <= self.max_staleness
            and self.current_session_id == chunk_record.session_id
        ):
            self.chunks.append(chunk_record)

            new_fetch_reqs = []
            for req, evt in self.fetch_reqs:
                if len(self.chunks) >= req:
                    evt.set()
                else:
                    new_fetch_reqs.append((req, evt))
            self.fetch_reqs = new_fetch_reqs
        else:
            self.reuse_metric.update(chunk_record.use_count, self.fragments_per_chunk)

    async def add_fragment(self, fragment_id: str, metadata: FragmentMetadata):
        fragment_record = self.fragment_manager.create_fragment_record(fragment_id=fragment_id, metadata=metadata)

        worker_uuid = metadata.worker_uuid
        if worker_uuid in self.recv_buffer:
            assert self.recv_buffer[worker_uuid][-1].metadata.fid_in_worker + 1 == metadata.fid_in_worker
            self.recv_buffer[worker_uuid].append(fragment_record)
        else:
            self.recv_buffer[worker_uuid] = deque([fragment_record])

        while len(self.recv_buffer[worker_uuid]) > 0 and (
            self.current_model_version - self.recv_buffer[worker_uuid][0].metadata.model_version > self.max_staleness
            or self.current_session_id != self.recv_buffer[worker_uuid][0].metadata.session_id
        ):
            self.reuse_metric.update(0.0, self.fragments_per_chunk)

            self.recv_buffer[worker_uuid].popleft()

        if len(self.recv_buffer[worker_uuid]) == 0:
            self.recv_buffer.pop(worker_uuid)
        elif len(self.recv_buffer[worker_uuid]) >= self.fragments_per_chunk:
            fragment_records = list(self.recv_buffer[worker_uuid])
            chunk_record = ChunkRecord(
                fragment_records=fragment_records,
                model_version=fragment_records[0].metadata.model_version,
                session_id=fragment_records[0].metadata.session_id,
                use_count=0,
            )
            self.recv_buffer.pop(worker_uuid)
            self.add_chunk(chunk_record)

        self.fragment_added += 1
        now = time.time()
        if self.fragments_per_report and self.fragment_added % self.fragments_per_report == 0:
            info = {
                "replay_buffer/fragments_per_second": self.fragments_per_report / (now - self._last_report_time),
                "replay_buffer/replay_buffer_size (chunk)": len(self.chunks),
                "replay_buffer/model_version": self.current_model_version,
            }
            wandb_logger.log(info)
            self._last_report_time = now

    async def update_model_version(self, session_id: str, model_version: int) -> None:
        self.current_model_version = model_version
        self.current_session_id = session_id
        while (
            len(self.chunks) > 0
            and self.chunks[0].model_version < self.current_model_version - self.max_staleness
            or len(self.chunks) > 0
            and self.chunks[0].session_id != self.current_session_id
        ):
            self.pop_chunk()

    async def fetch_fragments(self, num_fragments: int) -> List[Tuple[FragmentIndex, str]]:
        assert num_fragments % self.fragments_per_chunk == 0
        num_chunks = num_fragments // self.fragments_per_chunk
        assert num_chunks <= self.max_chunks

        while len(self.chunks) < num_chunks:
            evt = asyncio.Event()
            self.fetch_reqs.append((num_chunks, evt))
            await evt.wait()

        selected_idxs = np.random.choice(list(range(len(self.chunks))), num_chunks, replace=False)

        chunks_list: List[Optional[ChunkRecord]] = list(self.chunks)
        self.fragments_to_return = []
        for idx in selected_idxs:
            assert chunks_list[idx] is not None
            self.fragments_to_return += chunks_list[idx].fragment_records
            chunks_list[idx].use_count += 1
            if chunks_list[idx].use_count >= self.max_reuse:
                self.reuse_metric.update(chunks_list[idx].use_count, self.fragments_per_chunk)
                self.staleness_metric.update(
                    self.current_model_version - chunks_list[idx].model_version, self.fragments_per_chunk
                )
                chunks_list[idx] = None
        self.chunks = deque([chunk for chunk in chunks_list if chunk is not None])

        wandb_logger.log(
            {
                "replay_buffer/avg_reuse": self.reuse_metric.compute(),
                "replay_buffer/avg_staleness": self.staleness_metric.compute(),
            }
        )
        self.reuse_metric.reset()
        self.staleness_metric.reset()

        return [
            (
                FragmentIndex(
                    worker_uuid=fragment_record.metadata.worker_uuid,
                    fid_in_worker=fragment_record.metadata.fid_in_worker,
                ),
                fragment_record.fragment_id,
            )
            for fragment_record in self.fragments_to_return
        ]

    async def get_database_config(self):
        return self.database_config

    async def prepared_fragments(self) -> List[Tuple[FragmentIndex, str]]:
        return [
            (
                FragmentIndex(
                    worker_uuid=fragment_record.metadata.worker_uuid,
                    fid_in_worker=fragment_record.metadata.fid_in_worker,
                ),
                fragment_record.fragment_id,
            )
            for fragment_record in self.fragments_to_return
        ]
