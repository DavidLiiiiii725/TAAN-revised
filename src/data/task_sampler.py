"""
Type-aware batch sampler for multi-task GRPO training.

Guarantees that each batch contains at least ``min_per_type`` samples for
every task type present in the dataset.  When a type has fewer samples than
``min_per_type`` in a batch, it is excluded from TAAN normalisation (pure
GRPO fallback handled externally).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler

from .multitask_dataset import MultitaskDataset


class TypeAwareBatchSampler(Sampler):
    """Batch sampler that enforces per-type minimum sample counts.

    Algorithm per batch:
    1. For each type, maintain a shuffled index queue.
    2. Pull ``quota[type]`` indices from each type's queue.
    3. Fill remaining slots proportionally from all types.
    4. Shuffle the combined batch.

    Args:
        dataset: :class:`MultitaskDataset` instance.
        batch_size: Total number of samples per batch.
        min_per_type: Minimum samples required per type in each batch.
            Types with fewer available samples are included at their maximum.
        drop_last: If True, drop the final incomplete batch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: MultitaskDataset,
        batch_size: int,
        min_per_type: int = 10,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_per_type = min_per_type
        self.drop_last = drop_last
        self.seed = seed

        # Build {type_id: [indices]} mapping
        self._type_indices: Dict[str, List[int]] = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            self._type_indices[sample.type_id].append(i)

        self._type_ids: List[str] = sorted(self._type_indices.keys())
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each type
        type_queues: Dict[str, List[int]] = {
            tid: self._shuffled(indices)
            for tid, indices in self._type_indices.items()
        }
        # Pointers into each queue
        pointers: Dict[str, int] = {tid: 0 for tid in self._type_ids}

        total = len(self.dataset)
        steps = total // self.batch_size
        if not self.drop_last and total % self.batch_size:
            steps += 1

        for _ in range(steps):
            batch: List[int] = []

            # 1. Guaranteed minimum per type
            for tid in self._type_ids:
                q = type_queues[tid]
                n_available = len(q)
                n_take = min(self.min_per_type, n_available)
                ptr = pointers[tid]
                for _ in range(n_take):
                    batch.append(q[ptr % n_available])
                    ptr += 1
                pointers[tid] = ptr

            # 2. Fill remaining slots round-robin across all types
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                # Proportional fill based on type weight (equal by default)
                all_indices = [
                    q[pointers[tid] % len(q)]
                    for tid in self._type_ids
                    for q in [type_queues[tid]]
                ]
                # Cycle through types to fill remaining slots
                tid_cycle = self._type_ids * (remaining // len(self._type_ids) + 1)
                for tid in tid_cycle[:remaining]:
                    q = type_queues[tid]
                    ptr = pointers[tid]
                    batch.append(q[ptr % len(q)])
                    pointers[tid] = ptr + 1

            # Trim to exact batch_size
            batch = batch[: self.batch_size]
            self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shuffled(self, indices: List[int]) -> List[int]:
        shuffled = list(indices)
        self._rng.shuffle(shuffled)
        return shuffled

    def type_counts_in_batch(self, batch_indices: List[int]) -> Dict[str, int]:
        """Return {type_id: count} for a given list of sample indices."""
        counts: Dict[str, int] = defaultdict(int)
        for idx in batch_indices:
            counts[self.dataset.samples[idx].type_id] += 1
        return dict(counts)
