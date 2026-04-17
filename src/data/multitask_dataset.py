"""
Multi-task mixed dataset for TAAN-GRPO training.

Each sample is a :class:`TaskSample` that carries:
- ``prompt``: the instruction / question text
- ``type_id``: task type identifier (e.g. "math", "code", "writing", "chat")
- ``ground_truth``: reference answer for rule-based rewards (None for model rewards)
- ``reward_fn_name``: which reward function to apply

Data sources (HuggingFace datasets):
    math    → openai/gsm8k, hendrycks/math
    code    → openai/humaneval
    writing → tatsu-lab/alpaca_eval
    chat    → lmsys/chatbot_arena_conversations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TaskSample:
    """A single training sample."""

    prompt: str
    type_id: str
    ground_truth: Optional[str] = None
    reward_fn_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "type_id": self.type_id,
            "ground_truth": self.ground_truth,
            "reward_fn_name": self.reward_fn_name,
        }


class MultitaskDataset(Dataset):
    """Concatenated multi-task dataset that mixes samples from several sources.

    Args:
        samples: Pre-built list of :class:`TaskSample` objects.  Either pass
            this directly or use :meth:`from_config` to auto-load from
            HuggingFace datasets.
        transform: Optional callable applied to each sample at access time.
    """

    def __init__(
        self,
        samples: List[TaskSample],
        transform: Optional[Callable[[TaskSample], TaskSample]] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TaskSample:
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        task_types_config: Dict[str, Any],
        max_samples_per_type: Optional[int] = None,
        seed: int = 42,
    ) -> "MultitaskDataset":
        """Build a dataset by loading HuggingFace datasets as declared in config.

        Args:
            task_types_config: Dict mapping ``type_id → {datasets: [...], reward: ...}``.
            max_samples_per_type: Cap on samples loaded per type (for quick testing).
            seed: Random seed for shuffling.

        Returns:
            :class:`MultitaskDataset` with all samples concatenated.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required. "
                "Install it with: pip install datasets"
            ) from exc

        all_samples: List[TaskSample] = []

        for type_id, cfg in task_types_config.items():
            reward_fn_name = cfg.get("reward", "")
            dataset_names: List[str] = cfg.get("datasets", [])

            for ds_name in dataset_names:
                try:
                    samples = cls._load_hf_dataset(
                        ds_name,
                        type_id=type_id,
                        reward_fn_name=reward_fn_name,
                        max_samples=max_samples_per_type,
                        seed=seed,
                    )
                    all_samples.extend(samples)
                    logger.info(
                        "Loaded %d samples from %s (type=%s)",
                        len(samples), ds_name, type_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load dataset '%s': %s", ds_name, exc
                    )

        if not all_samples:
            logger.warning("No samples loaded — dataset is empty.")

        return cls(all_samples)

    @staticmethod
    def _load_hf_dataset(
        ds_name: str,
        type_id: str,
        reward_fn_name: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> List[TaskSample]:
        """Load and convert a single HuggingFace dataset to TaskSample list."""
        from datasets import load_dataset  # type: ignore[import]

        # Dataset-specific loading logic
        if ds_name == "openai/gsm8k":
            ds = load_dataset(ds_name, "main", split="train")
            samples = [
                TaskSample(
                    prompt=row["question"],
                    type_id=type_id,
                    ground_truth=row["answer"],
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        elif ds_name == "openai/humaneval":
            ds = load_dataset(ds_name, split="test")
            samples = [
                TaskSample(
                    prompt=row["prompt"],
                    type_id=type_id,
                    ground_truth=row.get("canonical_solution", ""),
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        elif ds_name == "tatsu-lab/alpaca_eval":
            ds = load_dataset(ds_name, split="eval")
            samples = [
                TaskSample(
                    prompt=row["instruction"],
                    type_id=type_id,
                    ground_truth=None,
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        elif ds_name == "lmsys/chatbot_arena_conversations":
            ds = load_dataset(ds_name, split="train")
            samples = []
            for row in ds:
                convo = row.get("conversation_a", [])
                if convo and convo[0].get("role") == "user":
                    samples.append(
                        TaskSample(
                            prompt=convo[0]["content"],
                            type_id=type_id,
                            ground_truth=None,
                            reward_fn_name=reward_fn_name,
                        )
                    )

        else:
            # Generic fallback: assume columns "prompt"/"question" and optional "answer"
            ds = load_dataset(ds_name, split="train")
            prompt_col = "prompt" if "prompt" in ds.column_names else "question"
            gt_col = "answer" if "answer" in ds.column_names else None
            samples = [
                TaskSample(
                    prompt=row[prompt_col],
                    type_id=type_id,
                    ground_truth=row[gt_col] if gt_col else None,
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        if max_samples is not None and len(samples) > max_samples:
            import random
            rng = random.Random(seed)
            samples = rng.sample(samples, max_samples)

        return samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def type_counts(self) -> Dict[str, int]:
        """Return {type_id: count} for all samples."""
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s.type_id] = counts.get(s.type_id, 0) + 1
        return counts

    def filter_by_type(self, type_id: str) -> "MultitaskDataset":
        """Return a new dataset containing only samples of *type_id*."""
        return MultitaskDataset(
            [s for s in self.samples if s.type_id == type_id],
            self.transform,
        )
