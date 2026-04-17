"""
Multi-task mixed dataset for TAAN-GRPO training.

Each sample is a :class:`TaskSample` that carries:
- ``prompt``: the instruction / question text
- ``type_id``: task type identifier (e.g. "math", "code", "writing", "chat")
- ``ground_truth``: reference answer for rule-based rewards (None for model rewards)
- ``reward_fn_name``: which reward function to apply

Data sources (HuggingFace datasets):
    math    → openai/gsm8k, hendrycks/competition_math
    code    → openai/humaneval
    writing → tatsu-lab/alpaca_eval
    chat    → lmsys/chatbot_arena_conversations
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        split: str = "train",
        validation_ratio: float = 0.1,
        max_samples_per_type: Optional[int] = None,
        seed: int = 42,
    ) -> "MultitaskDataset":
        """Build a dataset by loading HuggingFace datasets as declared in config.

        Args:
            task_types_config: Dict mapping ``type_id → {datasets: [...], reward: ...}``.
            split: Requested split, one of ``train`` or ``validation``.
            validation_ratio: Fallback validation ratio when no dedicated split
                exists for a dataset.
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

        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'validation'.")

        all_samples: List[TaskSample] = []

        for type_id, cfg in task_types_config.items():
            reward_fn_name = cfg.get("reward", "")
            dataset_entries: List[Any] = cfg.get("datasets", [])

            for ds_entry in dataset_entries:
                ds_name, source_split, use_holdout_split = cls._resolve_dataset_entry(ds_entry, split)
                try:
                    samples = cls._load_hf_dataset(
                        ds_name,
                        source_split=source_split,
                        requested_split=split,
                        use_holdout_split=use_holdout_split,
                        validation_ratio=validation_ratio,
                        type_id=type_id,
                        reward_fn_name=reward_fn_name,
                        max_samples=max_samples_per_type,
                        seed=seed,
                    )
                    all_samples.extend(samples)
                    logger.info(
                        "Loaded %d samples from %s:%s (requested=%s, type=%s)",
                        len(samples), ds_name, source_split, split, type_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load dataset '%s:%s' for split '%s': %s",
                        ds_name,
                        source_split,
                        split,
                        exc,
                    )

        if not all_samples:
            logger.warning("No samples loaded — dataset is empty.")

        return cls(all_samples)

    @staticmethod
    def _resolve_dataset_entry(ds_entry: Any, split: str) -> Tuple[str, str, bool]:
        """Return dataset name, source split, and whether holdout split is needed."""
        if isinstance(ds_entry, str):
            # Backward compatible default: use train split and fallback slicing.
            source_split = "train"
            return ds_entry, source_split, True

        if isinstance(ds_entry, dict):
            ds_name = ds_entry.get("name")
            if not ds_name:
                raise ValueError("Dataset entry dict must include a 'name' field.")
            split_key = "train_split" if split == "train" else "validation_split"
            source_split = ds_entry.get(split_key)
            use_holdout_split = False
            if source_split is None:
                source_split = ds_entry.get("split")
                if source_split is not None:
                    use_holdout_split = True
            if source_split is None:
                source_split = "train"
                use_holdout_split = True
                logger.warning(
                    "Dataset '%s' missing %s/split. Falling back to split='train'.",
                    ds_name,
                    split_key,
                )
            return ds_name, source_split, use_holdout_split

        raise TypeError("Dataset entry must be either a string or a mapping.")

    @staticmethod
    def _load_hf_dataset(
        ds_name: str,
        source_split: str,
        requested_split: str,
        use_holdout_split: bool,
        validation_ratio: float,
        type_id: str,
        reward_fn_name: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> List[TaskSample]:
        """Load and convert a single HuggingFace dataset to TaskSample list."""
        from datasets import load_dataset  # type: ignore[import]

        # Dataset-specific loading logic
        if ds_name == "openai/gsm8k":
            ds = load_dataset(ds_name, "main", split=source_split)
            samples = [
                TaskSample(
                    prompt=row["question"],
                    type_id=type_id,
                    ground_truth=row["answer"],
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        elif ds_name == "hendrycks/competition_math":
            ds = load_dataset(ds_name, split=source_split)
            samples = [
                TaskSample(
                    prompt=row["problem"],
                    type_id=type_id,
                    ground_truth=row.get("solution", ""),
                    reward_fn_name=reward_fn_name,
                )
                for row in ds
            ]

        elif ds_name == "google-research-datasets/mbpp":
            ds = load_dataset(ds_name, split=source_split)
            samples = []
            for row in ds:
                prompt = row.get("text", "")
                tests = row.get("test_list", [])
                test_blob = "\n".join(tests) if isinstance(tests, list) else str(tests)
                if not prompt:
                    continue
                samples.append(
                    TaskSample(
                        prompt=prompt,
                        type_id=type_id,
                        ground_truth=test_blob,
                        reward_fn_name=reward_fn_name,
                    )
                )

        elif ds_name == "openai/humaneval":
            ds = load_dataset(ds_name, split=source_split)
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
            ds = load_dataset(ds_name, split=source_split)
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
            ds = load_dataset(ds_name, split=source_split)
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
            ds = load_dataset(ds_name, split=source_split)
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

        # If caller requested validation but config only supplied train split,
        # create deterministic split from the same source split.
        if requested_split == "validation" and use_holdout_split:
            samples = MultitaskDataset._select_holdout_samples(
                samples=samples,
                validation_ratio=validation_ratio,
                seed=seed,
                keep_validation=True,
            )
        elif requested_split == "train" and use_holdout_split:
            samples = MultitaskDataset._select_holdout_samples(
                samples=samples,
                validation_ratio=validation_ratio,
                seed=seed,
                keep_validation=False,
            )

        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(seed)
            samples = rng.sample(samples, max_samples)

        return samples

    @staticmethod
    def _select_holdout_samples(
        samples: List[TaskSample],
        validation_ratio: float,
        seed: int,
        keep_validation: bool,
    ) -> List[TaskSample]:
        """Deterministically split train source into train/validation subsets."""
        val_count = int(len(samples) * validation_ratio)
        if keep_validation and samples and val_count <= 0:
            val_count = 1
        if val_count <= 0 and not keep_validation:
            return [] if keep_validation else samples

        rng = random.Random(seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        held_out = set(indices[:val_count])
        if keep_validation:
            return [s for i, s in enumerate(samples) if i in held_out]
        return [s for i, s in enumerate(samples) if i not in held_out]

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
