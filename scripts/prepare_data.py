"""
Data preparation script.

Downloads and preprocesses the HuggingFace datasets required for TAAN-GRPO
training and saves them as a local cache of :class:`TaskSample` lists.

Usage (shell)::

    python scripts/prepare_data.py
        --config configs/qwen3_4b_multitask.yaml
        --output_dir data/processed
        --max_samples_per_type 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.multitask_dataset import MultitaskDataset
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TAAN-GRPO training data")
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_multitask.yaml",
                        help="Training config YAML containing task_types.")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed JSON files.")
    parser.add_argument("--max_samples_per_type", type=int, default=None,
                        help="Cap on samples per task type (for quick testing).")
    parser.add_argument("--validation_split_ratio", type=float, default=0.1,
                        help="Fallback validation ratio when dataset has no validation split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    task_types_cfg = cfg_dict.get("task_types", {})
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Downloading and processing datasets…")
    train_dataset = MultitaskDataset.from_config(
        task_types_cfg,
        split="train",
        validation_ratio=args.validation_split_ratio,
        max_samples_per_type=args.max_samples_per_type,
        seed=args.seed,
    )
    val_dataset = MultitaskDataset.from_config(
        task_types_cfg,
        split="validation",
        validation_ratio=args.validation_split_ratio,
        max_samples_per_type=args.max_samples_per_type,
        seed=args.seed,
    )

    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Train type distribution: %s", train_dataset.type_counts())
    logger.info("Validation samples: %d", len(val_dataset))
    logger.info("Validation type distribution: %s", val_dataset.type_counts())

    # Save per-type JSON files for each split
    for split_name, split_dataset in [("train", train_dataset), ("validation", val_dataset)]:
        type_counts = split_dataset.type_counts()
        for type_id in type_counts:
            subset = split_dataset.filter_by_type(type_id)
            out_path = os.path.join(args.output_dir, f"{split_name}_{type_id}.json")
            with open(out_path, "w") as f:
                json.dump([s.to_dict() for s in subset.samples], f, indent=2)
            logger.info("Saved %d %s samples to %s", len(subset), type_id, out_path)

    # Save combined files
    train_path = os.path.join(args.output_dir, "train_all_tasks.json")
    with open(train_path, "w") as f:
        json.dump([s.to_dict() for s in train_dataset.samples], f, indent=2)
    logger.info("Combined train dataset saved to %s", train_path)

    val_path = os.path.join(args.output_dir, "validation_all_tasks.json")
    with open(val_path, "w") as f:
        json.dump([s.to_dict() for s in val_dataset.samples], f, indent=2)
    logger.info("Combined validation dataset saved to %s", val_path)

    # Backward-compatible combined file (train split)
    combined_path = os.path.join(args.output_dir, "all_tasks.json")
    with open(combined_path, "w") as f:
        json.dump([s.to_dict() for s in train_dataset.samples], f, indent=2)
    logger.info("Combined dataset (train) saved to %s", combined_path)


if __name__ == "__main__":
    main()
