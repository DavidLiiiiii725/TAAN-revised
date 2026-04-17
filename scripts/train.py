"""
Training entry point for TAAN-GRPO.

Usage:
    # Single GPU
    python scripts/train.py --config configs/qwen3_4b_multitask.yaml

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/qwen3_4b_multitask.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

# Allow importing from the repository root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.multitask_dataset import MultitaskDataset
from src.data.task_sampler import TypeAwareBatchSampler
from src.grpo.trainer import GRPOTAANTrainer, TrainingConfig
from src.utils.logging import MetricsLogger, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAAN-GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_4b_multitask.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config keys, e.g. --override lr=2e-6 batch_size=64",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r") as f:
        cfg: dict = yaml.safe_load(f)

    for kv in args.override:
        key, _, value_str = kv.partition("=")
        # Attempt numeric conversion
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
        cfg[key] = value
        logger.info("Config override: %s = %s", key, value)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg_dict = load_config(args)

    setup_logging(
        level=cfg_dict.get("log_level", "INFO"),
        wandb_project=cfg_dict.get("wandb_project"),
        wandb_run_name=cfg_dict.get("wandb_run_name"),
        config=cfg_dict,
    )

    config = TrainingConfig.from_dict(cfg_dict)
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────────
    task_types_cfg = cfg_dict.get("task_types", {})
    logger.info("Loading datasets…")
    dataset = MultitaskDataset.from_config(task_types_cfg, seed=config.seed)
    logger.info("Dataset loaded: %d total samples", len(dataset))
    logger.info("Type distribution: %s", dataset.type_counts())

    sampler = TypeAwareBatchSampler(
        dataset,
        batch_size=config.batch_size,
        min_per_type=10,
        seed=config.seed,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    logger.info("Initialising trainer…")
    trainer = GRPOTAANTrainer(config)

    metrics_logger = MetricsLogger(log_every=config.log_every)

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info("Starting training for %d steps.", config.max_steps)
    step = trainer.global_step

    for batch_indices in sampler:
        if step >= config.max_steps:
            break

        # Collate batch from dataset
        batch_samples = [dataset[i] for i in batch_indices]
        batch = {
            "prompts": [s.prompt for s in batch_samples],
            "type_ids": [s.type_id for s in batch_samples],
            "ground_truths": [s.ground_truth for s in batch_samples],
            "reward_fn_names": [s.reward_fn_name for s in batch_samples],
        }

        metrics = trainer.train_step(batch)
        metrics_logger.log(step, metrics)

        # Periodic evaluation
        if step % config.eval_every == 0 and step > 0:
            eval_results = trainer.evaluate()
            if eval_results:
                logger.info("Eval at step %d: %s", step, eval_results)

        # Checkpoint
        if step % config.save_every == 0 and step > 0:
            trainer.save_checkpoint(step)

        step = trainer.global_step

    # Final checkpoint
    trainer.save_checkpoint(step)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
