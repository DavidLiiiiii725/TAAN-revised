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
from collections import defaultdict

import yaml

# Allow importing from the repository root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.multitask_dataset import MultitaskDataset
from src.data.task_sampler import TypeAwareBatchSampler
from src.grpo.trainer import GRPOTAANTrainer, TrainingConfig
from src.rewards import CodeReward, MathReward, ModelReward
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


def build_reward_registry(task_types_cfg: dict) -> dict:
    registry = {}

    reward_factories = {
        "math_reward": MathReward,
        "code_reward": CodeReward,
        "model_reward": ModelReward,
    }

    for type_id, type_cfg in task_types_cfg.items():
        reward_name = type_cfg.get("reward", "")
        reward_cfg = type_cfg.get("reward_config", {})
        factory = reward_factories.get(reward_name)
        if factory is None:
            logger.warning("Unknown reward '%s' for type '%s'", reward_name, type_id)
            continue
        reward_fn = factory(**reward_cfg)
        # Keep both lookups because both GRPOTAANTrainer.train_step and
        # evaluate_on_validation first check reward function name and then
        # fall back to task type id.
        registry[type_id] = reward_fn
        registry[reward_name] = reward_fn

    return registry


def evaluate_on_validation(
    trainer: GRPOTAANTrainer,
    val_dataset: MultitaskDataset,
    max_samples: int = 128,
) -> dict:
    if len(val_dataset) == 0:
        return {}

    samples = val_dataset.samples[:max_samples]
    prompts = [s.prompt for s in samples]
    rollout = trainer.rollout_manager.generate(
        prompts,
        G=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=trainer.config.max_gen_tokens,
    )

    per_type_scores = defaultdict(list)
    for sample, response_list in zip(samples, rollout.responses):
        if not response_list:
            continue
        reward_fn = trainer.reward_registry.get(sample.reward_fn_name) or trainer.reward_registry.get(sample.type_id)
        if reward_fn is None:
            continue
        score = float(reward_fn(response_list[0], sample.ground_truth))
        per_type_scores[sample.type_id].append(score)

    if not per_type_scores:
        return {}

    per_type_mean = {k: sum(v) / len(v) for k, v in per_type_scores.items() if v}
    total_scores = sum(sum(v) for v in per_type_scores.values())
    total_count = sum(len(v) for v in per_type_scores.values())
    overall = total_scores / total_count
    return {"validation_mean_reward": overall, "validation_per_type": per_type_mean}


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
    validation_ratio = float(cfg_dict.get("validation_split_ratio", 0.1))
    logger.info("Loading datasets…")
    train_dataset = MultitaskDataset.from_config(
        task_types_cfg,
        split="train",
        validation_ratio=validation_ratio,
        seed=config.seed,
    )
    val_dataset = MultitaskDataset.from_config(
        task_types_cfg,
        split="validation",
        validation_ratio=validation_ratio,
        seed=config.seed,
    )
    logger.info("Train dataset loaded: %d total samples", len(train_dataset))
    logger.info("Train type distribution: %s", train_dataset.type_counts())
    logger.info("Validation dataset loaded: %d total samples", len(val_dataset))
    logger.info("Validation type distribution: %s", val_dataset.type_counts())

    sampler = TypeAwareBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        min_per_type=10,
        seed=config.seed,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    logger.info("Initialising trainer…")
    reward_registry = build_reward_registry(task_types_cfg)
    trainer = GRPOTAANTrainer(config, reward_registry=reward_registry)

    metrics_logger = MetricsLogger(log_every=config.log_every)

    # ── Training loop ────────────────────────────────────────────────────────
    logger.info("Starting training for %d steps.", config.max_steps)
    step = trainer.global_step

    for batch_indices in sampler:
        if step >= config.max_steps:
            break

        # Collate batch from dataset
        batch_samples = [train_dataset[i] for i in batch_indices]
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
            eval_results = evaluate_on_validation(trainer, val_dataset)
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
