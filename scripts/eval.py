"""
Evaluation script for TAAN-GRPO.

Usage:
    python scripts/eval.py \\
        --checkpoint outputs/qwen3-4b-multitask/step-1000 \\
        --config configs/qwen3_4b_multitask.yaml \\
        --task math
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grpo.trainer import GRPOTAANTrainer, TrainingConfig
from src.rewards.math_reward import MathReward
from src.rewards.code_reward import CodeReward
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REWARD_MAP = {
    "math": MathReward(),
    "code": CodeReward(),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TAAN-GRPO checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved checkpoint directory.")
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_multitask.yaml",
                        help="Training config YAML (for model_name etc.).")
    parser.add_argument("--task", type=str, default="math",
                        choices=list(REWARD_MAP.keys()),
                        help="Task type to evaluate on.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of evaluation samples.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="JSON file to save results.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    trainer: GRPOTAANTrainer,
    task: str,
    num_samples: int = 100,
) -> dict:
    """Run greedy-decoded evaluation on a small held-out set."""
    reward_fn = REWARD_MAP.get(task)
    if reward_fn is None:
        logger.warning("No reward function for task '%s'. Skipping.", task)
        return {}

    # Load a small eval set (GSM8K test for math, HumanEval for code)
    try:
        from datasets import load_dataset  # type: ignore[import]

        if task == "math":
            ds = load_dataset("openai/gsm8k", "main", split="test")
            samples = [(row["question"], row["answer"]) for row in ds]
        elif task == "code":
            ds = load_dataset("openai/humaneval", split="test")
            samples = [(row["prompt"], row.get("canonical_solution", "")) for row in ds]
        else:
            logger.warning("No eval dataset configured for task '%s'.", task)
            return {}
    except Exception as exc:
        logger.warning("Could not load eval dataset: %s", exc)
        return {}

    samples = samples[:num_samples]
    prompts = [p for p, _ in samples]
    ground_truths = [g for _, g in samples]

    # Greedy rollout (G=1, low temperature)
    rollout = trainer.rollout_manager.generate(
        prompts, G=1, temperature=0.0, top_p=1.0, max_tokens=512
    )

    correct = 0
    for i, (resp_list, gt) in enumerate(zip(rollout.responses, ground_truths)):
        score = reward_fn(resp_list[0], gt)
        if score >= 1.0:
            correct += 1

    accuracy = correct / len(samples) if samples else 0.0
    results = {"task": task, "accuracy": accuracy, "n_samples": len(samples)}
    logger.info("Eval results: %s", results)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    setup_logging()

    # Override model checkpoint path
    cfg_dict["resume_from"] = args.checkpoint
    config = TrainingConfig.from_dict(cfg_dict)

    trainer = GRPOTAANTrainer(config)
    results = evaluate(trainer, task=args.task, num_samples=args.num_samples)

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
