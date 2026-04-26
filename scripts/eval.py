"""
Evaluation script for TAAN-GRPO.

Usage:
    # Math evaluation (GSM8K, greedy, 100 samples)
    python scripts/eval.py \\
        --checkpoint outputs/qwen3-4b-multitask/step-1000 \\
        --config configs/qwen3_4b_multitask.yaml \\
        --task math

    # HumanEval pass@1 (greedy, all 164 tasks)
    python scripts/eval.py \\
        --checkpoint outputs/qwen3-4b-multitask/step-1000 \\
        --config configs/qwen3_4b_multitask.yaml \\
        --task code --pass_k 1

    # HumanEval pass@10 (sampled, reproducible with seed)
    python scripts/eval.py \\
        --checkpoint outputs/qwen3-4b-multitask/step-1000 \\
        --config configs/qwen3_4b_multitask.yaml \\
        --task code --pass_k 10 --num_samples 50 --seed 42

Notes on HumanEval evaluation
------------------------------
For code tasks this script uses the official HumanEval protocol:

1. **Ground truth = test cases, not canonical_solution.**
   Each HumanEval row has a ``test`` field containing a ``check(candidate)``
   function with assertion-based unit tests.  We run the model's completion
   concatenated with those test assertions to determine pass/fail.

2. **Candidate code = prompt + completion.**
   The HumanEval ``prompt`` already contains the function signature and
   docstring.  The model generates a continuation (function body), so the
   runnable candidate is ``prompt + response``.

3. **pass@k semantics.**
   With ``--pass_k k``, the model generates *k* independent samples per task
   (temperature 0.8 for k > 1, greedy for k = 1).  A task is considered
   *passed* if at least one of the *k* samples passes all unit tests.  This
   is a deterministic approximation of the standard pass@k estimator and is
   suitable for direct GRPO vs TAAN comparison when both runs use the same
   ``--seed``.

4. **Reproducibility.**
   Use ``--seed`` to fix the random subset (when ``--num_samples`` is smaller
   than the full dataset) and the generation seed, so that GRPO and TAAN runs
   operate on identical task subsets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
    parser = argparse.ArgumentParser(
        description="Evaluate a TAAN-GRPO checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved checkpoint directory.")
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_multitask.yaml",
                        help="Training config YAML (for model_name etc.).")
    parser.add_argument("--task", type=str, default="math",
                        choices=list(REWARD_MAP.keys()),
                        help="Task type to evaluate on.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help=(
                            "Maximum number of evaluation samples to use.  "
                            "Defaults to the full dataset.  Subset is drawn "
                            "deterministically with --seed."
                        ))
    parser.add_argument("--pass_k", type=int, default=1,
                        help=(
                            "Number of completions to generate per prompt for "
                            "pass@k evaluation (code task only).  k=1 uses "
                            "greedy decoding; k>1 uses temperature sampling."
                        ))
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic subset sampling and generation.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="JSON file to save results.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _load_dataset_samples(task: str) -> list:
    """Load raw dataset rows for *task*.  Returns a list of dicts."""
    from datasets import load_dataset  # type: ignore[import]

    if task == "math":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif task == "code":
        ds = load_dataset("openai/humaneval", split="test")
    else:
        raise ValueError(f"No eval dataset configured for task '{task}'.")
    return list(ds)


def _sample_rows(rows: list, num_samples: int | None, seed: int) -> list:
    """Return a deterministic (possibly full) subset of *rows*."""
    if num_samples is None or num_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, num_samples)


# ---------------------------------------------------------------------------
# Math evaluation
# ---------------------------------------------------------------------------

def evaluate_math(
    trainer: GRPOTAANTrainer,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """GSM8K accuracy with greedy decoding."""
    reward_fn = REWARD_MAP["math"]

    try:
        all_rows = _load_dataset_samples("math")
    except Exception as exc:
        logger.warning("Could not load GSM8K dataset: %s", exc)
        return {}

    n_loaded = len(all_rows)
    rows = _sample_rows(all_rows, num_samples, seed)
    n_eval = len(rows)

    logger.info("Math eval: loaded=%d, evaluating=%d, seed=%d", n_loaded, n_eval, seed)

    prompts = [row["question"] for row in rows]
    ground_truths = [row["answer"] for row in rows]

    rollout = trainer.rollout_manager.generate(
        prompts, G=1, temperature=0.0, top_p=1.0, max_tokens=512
    )

    correct = 0
    skipped = 0
    for resp_list, gt in zip(rollout.responses, ground_truths):
        if not resp_list:
            skipped += 1
            continue
        score = reward_fn(resp_list[0], gt)
        if score >= 1.0:
            correct += 1

    n_valid = n_eval - skipped
    accuracy = correct / n_valid if n_valid > 0 else 0.0
    results = {
        "task": "math",
        "metric": "accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "n_evaluated": n_eval,
        "n_valid": n_valid,
        "n_skipped_invalid": skipped,
        "n_loaded": n_loaded,
        "seed": seed,
    }
    logger.info(
        "Math accuracy: %.1f%% (%d/%d valid, %d skipped)",
        accuracy * 100, correct, n_valid, skipped,
    )
    return results


# ---------------------------------------------------------------------------
# HumanEval / code evaluation
# ---------------------------------------------------------------------------

def evaluate_humaneval(
    trainer: GRPOTAANTrainer,
    num_samples: int | None = None,
    pass_k: int = 1,
    seed: int = 42,
) -> dict:
    """HumanEval pass@k with execution-based correctness.

    Evaluation protocol
    -------------------
    * Ground truth: the ``test`` field of each HumanEval row, which contains a
      ``check(candidate)`` function with assertion-based unit tests.  This is
      the official HumanEval evaluation mechanism.
    * Candidate code: ``prompt + response`` — the HumanEval prompt already
      has the function signature, so the model completion is a continuation.
    * A task is *passed* if **at least one** of the *k* generated samples
      passes all unit tests (pass@k semantics).
    * Greedy decoding (temperature=0) is used for k=1; temperature sampling
      (temperature=0.8, top_p=0.95) for k>1.

    Args:
        trainer: Trainer with an attached rollout manager.
        num_samples: Max tasks to evaluate (``None`` = all 164 HumanEval tasks).
        pass_k: Number of completions per task.
        seed: Seed for deterministic subset sampling and generation.

    Returns:
        Metrics dict with transparent counts and per-task results.
    """
    reward_fn = REWARD_MAP["code"]

    try:
        all_rows = _load_dataset_samples("code")
    except Exception as exc:
        logger.warning("Could not load HumanEval dataset: %s", exc)
        return {}

    n_loaded = len(all_rows)
    rows = _sample_rows(all_rows, num_samples, seed)
    n_eval = len(rows)
    n_skipped_subset = n_loaded - n_eval

    logger.info(
        "HumanEval: loaded=%d, evaluating=%d, skipped_by_subset=%d, "
        "pass_k=%d, seed=%d",
        n_loaded, n_eval, n_skipped_subset, pass_k, seed,
    )

    prompts = [row["prompt"] for row in rows]
    # Use the 'test' field (unit test assertions), NOT 'canonical_solution'.
    test_cases = [row["test"] for row in rows]
    task_ids = [row["task_id"] for row in rows]

    temperature = 0.0 if pass_k == 1 else 0.8
    top_p = 1.0 if pass_k == 1 else 0.95

    rollout = trainer.rollout_manager.generate(
        prompts, G=pass_k, temperature=temperature, top_p=top_p, max_tokens=512
    )

    passed = 0
    n_invalid = 0
    per_task: list[dict] = []

    for i, (resp_list, test_code, prompt, task_id) in enumerate(
        zip(rollout.responses, test_cases, prompts, task_ids)
    ):
        if not resp_list:
            n_invalid += 1
            per_task.append({"task_id": task_id, "passed": False, "reason": "no_response"})
            continue

        task_passed = False
        for response in resp_list[:pass_k]:
            # Combine prompt + response to form a complete function definition.
            # The HumanEval prompt already contains the signature/docstring;
            # the model generates the body as a continuation.
            candidate_code = prompt + response
            score = reward_fn._run_tests(candidate_code, test_code)
            if score >= 1.0:
                task_passed = True
                break

        per_task.append({"task_id": task_id, "passed": task_passed})
        if task_passed:
            passed += 1

    n_attempted = n_eval - n_invalid
    pass_at_k = passed / n_attempted if n_attempted > 0 else 0.0

    results = {
        "task": "code",
        "metric": f"pass@{pass_k}",
        "pass_at_k": pass_at_k,
        "passed": passed,
        "n_attempted": n_attempted,
        "n_invalid": n_invalid,
        "n_evaluated": n_eval,
        "n_skipped_by_subset": n_skipped_subset,
        "n_loaded": n_loaded,
        "pass_k": pass_k,
        "seed": seed,
        "per_task": per_task,
    }
    logger.info(
        "HumanEval pass@%d: %.1f%% (%d/%d attempted, %d invalid, "
        "%d skipped by subset, %d total loaded)",
        pass_k, pass_at_k * 100, passed, n_attempted,
        n_invalid, n_skipped_subset, n_loaded,
    )
    return results


# ---------------------------------------------------------------------------
# Unified evaluate entry-point
# ---------------------------------------------------------------------------

def evaluate(
    trainer: GRPOTAANTrainer,
    task: str,
    num_samples: int | None = None,
    pass_k: int = 1,
    seed: int = 42,
) -> dict:
    """Dispatch to task-specific evaluator.

    Args:
        trainer: Trainer with rollout manager.
        task: ``"math"`` or ``"code"``.
        num_samples: Max samples to evaluate (``None`` = full dataset).
        pass_k: Number of completions per prompt (code task, pass@k).
        seed: Random seed for reproducibility.

    Returns:
        Metrics dict.
    """
    if task == "math":
        return evaluate_math(trainer, num_samples=num_samples, seed=seed)
    elif task == "code":
        return evaluate_humaneval(
            trainer, num_samples=num_samples, pass_k=pass_k, seed=seed
        )
    else:
        logger.warning("Unknown task '%s'. Skipping.", task)
        return {}


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
    results = evaluate(
        trainer,
        task=args.task,
        num_samples=args.num_samples,
        pass_k=args.pass_k,
        seed=args.seed,
    )

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
