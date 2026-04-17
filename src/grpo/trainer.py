"""
GRPO + TAAN training loop for multi-task RL fine-tuning.

Training loop per step:
1. Sample a batch from the multi-task dataset (each type ≥ min_type_samples).
2. Use vLLM to generate G responses per prompt.
3. Compute rewards using the per-type reward function.
4. Call TAANAdvantageNormalizer to obtain final advantages.
5. Compute GRPO policy loss (clip + KL penalty).
6. Backward pass + optimiser update.

Key hyper-parameters (from config):
    model_name:  "Qwen/Qwen3-4B"
    G:           8   (responses per prompt)
    batch_size:  128 (prompts per step)
    lr:          1e-6
    beta:        0.01 (KL penalty coefficient)
    clip_eps:    0.2  (PPO clip range)
    taan_alpha:  0.99
    taan_clip:   5.0
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from ..taan.advantage import TAANAdvantageNormalizer
from .policy_loss import compute_policy_loss, compute_sequence_log_prob
from .rollout import VLLMRolloutManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Flat configuration dataclass for :class:`GRPOTAANTrainer`."""

    # Model
    model_name: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"

    # Rollout
    G: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_gen_tokens: int = 2048

    # Training
    batch_size: int = 128
    lr: float = 1e-6
    weight_decay: float = 0.01
    max_steps: int = 5000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42

    # GRPO
    clip_eps: float = 0.2
    beta: float = 0.01

    # TAAN
    taan_alpha: float = 0.99
    taan_clip: float = 5.0
    taan_min_samples: int = 20
    taan_eps: float = 1e-8

    # Distributed
    tp_size: int = 1
    dp_size: int = 1
    weight_sync_every: int = 1   # sync vLLM weights every N steps

    # Checkpointing / logging
    output_dir: str = "outputs/taan-grpo"
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 200
    resume_from: Optional[str] = None
    wandb_project: str = "taan-grpo"
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


class GRPOTAANTrainer:
    """GRPO + TAAN multi-task RL trainer.

    Args:
        config: :class:`TrainingConfig` instance or a plain ``dict``.
        reward_registry: Mapping ``{type_id: callable(response, ground_truth) -> float}``.
            Required at training time; may be omitted for testing.
    """

    def __init__(
        self,
        config: TrainingConfig | Dict[str, Any],
        reward_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(config, dict):
            config = TrainingConfig.from_dict(config)
        self.config = config
        self.reward_registry = reward_registry or {}

        self._setup_seed()
        self._build_models()
        self._build_taan()
        self._build_optimizer()
        self.global_step = 0

        # Try to resume from checkpoint
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

    def _build_models(self) -> None:
        """Initialise policy model, reference model, and vLLM rollout engine."""
        cfg = self.config
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

            logger.info("Loading policy model: %s", cfg.model_name)
            self.policy_model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            self.policy_model.train()

            logger.info("Loading reference model (frozen): %s", cfg.model_name)
            self.ref_model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch_dtype,
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load HuggingFace models (%s). Trainer is in stub mode.", exc)
            self.policy_model = None  # type: ignore[assignment]
            self.ref_model = None     # type: ignore[assignment]
            self.tokenizer = None

        # vLLM rollout engine (lazy — only materialises on first generate call)
        self.rollout_manager = VLLMRolloutManager(
            model_name=cfg.model_name,
            tp_size=cfg.tp_size,
            dtype=cfg.dtype,
        )

    def _build_taan(self) -> None:
        cfg = self.config
        self.taan = TAANAdvantageNormalizer(
            alpha=cfg.taan_alpha,
            clip_value=cfg.taan_clip,
            eps=cfg.taan_eps,
            min_samples_robust=cfg.taan_min_samples,
        )

    def _build_optimizer(self) -> None:
        if self.policy_model is None:
            self.optimizer = None  # type: ignore[assignment]
            self.scheduler = None
            return

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            total_iters=self.config.warmup_steps,
        )
        constant = LinearLR(
            self.optimizer,
            start_factor=1.0,
            total_iters=self.config.max_steps - self.config.warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, constant],
            milestones=[self.config.warmup_steps],
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one GRPO+TAAN training step.

        Args:
            batch: Dict with keys:
                - ``prompts``: list of str, length B
                - ``type_ids``: list of str, length B
                - ``ground_truths``: list of str (or None), length B
                - ``reward_fn_names``: list of str, length B

        Returns:
            Metrics dict with keys ``loss``, ``mean_reward``, ``kl_div``,
            ``per_type_stats``, ``clipped_fraction``.
        """
        prompts: List[str] = batch["prompts"]
        type_ids: List[str] = batch["type_ids"]
        ground_truths: List[Optional[str]] = batch.get("ground_truths", [None] * len(prompts))
        reward_fn_names: List[str] = batch.get("reward_fn_names", [""] * len(prompts))

        cfg = self.config
        B = len(prompts)

        # ── Step 1: rollout ────────────────────────────────────────────
        rollout = self.rollout_manager.generate(
            prompts,
            G=cfg.G,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_gen_tokens,
        )
        # responses: (B, G)

        # ── Step 2: compute rewards ───────────────────────────────────
        rewards = torch.zeros(B, cfg.G)  # (B, G)
        for i, (type_id, gt, fn_name) in enumerate(
            zip(type_ids, ground_truths, reward_fn_names)
        ):
            reward_fn = self.reward_registry.get(fn_name) or self.reward_registry.get(type_id)
            for j, response in enumerate(rollout.responses[i]):
                if reward_fn is not None:
                    try:
                        rewards[i, j] = float(reward_fn(response, gt))
                    except Exception:
                        rewards[i, j] = 0.0

        # ── Step 3: TAAN advantage normalisation ──────────────────────
        advantages = self.taan(rewards, type_ids, step=self.global_step)  # (B, G)
        clipped_fraction = (advantages.abs() >= cfg.taan_clip * 0.99).float().mean().item()

        # ── Step 4: policy loss ───────────────────────────────────────
        if self.policy_model is None:
            # Stub mode (no models loaded) — return synthetic metrics
            return {
                "loss": 0.0,
                "mean_reward": rewards.mean().item(),
                "kl_div": 0.0,
                "per_type_stats": {},
                "clipped_fraction": clipped_fraction,
            }

        flat_advantages = advantages.reshape(-1)   # (B*G,)
        flat_log_probs = self._compute_log_probs(prompts, rollout.responses, use_policy=True)
        flat_ref_log_probs = self._compute_log_probs(prompts, rollout.responses, use_policy=False)

        total_loss, policy_loss, kl_loss = compute_policy_loss(
            flat_advantages,
            flat_log_probs,
            flat_ref_log_probs,
            clip_eps=cfg.clip_eps,
            beta=cfg.beta,
        )

        # ── Step 5: backward ──────────────────────────────────────────
        scaled_loss = total_loss / cfg.gradient_accumulation_steps
        scaled_loss.backward()

        if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), cfg.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # ── Weight sync ───────────────────────────────────────────────
        if (self.global_step + 1) % cfg.weight_sync_every == 0:
            try:
                self.rollout_manager.sync_weights(self.policy_model)
            except Exception as exc:
                logger.debug("Weight sync skipped: %s", exc)

        self.global_step += 1

        # ── Per-type stats ────────────────────────────────────────────
        per_type_stats: Dict[str, Dict[str, Any]] = {}
        for ttype in set(type_ids):
            indices = [k for k, t in enumerate(type_ids) if t == ttype]
            type_adv = advantages[indices].reshape(-1)
            n = type_adv.numel()
            per_type_stats[ttype] = {
                "mean_advantage": type_adv.mean().item(),
                "std_advantage": type_adv.std().item() if n > 1 else 0.0,
                "n_samples": n * cfg.G,
                "ema_mu": self.taan.ema_tracker._states.get(ttype, None) and
                          self.taan.ema_tracker._states[ttype].mu_ema,
            }

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
            "clipped_fraction": clipped_fraction,
            "per_type_stats": per_type_stats,
        }

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    def _compute_log_probs(
        self,
        prompts: List[str],
        responses: List[List[str]],
        use_policy: bool = True,
    ) -> torch.Tensor:  # (B*G,)
        """Compute mean-token log-prob for each (prompt, response) pair.

        Args:
            prompts: List of prompt strings, length B.
            responses: Nested list, shape (B, G).
            use_policy: If True, use ``self.policy_model``; else ``self.ref_model``.

        Returns:
            Flat tensor of shape (B*G,).
        """
        model = self.policy_model if use_policy else self.ref_model
        model_device = next(model.parameters()).device

        log_probs_list: List[float] = []

        for prompt, resp_list in zip(prompts, responses):
            for resp in resp_list:
                text = prompt + resp
                enc = self.tokenizer(text, return_tensors="pt").to(model_device)
                prompt_len = len(self.tokenizer(prompt)["input_ids"])

                with torch.no_grad() if not use_policy else torch.enable_grad():
                    out = model(**enc)
                logits = out.logits  # (1, T, V)

                T = enc["input_ids"].shape[1]
                response_mask = torch.zeros(1, T, dtype=torch.bool, device=model_device)
                response_mask[0, prompt_len:] = True

                lp = compute_sequence_log_prob(logits, enc["input_ids"], response_mask)
                log_probs_list.append(lp.item())

        return torch.tensor(log_probs_list, requires_grad=use_policy)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        """Save model and optimiser state to ``output_dir/step-{step}``."""
        if self.policy_model is None:
            return
        ckpt_dir = os.path.join(self.config.output_dir, f"step-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.policy_model.save_pretrained(ckpt_dir)  # type: ignore[attr-defined]
        self.tokenizer.save_pretrained(ckpt_dir)      # type: ignore[attr-defined]
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "taan_ema_states": self.taan.ema_tracker._states,
            },
            os.path.join(ckpt_dir, "trainer_state.pt"),
        )
        logger.info("Checkpoint saved to %s", ckpt_dir)

    def _load_checkpoint(self, ckpt_dir: str) -> None:
        """Resume from a saved checkpoint."""
        state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if not os.path.exists(state_path):
            logger.warning("No trainer_state.pt found in %s; starting fresh.", ckpt_dir)
            return
        state = torch.load(state_path, map_location="cpu")
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("global_step", 0)
        self.taan.ema_tracker._states = state.get("taan_ema_states", {})
        logger.info("Resumed from %s (step %d)", ckpt_dir, self.global_step)

    # ------------------------------------------------------------------
    # Evaluation (stub — override in subclasses)
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current policy. Override for real evaluation logic."""
        return {}
