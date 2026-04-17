"""
Hardened TAAN (Type-Aware Advantage Normalization)

Pipeline:
1. GRPO within-group normalization:
   For each prompt's G responses, compute A_i = (r_i - mean(r)) / (std(r, ddof=1) + eps)
   - Uses Bessel correction (ddof=1)
   - When std=0 (all rewards identical), A_i = 0

2. Group by type and compute robust statistics:
   - If the type has >= min_samples_robust samples (default 20): use median + IQR*0.7413
   - Otherwise: fall back to mean + std

3. Distributed AllReduce to sync statistics (sum, sum_sq, count per type)

4. EMA update with bias correction:
   - mu_ema  = alpha * mu_ema  + (1-alpha) * mu_batch
   - var_ema = alpha * var_ema + (1-alpha) * scale_batch^2
   - Bias correction: mu_corrected    = mu_ema  / (1 - alpha^t)
                      sigma_corrected = sqrt(var_ema / (1 - alpha^t))
   - alpha defaults to 0.99

5. TAAN normalization: A_taan = (A_grpo - mu_corrected) / (sigma_corrected + eps)

6. Advantage clipping: A_final = clip(A_taan, -c, c), c defaults to 5
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .ema_tracker import EMATracker
from .robust_stats import compute_location_scale


class TAANAdvantageNormalizer:
    """TAAN advantage normalizer.

    Args:
        alpha: EMA decay coefficient, default 0.99
        clip_value: Advantage clip threshold, default 5.0
        eps: Division epsilon, default 1e-8
        min_samples_robust: Minimum samples for robust statistics, default 20
        adaptive_clip: Whether to use adaptive clipping (Q99.5%), default False
    """

    def __init__(
        self,
        alpha: float = 0.99,
        clip_value: float = 5.0,
        eps: float = 1e-8,
        min_samples_robust: int = 20,
        adaptive_clip: bool = False,
    ) -> None:
        self.alpha = alpha
        self.clip_value = clip_value
        self.eps = eps
        self.min_samples_robust = min_samples_robust
        self.adaptive_clip = adaptive_clip

        # EMA tracker: manages per-type bias-corrected EMA state
        self.ema_tracker = EMATracker(alpha=alpha, eps=eps)

    # ------------------------------------------------------------------
    # Step 1: GRPO within-group normalization
    # ------------------------------------------------------------------

    def compute_grpo_advantages(
        self,
        rewards: torch.Tensor,   # (num_prompts, G)
        prompt_ids: Optional[torch.Tensor] = None,  # unused, kept for API compat
    ) -> torch.Tensor:           # (num_prompts, G)
        """GRPO 组内归一化 (ddof=1 Bessel correction).

        For each prompt row r of shape (G,):
            A_i = (r_i - mean(r)) / (std(r, ddof=1) + eps)
        When std == 0, A_i = 0 for all i.
        """
        mean = rewards.mean(dim=-1, keepdim=True)          # (num_prompts, 1)
        # unbiased=True → ddof=1
        std = rewards.std(dim=-1, keepdim=True, unbiased=True)  # (num_prompts, 1)

        # Replace zero-std rows with a sentinel that will produce A=0
        safe_std = torch.where(std < self.eps, torch.ones_like(std), std)
        advantages = (rewards - mean) / (safe_std + self.eps)

        # Zero out rows where the original std was essentially zero
        zero_mask = (std < self.eps).expand_as(advantages)
        advantages = advantages.masked_fill(zero_mask, 0.0)
        return advantages

    # ------------------------------------------------------------------
    # Steps 2–6: full TAAN pipeline
    # ------------------------------------------------------------------

    def compute_taan_advantages(
        self,
        grpo_advantages: torch.Tensor,   # (N,) flattened
        type_ids: List[str],             # length N
        step: int,
    ) -> torch.Tensor:                   # (N,) TAAN-normalized advantages
        """Apply TAAN normalization on flattened GRPO advantages.

        Steps:
        2. Compute robust per-type statistics.
        3. (Distributed sync — no-op in single-process; handled by EMATracker.)
        4. EMA update with bias correction.
        5. TAAN normalization.
        6. Advantage clipping.
        """
        device = grpo_advantages.device
        output = grpo_advantages.clone()

        unique_types: List[str] = list(dict.fromkeys(type_ids))  # preserve order

        type_id_tensor: Optional[torch.Tensor] = None
        if isinstance(type_ids, torch.Tensor):
            type_id_tensor = type_ids

        for ttype in unique_types:
            # Collect indices belonging to this type
            mask = torch.tensor(
                [t == ttype for t in type_ids],
                dtype=torch.bool,
                device=device,
            )
            values = grpo_advantages[mask]  # (n_type,)
            n = values.numel()

            if n == 0:
                continue

            # Step 2: batch statistics (robust or simple)
            mu_batch, scale_batch = compute_location_scale(
                values, min_samples_robust=self.min_samples_robust
            )

            # Step 4: EMA update
            mu_c, sigma_c = self.ema_tracker.update(
                type_id=ttype,
                mu=mu_batch.item(),
                scale=scale_batch.item(),
                step=step,
            )

            # Step 5: TAAN normalization
            normalized = (values - mu_c) / (sigma_c + self.eps)

            # Step 6: clipping
            if self.adaptive_clip:
                clip_val = float(torch.quantile(normalized.abs(), 0.995))
                clip_val = max(clip_val, self.clip_value)
            else:
                clip_val = self.clip_value

            normalized = torch.clamp(normalized, -clip_val, clip_val)
            output[mask] = normalized

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_robust_stats(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (median, IQR * 0.7413)."""
        median = torch.median(values)
        q75 = torch.quantile(values.float(), 0.75)
        q25 = torch.quantile(values.float(), 0.25)
        iqr_scaled = (q75 - q25) * 0.7413
        return median, iqr_scaled

    def _compute_simple_stats(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std)."""
        return values.mean(), values.std()

    def _update_ema(
        self, type_id: str, mu_batch: float, scale_batch: float, step: int
    ) -> Tuple[float, float]:
        """Delegate to EMATracker and return bias-corrected (mu, sigma)."""
        return self.ema_tracker.update(type_id, mu_batch, scale_batch, step)

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    def forward(
        self,
        rewards: torch.Tensor,   # (num_prompts, G)
        type_ids: List[str],     # length num_prompts
        step: int,
    ) -> torch.Tensor:           # (num_prompts, G) final advantages
        """Full GRPO + TAAN pipeline.

        Returns advantages shaped (num_prompts, G).
        """
        num_prompts, G = rewards.shape

        # Step 1: GRPO within-group normalization
        grpo_adv = self.compute_grpo_advantages(rewards)  # (num_prompts, G)

        # Expand type_ids to match flattened sample dimension
        # Each prompt has G samples, all with the same type
        expanded_types: List[str] = []
        for t in type_ids:
            expanded_types.extend([t] * G)

        # Step 2–6: TAAN on flattened tensor
        flat_adv = grpo_adv.reshape(-1)  # (num_prompts * G,)
        flat_taan = self.compute_taan_advantages(flat_adv, expanded_types, step)

        return flat_taan.reshape(num_prompts, G)

    # Make the object callable
    __call__ = forward
