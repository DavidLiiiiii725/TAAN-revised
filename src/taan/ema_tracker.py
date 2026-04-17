"""
EMA (Exponential Moving Average) statistics tracker for TAAN.

Maintains per-type EMA state with Adam-style bias correction so that
early steps do not under-estimate the location/scale estimates.

Optionally synchronises state across distributed workers via AllReduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class EMAState:
    """Per-type EMA state."""

    mu_ema: float = 0.0    # running mean
    var_ema: float = 0.0   # running variance (of the scale)
    step_count: int = 0    # number of updates applied (for bias correction)


class EMATracker:
    """Cross-batch exponential moving average tracker.

    Features:
    - Bias correction (Adam-style) to prevent under-estimation in early steps.
    - Optional distributed AllReduce synchronisation.
    - Independent EMA state per task type.

    Args:
        alpha: EMA decay coefficient (default 0.99).
        eps: Division epsilon (default 1e-8).
    """

    def __init__(self, alpha: float = 0.99, eps: float = 1e-8) -> None:
        self.alpha = alpha
        self.eps = eps
        self._states: Dict[str, EMAState] = {}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        type_id: str,
        mu: float,
        scale: float,
        step: int,  # global training step (used only for API clarity; internal counter used)
    ) -> Tuple[float, float]:
        """Update EMA for *type_id* and return bias-corrected (mu, sigma).

        Args:
            type_id: Task type identifier (e.g. "math", "code").
            mu: Batch location estimate.
            scale: Batch scale estimate (std-like value, must be ≥ 0).
            step: Current global training step (not used internally for bias
                  correction — we use *step_count* instead, which counts only
                  the actual updates per type).

        Returns:
            Tuple of (mu_corrected, sigma_corrected).
        """
        if type_id not in self._states:
            self._states[type_id] = EMAState()

        state = self._states[type_id]
        alpha = self.alpha

        # EMA update
        state.mu_ema = alpha * state.mu_ema + (1.0 - alpha) * mu
        state.var_ema = alpha * state.var_ema + (1.0 - alpha) * (scale ** 2)
        state.step_count += 1

        # Bias correction factor
        correction = 1.0 - alpha ** state.step_count

        mu_corrected = state.mu_ema / correction
        # Variance is corrected independently; take sqrt for sigma
        var_corrected = state.var_ema / correction
        sigma_corrected = (var_corrected ** 0.5) if var_corrected >= 0 else 0.0

        return mu_corrected, sigma_corrected

    # ------------------------------------------------------------------
    # Distributed synchronisation
    # ------------------------------------------------------------------

    def sync_across_workers(self) -> None:
        """Synchronise EMA states across all distributed workers via AllReduce.

        Transmits 3 scalars per type: (sum_mu_ema, sum_var_ema, sum_step_count).
        After averaging, each worker's state is identical.

        No-op when distributed training is not initialised.
        """
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        type_ids = sorted(self._states.keys())
        if not type_ids:
            return

        # Pack: [mu_ema, var_ema, step_count] per type
        data = []
        for tid in type_ids:
            s = self._states[tid]
            data.extend([s.mu_ema, s.var_ema, float(s.step_count)])

        tensor = torch.tensor(data, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size

        # Unpack
        for i, tid in enumerate(type_ids):
            base = i * 3
            state = self._states[tid]
            state.mu_ema = float(tensor[base])
            state.var_ema = float(tensor[base + 1])
            state.step_count = int(round(float(tensor[base + 2])))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_state(self, type_id: str) -> Optional[EMAState]:
        """Return the EMA state for *type_id*, or None if unseen."""
        return self._states.get(type_id)

    def reset(self, type_id: Optional[str] = None) -> None:
        """Reset state for one type or all types (if *type_id* is None)."""
        if type_id is None:
            self._states.clear()
        elif type_id in self._states:
            del self._states[type_id]
