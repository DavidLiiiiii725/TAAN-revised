"""
Training logging utilities.

Provides:
- :func:`setup_logging`: configure Python logging + optional wandb.
- :class:`MetricsLogger`: accumulate and flush per-step metrics to wandb /
  console.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Configure Python logging and (optionally) initialise wandb.

    Args:
        level: Python logging level string (default ``"INFO"``).
        log_file: If provided, also write logs to this file.
        wandb_project: wandb project name.  If None, wandb is not initialised.
        wandb_run_name: Optional wandb run name.
        config: Config dict to pass to ``wandb.init(config=…)``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    if wandb_project:
        try:
            import wandb  # type: ignore[import]

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config or {},
                resume="allow",
            )
            logger.info("wandb initialised: project=%s", wandb_project)
        except Exception as exc:
            logger.warning("wandb initialisation failed: %s", exc)


class MetricsLogger:
    """Accumulate per-step metrics and log them to console and wandb.

    Args:
        log_every: How often (in steps) to flush accumulated metrics.
        use_wandb: If True, attempt to log metrics to wandb.
    """

    def __init__(self, log_every: int = 10, use_wandb: bool = True) -> None:
        self.log_every = log_every
        self.use_wandb = use_wandb
        self._buffer: Dict[str, List[float]] = defaultdict(list)
        self._step_start: float = time.time()

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """Buffer *metrics* and flush every ``log_every`` steps.

        Nested dicts (e.g. ``per_type_stats``) are flattened with ``/`` separators.

        Args:
            step: Current global training step.
            metrics: Dict of metric name → value (numeric or nested dict).
        """
        flat = self._flatten(metrics)
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                self._buffer[k].append(float(v))

        if (step + 1) % self.log_every == 0:
            self._flush(step)

    def _flush(self, step: int) -> None:
        """Average buffered metrics and log them."""
        averaged: Dict[str, float] = {
            k: sum(vs) / len(vs) for k, vs in self._buffer.items() if vs
        }
        elapsed = time.time() - self._step_start
        averaged["steps_per_sec"] = self.log_every / max(elapsed, 1e-6)
        self._step_start = time.time()

        # Console
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(averaged.items()))
        logger.info("step=%d  %s", step, metric_str)

        # wandb
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                if wandb.run is not None:
                    wandb.log({"step": step, **averaged}, step=step)
            except Exception:
                pass

        self._buffer.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(
        d: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten a nested dict with ``/`` as separator."""
        out: Dict[str, Any] = {}
        for k, v in d.items():
            full_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                out.update(MetricsLogger._flatten(v, prefix=full_key))
            else:
                out[full_key] = v
        return out
