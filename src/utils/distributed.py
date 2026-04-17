"""
Distributed training utilities.

Thin wrappers around ``torch.distributed`` that are no-ops in single-process
mode so that all other modules can call them unconditionally.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return True if torch.distributed is initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current process rank (0 in single-process mode)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the total number of processes (1 in single-process mode)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Return True only on rank 0."""
    return get_rank() == 0


def allreduce_scalar(value: float, op: str = "mean") -> float:
    """AllReduce a scalar float across all workers.

    Args:
        value: Local scalar value.
        op: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Reduced scalar.
    """
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= get_world_size()
    return float(tensor)


def allreduce_dict(
    metrics: Dict[str, Any], op: str = "mean"
) -> Dict[str, Any]:
    """AllReduce every numeric value in *metrics* across all workers.

    Non-numeric values are passed through unchanged (taken from rank 0).

    Args:
        metrics: Dict of metric name → value.
        op: ``"mean"`` or ``"sum"``.

    Returns:
        New dict with reduced values.
    """
    if not is_distributed():
        return metrics

    result = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            result[k] = allreduce_scalar(float(v), op=op)
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            result[k] = allreduce_scalar(v.item(), op=op)
        else:
            result[k] = v  # pass through (e.g. nested dicts)
    return result


def barrier() -> None:
    """Synchronise all distributed workers at this point."""
    if is_distributed():
        dist.barrier()
