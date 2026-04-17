"""
Robust statistics for TAAN advantage normalization.

The key function :func:`compute_location_scale` returns a (location, scale)
pair using:
- **median + IQR × 0.7413** when enough samples are available (≥ min_samples_robust),
  which makes the scale estimate consistent with the standard deviation under a
  Gaussian distribution while being resistant to outliers.
- **mean + std** as a fallback for small batches.

The 0.7413 correction factor comes from the relationship between IQR and σ for
a normal distribution: σ ≈ IQR / 1.3490, so IQR × (1/1.3490) ≈ IQR × 0.7413.
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_location_scale(
    values: torch.Tensor,
    min_samples_robust: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate (location, scale) for a 1-D tensor.

    Selects the estimator based on the number of available samples:

    - ``n >= min_samples_robust``: **median** as location, **IQR × 0.7413**
      as scale (robust to outliers).
    - ``n < min_samples_robust``: **mean** as location, **std** as scale.

    In both cases a zero scale (e.g. constant input) is left as-is; callers
    should add an epsilon before dividing.

    Args:
        values: 1-D float tensor of advantage values.
        min_samples_robust: Threshold for switching to robust estimators.

    Returns:
        Tuple ``(location, scale)`` as scalar tensors on the same device/dtype
        as *values*.
    """
    n = values.numel()

    if n >= min_samples_robust:
        return _robust_stats(values)
    else:
        return _simple_stats(values)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _robust_stats(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Median and IQR-based scale (× 0.7413 consistency factor).

    Args:
        values: 1-D tensor with at least 1 element.

    Returns:
        ``(median, iqr_scaled)`` as scalar tensors.
    """
    # torch.median / torch.quantile require float; cast if needed
    v = values.float()
    location = torch.median(v)
    q75 = torch.quantile(v, 0.75)
    q25 = torch.quantile(v, 0.25)
    scale = (q75 - q25) * 0.7413
    # Cast back to original dtype for consistency
    return location.to(values.dtype), scale.to(values.dtype)


def _simple_stats(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation (unbiased, ddof=1 when n > 1, else 0).

    Args:
        values: 1-D tensor with at least 1 element.

    Returns:
        ``(mean, std)`` as scalar tensors.
    """
    location = values.mean()
    if values.numel() > 1:
        scale = values.std(unbiased=True)
    else:
        scale = torch.zeros([], dtype=values.dtype, device=values.device)
    return location, scale


def mad(values: torch.Tensor) -> torch.Tensor:
    """Median Absolute Deviation (MAD).

    Returns the MAD scaled by 1.4826 for consistency with σ under Gaussian
    distributions.

    Args:
        values: 1-D tensor.

    Returns:
        Scalar tensor: 1.4826 * median(|x - median(x)|).
    """
    v = values.float()
    med = torch.median(v)
    return (1.4826 * torch.median((v - med).abs())).to(values.dtype)
