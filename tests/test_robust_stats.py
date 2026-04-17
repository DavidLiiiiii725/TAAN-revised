"""
Unit tests for robust statistics (src/taan/robust_stats.py).
"""

from __future__ import annotations

import math

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.taan.robust_stats import (
    compute_location_scale,
    _robust_stats,
    _simple_stats,
    mad,
)


# ---------------------------------------------------------------------------
# compute_location_scale: routing logic
# ---------------------------------------------------------------------------

class TestComputeLocationScale:
    def test_routes_to_robust_when_enough_samples(self):
        """With n >= min_samples_robust, should return median-based estimates."""
        torch.manual_seed(0)
        values = torch.randn(50)
        loc, scale = compute_location_scale(values, min_samples_robust=20)

        expected_loc = torch.median(values.float())
        assert math.isclose(loc.item(), expected_loc.item(), rel_tol=1e-5)

    def test_routes_to_simple_when_few_samples(self):
        """With n < min_samples_robust, should return mean-based estimates."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loc, scale = compute_location_scale(values, min_samples_robust=20)

        assert math.isclose(loc.item(), values.mean().item(), rel_tol=1e-5)
        assert math.isclose(scale.item(), values.std(unbiased=True).item(), rel_tol=1e-5)

    def test_exactly_at_threshold(self):
        """Exactly min_samples_robust samples should use robust estimator."""
        values = torch.arange(20, dtype=torch.float32)
        loc, scale = compute_location_scale(values, min_samples_robust=20)

        expected_loc = torch.median(values.float())
        assert math.isclose(loc.item(), expected_loc.item(), rel_tol=1e-5)

    def test_single_sample(self):
        """Single sample: no crash, scale should be 0."""
        values = torch.tensor([3.14])
        loc, scale = compute_location_scale(values, min_samples_robust=20)
        assert math.isclose(loc.item(), 3.14, rel_tol=1e-5)
        assert scale.item() == 0.0

    def test_output_on_same_device(self):
        """Output tensors should be on the same device as input."""
        values = torch.tensor([1.0, 2.0, 3.0])
        loc, scale = compute_location_scale(values, min_samples_robust=2)
        assert loc.device == values.device
        assert scale.device == values.device


# ---------------------------------------------------------------------------
# _robust_stats
# ---------------------------------------------------------------------------

class TestRobustStats:
    def test_median_correct(self):
        """Median of [1,2,3,4,5] is 3."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loc, _ = _robust_stats(values)
        assert math.isclose(loc.item(), 3.0, rel_tol=1e-5)

    def test_iqr_scaling(self):
        """IQR * 0.7413 for a standard normal should be close to 1.0."""
        torch.manual_seed(42)
        values = torch.randn(10000)
        _, scale = _robust_stats(values)
        # For standard normal: IQR ≈ 1.3490, so IQR * 0.7413 ≈ 1.0
        assert abs(scale.item() - 1.0) < 0.05, (
            f"IQR-based scale should be ~1.0 for N(0,1), got {scale.item():.4f}"
        )

    def test_outlier_resistance(self):
        """Adding extreme outliers should barely change median or IQR."""
        torch.manual_seed(7)
        base = torch.randn(100)
        with_outliers = base.clone()
        with_outliers[:5] = 1000.0

        loc_base, scale_base = _robust_stats(base)
        loc_out, scale_out = _robust_stats(with_outliers)

        # Location shift must be small
        assert abs(loc_out.item() - loc_base.item()) < 0.5, (
            f"Median shifted too much: {loc_base.item():.4f} → {loc_out.item():.4f}"
        )
        # Scale change must be moderate
        assert abs(scale_out.item() - scale_base.item()) < 0.5, (
            f"IQR shifted too much: {scale_base.item():.4f} → {scale_out.item():.4f}"
        )

    def test_constant_values(self):
        """Constant input → scale = 0."""
        values = torch.full((30,), 5.0)
        loc, scale = _robust_stats(values)
        assert math.isclose(loc.item(), 5.0, rel_tol=1e-5)
        assert math.isclose(scale.item(), 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# _simple_stats
# ---------------------------------------------------------------------------

class TestSimpleStats:
    def test_mean_std(self):
        """Mean and std of [0, 1, 2, 3, 4]."""
        values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        loc, scale = _simple_stats(values)
        assert math.isclose(loc.item(), 2.0, rel_tol=1e-5)
        assert math.isclose(scale.item(), values.std(unbiased=True).item(), rel_tol=1e-5)

    def test_single_value_zero_std(self):
        """Single value → std = 0 (no crash)."""
        values = torch.tensor([42.0])
        loc, scale = _simple_stats(values)
        assert math.isclose(loc.item(), 42.0, rel_tol=1e-5)
        assert scale.item() == 0.0


# ---------------------------------------------------------------------------
# mad
# ---------------------------------------------------------------------------

class TestMAD:
    def test_mad_standard_normal(self):
        """Scaled MAD of N(0,1) should be close to 1.0."""
        torch.manual_seed(0)
        values = torch.randn(10000)
        result = mad(values)
        assert abs(result.item() - 1.0) < 0.05, (
            f"Scaled MAD should be ~1.0 for N(0,1), got {result.item():.4f}"
        )

    def test_mad_constant(self):
        """MAD of constant values is 0."""
        values = torch.full((20,), 3.0)
        assert mad(values).item() == 0.0
