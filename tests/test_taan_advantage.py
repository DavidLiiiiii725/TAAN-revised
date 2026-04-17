"""
Unit tests for TAAN advantage normalisation (src/taan/advantage.py).

Covered scenarios:
1. test_grpo_basic               – verify values and signs for a simple case
2. test_grpo_all_equal           – all rewards equal → advantages all zero
3. test_taan_preserves_order     – TAAN does not change within-group ranking
4. test_taan_equalizes_scale     – TAAN reduces cross-type scale disparity
5. test_robust_stats_resistance  – median+IQR more stable than mean+std under outliers
6. test_ema_bias_correction      – early steps have bias-corrected estimates
7. test_clipping                 – clip threshold is enforced
8. test_small_type_fallback      – few samples → falls back to mean+std (no crash)
9. test_grpo_bessel_correction   – ddof=1 used in within-group normalization
10. test_forward_shape           – output shape matches input
"""

from __future__ import annotations

import math

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.taan.advantage import TAANAdvantageNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_normalizer(**kwargs) -> TAANAdvantageNormalizer:
    defaults = dict(alpha=0.99, clip_value=5.0, eps=1e-8, min_samples_robust=20)
    defaults.update(kwargs)
    return TAANAdvantageNormalizer(**defaults)


# ---------------------------------------------------------------------------
# 1. Basic GRPO normalisation
# ---------------------------------------------------------------------------

class TestGRPOAdvantages:
    def test_grpo_basic_values(self):
        """Rewards [[1,1,0,0]] → advantages have correct signs and magnitude."""
        norm = make_normalizer()
        rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # (1, 4)
        adv = norm.compute_grpo_advantages(rewards)
        # Higher rewards → positive advantages
        assert adv[0, 0].item() > 0
        assert adv[0, 1].item() > 0
        # Lower rewards → negative advantages
        assert adv[0, 2].item() < 0
        assert adv[0, 3].item() < 0
        # Symmetry: advantages for 1.0 and 0.0 should be mirror images
        assert math.isclose(adv[0, 0].item(), -adv[0, 2].item(), rel_tol=1e-5)

    def test_grpo_basic_two_prompts(self):
        """Multiple prompts are normalised independently."""
        norm = make_normalizer()
        # Prompt 0: rewards 1,1,0,0   Prompt 1: rewards 3.1,3.2,3.0,3.3
        rewards = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [3.1, 3.2, 3.0, 3.3],
        ])
        adv = norm.compute_grpo_advantages(rewards)
        # Prompt 0: mean=0.5, highest advantage at indices 0,1
        assert adv[0, 0].item() > 0 and adv[0, 1].item() > 0
        # Prompt 1: highest reward is 3.3 at index 3
        assert adv[1, 3].item() > adv[1, 0].item()
        # Both rows should have approximately zero mean
        assert abs(adv[0].mean().item()) < 1e-5
        assert abs(adv[1].mean().item()) < 1e-5

    def test_grpo_all_equal(self):
        """When all rewards are the same, all advantages should be zero."""
        norm = make_normalizer()
        rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        adv = norm.compute_grpo_advantages(rewards)
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_grpo_bessel_correction(self):
        """Verify ddof=1 Bessel correction (unbiased std) is used."""
        norm = make_normalizer()
        # For [0, 1]: unbiased std = 1/sqrt(1) * sqrt(0.5) = sqrt(0.5) ≈ 0.7071
        # biased std = sqrt(0.25) = 0.5
        rewards = torch.tensor([[0.0, 1.0]])
        adv = norm.compute_grpo_advantages(rewards)
        # With ddof=1: std = 0.7071, adv[0] ≈ (0-0.5)/0.7071 ≈ -0.7071
        expected = (0.0 - 0.5) / (math.sqrt(0.5) + 1e-8)
        assert math.isclose(adv[0, 0].item(), expected, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 3. TAAN preserves within-group order
# ---------------------------------------------------------------------------

class TestTAANPreservesOrder:
    def test_taan_preserves_intra_group_order(self):
        """TAAN normalisation must not change the ranking within each prompt."""
        norm = make_normalizer(min_samples_robust=5)  # lower threshold for test
        rewards = torch.tensor([
            [1.0, 3.0, 2.0, 0.5],
            [0.1, 0.9, 0.5, 0.3],
        ])
        type_ids = ["math", "math"]

        grpo_adv = norm.compute_grpo_advantages(rewards)
        taan_adv = norm.forward(rewards, type_ids, step=50)

        for i in range(rewards.shape[0]):
            grpo_order = grpo_adv[i].argsort().tolist()
            taan_order = taan_adv[i].argsort().tolist()
            assert grpo_order == taan_order, (
                f"Prompt {i}: TAAN changed ranking "
                f"(GRPO={grpo_order}, TAAN={taan_order})"
            )


# ---------------------------------------------------------------------------
# 4. TAAN equalises cross-type scale
# ---------------------------------------------------------------------------

class TestTAANEqualizesScale:
    def test_taan_reduces_cross_type_variance_gap(self):
        """After many steps the per-type variance should be closer to 1 than before TAAN."""
        norm = make_normalizer(alpha=0.9, min_samples_robust=5)
        torch.manual_seed(0)

        # Simulate 100 steps of training with two types having different variances
        for step in range(100):
            # Type "high_var": rewards spread over [0, 10]
            r_high = torch.rand(4, 8) * 10
            # Type "low_var": rewards close to 0 with tiny spread
            r_low = torch.rand(4, 8) * 0.1

            rewards = torch.cat([r_high, r_low], dim=0)   # (8, 8)
            type_ids = ["high_var"] * 4 + ["low_var"] * 4

            taan_adv = norm.forward(rewards, type_ids, step=step)

        # After warmup, advantages from both types should have similar scales
        high_adv = taan_adv[:4].reshape(-1)
        low_adv = taan_adv[4:].reshape(-1)

        high_std = high_adv.std().item()
        low_std = low_adv.std().item()

        # Both should be in a reasonable range (not wildly different)
        ratio = max(high_std, low_std) / (min(high_std, low_std) + 1e-6)
        assert ratio < 10.0, (
            f"Cross-type std ratio is too large after TAAN: "
            f"high_std={high_std:.3f}, low_std={low_std:.3f}, ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# 5. Robust statistics: outlier resistance
# ---------------------------------------------------------------------------

class TestRobustStatsOutlierResistance:
    def test_robust_stats_outlier_resistance(self):
        """median+IQR should be less affected by 10% outliers than mean+std."""
        from src.taan.robust_stats import _robust_stats, _simple_stats

        torch.manual_seed(42)
        clean = torch.randn(100)

        # 10% outliers with magnitude 100×
        with_outliers = clean.clone()
        n_outliers = 10
        outlier_idx = torch.randperm(100)[:n_outliers]
        with_outliers[outlier_idx] = 100.0

        # Robust estimators
        med_clean, iqr_clean = _robust_stats(clean)
        med_out, iqr_out = _robust_stats(with_outliers)

        # Mean/std estimators
        mean_clean, std_clean = _simple_stats(clean)
        mean_out, std_out = _simple_stats(with_outliers)

        # Change in location estimate
        robust_shift = abs(med_out.item() - med_clean.item())
        simple_shift = abs(mean_out.item() - mean_clean.item())

        assert robust_shift < simple_shift, (
            f"Robust estimator shifted more than simple: "
            f"robust_shift={robust_shift:.4f}, simple_shift={simple_shift:.4f}"
        )

        # Change in scale estimate
        robust_scale_change = abs(iqr_out.item() - iqr_clean.item())
        simple_scale_change = abs(std_out.item() - std_clean.item())

        assert robust_scale_change < simple_scale_change, (
            f"Robust scale shifted more than simple scale: "
            f"robust={robust_scale_change:.4f}, simple={simple_scale_change:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. EMA bias correction
# ---------------------------------------------------------------------------

class TestEMABiasCorrection:
    def test_ema_bias_correction_early_steps(self):
        """In early steps, bias correction should push mu away from zero."""
        from src.taan.ema_tracker import EMATracker

        tracker = EMATracker(alpha=0.99, eps=1e-8)

        # Feed constant mu=1.0, scale=1.0 for 3 steps
        results = []
        for step in range(1, 4):
            mu_c, sigma_c = tracker.update("math", mu=1.0, scale=1.0, step=step)
            results.append((mu_c, sigma_c))

        # After step 1: raw EMA = 0.01, correction = 1-0.99^1 = 0.01 → corrected ≈ 1.0
        mu_step1, _ = results[0]
        assert math.isclose(mu_step1, 1.0, rel_tol=1e-5), (
            f"After step 1, bias-corrected mu should be ~1.0, got {mu_step1}"
        )

        # After step 2: raw EMA = 0.01 + 0.99*0.01 = 0.0199, correction = 0.0199 → ≈1.0
        mu_step2, _ = results[1]
        assert math.isclose(mu_step2, 1.0, rel_tol=1e-3), (
            f"After step 2, bias-corrected mu should be ~1.0, got {mu_step2}"
        )

    def test_ema_converges_over_time(self):
        """With constant input, EMA should converge to that value."""
        from src.taan.ema_tracker import EMATracker

        tracker = EMATracker(alpha=0.9)
        for step in range(1, 200):
            mu_c, _ = tracker.update("code", mu=5.0, scale=2.0, step=step)
        assert math.isclose(mu_c, 5.0, rel_tol=1e-2), (
            f"EMA should converge to 5.0, got {mu_c}"
        )


# ---------------------------------------------------------------------------
# 7. Clipping
# ---------------------------------------------------------------------------

class TestClipping:
    def test_clip_threshold_enforced(self):
        """All advantages should lie within [-clip_value, clip_value]."""
        norm = make_normalizer(clip_value=2.0, min_samples_robust=5)
        torch.manual_seed(0)
        # Large variance rewards to force large raw advantages
        rewards = torch.randn(16, 8) * 10
        type_ids = ["math"] * 16

        adv = norm.forward(rewards, type_ids, step=50)
        assert adv.abs().max().item() <= 2.0 + 1e-6, (
            f"Max advantage {adv.abs().max().item():.4f} exceeds clip_value=2.0"
        )

    def test_default_clip_value(self):
        """Default clip value is 5.0."""
        norm = make_normalizer()
        assert norm.clip_value == 5.0


# ---------------------------------------------------------------------------
# 8. Small-type fallback
# ---------------------------------------------------------------------------

class TestSmallTypeFallback:
    def test_small_type_no_crash(self):
        """With only 3 samples for a type, TAAN should not crash."""
        norm = make_normalizer(min_samples_robust=20)
        rewards = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],  # math (3 prompts)
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 0.0, 0.0],
        ])
        type_ids = ["math", "math", "math"]
        adv = norm.forward(rewards, type_ids, step=1)
        assert adv.shape == rewards.shape

    def test_small_type_uses_mean_std(self):
        """With < min_samples_robust samples, mean+std estimator is used."""
        from src.taan.robust_stats import compute_location_scale

        values = torch.tensor([1.0, 2.0, 3.0])  # only 3 samples
        loc, scale = compute_location_scale(values, min_samples_robust=20)

        # Should use mean=2.0, std=1.0
        assert math.isclose(loc.item(), 2.0, rel_tol=1e-5)
        assert math.isclose(scale.item(), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 9. Forward output shape
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_output_shape(self):
        """forward() must return a tensor of the same shape as rewards."""
        norm = make_normalizer(min_samples_robust=5)
        B, G = 6, 8
        rewards = torch.rand(B, G)
        type_ids = ["math"] * 3 + ["code"] * 3

        out = norm.forward(rewards, type_ids, step=10)
        assert out.shape == (B, G), f"Expected ({B}, {G}), got {out.shape}"

    def test_callable_interface(self):
        """Normalizer should be callable (TAANAdvantageNormalizer(...)(rewards, ...))."""
        norm = make_normalizer(min_samples_robust=5)
        rewards = torch.rand(4, 4)
        out = norm(rewards, ["math"] * 4, step=1)
        assert out.shape == rewards.shape
