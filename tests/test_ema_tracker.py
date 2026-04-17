"""
Unit tests for EMA tracker (src/taan/ema_tracker.py).
"""

from __future__ import annotations

import math

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.taan.ema_tracker import EMATracker, EMAState


# ---------------------------------------------------------------------------
# Basic update behaviour
# ---------------------------------------------------------------------------

class TestEMATrackerUpdate:
    def test_first_update_bias_corrected(self):
        """After step 1 with mu=1.0: corrected value should equal ~1.0."""
        tracker = EMATracker(alpha=0.99)
        mu_c, sigma_c = tracker.update("math", mu=1.0, scale=1.0, step=1)
        # raw_ema = (1-0.99)*1.0 = 0.01; correction = 1 - 0.99^1 = 0.01
        # corrected = 0.01 / 0.01 = 1.0
        assert math.isclose(mu_c, 1.0, rel_tol=1e-5), f"Got {mu_c}"

    def test_multiple_types_independent(self):
        """Each type has independent EMA state."""
        tracker = EMATracker(alpha=0.9)
        tracker.update("math", mu=1.0, scale=1.0, step=1)
        tracker.update("code", mu=5.0, scale=2.0, step=1)

        state_math = tracker.get_state("math")
        state_code = tracker.get_state("code")

        assert state_math is not None
        assert state_code is not None
        assert state_math.step_count == 1
        assert state_code.step_count == 1
        # Means should differ
        assert not math.isclose(state_math.mu_ema, state_code.mu_ema)

    def test_convergence_to_target(self):
        """EMA with alpha=0.9 should converge to constant input after many steps."""
        tracker = EMATracker(alpha=0.9)
        TARGET = 3.7
        for step in range(1, 300):
            mu_c, _ = tracker.update("writing", mu=TARGET, scale=0.5, step=step)
        assert math.isclose(mu_c, TARGET, rel_tol=1e-2), (
            f"EMA did not converge to {TARGET}, got {mu_c}"
        )

    def test_sigma_positive(self):
        """sigma_corrected should always be non-negative."""
        tracker = EMATracker(alpha=0.99)
        for step in range(1, 20):
            _, sigma_c = tracker.update("chat", mu=0.0, scale=0.0, step=step)
            assert sigma_c >= 0.0, f"sigma_corrected negative at step {step}: {sigma_c}"

    def test_step_count_increments(self):
        """step_count should increment with each call for a given type."""
        tracker = EMATracker(alpha=0.99)
        for step in range(1, 6):
            tracker.update("math", mu=1.0, scale=1.0, step=step)
        state = tracker.get_state("math")
        assert state.step_count == 5


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------

class TestBiasCorrection:
    def test_bias_correction_formula(self):
        """Verify the bias correction formula mu / (1 - alpha^t) manually."""
        alpha = 0.9
        tracker = EMATracker(alpha=alpha)

        mu_c, _ = tracker.update("math", mu=2.0, scale=1.0, step=1)

        # Manual calculation
        mu_ema = (1 - alpha) * 2.0           # = 0.2
        correction = 1 - alpha ** 1          # = 0.1
        expected = mu_ema / correction        # = 2.0

        assert math.isclose(mu_c, expected, rel_tol=1e-6), (
            f"Expected {expected}, got {mu_c}"
        )

    def test_bias_correction_step2(self):
        """Two-step bias correction check."""
        alpha = 0.9
        tracker = EMATracker(alpha=alpha)
        tracker.update("math", mu=2.0, scale=1.0, step=1)
        mu_c, _ = tracker.update("math", mu=2.0, scale=1.0, step=2)

        # Step 1: mu_ema = 0.2
        # Step 2: mu_ema = 0.9*0.2 + 0.1*2.0 = 0.18 + 0.2 = 0.38
        # correction = 1 - 0.9^2 = 0.19
        # corrected = 0.38 / 0.19 = 2.0
        assert math.isclose(mu_c, 2.0, rel_tol=1e-5), (
            f"Expected 2.0 at step 2, got {mu_c}"
        )

    def test_sigma_bias_correction(self):
        """Sigma bias correction: sqrt(var_corrected)."""
        alpha = 0.9
        tracker = EMATracker(alpha=alpha)
        _, sigma_c = tracker.update("math", mu=0.0, scale=2.0, step=1)

        # var_ema = (1-0.9) * 4.0 = 0.4
        # correction = 0.1
        # var_corrected = 0.4 / 0.1 = 4.0
        # sigma = sqrt(4.0) = 2.0
        assert math.isclose(sigma_c, 2.0, rel_tol=1e-5), (
            f"Expected sigma=2.0, got {sigma_c}"
        )


# ---------------------------------------------------------------------------
# Reset and state management
# ---------------------------------------------------------------------------

class TestEMATrackerReset:
    def test_reset_single_type(self):
        """Reset a single type while keeping others."""
        tracker = EMATracker(alpha=0.99)
        tracker.update("math", mu=1.0, scale=1.0, step=1)
        tracker.update("code", mu=2.0, scale=1.0, step=1)

        tracker.reset("math")
        assert tracker.get_state("math") is None
        assert tracker.get_state("code") is not None

    def test_reset_all(self):
        """Reset all types at once."""
        tracker = EMATracker(alpha=0.99)
        for tid in ["math", "code", "writing"]:
            tracker.update(tid, mu=1.0, scale=1.0, step=1)

        tracker.reset()
        for tid in ["math", "code", "writing"]:
            assert tracker.get_state(tid) is None

    def test_get_state_unknown_type(self):
        """get_state returns None for an unseen type."""
        tracker = EMATracker(alpha=0.99)
        assert tracker.get_state("unknown_type") is None


# ---------------------------------------------------------------------------
# EMA state dataclass
# ---------------------------------------------------------------------------

class TestEMAState:
    def test_default_values(self):
        """EMAState defaults to zeros."""
        state = EMAState()
        assert state.mu_ema == 0.0
        assert state.var_ema == 0.0
        assert state.step_count == 0
