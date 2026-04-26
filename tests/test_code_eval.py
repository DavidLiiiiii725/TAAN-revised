"""
Tests for the HumanEval evaluation protocol in scripts/eval.py and
the CodeReward execution backend.

Covered scenarios
-----------------
1. test_code_reward_passes_correct_function
   – CodeReward._run_tests returns 1.0 when code + tests both pass.
2. test_code_reward_fails_wrong_implementation
   – CodeReward._run_tests returns 0.0 when the function returns wrong results.
3. test_code_reward_handles_syntax_error
   – CodeReward._run_tests returns 0.0 on syntax-broken code.
4. test_code_reward_handles_timeout
   – CodeReward._run_tests returns 0.0 on infinite loops (short timeout).
5. test_humaneval_style_prompt_plus_response
   – Simulates the HumanEval protocol: prompt + response is tested against
     the ``test`` field (not canonical_solution).
6. test_canonical_solution_is_not_valid_test
   – Confirms that using canonical_solution as tests gives an unreliable
     (vacuous) result — the old broken behaviour.
7. test_sample_rows_deterministic
   – _sample_rows returns the same subset for the same seed.
8. test_sample_rows_different_seeds_differ
   – Different seeds produce different subsets.
9. test_sample_rows_full_dataset_unchanged
   – When num_samples >= len(rows), all rows are returned.
10. test_evaluate_counts_invalid_responses
    – evaluate_humaneval correctly tracks tasks with no responses.
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import CodeReward directly to avoid torch dependency via src/rewards/__init__.py
import importlib.util as _ilu

def _import_code_reward():
    import unittest.mock as _mock

    # Build a stub BaseReward class that doesn't require torch
    base_cls = type("BaseReward", (), {
        "__init_subclass__": classmethod(lambda cls, **kw: None),
        "__call__": lambda self, *a, **kw: 0.0,
        "batch_call": lambda self, *a, **kw: [],
    })

    base_mod = _mock.MagicMock()
    base_mod.BaseReward = base_cls

    # Patch the src.rewards package and its base module so the relative import works
    rewards_pkg = _mock.MagicMock()
    rewards_pkg.BaseReward = base_cls

    with _mock.patch.dict("sys.modules", {
        "src": _mock.MagicMock(),
        "src.rewards": rewards_pkg,
        "src.rewards.base": base_mod,
    }):
        spec = _ilu.spec_from_file_location(
            "src.rewards.code_reward",
            os.path.join(os.path.dirname(__file__), "..", "src", "rewards", "code_reward.py"),
        )
        mod = _ilu.module_from_spec(spec)
        mod.__package__ = "src.rewards"
        sys.modules["src.rewards.code_reward"] = mod
        spec.loader.exec_module(mod)
    return mod

_cr_mod = _import_code_reward()
CodeReward = _cr_mod.CodeReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reward(timeout: int = 5) -> CodeReward:
    return CodeReward(timeout=timeout)


# ---------------------------------------------------------------------------
# 1–4: CodeReward execution tests
# ---------------------------------------------------------------------------

class TestCodeRewardRunTests:
    def test_passes_correct_function(self):
        """_run_tests returns 1.0 for a correct implementation."""
        reward = make_reward()
        code = "def add(a, b):\n    return a + b\n"
        tests = (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 1) == 0\n"
            "check(add)\n"
        )
        assert reward._run_tests(code, tests) == 1.0

    def test_fails_wrong_implementation(self):
        """_run_tests returns 0.0 when assertions fail."""
        reward = make_reward()
        code = "def add(a, b):\n    return a - b\n"  # wrong: subtraction
        tests = (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "check(add)\n"
        )
        assert reward._run_tests(code, tests) == 0.0

    def test_handles_syntax_error(self):
        """_run_tests returns 0.0 when the code has a syntax error."""
        reward = make_reward()
        code = "def add(a, b)\n    return a + b\n"  # missing colon
        tests = "assert add(1, 2) == 3\n"
        assert reward._run_tests(code, tests) == 0.0

    def test_handles_timeout(self):
        """_run_tests returns 0.0 for code that exceeds the timeout."""
        reward = CodeReward(timeout=1)
        code = "def spin():\n    while True: pass\n"
        tests = "spin()\n"
        assert reward._run_tests(code, tests) == 0.0


# ---------------------------------------------------------------------------
# 5–6: HumanEval protocol correctness
# ---------------------------------------------------------------------------

class TestHumanEvalProtocol:
    # Minimal HumanEval-style fixtures
    PROMPT = (
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b."""\n'
    )
    CORRECT_BODY = "    return a + b\n"
    WRONG_BODY = "    return a * b\n"
    TEST_FIELD = (
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(0, 0) == 0\n"
        "    assert candidate(-1, 1) == 0\n"
        "check(add)\n"
    )
    CANONICAL_SOLUTION = "    return a + b\n"  # same as correct body

    def test_humaneval_style_prompt_plus_response_passes(self):
        """prompt + correct_response tested against 'test' field → 1.0."""
        reward = make_reward()
        candidate = self.PROMPT + self.CORRECT_BODY
        score = reward._run_tests(candidate, self.TEST_FIELD)
        assert score == 1.0

    def test_humaneval_style_prompt_plus_response_fails(self):
        """prompt + wrong_response tested against 'test' field → 0.0."""
        reward = make_reward()
        candidate = self.PROMPT + self.WRONG_BODY
        score = reward._run_tests(candidate, self.TEST_FIELD)
        assert score == 0.0

    def test_canonical_solution_as_tests_is_unreliable(self):
        """Using canonical_solution as 'tests' is unreliable and non-standard.

        When the old evaluation ran ``code + canonical_solution``, it executed
        a script where the "tests" were either:

        (a) Another function definition — no assertions executed, so exit code
            is always 0 → vacuous 1.0 regardless of implementation correctness.
        (b) Indented function-body code — after textwrap.dedent, ``return``
            becomes a top-level statement → SyntaxError → always 0.0, even for
            correct implementations.

        Both cases produce incorrect and misleading results.  This test
        demonstrates case (a): a *wrong* implementation appears to pass because
        the "test" is just another function definition with no assertions.
        """
        reward = make_reward()
        wrong_impl = self.PROMPT + self.WRONG_BODY

        # Simulate a canonical_solution that is itself a complete function
        # definition (no assertions) — this is the vacuous-pass scenario.
        canonical_as_function = (
            "def add_reference(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
        vacuous_score = reward._run_tests(wrong_impl, canonical_as_function)
        # The script runs without error because there are no assertions —
        # the wrong implementation silently "passes".
        assert vacuous_score == 1.0, (
            "canonical_solution as a function definition gives vacuous 1.0 "
            "even for wrong implementations — confirming the old evaluation "
            "was unreliable."
        )

        # By contrast, using the proper 'test' field catches the wrong impl.
        correct_score = reward._run_tests(wrong_impl, self.TEST_FIELD)
        assert correct_score == 0.0, (
            "Proper test assertions correctly reject the wrong implementation."
        )


# ---------------------------------------------------------------------------
# 7–9: _sample_rows determinism
# ---------------------------------------------------------------------------

class TestSampleRows:
    def _import(self):
        # Import inside test to avoid importing scripts/ at module level
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_script",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "eval.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch heavy imports so we can import the helpers without side effects
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {
            "src.grpo.trainer": mock.MagicMock(),
            "src.rewards.math_reward": mock.MagicMock(),
            "src.rewards.code_reward": mock.MagicMock(),
            "src.utils.logging": mock.MagicMock(),
            "yaml": mock.MagicMock(),
        }):
            spec.loader.exec_module(mod)
        return mod

    def test_deterministic_same_seed(self):
        """Same seed → identical subset."""
        mod = self._import()
        rows = list(range(50))
        s1 = mod._sample_rows(rows, 10, seed=7)
        s2 = mod._sample_rows(rows, 10, seed=7)
        assert s1 == s2

    def test_different_seeds_differ(self):
        """Different seeds → (very likely) different subsets."""
        mod = self._import()
        rows = list(range(100))
        s1 = mod._sample_rows(rows, 10, seed=1)
        s2 = mod._sample_rows(rows, 10, seed=999)
        assert s1 != s2

    def test_full_dataset_unchanged(self):
        """num_samples >= len(rows) → all rows returned (order preserved)."""
        mod = self._import()
        rows = list(range(20))
        assert mod._sample_rows(rows, None, seed=0) == rows
        assert mod._sample_rows(rows, 20, seed=0) == rows
        assert mod._sample_rows(rows, 100, seed=0) == rows


# ---------------------------------------------------------------------------
# 10: Invalid-response accounting
# ---------------------------------------------------------------------------

class TestEvaluateHumanevalCounting:
    """Lightweight test for the accounting logic in evaluate_humaneval.

    We mock the trainer and dataset loader so no model or network I/O occurs.
    """

    def test_counts_invalid_responses(self):
        """Tasks with empty response lists are counted as invalid."""
        import unittest.mock as mock
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "eval_script2",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "eval.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        with mock.patch.dict("sys.modules", {
            "src.grpo.trainer": mock.MagicMock(),
            "src.rewards.math_reward": mock.MagicMock(),
            "src.rewards.code_reward": mock.MagicMock(),
            "src.utils.logging": mock.MagicMock(),
            "yaml": mock.MagicMock(),
        }):
            spec.loader.exec_module(mod)

        # Patch REWARD_MAP to use a real CodeReward (no actual model needed)
        real_reward = CodeReward(timeout=5)
        mod.REWARD_MAP["code"] = real_reward

        # Build fake dataset rows (3 tasks)
        fake_rows = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n    ",
                "test": "assert add(1,2)==3\n",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def sub(a, b):\n    ",
                "test": "assert sub(3,1)==2\n",
            },
            {
                "task_id": "HumanEval/2",
                "prompt": "def mul(a, b):\n    ",
                "test": "assert mul(2,3)==6\n",
            },
        ]

        # Fake rollout: task 0 has a valid response, tasks 1 and 2 are empty
        class FakeRollout:
            responses = [
                ["return a + b\n"],  # valid (will be prepended with prompt)
                [],                  # invalid
                [],                  # invalid
            ]

        class FakeRolloutManager:
            def generate(self, prompts, **kwargs):
                return FakeRollout()

        class FakeTrainer:
            rollout_manager = FakeRolloutManager()

        with mock.patch.object(mod, "_load_dataset_samples", return_value=fake_rows):
            results = mod.evaluate_humaneval(FakeTrainer(), num_samples=None, pass_k=1, seed=0)

        assert results["n_loaded"] == 3
        assert results["n_evaluated"] == 3
        assert results["n_invalid"] == 2
        assert results["n_attempted"] == 1
        # Task 0's code is "def add(a, b):\n    return a + b\n" — should pass
        assert results["passed"] == 1
        assert results["pass_at_k"] == pytest.approx(1.0)
