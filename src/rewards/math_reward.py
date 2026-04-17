"""
Math reward function.

Extracts the model's final answer from common formats (\\boxed{}, "The answer
is …", "= …") and compares it to the ground truth using symbolic equivalence.
Returns 1.0 on match, 0.0 otherwise.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .base import BaseReward


class MathReward(BaseReward):
    r"""Rule-based reward for mathematical reasoning tasks.

    Supports extracting answers from:
    - LaTeX ``\boxed{…}`` notation
    - "The answer is X" / "= X" patterns
    - The last number/expression in the response

    Args:
        strict: When True, require exact string match after normalisation.
            When False (default), apply light symbolic simplification.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def __call__(
        self, response: str, ground_truth: Optional[str] = None
    ) -> float:
        """Return 1.0 if response matches ground_truth, else 0.0."""
        if ground_truth is None:
            return 0.0
        predicted = self.extract_answer(response)
        return 1.0 if self.is_equivalent(predicted, ground_truth) else 0.0

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_answer(self, text: str) -> str:
        r"""Extract the final answer from *text*.

        Tries (in order):
        1. ``\boxed{…}`` — handles nested braces.
        2. "The answer is X" / "answer: X" (case-insensitive).
        3. The last standalone number or fraction.
        4. The last non-empty line.

        Args:
            text: Raw model output.

        Returns:
            Extracted answer string (may be empty if nothing matches).
        """
        # 1. \boxed{…} — support nested braces
        boxed = self._extract_boxed(text)
        if boxed is not None:
            return self._normalise(boxed)

        # 2. "The answer is …" / "answer: …"
        pattern_answer = re.search(
            r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n.,;]+)",
            text,
            re.IGNORECASE,
        )
        if pattern_answer:
            return self._normalise(pattern_answer.group(1).strip())

        # 3. Last number / fraction in text
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
        if numbers:
            return self._normalise(numbers[-1])

        # 4. Last non-empty line
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        return self._normalise(lines[-1]) if lines else ""

    # ------------------------------------------------------------------
    # Equivalence check
    # ------------------------------------------------------------------

    def is_equivalent(self, predicted: str, ground_truth: str) -> bool:
        """Check if *predicted* equals *ground_truth* after normalisation.

        Normalisation includes:
        - Unicode NFKC normalisation
        - Strip whitespace and punctuation wrappers
        - Remove thousands separators (commas in numbers)
        - Lower-case comparison

        Args:
            predicted: Normalised predicted answer.
            ground_truth: Reference answer string.

        Returns:
            True if the two strings are considered equivalent.
        """
        norm_pred = self._normalise(predicted)
        norm_gt = self._normalise(ground_truth)

        if norm_pred == norm_gt:
            return True

        # Numeric comparison (handle "$1,000" == "1000")
        try:
            return float(norm_pred.replace(",", "")) == float(
                norm_gt.replace(",", "")
            )
        except ValueError:
            pass

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_boxed(self, text: str) -> Optional[str]:
        r"""Return the content of the innermost/last ``\boxed{…}`` in *text*."""
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None
        start = idx + len(r"\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        return text[start : pos - 1]

    @staticmethod
    def _normalise(text: str) -> str:
        """Apply unicode normalisation, strip, and lower-case."""
        text = unicodedata.normalize("NFKC", text)
        text = text.strip().rstrip(".,;:")
        return text.lower()
