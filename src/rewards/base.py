"""Base class for all reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseReward(ABC):
    """Abstract base class for reward functions.

    All reward functions must implement :meth:`__call__`, which takes a model
    response and (optionally) a ground-truth reference, and returns a float
    reward in the range [0, 1] (or an unbounded continuous score for model-based
    rewards that are later normalised by TAAN).
    """

    @abstractmethod
    def __call__(self, response: str, ground_truth: Optional[str] = None) -> float:
        """Compute a scalar reward for the given response.

        Args:
            response: The model-generated text to evaluate.
            ground_truth: Reference answer (required for rule-based rewards,
                may be ``None`` for model-scored rewards).

        Returns:
            A float reward value.
        """
        ...

    def batch_call(
        self, responses: list[str], ground_truths: list[Optional[str]]
    ) -> list[float]:
        """Compute rewards for a batch.  Default: loop over :meth:`__call__`."""
        return [self(r, g) for r, g in zip(responses, ground_truths)]
