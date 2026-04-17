"""
Reward model scoring for open-ended tasks (writing, chat, …).

Uses a HuggingFace sequence-classification model (reward model) to score
each (prompt, response) pair.  Returns a continuous value that TAAN will
normalise across task types.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from .base import BaseReward

logger = logging.getLogger(__name__)


class ModelReward(BaseReward):
    """Reward model-based scoring using a HuggingFace classifier.

    The model is expected to produce a scalar score for each
    ``(prompt, response)`` pair.  Typical choice: an instruction-following
    quality model such as ``Qwen/Qwen2.5-7B-Instruct`` with a reward head.

    Args:
        model_name: HuggingFace model ID for the reward model.
        batch_size: Internal batch size for reward model inference.
        device: Torch device string.  Defaults to ``"cuda"`` if available.
        dtype: Weight dtype for the reward model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        batch_size: int = 32,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # type: ignore[import]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

            logger.info("Loading reward model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                num_labels=1,
            ).to(self.device)
            self._model.eval()
        except Exception as exc:
            logger.warning(
                "Could not load reward model '%s': %s. "
                "ModelReward will return 0.0 for all inputs.",
                self.model_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def __call__(
        self, response: str, ground_truth: Optional[str] = None
    ) -> float:
        """Score a single (response) with an optional prompt context.

        Args:
            response: Model-generated text to score.
            ground_truth: Optional prompt / instruction context.

        Returns:
            Scalar float reward.
        """
        scores = self.batch_call([response], [ground_truth])
        return scores[0]

    def batch_call(
        self,
        responses: List[str],
        ground_truths: List[Optional[str]],
    ) -> List[float]:
        """Batch reward scoring.

        Args:
            responses: List of generated texts.
            ground_truths: Corresponding prompt/context strings (may be None).

        Returns:
            List of float rewards.
        """
        self._ensure_loaded()
        if self._model is None:
            return [0.0] * len(responses)

        results: List[float] = []
        for i in range(0, len(responses), self.batch_size):
            batch_resp = responses[i : i + self.batch_size]
            batch_gt = ground_truths[i : i + self.batch_size]

            # Build input texts: "prompt\nresponse" or just "response"
            texts = [
                (f"{gt}\n{r}" if gt else r)
                for r, gt in zip(batch_resp, batch_gt)
            ]

            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**enc).logits.squeeze(-1)  # (B,)

            results.extend(logits.float().cpu().tolist())

        return results
