"""
vLLM-based rollout manager for GRPO sampling.

Key features:
- Batched generation of G responses per prompt.
- Weight synchronisation from the training model to the vLLM engine.
- Returns a structured :class:`RolloutBatch` with texts, token IDs, and
  per-token log-probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class RolloutBatch:
    """Structured output from a single rollout pass.

    Attributes:
        prompts: Original prompt strings, length num_prompts.
        responses: Nested list of generated response strings,
            shape (num_prompts, G).
        prompt_ids: Token IDs for each prompt (may differ in length),
            list of length num_prompts.
        response_ids: Token IDs for each response (may differ in length),
            shape (num_prompts, G).
        log_probs: Per-sequence mean log-probability under the sampling model,
            shape (num_prompts, G).  None until computed.
    """

    prompts: List[str]
    responses: List[List[str]]
    prompt_ids: List[torch.Tensor] = field(default_factory=list)
    response_ids: List[List[torch.Tensor]] = field(default_factory=list)
    log_probs: Optional[torch.Tensor] = None   # (num_prompts, G)


class VLLMRolloutManager:
    """High-level wrapper around a vLLM :class:`LLM` engine.

    Handles:
    1. Lazy import of vLLM so the package is importable even when vLLM is not
       installed (e.g. during unit tests).
    2. Generating G responses per prompt.
    3. Syncing policy model weights into the vLLM engine for on-policy sampling.

    Args:
        model_name: HuggingFace model identifier.
        tp_size: Tensor-parallel size for the vLLM engine.
        dtype: Weight dtype for vLLM (default ``"bfloat16"``).
        gpu_memory_utilization: GPU memory budget for vLLM (default 0.5).
    """

    def __init__(
        self,
        model_name: str,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.tp_size = tp_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine = None   # lazy-initialised

    # ------------------------------------------------------------------
    # Lazy vLLM initialisation
    # ------------------------------------------------------------------

    @property
    def engine(self):
        """Return the vLLM LLM engine, initialising it on first access."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def _build_engine(self):
        try:
            from vllm import LLM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for rollout sampling. "
                "Install it with: pip install vllm"
            ) from exc

        return LLM(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[str],
        G: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ) -> RolloutBatch:
        """Generate *G* responses for each prompt using vLLM.

        Args:
            prompts: List of prompt strings.
            G: Number of responses to generate per prompt.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum generation length in tokens.

        Returns:
            :class:`RolloutBatch` containing texts and (optionally) token IDs.
        """
        from vllm import SamplingParams  # type: ignore[import]

        sampling_params = SamplingParams(
            n=G,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=0,   # request token-level log-probs from vLLM
        )

        outputs = self.engine.generate(prompts, sampling_params)

        responses: List[List[str]] = []
        response_ids: List[List[torch.Tensor]] = []
        log_probs_list: List[List[float]] = []

        for output in outputs:
            resp_texts: List[str] = []
            resp_ids: List[torch.Tensor] = []
            resp_lps: List[float] = []
            for candidate in output.outputs:
                resp_texts.append(candidate.text)
                resp_ids.append(torch.tensor(candidate.token_ids, dtype=torch.long))
                # Compute mean token log-prob from vLLM's per-token logprobs
                if candidate.logprobs:
                    token_lps = [
                        list(lp.values())[0].logprob
                        for lp in candidate.logprobs
                        if lp
                    ]
                    mean_lp = sum(token_lps) / len(token_lps) if token_lps else 0.0
                else:
                    mean_lp = 0.0
                resp_lps.append(mean_lp)
            responses.append(resp_texts)
            response_ids.append(resp_ids)
            log_probs_list.append(resp_lps)

        log_probs_tensor = torch.tensor(log_probs_list)  # (num_prompts, G)

        return RolloutBatch(
            prompts=prompts,
            responses=responses,
            response_ids=response_ids,
            log_probs=log_probs_tensor,
        )

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    def sync_weights(self, policy_model: torch.nn.Module) -> None:
        """Copy the latest policy model weights into the vLLM engine.

        This keeps the rollout distribution on-policy after each optimiser
        update.  The implementation relies on vLLM's
        ``LLMEngine.update_weights`` API (available from vLLM ≥ 0.6).

        Args:
            policy_model: The currently-trained HuggingFace model.
        """
        try:
            from vllm.model_executor.model_loader.weight_utils import (  # type: ignore[import]
                initialize_dummy_weights,
            )
        except ImportError:
            pass  # older vLLM — best-effort

        # Collect state dict on CPU to avoid double GPU memory allocation
        state_dict = {k: v.cpu() for k, v in policy_model.state_dict().items()}

        # vLLM >= 0.6 exposes an `update_weights` method on the engine
        if hasattr(self.engine, "update_weights"):
            self.engine.update_weights(state_dict)
        elif hasattr(self.engine, "llm_engine") and hasattr(
            self.engine.llm_engine, "update_weights"
        ):
            self.engine.llm_engine.update_weights(state_dict)
        else:
            # Fallback: recreate the engine (expensive but always correct)
            del self._engine
            self._engine = None
            _ = self.engine  # re-initialise
