"""
GRPO policy loss with PPO-style clipping and KL divergence penalty.

Loss formulation:
    L = -E[A * min(ratio, clip(ratio, 1-ε, 1+ε))] + β * KL(policy ‖ ref)

where:
    ratio = exp(log_probs - ref_log_probs)
    KL is approximated as (log_probs - ref_log_probs)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_policy_loss(
    advantages: torch.Tensor,       # (N,)
    log_probs: torch.Tensor,        # (N,)  policy log-probs
    ref_log_probs: torch.Tensor,    # (N,)  reference model log-probs
    clip_eps: float = 0.2,
    beta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the GRPO policy loss.

    Args:
        advantages: Per-sample advantage values, shape (N,).
        log_probs: Log-probabilities under the current policy, shape (N,).
        ref_log_probs: Log-probabilities under the frozen reference model, (N,).
        clip_eps: PPO clip range ε (default 0.2).
        beta: KL penalty coefficient β (default 0.01).

    Returns:
        Tuple of scalar tensors: ``(total_loss, policy_loss, kl_loss)``.
    """
    log_ratio = log_probs - ref_log_probs          # (N,)
    ratio = torch.exp(log_ratio)

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = -advantages * torch.min(ratio, clipped_ratio)
    policy_loss = surrogate.mean()

    # Approximate KL: E[π/ref - 1 - log(π/ref)] ≈ E[log_ratio] for small KL
    # Use a simple direct approximation: KL ≈ log_probs - ref_log_probs
    kl_loss = beta * log_ratio.mean()

    total_loss = policy_loss + kl_loss
    return total_loss, policy_loss, kl_loss


def compute_sequence_log_prob(
    model_output_logits: torch.Tensor,   # (B, T, V)
    input_ids: torch.Tensor,             # (B, T)
    response_mask: torch.Tensor,         # (B, T) — 1 for response tokens, 0 for prompt
) -> torch.Tensor:                       # (B,)
    """Compute the mean token log-probability over response tokens.

    Args:
        model_output_logits: Raw logits from the language model.
        input_ids: Token IDs used as labels (shifted by 1 inside this function).
        response_mask: Boolean mask selecting response tokens.

    Returns:
        Per-sequence mean log-probability of response tokens, shape (B,).
    """
    # Shift: predict token t+1 from token t
    shift_logits = model_output_logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()                   # (B, T-1)
    shift_mask = response_mask[:, 1:].contiguous()                 # (B, T-1)

    # Token-level log-probs via cross-entropy (nll)
    B, T, V = shift_logits.shape
    log_probs_all = -F.cross_entropy(
        shift_logits.view(B * T, V),
        shift_labels.view(B * T),
        reduction="none",
    ).view(B, T)                                                    # (B, T-1)

    # Mean over response tokens
    denom = shift_mask.float().sum(dim=-1).clamp(min=1.0)
    seq_log_prob = (log_probs_all * shift_mask.float()).sum(dim=-1) / denom
    return seq_log_prob                                             # (B,)
