"""Utilities for a minimal Group Relative Policy Optimization (GRPO) setup.

This module keeps the implementation intentionally small so that block-level
benchmarks can exercise the core GRPO math without depending on a full RLHF
pipeline.  The primitives below operate on pre-computed token log-probabilities
and scalar rewards grouped by prompt, closely mirroring the algorithm described
in the original GRPO announcement:

  * rewards belonging to the same prompt are normalized to zero mean / unit
    variance (Group Relative Advantage computation)
  * policy updates leverage a clipped PPO-style surrogate objective
  * a lightweight KL penalty against a frozen reference model stabilises the
    optimisation

The expectation is that upstream code prepares tensors with shape
``[num_groups, group_size, seq_len]`` for token-wise log-probabilities
(`policy_log_probs`, `reference_log_probs`) and
``[num_groups, group_size]`` for scalar rewards.  Optional masks allow handling
ragged batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

__all__ = [
    "GRPOBenchmarkBlock",
    "GRPOConfig",
    "compute_group_advantages",
    "count_flops",
    "create_block",
    "create_config",
    "exact_forward_kl",
    "get_presets",
    "grpo_loss",
    "grpo_step",
    "list_presets",
    "run_block",
]


@dataclass
class GRPOConfig:
    """Configuration for the basic GRPO objective.

    Attributes:
        clip_range:        PPO-style clip range for the probability ratio.
        kl_weight:         Scaling factor applied to the KL penalty term.
        reward_scale:      Optional multiplicative scaling for the normalised
                           advantages.  Tweak to control update magnitude.
        eps:               Numerical stability epsilon used in normalisation.
    """

    clip_range: float = 0.2
    kl_weight: float = 0.01
    reward_scale: float = 1.0
    eps: float = 1e-8


def _reduce_token_log_probs(
    log_probs: torch.Tensor, token_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Collapse per-token log-probabilities to per-sample sums.

    Args:
        log_probs:   Tensor shaped ``[..., seq_len]`` containing token log-probs.
        token_mask:  Optional mask aligned with ``log_probs`` (1.0 for valid
                     tokens, 0.0 otherwise).  Used to support ragged batches.

    Returns:
        Tensor with the sequence dimension reduced via a masked sum.
    """

    if log_probs.ndim < 1:
        raise ValueError("log_probs must have at least one dimension")

    if token_mask is not None:
        if token_mask.shape != log_probs.shape:
            raise ValueError("token_mask must have the same shape as log_probs")
        token_mask = token_mask.to(dtype=log_probs.dtype)
        log_probs = log_probs * token_mask

    return log_probs.sum(dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a mean that gracefully handles masks and empty inputs."""

    if mask is None:
        return values.mean()

    mask = mask.to(dtype=values.dtype)
    total_weight = mask.sum()
    if torch.all(total_weight == 0):
        return torch.zeros((), dtype=values.dtype, device=values.device)

    return (values * mask).sum() / total_weight


def compute_group_advantages(
    rewards: torch.Tensor,
    group_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalise rewards within each prompt group to obtain GRPO advantages.

    Args:
        rewards:     Tensor of shape ``[num_groups, group_size]`` (or any tensor
                     where the last dimension indexes the different samples for
                     the same prompt).
        group_mask:  Optional binary mask matching ``rewards`` to ignore padded
                     samples when computing statistics.
        eps:         Small value for numerical stability during normalisation.

    Returns:
        Tensor of normalised advantages with the same shape as ``rewards``.
    """

    if rewards.ndim < 1:
        raise ValueError("rewards must have at least one dimension")

    if group_mask is not None and group_mask.shape != rewards.shape:
        raise ValueError("group_mask must have the same shape as rewards")

    if group_mask is not None:
        mask = group_mask.to(dtype=rewards.dtype)
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (rewards * mask).sum(dim=-1, keepdim=True) / counts
        variance = (((rewards - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / counts
        std = torch.sqrt(variance + eps)
        advantages = (rewards - mean) / std
        advantages = advantages * mask
    else:
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True, unbiased=False).clamp(min=eps)
        advantages = (rewards - mean) / std

    return torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)


def exact_forward_kl(
    policy_log_probs_full: torch.Tensor,
    reference_log_probs_full: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact forward KL averaged over tokens (and optional mask)."""

    p = policy_log_probs_full.exp()
    kl_token = (p * (policy_log_probs_full - reference_log_probs_full)).sum(dim=-1)

    if token_mask is not None:
        mask = token_mask.to(dtype=kl_token.dtype)
        return (kl_token * mask).sum() / mask.sum().clamp(min=1)

    return kl_token.mean()


def grpo_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    config: GRPOConfig | None = None,
    token_mask: torch.Tensor | None = None,
    group_mask: torch.Tensor | None = None,
    *,
    old_policy_log_probs_selected: torch.Tensor,
    policy_log_probs_full: torch.Tensor,
    reference_log_probs_full: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO objective for a batch of grouped samples.

    Args:
        policy_log_probs:    Token log-probabilities under the policy being
                              optimised.  Expected shape ``[num_groups,
                              group_size, seq_len]``.
        reference_log_probs: Token log-probabilities under the frozen reference
                              policy, same shape as ``policy_log_probs``.
        rewards:              Scalar rewards per sample with shape
                              ``[num_groups, group_size]``.
        config:               :class:`GRPOConfig` controlling clipping / KL / eps.
        token_mask:           Optional mask aligned with ``policy_log_probs`` to
                              handle variable-length sequences (1.0 = keep).
        group_mask:           Optional mask aligned with ``rewards`` to drop
                              padded samples from the per-group statistics.

    Returns:
        ``(loss, metrics)`` where ``metrics`` contains detached diagnostic
        tensors that callers can log without breaking autograd.
    """

    if config is None:
        config = GRPOConfig()

    _ = reference_log_probs

    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy_log_probs and reference_log_probs must match in shape")

    if policy_log_probs.ndim < 2:
        raise ValueError("log-prob tensors must include at least batch and token dimensions")

    sample_shape = policy_log_probs.shape[:-1]
    if rewards.shape != sample_shape:
        raise ValueError(
            "rewards must match policy/ref log-prob shapes excluding the token dimension"
        )

    if token_mask is not None and token_mask.shape != policy_log_probs.shape:
        raise ValueError("token_mask must share the same shape as log-prob tensors")

    if group_mask is not None and group_mask.shape != rewards.shape:
        raise ValueError("group_mask must share the same shape as rewards")

    policy_sums = _reduce_token_log_probs(policy_log_probs, token_mask=token_mask)

    if policy_log_probs_full.shape != reference_log_probs_full.shape:
        raise ValueError("Full policy/reference log-prob tensors must match in shape")

    if policy_log_probs_full.shape[:-1] != policy_log_probs.shape:
        raise ValueError(
            "policy_log_probs_full must align with policy_log_probs across non-vocab dimensions"
        )

    if old_policy_log_probs_selected.shape != policy_log_probs.shape:
        raise ValueError("old_policy_log_probs_selected must match policy_log_probs in shape")

    old_sums = _reduce_token_log_probs(old_policy_log_probs_selected, token_mask=token_mask)

    advantages = compute_group_advantages(rewards, group_mask=group_mask, eps=config.eps)
    advantages = advantages * config.reward_scale

    log_ratio = policy_sums - old_sums.detach()
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
    clipped = clipped_ratio * advantages

    kl_term = exact_forward_kl(policy_log_probs_full, reference_log_probs_full, token_mask)

    if group_mask is not None:
        mask = group_mask.to(dtype=policy_sums.dtype)
        surrogate = torch.minimum(unclipped, clipped) * mask
        policy_loss = -_masked_mean(surrogate, mask)
        ratio_mean = _masked_mean(ratio, mask)
    else:
        surrogate = torch.minimum(unclipped, clipped)
        policy_loss = -surrogate.mean()
        ratio_mean = ratio.mean()

    total_loss = policy_loss + config.kl_weight * kl_term

    with torch.no_grad():
        metrics: dict[str, torch.Tensor] = {
            "policy_loss": policy_loss.detach(),
            "kl": kl_term.detach(),
            "ratio_mean": ratio_mean.detach(),
            "adv_mean": advantages.detach().mean(),
            "adv_std": advantages.detach().std(unbiased=False),
        }

    return total_loss, metrics


def grpo_step(
    optimizer: torch.optim.Optimizer,
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    config: GRPOConfig | None = None,
    token_mask: torch.Tensor | None = None,
    group_mask: torch.Tensor | None = None,
    *,
    old_policy_log_probs_selected: torch.Tensor,
    policy_log_probs_full: torch.Tensor,
    reference_log_probs_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Utility that wraps :func:`grpo_loss` with an optimisation step."""

    loss, metrics = grpo_loss(
        policy_log_probs,
        reference_log_probs,
        rewards,
        config=config,
        token_mask=token_mask,
        group_mask=group_mask,
        old_policy_log_probs_selected=old_policy_log_probs_selected,
        policy_log_probs_full=policy_log_probs_full,
        reference_log_probs_full=reference_log_probs_full,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    metrics = dict(metrics)
    metrics["loss"] = loss.detach()
    return metrics


class _GRPOOutput:
    """Container mimicking typical transformer outputs for benchmark plumbing."""

    def __init__(self, loss: torch.Tensor):
        self.loss = loss
        self.last_hidden_state = loss.view(1, 1)


class GRPOBenchmarkBlock(nn.Module):
    """Self-contained GRPO workload used by the bench harness.

    The module keeps a tiny set of learnable policy logits along with frozen
    reference logits, sampled token indices, rewards, and masks.  Forward passes
    compute the GRPO objective defined above and expose the scalar loss through
    ``last_hidden_state`` so the generic benchmark runner can reuse its loss
    handling code.
    """

    def __init__(self, config: dict[str, float], device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.config = config
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

        self.num_groups = int(config["num_groups"])
        self.group_size = int(config["group_size"])
        self.seq_len = int(config["seq_len"])
        self.vocab_size = int(config["vocab_size"])

        policy_logits = torch.randn(
            self.num_groups,
            self.group_size,
            self.seq_len,
            self.vocab_size,
            device=device,
            dtype=self.dtype,
        )
        self.policy_logits = nn.Parameter(policy_logits)

        reference_logits = torch.randn(
            self.num_groups,
            self.group_size,
            self.seq_len,
            self.vocab_size,
            device=device,
            dtype=self.dtype,
        )
        self.register_buffer("reference_logits", reference_logits)

        actions = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_groups, self.group_size, self.seq_len),
            device=device,
        )
        self.register_buffer("actions", actions)

        seq_lengths = torch.randint(
            low=max(1, self.seq_len // 2),
            high=self.seq_len + 1,
            size=(self.num_groups, self.group_size),
            device=device,
        )
        positions = torch.arange(self.seq_len, device=device).view(1, 1, -1)
        token_mask = positions < seq_lengths.unsqueeze(-1)
        self.register_buffer("token_mask", token_mask)

        group_mask = torch.ones(self.num_groups, self.group_size, device=device, dtype=torch.bool)
        drop_mask = torch.rand_like(group_mask, dtype=torch.float32) < 0.25
        group_mask = group_mask & (~drop_mask)
        self.register_buffer("group_mask", group_mask)

        rewards = torch.randn(self.num_groups, self.group_size, device=device, dtype=self.dtype)
        self.register_buffer("rewards", rewards)

        self._latest_metrics: dict[str, torch.Tensor] = {}

    def forward(self, *_args, **_kwargs):
        policy_logits = self.policy_logits.float()
        reference_logits = self.reference_logits.float()

        policy_log_probs = functional.log_softmax(policy_logits, dim=-1)
        reference_log_probs = functional.log_softmax(reference_logits, dim=-1)

        index = self.actions.unsqueeze(-1)
        policy_selected = policy_log_probs.gather(dim=-1, index=index).squeeze(-1)
        reference_selected = reference_log_probs.gather(dim=-1, index=index).squeeze(-1)

        old_policy_selected = policy_selected.detach()

        token_mask = self.token_mask
        if token_mask is not None and token_mask.dtype != policy_selected.dtype:
            token_mask = token_mask.to(dtype=policy_selected.dtype)

        group_mask = self.group_mask

        cfg = GRPOConfig(
            clip_range=self.config.get("clip_range", 0.2),
            kl_weight=self.config.get("kl_weight", 0.01),
            reward_scale=self.config.get("reward_scale", 1.0),
            eps=self.config.get("eps", 1e-8),
        )

        loss, metrics = grpo_loss(
            policy_selected,
            reference_selected,
            self.rewards.float(),
            config=cfg,
            token_mask=token_mask,
            group_mask=group_mask,
            old_policy_log_probs_selected=old_policy_selected,
            policy_log_probs_full=policy_log_probs,
            reference_log_probs_full=reference_log_probs,
        )

        self._latest_metrics = metrics

        return _GRPOOutput(loss)

    def latest_metrics(self) -> dict[str, torch.Tensor]:
        return dict(self._latest_metrics)


_PRESETS: dict[str, dict[str, float]] = {
    "grpo-small": {
        "group_size": 4,
        "vocab_size": 32000,
        "clip_range": 0.2,
        "kl_weight": 0.01,
        "reward_scale": 1.0,
    },
    "grpo-medium": {
        "group_size": 6,
        "vocab_size": 64000,
        "clip_range": 0.2,
        "kl_weight": 0.02,
        "reward_scale": 1.0,
    },
}


def list_presets():
    return list(_PRESETS.keys())


def get_presets():
    return list_presets()


def create_config(preset_name: str, num_groups: int, seq_len: int) -> dict[str, float]:
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown GRPO preset '{preset_name}'")

    base = dict(_PRESETS[preset_name])
    base.update(
        {
            "num_groups": int(num_groups),
            "batch_size": int(num_groups),
            "seq_len": int(seq_len),
            "eps": 1e-8,
        }
    )
    return base


def create_block(
    config: dict[str, float],
    device: torch.device,
    dtype: torch.dtype | None = None,
    tp_mesh=None,
    dp_mesh=None,
) -> nn.Module:
    dtype = dtype if dtype is not None else torch.float32
    block = GRPOBenchmarkBlock(config, device=device, dtype=dtype)
    block = block.to(device)
    block.eval()
    return block


def run_block(block, input_tensor, **kwargs):
    """Execute GRPO block — ignores input_tensor as the block is self-contained."""
    return block()


def count_flops(config: dict[str, float]) -> int:
    num_groups = int(config["num_groups"])
    group_size = int(config["group_size"])
    seq_len = int(config["seq_len"])
    vocab_size = int(config["vocab_size"])

    total_tokens = num_groups * group_size * seq_len
    return total_tokens * vocab_size * 2
