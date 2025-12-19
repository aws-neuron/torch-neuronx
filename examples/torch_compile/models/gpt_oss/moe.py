# ruff: noqa: N812, SIM108

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from tests.neuron_dynamo_backend.integration.models.gpt_oss.utils import indices_padding_wrapper


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Compile-friendly expert computation using masked operations instead of dynamic slicing.
    This avoids data-dependent control flow that torch.compile can't handle.
    """
    num_experts = w1.shape[0]
    batch_size, hidden_dim = x.shape

    # Convert to long for indexing
    num_tokens_per_expert_int = num_tokens_per_expert.long()

    # Create expert assignment for each token (which expert processes which token)
    # This is compile-friendly because we process all tokens through all experts
    # and use masking instead of dynamic slicing
    cumsum = torch.cumsum(num_tokens_per_expert_int, dim=0)
    cumsum_shifted = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=cumsum.device), cumsum[:-1]]
    )

    # Initialize output
    out = torch.zeros_like(x)

    # Process each expert
    for expert_idx in range(num_experts):
        # Get the range of tokens for this expert
        start_idx = cumsum_shifted[expert_idx]
        end_idx = cumsum[expert_idx]

        # Create a mask for tokens belonging to this expert
        # This is static shape, so compile-friendly
        token_indices = torch.arange(batch_size, device=x.device)
        mask = (token_indices >= start_idx) & (token_indices < end_idx)

        # Compute for all tokens, mask will zero out non-expert tokens
        # This avoids data-dependent branching
        mask_expanded = mask.unsqueeze(1).float()  # [batch_size, 1]
        x_masked = x * mask_expanded  # Zero out tokens not for this expert

        # Compute expert output (will be zeros for masked-out tokens)
        h = F.silu(torch.matmul(x_masked, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_masked, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))

        # Add to output (masked tokens contribute zeros)
        out = out + h * mask_expanded

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
    h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            # NOTE: If EP is not used, we need to pad the indices
            #       to prepare for grouped_mm;
            #       otherwise, EP will handle the padding.
            if not isinstance(self.w1, DTensor) or "ep" not in self.w1.device_mesh.mesh_dim_names:
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use
            sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(n_tokens * self.top_k, device=scores.device, dtype=torch.int64).reshape(
                n_tokens, self.top_k
            )
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional):
                Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.

        # Manual topk implementation to avoid torch.topk which has torch-mlir issues
        # This works for small k values (typically k=1 or k=2 in MoE)
        routing_scores = scores + expert_bias if expert_bias is not None else scores

        if self.top_k == 1:
            # Simple case: just find the max
            top_scores_routing, selected_experts_indices = routing_scores.max(dim=1, keepdim=True)
            if expert_bias is not None:
                # Get actual scores (without bias)
                top_scores = scores.gather(dim=1, index=selected_experts_indices)
            else:
                top_scores = top_scores_routing
        else:
            # For k > 1, we need to iteratively find top-k elements
            # This is less efficient but avoids torch.topk
            # Note: This path isn't currently used, since the default top k is set to 1.
            batch_size, num_experts = routing_scores.shape
            selected_experts_indices = torch.zeros(
                batch_size, self.top_k, dtype=torch.long, device=routing_scores.device
            )
            top_scores = torch.zeros(
                batch_size, self.top_k, dtype=routing_scores.dtype, device=routing_scores.device
            )

            # Create a copy to mask out selected experts
            remaining_scores = routing_scores.clone()

            for k_idx in range(self.top_k):
                # Find max in remaining scores
                max_scores, max_indices = remaining_scores.max(dim=1, keepdim=True)
                selected_experts_indices[:, k_idx : k_idx + 1] = max_indices

                # Get actual scores (without bias) if needed
                if expert_bias is not None:
                    top_scores[:, k_idx : k_idx + 1] = scores.gather(dim=1, index=max_indices)
                else:
                    top_scores[:, k_idx : k_idx + 1] = max_scores

                # Mask out the selected expert by setting its score to -inf
                remaining_scores.scatter_(1, max_indices, float("-inf"))

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass
        # that to experts forward
        # Use scatter_add instead of histc for torch-mlir compatibility
        num_tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.float32, device=selected_experts_indices.device
        )
        ones = torch.ones_like(selected_experts_indices.view(-1), dtype=torch.float32)
        num_tokens_per_expert.scatter_add_(0, selected_experts_indices.view(-1).long(), ones)

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass
        # that to experts forward
        # Use scatter_add instead of histc for torch-mlir compatibility
        num_tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.float32, device=selected_experts_indices.device
        )
        ones = torch.ones_like(selected_experts_indices.view(-1), dtype=torch.float32)
        num_tokens_per_expert.scatter_add_(0, selected_experts_indices.view(-1).long(), ones)

        # Reorder the token indices to match the order of the experts
        # Build permutation using vectorized scatter operations (compile-friendly)
        # token_indices_experts_sorted shape (bs*slen*top_k,)

        expert_indices_flat = selected_experts_indices.view(-1).long()
        total_tokens = expert_indices_flat.shape[0]

        # Calculate the starting position for each expert's tokens in the output
        expert_offsets = torch.zeros(
            self.num_experts, dtype=torch.long, device=expert_indices_flat.device
        )
        expert_offsets[1:] = torch.cumsum(num_tokens_per_expert[:-1].long(), dim=0)

        # For each token, calculate its position within its expert group
        # We do this by counting how many times each expert appears before each position
        token_positions = torch.arange(
            total_tokens, device=expert_indices_flat.device, dtype=torch.long
        )

        # Create a matrix where each row is a mask for one expert
        # expert_masks[i, j] = 1 if token j belongs to expert i
        expert_masks = (
            expert_indices_flat.unsqueeze(0)
            == torch.arange(
                self.num_experts, device=expert_indices_flat.device, dtype=torch.long
            ).unsqueeze(1)
        ).long()

        # Cumulative sum along tokens dimension gives us the position within each expert
        # cumsum[i, j] = how many tokens for expert i have we seen up to position j
        expert_cumsum = torch.cumsum(expert_masks, dim=1)

        # For each token, get its position within its expert group (0-indexed)
        # We need to select the right row based on expert_indices_flat
        within_expert_position = (
            torch.gather(expert_cumsum, 0, expert_indices_flat.unsqueeze(0)).squeeze(0) - 1
        )

        # Calculate output position: expert's offset + position within expert
        output_positions = (
            torch.gather(expert_offsets, 0, expert_indices_flat) + within_expert_position
        )

        # Build the inverse permutation: output_positions[i] tells us where token i should go
        # We want token_indices_experts_sorted[j] to tell us which
        # input token goes to output position j
        token_indices_experts_sorted = torch.zeros(
            total_tokens, dtype=torch.long, device=expert_indices_flat.device
        )
        token_indices_experts_sorted.scatter_(0, output_positions, token_positions)

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class GptOssMoE(nn.Module):
    """Alias for MoE to match expected import name."""

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.top_k = moe_args.top_k
        self.score_before_experts = moe_args.score_before_experts

        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update
        # the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # Simplified path for torch.compile compatibility
        # Process all tokens through all experts, weighted by routing scores
        # This avoids boolean masking and complex operations

        # shared expert
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        # Create a score matrix: (bs*slen, num_experts)
        # For each token, this will have non-zero scores only for selected experts
        num_experts = self.experts.num_experts
        batch_size = x.shape[0]

        # Initialize score matrix with zeros
        expert_score_matrix = torch.zeros(
            batch_size, num_experts, dtype=top_scores.dtype, device=x.device
        )

        # Fill in the scores for selected experts
        # selected_experts_indices: (bs*slen, top_k)
        # top_scores: (bs*slen, top_k)
        for k_idx in range(self.top_k):
            expert_indices = selected_experts_indices[:, k_idx]  # (bs*slen,)
            scores_k = top_scores[:, k_idx]  # (bs*slen,)
            # Scatter the scores into the matrix
            expert_score_matrix.scatter_add_(1, expert_indices.unsqueeze(1), scores_k.unsqueeze(1))

        # Now process each expert
        for expert_idx in range(num_experts):
            # Get scores for this expert: (bs*slen, 1)
            token_scores = expert_score_matrix[:, expert_idx : expert_idx + 1]

            # Apply scores before expert computation (zeros out non-assigned tokens)
            if self.score_before_experts:
                x_scored = x * token_scores
            else:
                x_scored = x

            # Get expert weights
            if isinstance(self.experts.w1, DTensor):
                w1 = self.experts.w1.to_local()[expert_idx]
                w2 = self.experts.w2.to_local()[expert_idx]
                w3 = self.experts.w3.to_local()[expert_idx]
            else:
                w1 = self.experts.w1[expert_idx]
                w2 = self.experts.w2[expert_idx]
                w3 = self.experts.w3[expert_idx]

            # Compute expert output for all tokens (zeros for non-assigned tokens due to scoring)
            h = F.silu(torch.matmul(x_scored, w1.transpose(-2, -1)))
            h = h * torch.matmul(x_scored, w3.transpose(-2, -1))
            expert_out = torch.matmul(h, w2.transpose(-2, -1))

            # Apply scores after expert computation if needed
            if not self.score_before_experts:
                expert_out = expert_out * token_scores

            # Add to output (non-assigned tokens contribute zeros)
            out = out + expert_out

        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
