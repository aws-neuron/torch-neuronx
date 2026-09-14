"""MLIR implementation of scaled_dot_product_attention"""

import math

import torch
from torch._meta_registrations import alloc_with_matching_layout

from torch_neuronx._sdpa_math import compute_attention, compute_attention_backward

from ...base import AttentionOpImpl, ExecutionResult
from ...tensor_layout_utils import match_layout
from ..kernel import TorchMlirKernel


def generate_dropout_mask(
    dropout_mask_shape, dropout_p, device, philox_seed, philox_offset, backward=False
):
    if dropout_p < 0.0 or dropout_p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {dropout_p}")

    if dropout_p == 0.0:
        dropout_mask = None
    elif float(dropout_p) == 1.0:
        dropout_mask = torch.zeros(dropout_mask_shape, dtype=torch.bool, device=device)
    else:
        # Generate this on CPU and bring to neuron
        dropout_mask = torch.empty(dropout_mask_shape, dtype=torch.bool)
        keep_prob = 1.0 - float(dropout_p)
        generator = torch.Generator()
        generator.set_state(philox_seed)
        dropout_mask = torch.bernoulli(dropout_mask, keep_prob, generator=generator).to(device)
        if not backward:
            # set rng state to reproduce same dropout mask as running on CPU
            torch.set_rng_state(generator.get_state())

    return dropout_mask


class ScaledDotProductAttnMLIRImpl(AttentionOpImpl):
    """Native torch attention decomposed"""

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_attention,
            op_name,
            static_argnums=(5, 6, 7),  # scale, is_causal, dropout_p
        )
        self.op_name = op_name

    def can_handle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> bool:
        """Check if MLIR implementation can handle inputs"""
        if not super().can_handle(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        ):
            return False

        # Check tensors are on neuron device
        if (
            query.device.type != "neuron"
            or key.device.type != "neuron"
            or value.device.type != "neuron"
        ):
            return False

        # Check dtype
        if not (query.dtype == key.dtype == value.dtype):
            return False

        # Check 4D tensors
        return not (query.ndim != 4 or key.ndim != 4 or value.ndim != 4)

    def _execute_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> ExecutionResult:
        try:
            batch_size, q_heads, seq_len_q, head_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape

            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # GQA
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            philox_seed = torch.get_rng_state()
            philox_offset = torch.empty((0,), dtype=torch.int32, device=query.device)

            # dropout mask
            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape, dropout_p, query.device, philox_seed, philox_offset
            )

            # lowering issues when inputs to kernel were None
            if attn_bias is None:
                attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)
            if dropout_mask is None:
                dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            attn_output, lse = self.kernel(
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                scale,
                is_causal,
                dropout_p,
            )

            # Match output layout to query, same as
            # meta__scaled_dot_product_fused_attention_overrideable.
            # Pass query.shape (not a reconstructed tuple) so alloc_with_matching_layout
            # takes the empty_like path and preserves non-contiguous strides (e.g. transpose).
            embed_v = value.shape[-1]
            out = alloc_with_matching_layout(query, (batch_size, q_heads, seq_len_q, embed_v))
            attn_output = match_layout(attn_output, out, out)

            result = (
                attn_output,
                lse,
                None,  # cum_seq_q
                None,  # cum_seq_k
                seq_len_q,
                seq_len_k,
                philox_seed,
                philox_offset,
                None,  # debug_attn_mask
            )
            return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of MLIR SDPA forward."""

        def _mlir_sdpa_fwd_traceable(
            query,
            key,
            value,
            attn_bias,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=None,
        ):
            head_dim = query.shape[-1]
            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            batch_size, q_heads = query.shape[0], query.shape[1]
            kv_heads = key.shape[1]
            seq_len_q, seq_len_k = query.shape[2], key.shape[2]

            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            if attn_bias is None:
                attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)

            # Generate dropout mask using traceable ops
            if dropout_p > 0.0 and dropout_p < 1.0:
                dropout_mask = torch.bernoulli(
                    torch.full(
                        (batch_size, q_heads, seq_len_q, seq_len_k),
                        1.0 - dropout_p,
                        dtype=query.dtype,
                        device=query.device,
                    )
                )
            elif dropout_p >= 1.0:
                dropout_mask = torch.zeros(
                    (batch_size, q_heads, seq_len_q, seq_len_k),
                    dtype=torch.bool,
                    device=query.device,
                )
            else:
                dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            output, lse = compute_attention(
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                scale,
                is_causal,
                dropout_p,
            )

            # Match the 7-element tuple format expected by DispatchNkiImpl
            placeholder = query.new_zeros(())
            if dropout_p > 0.0:
                philox_seed = torch.randint(
                    0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32
                )
                philox_offset = torch.zeros(1, device=query.device, dtype=torch.int32)
            else:
                philox_seed = (query.flatten()[0] * 0).to(torch.int32)
                philox_offset = (query.flatten()[0] * 0).to(torch.int32)
            return (output, lse, placeholder, placeholder, philox_seed, philox_offset, placeholder)

        from ...lowering_result import LoweringResult
        from ..kernel import TorchMlirKernel

        mlir_kernel = TorchMlirKernel(
            _mlir_sdpa_fwd_traceable,
            self.op_name,
            static_argnums=(4, 5, 6, 7),
        )

        def mlir_sdpa_fwd_lower(*args, **kwargs):
            query = args[0] if args else kwargs.get("query")
            if query.device.type == "meta":
                return LoweringResult(
                    op_name=self.op_name,
                    original_output_specs=[],
                    is_empty_tensor=True,
                    stablehlo_bytes=b"",
                )
            return mlir_kernel.lower(*args, **kwargs)

        return mlir_sdpa_fwd_lower


class ScaledDotProductAttnBackwardMLIRImpl(AttentionOpImpl):
    """Attention backward compute"""

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_attention_backward,
            op_name,
            static_argnums=(8, 9, 10),  # scale, is_causal, dropout_p
        )
        self.op_name = op_name

    def can_handle(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> bool:
        """Check if MLIR backward can handle inputs."""
        if not super().can_handle(query, key, value):
            return False

        # Check tensors are on neuron device
        tensors = [grad_out, out, logsumexp]
        if not all(t.device.type == "neuron" for t in tensors):
            return False

        # Check same dtype
        if not (query.dtype == grad_out.dtype == out.dtype):
            return False

        # Check 4D tensors
        if grad_out.ndim != 4 or out.ndim != 4:
            return False

        # Check shapes match
        batch_size, q_heads, seq_len_q, _ = query.shape
        batch_go, go_heads, seq_len_go, _ = grad_out.shape
        if seq_len_q != seq_len_go or q_heads != go_heads or batch_size != batch_go:
            return False

        # LSE shape validation
        return logsumexp.ndim == 3 and logsumexp.shape == (batch_size, q_heads, seq_len_q)

    def _execute_impl(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> ExecutionResult:
        try:
            batch_size, q_heads, seq_len_q, head_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape

            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # GQA
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            # dropout mask
            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape,
                dropout_p,
                query.device,
                philox_seed,
                philox_offset,
                backward=True,
            )

            if dropout_mask is None:
                dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            grad_q, grad_k, grad_v, grad_attn_bias = self.kernel(
                grad_out,
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                out,
                logsumexp,
                scale,
                is_causal,
                dropout_p,
            )

            # Reduce GQA gradients
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                grad_k = grad_k.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )
                grad_v = grad_v.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )

            return ExecutionResult(success=True, output=(grad_q, grad_k, grad_v, grad_attn_bias))

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of MLIR SDPA backward."""

        def _mlir_sdpa_bwd_traceable(
            grad_out,
            query,
            key,
            value,
            attn_bias,
            grad_input_mask=None,
            out=None,
            logsumexp=None,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=0,
            max_k=0,
            dropout_p=0.0,
            is_causal=False,
            philox_seed=None,
            philox_offset=None,
            scale=None,
        ):
            head_dim = query.shape[-1]
            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            batch_size = query.shape[0]
            q_heads, kv_heads = query.shape[1], key.shape[1]
            seq_len_k = key.shape[2]

            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            if attn_bias is None or attn_bias.numel() == 0:
                attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)
            dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            grad_q, grad_k, grad_v, grad_attn_bias = compute_attention_backward(
                grad_out,
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                out,
                logsumexp,
                scale,
                is_causal,
                dropout_p,
            )

            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                grad_k = grad_k.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )
                grad_v = grad_v.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )

            if grad_attn_bias is None:
                grad_attn_bias = query.new_zeros(())

            return grad_q, grad_k, grad_v, grad_attn_bias

        from ...lowering_result import LoweringResult
        from ..kernel import TorchMlirKernel

        mlir_kernel = TorchMlirKernel(
            _mlir_sdpa_bwd_traceable,
            self.op_name,
            static_argnums=(5, 10, 11, 12, 13, 16),
        )

        def mlir_sdpa_bwd_lower(*args, **kwargs):
            query = args[1] if len(args) > 1 else kwargs.get("query")
            if query.device.type == "meta":
                return LoweringResult(
                    op_name=self.op_name,
                    original_output_specs=[],
                    is_empty_tensor=True,
                    stablehlo_bytes=b"",
                )
            return mlir_kernel.lower(*args, **kwargs)

        return mlir_sdpa_bwd_lower
