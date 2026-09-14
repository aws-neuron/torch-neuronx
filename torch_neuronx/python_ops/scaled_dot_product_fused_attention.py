"""
Fused scaled dot-product attention implementation for Neuron devices.

This implements the overrideable fused attention interface which allows
custom backends to provide their own optimized attention implementations.
"""

import math

import torch
from torch._meta_registrations import alloc_with_matching_layout

from torch_neuronx._C import _is_epilogue_enabled
from torch_neuronx.neuron_dynamo_backend.decompositions import (
    _can_use_nki_flash_attention,
    _can_use_nki_flash_attention_backward,
    _compute_attention_backward_nki,
    _compute_attention_nki,
    _prepare_backward_inputs,
    _reduce_gqa_gradients,
    compute_attention_backward_decomposition,
    scaled_dot_product_attention_decomposition,
)

from .base import AttentionOpImpl, ExecutionResult, Operation
from .lowering_result import LoweringResult
from .tensor_layout_utils import match_layout
from .torch_mlir.kernel import TorchMlirKernel


def _compute_output_strides_from_ref(ref_tensor, output_shape):
    """Compute dense output strides matching ref_tensor's dimension ordering.

    Returns a tuple of strides for output_shape that preserves the memory layout
    (dimension ordering) of ref_tensor, or None if:
    - ref_tensor has zero strides (from expand)
    - ref_tensor has duplicate strides (ambiguous ordering)
    - The resulting strides would be contiguous (no epilogue needed)
    """
    strides = ref_tensor.stride()
    ndim = len(strides)
    if ndim != len(output_shape):
        return None
    # Reject zero strides (expand) and duplicate strides (ambiguous)
    if 0 in strides or len(set(strides)) != ndim:
        return None
    # Dimension ordering: sort by stride descending (outermost first)
    perm = sorted(range(ndim), key=lambda d: strides[d], reverse=True)
    # Compute dense strides for output_shape in this order
    out_strides = [0] * ndim
    s = 1
    for d in reversed(perm):
        out_strides[d] = s
        s *= output_shape[d]
    out_strides = tuple(out_strides)
    # Skip if already contiguous
    contiguous = []
    s = 1
    for d in reversed(range(ndim)):
        contiguous.append(s)
        s *= output_shape[d]
    contiguous.reverse()
    if out_strides == tuple(contiguous):
        return None
    return out_strides


def _sdpa_fwd_traceable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
):
    """Traceable forward wrapper for make_fx. Mirrors sdpa_fused_overrideable decomposition."""
    if dropout_p > 0.0:
        philox_seed = torch.randint(0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32)
        philox_offset = torch.zeros(1, device=query.device, dtype=torch.int32)
    else:
        philox_seed = (query.flatten()[0] * 0).to(torch.int32)
        philox_offset = (query.flatten()[0] * 0).to(torch.int32)

    # Use 0-dim tensor placeholders for None slots (make_fx can't trace None)
    placeholder = query.new_zeros(())

    # max_q and max_k are plain ints in the ATen schema output — they cannot
    # be part of the traced HLO.  Return only tensor outputs from the traceable
    # function; the lower_func wrapper will splice max_q/max_k back in.
    if _can_use_nki_flash_attention(query, key, value, attn_bias, dropout_p):
        output, logsumexp = _compute_attention_nki(
            query, key, value, is_causal, scale, training=True
        )
        return (
            output,
            logsumexp,
            placeholder,
            placeholder,
            philox_seed,
            philox_offset,
            placeholder,
        )

    output, logsumexp, dropout_mask = scaled_dot_product_attention_decomposition(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        scale,
        enable_gqa=True,
        philox_seed=philox_seed,
    )
    philox_seed = dropout_mask
    return (output, logsumexp, placeholder, placeholder, philox_seed, philox_offset, placeholder)


def _sdpa_bwd_traceable(
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
):
    """Traceable backward wrapper for make_fx. Mirrors the backward decomposition."""
    if _can_use_nki_flash_attention_backward(query, key, value, attn_bias, dropout_p):
        grad_q, grad_k, grad_v = _compute_attention_backward_nki(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            is_causal,
            scale,
        )
        grad_attn_bias = query.new_zeros(())
        return grad_q, grad_k, grad_v, grad_attn_bias

    # Fallback
    orig_k_shape = key.shape

    b, q_heads, seq_q, e, seq_k, orig_kv_heads, scale_val, is_gqa, key, value = (
        _prepare_backward_inputs(query, key, value, orig_k_shape, scale)
    )

    if attn_bias is None or attn_bias.numel() == 0:
        attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)

    grad_q, grad_k, grad_v, grad_attn_bias = compute_attention_backward_decomposition(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        out,
        logsumexp,
        scale_val,
        is_causal,
        dropout_p,
        philox_seed,
    )

    grad_k, grad_v = _reduce_gqa_gradients(
        grad_k, grad_v, is_gqa, b, orig_kv_heads, q_heads, seq_k, e
    )

    if grad_attn_bias is None or grad_attn_bias.numel() == 1:
        grad_attn_bias = query.new_zeros(())

    return grad_q, grad_k, grad_v, grad_attn_bias


class ScaledDotProductFusedAttentionNKIImpl(AttentionOpImpl):
    """NKI implementation of fused scaled dot-product attention (returns tuple for training)"""

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
        """Check if NKI implementation can handle the given inputs"""
        if not super().can_handle(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        ):
            return False

        return _can_use_nki_flash_attention(query, key, value, attn_bias, dropout_p=dropout_p)

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
        """Execute the fused scaled dot-product attention operation"""
        try:
            from .nki_kernels.scaled_dot_product_attention import (
                scaled_dot_product_attention_kernel,
            )

            # Extract dimensions
            batch_size, q_heads, seq_len_q, embed_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape
            _, _, _, embed_v = value.shape

            # Calculate scale if not provided
            if scale is None:
                scale = 1.0 / math.sqrt(embed_dim)

            if hasattr(self, "_returns_single_tensor"):
                training = dropout_p > 0.0 or torch.is_grad_enabled()
            else:
                # overrideable API always returns LSE per ATen contract
                training = True

            # Prepare RNG state for dropout
            if dropout_p > 0.0:
                # Generate philox seed and offset for reproducible dropout
                philox_seed = torch.randint(
                    0, 2**31, (1, 1), dtype=torch.int32, device=query.device
                )
                philox_offset = torch.zeros((1, 1), dtype=torch.int32, device=query.device)
            else:
                # Empty tensors when no dropout
                philox_seed = torch.empty((0,), dtype=torch.int32, device=query.device)
                philox_offset = torch.empty((0,), dtype=torch.int32, device=query.device)

            # Allocate LSE buffer for training/backward pass with ATen spec shape [B, H, S]
            if training:
                lse = torch.empty(
                    batch_size,
                    q_heads,
                    seq_len_q,
                    dtype=torch.float32,
                    device=query.device,
                )
            else:
                lse = torch.empty((0,), dtype=torch.float32, device=query.device)

            # Create cumulative sequence tensors, using None since they are not used
            # in backward.
            cum_seq_q = None
            cum_seq_k = None

            # Max sequence lengths
            max_q = seq_len_q
            max_k = seq_len_k

            # Debug mask (not supported)
            debug_attn_mask = None

            # Call the NKI kernel — inputs in ATen format, kernel handles transpose and GQA
            attn_output = scaled_dot_product_attention_kernel(
                query,
                key,
                value,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
                training=training,
                seed=philox_seed if dropout_p > 0.0 else None,
                lse=lse if training else None,
            )

            # Create output tensor as done in meta__scaled_dot_product_fused_attention_overrideable
            # Output shape: (batch_size, q_heads, seq_len_q, embed_v)
            # https://github.com/pytorch/pytorch/blob/ca36c84c34589f26f98510addb89ad3e20acfe8b/torch/_meta_registrations.py#L5964
            # Expected behavior is output should have same stride as input query tensor. So,
            # attn_output should be contiguous if input query tensor is contiguous. Similarly,
            # if input query is non-contiguous (most common), result should be non-contiguous.

            # Previously, we did this, which works for non-contiguous input query
            # tensor but runs into error for contiguous case.
            # attn_output = attn_output.transpose(1, 2).contiguous().transpose(1, 2)

            out = alloc_with_matching_layout(query, (batch_size, q_heads, seq_len_q, embed_v))
            attn_output = match_layout(attn_output, out, out)

            # Check if this is the single-tensor version
            if hasattr(self, "_returns_single_tensor"):
                # For aten::scaled_dot_product_attention, only return the attention output
                return ExecutionResult(success=True, output=attn_output)
            else:
                # Return tuple matching the overrideable API
                result = (
                    attn_output,
                    lse,
                    cum_seq_q,
                    cum_seq_k,
                    max_q,
                    max_k,
                    philox_seed,
                    philox_offset,
                    debug_attn_mask,
                )
                return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of NKI SDPA forward."""
        op_name = "_scaled_dot_product_fused_attention_overrideable"
        # static_argnums: 4=dropout_p, 5=is_causal, 6=return_debug_mask, 7=scale
        kernel = TorchMlirKernel(
            _sdpa_fwd_traceable,
            op_name,
            static_argnums=(4, 5, 6, 7),
        )

        def sdpa_fwd_lower(*args, **kwargs):
            query = args[0] if args else kwargs.get("query")
            if query.device.type == "meta":
                return LoweringResult(
                    op_name=op_name,
                    original_output_specs=[],
                    is_empty_tensor=True,
                    stablehlo_bytes=b"",
                )
            result = kernel.lower(*args, **kwargs)
            # Inject target_strides for output[0] (attention) so the epilogue
            # fuses the stride relayout into the NEFF instead of a post-hoc copy.
            # Only valid when query strides represent a simple permutation (no gaps/zeros).
            if _is_epilogue_enabled() and result.output_specs and len(result.output_specs) > 0:
                target = _compute_output_strides_from_ref(query, result.output_specs[0][0])
                if target is not None:
                    spec = result.output_specs[0]
                    result.output_specs[0] = (spec[0], spec[1], target)
            return result

        return sdpa_fwd_lower


class ScaledDotProductFusedAttentionOp(Operation):
    """Fused scaled dot-product attention operation with NKI implementation"""

    def _setup_implementations(self):
        self._implementations.append(ScaledDotProductFusedAttentionNKIImpl())
        from .torch_mlir.ops.scaled_dot_product_attention import ScaledDotProductAttnMLIRImpl

        self._implementations.append(ScaledDotProductAttnMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        return "_scaled_dot_product_fused_attention_overrideable"

    def _unhandled_cpu_fallback(self, *args, **kwargs):
        # remap altered kwargs to original names - still breaks due to incompatible outputs
        kwargs["attn_mask"] = kwargs.pop("attn_bias", None)
        kwargs.pop("return_debug_mask", None)
        super()._unhandled_cpu_fallback(*args, **kwargs)


# Create singleton instance
_fused_sdpa_op = ScaledDotProductFusedAttentionOp()


def scaled_dot_product_fused_attention_overrideable_neuron(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Implements the overrideable fused attention interface for Neuron devices.

    This is called by PyTorch when it needs a custom fused attention implementation
    for training or when dropout is enabled.

    Returns:
        Tuple of (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                  philox_seed, philox_offset, debug_attn_mask)
    """
    # Validate device
    if query.device.type != "neuron":
        raise ValueError(f"Expected Neuron device, got {query.device}")

    if return_debug_mask:
        raise NotImplementedError("Debug mask not supported in Neuron implementation.")

    # Use the operation
    result = _fused_sdpa_op(
        query,
        key,
        value,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )

    return result


class ScaledDotProductAttentionBackwardNKIImpl(AttentionOpImpl):
    """NKI implementation of scaled dot-product attention backward"""

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        query = args[1] if len(args) > 1 else kwargs.get("query")
        key = args[2] if len(args) > 2 else kwargs.get("key")
        value = args[3] if len(args) > 3 else kwargs.get("value")
        attn_bias = args[4] if len(args) > 4 else kwargs.get("attn_bias")
        dropout_p = args[12] if len(args) > 12 else kwargs.get("dropout_p", 0.0)

        if not super().can_handle(query, key, value, attn_bias):
            return False

        return _can_use_nki_flash_attention_backward(query, key, value, attn_bias, dropout_p)

    def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
        """Execute the backward pass"""
        # Extract all arguments
        grad_out = args[0] if len(args) > 0 else kwargs.get("grad_out")
        query = args[1] if len(args) > 1 else kwargs.get("query")
        key = args[2] if len(args) > 2 else kwargs.get("key")
        value = args[3] if len(args) > 3 else kwargs.get("value")
        attn_bias = args[4] if len(args) > 4 else kwargs.get("attn_bias")
        # grad_input_mask = args[5] if len(args) > 5 else kwargs.get("grad_input_mask")  # unused
        out = args[6] if len(args) > 6 else kwargs.get("out")
        logsumexp = args[7] if len(args) > 7 else kwargs.get("logsumexp")
        # cum_seq_q = args[8] if len(args) > 8 else kwargs.get("cum_seq_q")  # unused
        # cum_seq_k = args[9] if len(args) > 9 else kwargs.get("cum_seq_k")  # unused
        # max_q = args[10] if len(args) > 10 else kwargs.get("max_q")  # unused
        # max_k = args[11] if len(args) > 11 else kwargs.get("max_k")  # unused
        dropout_p = args[12] if len(args) > 12 else kwargs.get("dropout_p", 0.0)
        is_causal = args[13] if len(args) > 13 else kwargs.get("is_causal", False)
        philox_seed = args[14] if len(args) > 14 else kwargs.get("philox_seed")
        scale = args[16] if len(args) > 16 else kwargs.get("scale")

        try:
            from .nki_kernels.scaled_dot_product_attention_backward import (
                scaled_dot_product_attention_backward_kernel,
            )

            # Check for unsupported features
            if attn_bias is not None:
                return ExecutionResult(
                    success=False, error_msg="Attention bias gradient not yet supported"
                )

            # Calculate scale if not provided
            if scale is None:
                scale = 1.0 / math.sqrt(query.shape[-1])

            # Convert philox seed for NKI kernel (if dropout was used)
            seed = philox_seed if dropout_p > 0.0 and philox_seed.numel() > 0 else None

            # Call the backward kernel — inputs in ATen format, kernel handles transpose
            grad_q, grad_k, grad_v = scaled_dot_product_attention_backward_kernel(
                grad_out=grad_out,
                q=query,
                k=key,
                v=value,
                out=out,
                lse=logsumexp,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
                seed=seed,
            )

            grad_q = match_layout(grad_q, query)
            grad_k = match_layout(grad_k, key)
            grad_v = match_layout(grad_v, value)

            # grad_attn_bias is None since we don't support it yet
            grad_attn_bias = None

            return ExecutionResult(success=True, output=(grad_q, grad_k, grad_v, grad_attn_bias))
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        """Check for empty tensors and reject them"""
        # Extract the main tensor arguments
        if len(args) >= 4:
            grad_out = args[0]
            query = args[1]
            key = args[2]
            value = args[3]
            if (
                grad_out.numel() == 0
                or query.numel() == 0
                or key.numel() == 0
                or value.numel() == 0
            ):
                return ExecutionResult(
                    success=False,
                    error_msg="Attention backward operations do not support empty tensors. "
                    "Please ensure all input tensors have non-zero elements.",
                )
        return None

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of NKI SDPA backward."""
        op_name = "_scaled_dot_product_fused_attention_overrideable_backward"
        # static_argnums: 5=grad_input_mask, 10=max_q, 11=max_k,
        #                 12=dropout_p, 13=is_causal, 16=scale
        kernel = TorchMlirKernel(
            _sdpa_bwd_traceable,
            op_name,
            static_argnums=(5, 10, 11, 12, 13, 16),
        )

        def sdpa_bwd_lower(*args, **kwargs):
            query = args[1] if len(args) > 1 else kwargs.get("query")
            if query.device.type == "meta":
                return LoweringResult(
                    op_name=op_name,
                    original_output_specs=[],
                    is_empty_tensor=True,
                    stablehlo_bytes=b"",
                )
            result = kernel.lower(*args, **kwargs)
            # Inject target_strides for grad outputs [0,1,2] matching query/key/value strides
            # so the epilogue fuses the stride relayout into the NEFF.
            if _is_epilogue_enabled() and result.output_specs and len(result.output_specs) >= 3:
                refs = [args[1], args[2], args[3]]  # query, key, value
                for i in range(3):
                    spec = result.output_specs[i]
                    if isinstance(spec, tuple) and len(spec) >= 2:
                        target = _compute_output_strides_from_ref(refs[i], spec[0])
                        if target is not None:
                            result.output_specs[i] = (spec[0], spec[1], target)
            return result

        return sdpa_bwd_lower


class ScaledDotProductAttentionBackwardOp(Operation):
    """Scaled dot-product attention backward operation with NKI implementation"""

    def _setup_implementations(self):
        self._implementations.append(ScaledDotProductAttentionBackwardNKIImpl())
        from .torch_mlir.ops.scaled_dot_product_attention import (
            ScaledDotProductAttnBackwardMLIRImpl,
        )

        self._implementations.append(ScaledDotProductAttnBackwardMLIRImpl(self.op_name))

    @property
    def op_name(self) -> str:
        return "_scaled_dot_product_fused_attention_overrideable_backward"


# Create singleton instance for backward
_sdpa_backward_op = ScaledDotProductAttentionBackwardOp()


def scaled_dot_product_fused_attention_overrideable_backward_neuron(
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
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for fused attention using the Operation pattern.

    Returns:
        Tuple of (grad_query, grad_key, grad_value, grad_attn_bias)
    """
    # Validate device
    if query.device.type != "neuron":
        raise ValueError(f"Expected Neuron device, got {query.device}")

    # Use the operation
    return _sdpa_backward_op(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        grad_input_mask,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )


def scaled_dot_product_fused_attention_overrideable_backward_meta(
    grad_out,
    query,
    key,
    value,
    attn_bias,
    grad_input_mask,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    """Meta implementation for fake tensor dispatch"""
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    grad_attn_bias = torch.empty_like(attn_bias) if attn_bias is not None else None
    return grad_query, grad_key, grad_value, grad_attn_bias
