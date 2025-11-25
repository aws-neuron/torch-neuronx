"""Tests for return-tensor allocation and overrides in NKI kernels.

These tests validate that TorchNeuronNKIKernel reuses preallocated output buffers
via `return_tensor_overrides` when the underlying vendor kernels return tensors
instead of writing into provided arguments. This covers the copy-avoidance
behavior added for flash attention forward/backward wrappers.
"""

import math

import torch

from torch_neuronx import TorchNeuronNKIKernel


def test_flash_fwd_returns_into_override_buffer():
    """flash_fwd should write directly into the provided output buffer.

    Verifies that when `return_tensor_overrides` is passed, the returned tensor
    is exactly the same object as the override buffer (no extra copy).
    """
    from neuronxcc.nki.kernels.attention import FlashConfig, flash_fwd

    device = torch.device("neuron", 0)

    # Small, tile-friendly problem size
    B, H, S, D = 1, 4, 512, 64  # noqa: N806

    # Build inputs: Q/K in NKI layout (B, H, D, S), V in ATen layout (B, H, S, D)
    q_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

    q_nki = q_aten.transpose(-2, -1).contiguous()
    k_nki = k_aten.transpose(-2, -1).contiguous()

    # Preallocate output buffer (ATen layout)
    out_buf = torch.empty(B, H, S, D, device=device, dtype=q_aten.dtype)

    # Build config (no training/LSE, do not transpose V)
    cfg = FlashConfig(training=False, should_transpose_v=False, seq_tile_size=512)

    # Construct kernel with override for return tensor
    k = TorchNeuronNKIKernel(
        func=getattr(flash_fwd, "func", flash_fwd),
        grid=(B, H),
        kernel_return=True,
        return_tensor_overrides=out_buf,
    )

    scale = 1.0 / math.sqrt(D)

    out_ret = k(
        q_nki,
        k_nki,
        v_aten,
        None,  # seed
        use_causal_mask=False,
        mixed_precision=True,
        softmax_scale=scale,
        dropout_p=0.0,
        config=cfg,
    )

    # The returned tensor must be the same object as the override buffer
    assert out_ret is out_buf
    # Sanity: shape/dtype/device unchanged
    assert tuple(out_ret.shape) == (B, H, S, D)
    assert out_ret.dtype == q_aten.dtype and out_ret.device == device


def test_flash_attn_bwd_returns_into_override_buffers():
    """flash_attn_bwd should write gradients into provided buffers.

    Runs a forward pass to create `out` and then validates that the backward
    kernel returns (dq, dk, dv) that are the exact same objects as the
    preallocated gradient buffers passed via `return_tensor_overrides`.
    """
    from neuronxcc.nki.kernels.attention import FlashConfig, flash_attn_bwd, flash_fwd

    device = torch.device("neuron", 0)

    # Problem size
    B, H, S, D = 1, 4, 512, 64  # noqa: N806

    # Inputs
    q_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v_aten = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

    q_nki = q_aten.transpose(-2, -1).contiguous()
    k_nki = k_aten.transpose(-2, -1).contiguous()

    # Forward with training=True to produce LSE
    out_buf = torch.empty(B, H, S, D, device=device, dtype=q_aten.dtype)
    cfg = FlashConfig(training=True, should_transpose_v=False, seq_tile_size=512)
    scale = 1.0 / math.sqrt(D)

    fwd = TorchNeuronNKIKernel(
        func=getattr(flash_fwd, "func", flash_fwd),
        grid=(B, H),
        kernel_return=True,
        return_tensor_overrides=out_buf,  # let forward write directly into out_buf
    )

    out_ret, lse_ret = fwd(
        q_nki,
        k_nki,
        v_aten,
        None,  # seed
        use_causal_mask=False,
        mixed_precision=True,
        softmax_scale=scale,
        dropout_p=0.0,
        config=cfg,
    )

    # Forward must have returned the same output buffer; keep both outputs
    assert out_ret is out_buf

    # Prepare grad_out and expected gradient buffer shapes (NKI layout for dq/dk/dv)
    grad_out_aten = torch.randn_like(out_buf)
    grad_out_nki = grad_out_aten.transpose(-2, -1).contiguous()
    out_nki = out_ret.transpose(-2, -1).contiguous()

    # Preallocate gradient buffers that should be written into by the kernel
    dq_buf = torch.empty_like(q_nki)
    dk_buf = torch.empty_like(k_nki)
    dv_buf = torch.empty_like(k_nki)  # same NKI layout as K

    bwd = TorchNeuronNKIKernel(
        func=getattr(flash_attn_bwd, "func", flash_attn_bwd),
        grid=(B, H),
        kernel_return=True,
        return_tensor_overrides=(dq_buf, dk_buf, dv_buf),
    )

    dq_ret, dk_ret, dv_ret = bwd(
        q_nki,
        k_nki,
        v_aten.transpose(-2, -1).contiguous(),
        out_nki,
        grad_out_nki,
        lse_ret,
        None,  # seed for dropout
        use_causal_mask=False,
        mixed_precision=True,
        dropout_p=0.0,
        softmax_scale=scale,
        logit_bias_ref=None,
    )

    # Returned tensors must be the same objects as our overrides
    assert dq_ret is dq_buf
    assert dk_ret is dk_buf
    assert dv_ret is dv_buf
