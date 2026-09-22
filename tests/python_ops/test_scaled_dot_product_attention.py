"""Tests for scaled_dot_product_attention operation"""

import math
import time
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import convert_fx_to_stablehlo
from torch_neuronx.python_ops.base import ExecutionResult
from torch_neuronx.python_ops.compilation import hlo_cache
from torch_neuronx.utils import use_mlir_aten_ops


def test_scaled_dot_product_attention_basic():
    """Basic test that SDPA runs on Neuron"""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(42)
    device = torch.device("neuron", 0)
    batch_size, num_heads, seq_len, d_head = 1, 8, 2048, 128

    with track_neuron_ops():
        q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)

        with torch.no_grad():
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        assert output.shape == (batch_size, num_heads, seq_len, d_head)
        assert output.device.type == "neuron"
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")


def test_scaled_dot_product_attention_different_head_dims():
    """Test SDPA with different KV head dims"""

    torch.manual_seed(42)
    device = torch.device("neuron", 0)

    batch_size, num_heads, seq_len = 1, 32, 2048
    qk_head_dim, v_head_dim = 192, 128

    q_cpu = torch.randn(
        batch_size, num_heads, seq_len, qk_head_dim, dtype=torch.float32, requires_grad=True
    )
    k_cpu = torch.randn(
        batch_size, num_heads, seq_len, qk_head_dim, dtype=torch.float32, requires_grad=True
    )
    v_cpu = torch.randn(
        batch_size, num_heads, seq_len, v_head_dim, dtype=torch.float32, requires_grad=True
    )

    output_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, dropout_p=0.0, is_causal=True)
    grad_output = torch.randn_like(output_cpu)
    output_cpu.backward(grad_output)
    q_grad_cpu, k_grad_cpu, v_grad_cpu = q_cpu.grad, k_cpu.grad, v_cpu.grad

    with track_neuron_ops():
        q_neuron = q_cpu.detach().clone().to(device).requires_grad_()
        k_neuron = k_cpu.detach().clone().to(device).requires_grad_()
        v_neuron = v_cpu.detach().clone().to(device).requires_grad_()

        output_neuron = F.scaled_dot_product_attention(
            q_neuron, k_neuron, v_neuron, dropout_p=0.0, is_causal=True
        )
        output_neuron.backward(grad_output.to(device))

        assert output_neuron.shape == (batch_size, num_heads, seq_len, v_head_dim)
        assert q_neuron.grad.shape == q_neuron.shape
        assert k_neuron.grad.shape == k_neuron.shape
        assert v_neuron.grad.shape == v_neuron.shape

        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

        rtol, atol = 5e-2, 5e-3
        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=rtol, atol=atol)
        torch.testing.assert_close(q_neuron.grad.cpu(), q_grad_cpu, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_neuron.grad.cpu(), k_grad_cpu, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_neuron.grad.cpu(), v_grad_cpu, rtol=rtol, atol=atol)


@pytest.fixture
def qkv_inputs(request):
    """Generate Q, K, V tensors for attention tests"""
    # Default parameters
    batch_size = 1
    q_heads = 8
    kv_heads = 8
    seq_len = 512
    d_head = 128
    dtype = torch.bfloat16
    requires_grad = False

    # Get parameters from test parametrization
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        callspec = request.node.callspec
        if "q_heads" in callspec.params:
            q_heads = callspec.params["q_heads"]
        if "kv_heads" in callspec.params:
            kv_heads = callspec.params["kv_heads"]
        if "d_head" in callspec.params:
            d_head = callspec.params["d_head"]
        if "seq_len" in callspec.params:
            seq_len = callspec.params["seq_len"]
        if "dtype" in callspec.params:
            dtype = callspec.params["dtype"]

    # Generate tensors
    q = torch.randn(batch_size, q_heads, seq_len, d_head, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype, requires_grad=requires_grad)

    return q, k, v


@pytest.mark.parametrize(
    "dtype,q_heads,kv_heads",
    [
        pytest.param(
            torch.float32,
            8,
            8,
            marks=pytest.mark.xfail(
                reason="MHA float32 flaky test",
                strict=False,
            ),
        ),  # MHA float32
        (torch.float16, 8, 8),  # MHA float16
        (torch.bfloat16, 8, 8),  # MHA bfloat16
        (torch.float32, 24, 6),  # GQA float32
        (torch.float16, 24, 6),  # GQA float16
        (torch.bfloat16, 24, 6),  # GQA bfloat16
        (torch.float32, 8, 1),  # MQA float32
        (torch.float16, 8, 1),  # MQA float16
        (torch.bfloat16, 8, 1),  # MQA bfloat16
    ],
)
def test_scaled_dot_product_attention_accuracy(dtype, q_heads, kv_heads):
    """Test SDPA accuracy across different dtypes and head configurations"""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch_neuronx.synchronize()

    torch.manual_seed(42)
    device = torch.device("neuron", 0)
    batch_size, seq_len, d_head = 1, 512, 64

    # Generate tensors and clamp to reasonable range
    q_cpu = torch.randn(batch_size, q_heads, seq_len, d_head, dtype=dtype)
    k_cpu = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype)
    v_cpu = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype)

    q_cpu.requires_grad = True
    k_cpu.requires_grad = True
    v_cpu.requires_grad = True

    # CPU forward/backward
    output_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, dropout_p=0.0, enable_gqa=True)

    # Clamp grad_output to avoid tiny values
    grad_output = torch.randn_like(output_cpu).clamp(-1, 1) * 0.1

    output_cpu.backward(grad_output)
    q_grad_cpu, k_grad_cpu, v_grad_cpu = q_cpu.grad, k_cpu.grad, v_cpu.grad

    with track_neuron_ops():
        # Neuron forward/backward
        q_neuron = q_cpu.detach().clone().to(device).requires_grad_()
        k_neuron = k_cpu.detach().clone().to(device).requires_grad_()
        v_neuron = v_cpu.detach().clone().to(device).requires_grad_()

        output_neuron = F.scaled_dot_product_attention(
            q_neuron, k_neuron, v_neuron, dropout_p=0.0, enable_gqa=True
        )
        torch_neuronx.synchronize()
        output_neuron_snapshot = output_neuron.detach().cpu()

        output_neuron.backward(grad_output.to(device))

        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

        # Ensure backward has fully completed before reading gradients off-device.
        torch_neuronx.synchronize()
        q_grad_neuron = q_neuron.grad.detach().cpu()
        k_grad_neuron = k_neuron.grad.detach().cpu()
        v_grad_neuron = v_neuron.grad.detach().cpu()

    # Compare results with dtype-appropriate tolerances
    # C++ dispatch compiles via make_fx tracing which introduces slightly more
    # float32 rounding than direct kernel execution, so float32 needs wider tolerance.
    rtol, atol = (1e-2, 1e-2) if dtype == torch.float32 else (5e-2, 5e-3)

    torch.testing.assert_close(output_neuron_snapshot, output_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(q_grad_neuron, q_grad_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_grad_neuron, k_grad_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(v_grad_neuron, v_grad_cpu, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "case,expected_error",
    [
        ("wrong_device", RuntimeError),
        ("wrong_dtype", RuntimeError),
        ("invalid_gqa", RuntimeError),
    ],
)
def test_scaled_dot_product_attention_runtime_errors(case, expected_error):
    """Test RuntimeError cases through F.scaled_dot_product_attention"""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(42)
    device = torch.device("neuron", 0)

    if case == "wrong_device":
        q = torch.randn(1, 8, 512, 64, dtype=torch.float32)
        k = torch.randn(1, 8, 512, 64, dtype=torch.float32, device=device)
        v = torch.randn(1, 8, 512, 64, dtype=torch.float32, device=device)

        @assert_raises(
            expected_error, match="Expected query, key, and value to have the same device type"
        )
        def _test_wrong_device():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        _test_wrong_device()

    elif case == "wrong_dtype":
        q = torch.randn(1, 8, 512, 64, dtype=torch.float32, device=device)
        k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=device)
        v = torch.randn(1, 8, 512, 64, dtype=torch.float32, device=device)

        @assert_raises(
            expected_error, match="Expected query, key, and value to have the same dtype"
        )
        def _test_wrong_dtype():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        _test_wrong_dtype()

    elif case == "invalid_gqa":
        q = torch.randn(1, 12, 512, 64, dtype=torch.float32, device=device)
        k = torch.randn(1, 5, 512, 64, dtype=torch.float32, device=device)  # 12 not divisible by 5
        v = torch.randn(1, 5, 512, 64, dtype=torch.float32, device=device)

        @assert_raises(
            expected_error,
            match="Number of heads in key and value must divide the number of heads in query",
        )
        def _test_invalid_gqa():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        _test_invalid_gqa()


@pytest.mark.parametrize(
    "q_heads,kv_heads,d_head,seq_len,dropout_p,dtype,is_causal,use_attn_bias",
    [
        (4, 4, 64, 256, 0.0, torch.bfloat16, False, False),
        (4, 2, 64, 512, 0.0, torch.float32, True, False),
        (4, 4, 256, 256, 0.0, torch.bfloat16, False, True),
        (4, 2, 256, 512, 0.0, torch.float32, True, False),
        (2, 1, 64, 577, 0.0, torch.bfloat16, False, False),
        pytest.param(
            2,
            2,
            64,
            256,
            0.1,
            torch.float32,
            False,
            True,
            marks=pytest.mark.xfail(
                reason="C++ dispatch traces dropout via torch.bernoulli in the HLO graph, "
                "which uses device-side RNG and cannot reproduce the CPU dropout mask "
                "from torch.manual_seed. Dropout is functionally correct for training "
                "(random mask each call) but won't match CPU reference numerically."
            ),
        ),
    ],
)
@patch(
    "torch_neuronx.python_ops.scaled_dot_product_fused_attention."
    "ScaledDotProductFusedAttentionNKIImpl.can_handle",
    return_value=False,
)
def test_scaled_dot_product_attention_mlir_fallback(
    mocked_handle,
    q_heads,
    kv_heads,
    d_head,
    seq_len,
    dropout_p,
    dtype,
    is_causal,
    use_attn_bias,
    qkv_inputs,
    monkeypatch,
):
    """Test MLIR fallback with CPU accuracy comparison (including GQA variants)"""

    # Check if Neuron device is available
    if not torch.neuron.is_available():
        print("Neuron device not available, skipping test")
        return

    device = torch.device("neuron", 0)

    seed = 42

    # Get tensors from fixture (will use parametrized q_heads, kv_heads)
    q_cpu, k_cpu, v_cpu = qkv_inputs

    # Handle GQA for CPU reference - expand KV to match Q heads
    batch_size, q_heads_actual, q_seq_len_actual, d_head_actual = q_cpu.shape
    _, kv_heads_actual, k_seq_len_actual, _ = k_cpu.shape

    # Create attention bias if needed
    attn_bias_cpu = None
    if use_attn_bias:
        attn_bias_cpu = torch.randn(
            batch_size, q_heads_actual, q_seq_len_actual, k_seq_len_actual, dtype=dtype
        )

    # Compute CPU reference
    torch.manual_seed(seed)
    with torch.no_grad():
        output_cpu = F.scaled_dot_product_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            attn_mask=attn_bias_cpu,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=(q_heads_actual != kv_heads_actual),
        )
        output_cpu_2 = F.scaled_dot_product_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            attn_mask=attn_bias_cpu,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=(q_heads_actual != kv_heads_actual),
        )

    # Move to Neuron
    q_neuron = q_cpu.to(device)
    k_neuron = k_cpu.to(device)
    v_neuron = v_cpu.to(device)
    attn_bias_neuron = attn_bias_cpu.to(device) if attn_bias_cpu is not None else None

    torch.manual_seed(seed)
    with (
        torch.no_grad(),
        track_neuron_ops(),
    ):
        output_neuron = F.scaled_dot_product_attention(
            q_neuron,
            k_neuron,
            v_neuron,
            attn_mask=attn_bias_neuron,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Call SDPA twice to verify RNG state changes match CPU behavior for dropout
        output_neuron_2 = F.scaled_dot_product_attention(
            q_neuron,
            k_neuron,
            v_neuron,
            attn_mask=attn_bias_neuron,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Verify the output is valid
        batch_size, q_heads_actual, seq_len, d_head = q_cpu.shape
        assert output_neuron.shape == (batch_size, q_heads_actual, seq_len, d_head)
        assert output_neuron.device.type == "neuron"

        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")

    # Compare MLIR fallback output with CPU reference
    output_neuron_cpu = output_neuron.cpu()
    output_neuron_cpu_2 = output_neuron_2.cpu()

    max_diff = torch.max(torch.abs(output_neuron_cpu - output_cpu)).item()
    mean_diff = torch.mean(torch.abs(output_neuron_cpu - output_cpu)).item()

    print(
        f"MLIR Fallback (Q:{q_heads}, KV:{kv_heads},"
        f" dropout:{dropout_p}, dtype:{dtype}, causal:{is_causal}) - Max difference: {max_diff:.6f}"
    )
    print(
        f"MLIR Fallback (Q:{q_heads}, KV:{kv_heads},"
        f" dropout:{dropout_p}, dtype:{dtype}, causal:{is_causal})"
        f" - Mean difference: {mean_diff:.6f}"
    )

    # Use reasonable tolerances for MLIR fallback comparison
    rtol = 5e-2  # 5% relative tolerance
    atol = 1e-2  # 0.01 absolute tolerance
    torch.testing.assert_close(output_neuron_cpu, output_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(output_neuron_cpu_2, output_cpu_2, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "q_heads,kv_heads,d_head,seq_len,dropout_p,dtype,is_causal,use_attn_bias",
    [
        (4, 4, 64, 256, 0.0, torch.bfloat16, False, False),
        (4, 2, 64, 512, 0.0, torch.float32, True, False),
        (4, 4, 256, 256, 0.0, torch.bfloat16, False, True),
        (4, 2, 256, 512, 0.0, torch.float32, True, False),
        (2, 1, 64, 577, 0.0, torch.bfloat16, False, False),
        pytest.param(
            2,
            2,
            64,
            256,
            0.1,
            torch.float32,
            False,
            True,
            marks=pytest.mark.xfail(
                reason="grad_q and grad_k are not matching CPU when dropout is enabled"
            ),
        ),
    ],
)
@patch(
    "torch_neuronx.python_ops.scaled_dot_product_fused_attention."
    "ScaledDotProductFusedAttentionNKIImpl.can_handle",
    return_value=False,
)
def test_scaled_dot_product_attention_backward_mlir_fallback(
    mocked_handle,
    q_heads,
    kv_heads,
    d_head,
    seq_len,
    dropout_p,
    dtype,
    is_causal,
    use_attn_bias,
    qkv_inputs,
    monkeypatch,
):
    """Test MLIR fallback backward pass with gradient comparison to CPU baseline"""

    # Check if Neuron device is available
    if not torch.neuron.is_available():
        print("Neuron device not available, skipping test")
        return

    device = torch.device("neuron", 0)
    seed = 42

    # Get tensors from fixture and enable gradients
    q_cpu, k_cpu, v_cpu = qkv_inputs
    q_cpu = q_cpu.requires_grad_(True)
    k_cpu = k_cpu.requires_grad_(True)
    v_cpu = v_cpu.requires_grad_(True)

    # Create gradient output tensor and attention bias
    batch_size, q_heads_actual, q_seq_len_actual, d_head_actual = q_cpu.shape
    _, kv_heads_actual, k_seq_len_actual, _ = k_cpu.shape
    grad_output = torch.randn(batch_size, q_heads_actual, q_seq_len_actual, d_head_actual)

    attn_bias_cpu = None
    if use_attn_bias:
        attn_bias_cpu = torch.randn(
            batch_size, q_heads_actual, q_seq_len_actual, k_seq_len_actual, dtype=dtype
        )

    # Compute CPU reference forward and backward
    torch.manual_seed(seed)
    output_cpu = F.scaled_dot_product_attention(
        q_cpu,
        k_cpu,
        v_cpu,
        attn_mask=attn_bias_cpu,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=(q_heads_actual != kv_heads_actual),
    )
    output_cpu.backward(grad_output)

    grad_q_cpu = q_cpu.grad.clone()
    grad_k_cpu = k_cpu.grad.clone()
    grad_v_cpu = v_cpu.grad.clone()

    # Move to Neuron and enable gradients
    q_neuron = q_cpu.detach().clone().to(device).requires_grad_(True)
    k_neuron = k_cpu.detach().clone().to(device).requires_grad_(True)
    v_neuron = v_cpu.detach().clone().to(device).requires_grad_(True)
    attn_bias_neuron = attn_bias_cpu.to(device) if attn_bias_cpu is not None else None
    grad_output_neuron = grad_output.to(device)

    torch.manual_seed(seed)
    output_neuron = F.scaled_dot_product_attention(
        q_neuron,
        k_neuron,
        v_neuron,
        attn_mask=attn_bias_neuron,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=(q_heads_actual != kv_heads_actual),
    )
    output_neuron.backward(grad_output_neuron)
    assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    # Move gradients back to CPU
    grad_q_neuron = q_neuron.grad.cpu()
    grad_k_neuron = k_neuron.grad.cpu()
    grad_v_neuron = v_neuron.grad.cpu()

    # Compare gradients
    rtol = 5e-2  # 5% relative tolerance
    atol = 5e-2  # 0.05 absolute tolerance

    torch.testing.assert_close(grad_q_neuron, grad_q_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_k_neuron, grad_k_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_v_neuron, grad_v_cpu, rtol=rtol, atol=atol)

    print(f"MLIR fallback gradients match CPU reference (Q:{q_heads}, KV:{kv_heads})")


@patch(
    "torch_neuronx.neuron_dynamo_backend.fx.fx_transform.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_correctness(mock_compiler):
    """Test that IR cache is properly incremented and hit on subsequent calls."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    # Disable persistent cache
    with patch.object(hlo_cache, "_enabled", False):
        device = torch.device("neuron", 0)
        batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 128

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        initial_count = mock_compiler.call_count

        # First call - should trace and cache IR
        with torch.no_grad():
            output1 = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        assert mock_compiler.call_count >= initial_count + 1, "Compiler was not called as expected"
        traced_call_count = mock_compiler.call_count
        # Second call - should use cached IR (cache size unchanged)
        with torch.no_grad():
            output2 = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        assert mock_compiler.call_count == traced_call_count, "Compiler called again"

        # Third call - should also use cached IR (cache size unchanged)
        with torch.no_grad():
            output3 = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        assert mock_compiler.call_count == traced_call_count, "Compiler called again"

        # Verify outputs are consistent
        torch.testing.assert_close(output1.cpu(), output2.cpu(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output1.cpu(), output3.cpu(), rtol=1e-5, atol=1e-5)


@patch(
    "torch_neuronx.neuron_dynamo_backend.fx.fx_transform.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_different_inputs_same_shape(mock_compiler):
    """Test IR cache hit with different inputs, same shapes."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    # Disable persistent cache
    with patch.object(hlo_cache, "_enabled", False):
        device = torch.device("neuron", 0)
        batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 64

        # First call with specific input values
        torch.manual_seed(42)
        q1 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k1 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v1 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        initial_count = mock_compiler.call_count

        with torch.no_grad():
            output1 = F.scaled_dot_product_attention(q1, k1, v1, dropout_p=0.0, is_causal=False)

        assert mock_compiler.call_count >= initial_count + 1, "Compiler was not called as expected"
        traced_call_count = mock_compiler.call_count

        # Second call with different input values but same shapes - should hit cache
        torch.manual_seed(123)
        q2 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k2 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v2 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        with torch.no_grad():
            output2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=False)

        assert mock_compiler.call_count == traced_call_count, "Compiler called again"

        # Verify shapes are the same
        assert output1.shape == output2.shape

        # Verify outputs are different (since inputs are different)
        # Use a threshold to ensure they're meaningfully different
        # TODO: sub fails with non-contiguous tensors when MLIR is enabled, needs fix
        diff = torch.abs(output1.cpu() - output2.cpu()).mean().item()
        assert diff > 1e-3, f"Outputs should be different with different inputs, but diff={diff}"


@patch(
    "torch_neuronx.neuron_dynamo_backend.fx.fx_transform.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_different_shapes(mock_compiler):
    """Test that IR cache creates separate entries for different shapes and different inputs."""
    # Disable HLO persistent cache — this test asserts on tracer call counts
    from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod

    _orig_enabled = _hlo_cache_mod._enabled
    _hlo_cache_mod._enabled = False

    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    device = torch.device("neuron", 0)

    initial_count = mock_compiler.call_count

    # First shape: batch_size=2, num_heads=4, seq_len=128, head_dim=64
    torch.manual_seed(42)
    q1 = torch.randn(1, 4, 512, 128, device=device)
    k1 = torch.randn(1, 4, 512, 128, device=device)
    v1 = torch.randn(1, 4, 512, 128, device=device)

    with torch.no_grad():
        output1 = F.scaled_dot_product_attention(q1, k1, v1, dropout_p=0.0, is_causal=False)

    assert mock_compiler.call_count >= initial_count + 1, "Compiler was not called as expected"
    traced_call_count = mock_compiler.call_count

    # Second shape: batch_size=1, num_heads=8, seq_len=256, head_dim=64 (different shape)
    torch.manual_seed(123)
    q2 = torch.randn(1, 8, 512, 128, device=device)
    k2 = torch.randn(1, 8, 512, 128, device=device)
    v2 = torch.randn(1, 8, 512, 128, device=device)

    with torch.no_grad():
        output2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0, is_causal=False)

    assert mock_compiler.call_count >= traced_call_count + 1, "Compiler was not called as expected"
    traced_call_count = mock_compiler.call_count
    # Third call with first shape again - should hit cache (size stays at 2)
    torch.manual_seed(456)
    q3 = torch.randn(1, 4, 512, 128, device=device)
    k3 = torch.randn(1, 4, 512, 128, device=device)
    v3 = torch.randn(1, 4, 512, 128, device=device)

    with torch.no_grad():
        output3 = F.scaled_dot_product_attention(q3, k3, v3, dropout_p=0.0, is_causal=False)

    assert mock_compiler.call_count == traced_call_count, "Compiler was not called as expected"

    # Verify output shapes
    assert output1.shape == (1, 4, 512, 128)
    assert output2.shape == (1, 8, 512, 128)
    assert output3.shape == (1, 4, 512, 128)

    _hlo_cache_mod._enabled = _orig_enabled


def test_scaled_dot_product_attention_uses_mlir_with_dropout():
    """Test that dropout SDPA runs on Neuron (not NKI, which doesn't support dropout)."""

    device = "neuron"
    batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    with torch.no_grad(), track_neuron_ops():
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert output.device.type == "neuron"
    assert_op_runs_on_neuron("_scaled_dot_product_fused_attention_overrideable")


def test_sdpa_stride_cache_hit_for_views():
    """Test that SDPA with different input strides (from slicing) shares cache entries.

    Views of different-sized parent tensors produce the same logical shape but
    different stride[0]. The cache key should normalize contiguous strides so
    these share a single compiled NEFF.
    """
    device = "neuron"
    b, h, s, d = 1, 8, 512, 64

    # Warmup: standard strides + non-standard strides (run twice to warm NRT path)
    for _ in range(2):
        q0 = torch.randn(b, h, s, d, device=device)
        with torch.no_grad():
            F.scaled_dot_product_attention(q0, q0, q0)
        parent0 = torch.randn(b, h + 2, s, d, device=device)
        q0 = parent0[:, :h, :, :]
        with torch.no_grad():
            F.scaled_dot_product_attention(q0, q0, q0)
        torch.neuron.synchronize()

    # Test: multiple different stride patterns should all be fast (<2ms)
    times = []
    for i in range(5):
        parent = torch.randn(b, h + i, s, d, device=device)
        q = parent[:, :h, :, :]
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            F.scaled_dot_product_attention(q, q, q)
        torch.neuron.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    # Median should be fast — no recompilation. Use median to filter jitter.
    median_time = sorted(times)[len(times) // 2]
    assert median_time < 10, (
        f"Median SDPA call took {median_time:.1f}ms (expected <10ms, "
        f"cache key may include strides). All times: " + ", ".join(f"{t:.1f}" for t in times)
    )


def test_fused_attention_overrideable_lse_shape():
    """Test that _scaled_dot_product_fused_attention_overrideable returns LSE with
    ATen-spec shape [B, H, S] (not the NKI tiled layout [B, H, 128, S/128]).

    Required for ring attention (diffusers context parallelism) which expects
    LSE shape [B, H, S] to accumulate partial attention results across ranks.
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    device = torch.device("neuron", 0)
    batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

    # Call the overrideable op directly (training=True since requires_grad=True)
    result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        q, k, v, attn_bias=None, dropout_p=0.0, is_causal=False, return_debug_mask=False
    )
    out, lse = result[0], result[1]

    # Output shape: [B, H, S, D]
    assert out.shape == (
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    ), f"Unexpected output shape: {out.shape}"
    # LSE shape must match ATen spec: [B, H, S]
    assert lse.shape == (batch_size, num_heads, seq_len), (
        f"LSE shape {lse.shape} does not match ATen spec [B, H, S] = "
        f"[{batch_size}, {num_heads}, {seq_len}]. "
        f"NKI tiled layout [B, H, 128, S/128] must be reshaped before returning."
    )
    assert lse.dtype == torch.float32, f"LSE dtype should be float32, got {lse.dtype}"


def test_fused_attention_overrideable_lse_inside_autograd_function():
    """Test that LSE is non-empty inside torch.autograd.Function.forward.

    torch.autograd.Function.forward runs with grad disabled, so torch.is_grad_enabled()
    is False. The NKI kernel must still compute and return LSE because
    _scaled_dot_product_fused_attention_overrideable always returns LSE per its
    ATen output contract — ring attention depends on this.
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    device = torch.device("neuron", 0)
    batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64
    lse_results = {}

    class _TestFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            # grad is disabled here — LSE must still be computed
            assert (
                not torch.is_grad_enabled()
            ), "Expected grad to be disabled in autograd.Function.forward"
            result = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                q, k, v, attn_bias=None, dropout_p=0.0, is_causal=False, return_debug_mask=False
            )
            lse_results["lse"] = result[1]
            return result[0]

        @staticmethod
        def backward(ctx, grad):
            return grad, grad, grad

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

    _TestFn.apply(q, k, v)

    lse = lse_results["lse"]
    assert lse.numel() > 0, "LSE must not be empty inside autograd.Function.forward (grad disabled)"
    assert lse.shape == (
        batch_size,
        num_heads,
        seq_len,
    ), f"LSE shape {lse.shape} != expected [{batch_size}, {num_heads}, {seq_len}]"


# ---------------------------------------------------------------------------
# Non-contiguous input stride tests
# ---------------------------------------------------------------------------
# MLIR path: D > 128 (NKI requires D <= 128)
# NKI path:  D <= 128, L % 512 == 0
#
# For transpose(1,2) to hit NKI the original tensor must be (B, L, H, D)
# so that after transpose the shape is (B, H, L, D) with L%512==0, D<=128.
# ---------------------------------------------------------------------------

_MLIR_NONCONTIG_CASES = [
    ("chunk_last_dim", lambda: torch.randn(1, 1, 512, 512 * 3).chunk(3, dim=-1)[0]),
    ("transpose_1_2", lambda: torch.randn(2, 4, 512, 512).transpose(1, 2)),
    (
        "as_strided_gaps",
        lambda: torch.randn(1, 1, 512, 1024).as_strided(
            (1, 1, 512, 512), (524288, 524288, 1024, 1)
        ),
    ),
    ("expand_zero_stride", lambda: torch.randn(1, 1, 512, 512).expand(2, 1, 512, 512)),
]

_NKI_NONCONTIG_CASES = [
    ("chunk_last_dim", lambda: torch.randn(1, 1, 512, 64 * 3).chunk(3, dim=-1)[0]),
    ("transpose_1_2", lambda: torch.randn(2, 512, 4, 64).transpose(1, 2)),  # (B,L,H,D) → (B,H,L,D)
    (
        "as_strided_gaps",
        lambda: torch.randn(1, 1, 512, 128).as_strided((1, 1, 512, 64), (32768, 32768, 128, 1)),
    ),
    ("expand_zero_stride", lambda: torch.randn(1, 1, 512, 64).expand(2, 1, 512, 64)),
]


def _run_noncontig_sdpa(q_cpu_fn, cpp_dispatch):
    """Run SDPA on neuron and assert output strides and values match CPU."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    import os

    orig = os.environ.get("TORCH_NEURONX_CPP_DISPATCH")
    os.environ["TORCH_NEURONX_CPP_DISPATCH"] = "1" if cpp_dispatch else "0"
    try:
        q_cpu = q_cpu_fn()
        q_neu = q_cpu.to("neuron")
        with torch.no_grad():
            out_cpu = F.scaled_dot_product_attention(q_cpu, q_cpu, q_cpu)
            out_neu = F.scaled_dot_product_attention(q_neu, q_neu, q_neu)
        assert (
            out_neu.stride() == out_cpu.stride()
        ), f"stride mismatch: neuron={out_neu.stride()} cpu={out_cpu.stride()}"
        torch.testing.assert_close(out_neu.cpu(), out_cpu, rtol=5e-2, atol=5e-3)
    finally:
        if orig is None:
            os.environ.pop("TORCH_NEURONX_CPP_DISPATCH", None)
        else:
            os.environ["TORCH_NEURONX_CPP_DISPATCH"] = orig


@pytest.mark.parametrize(
    "name,q_cpu_fn", _MLIR_NONCONTIG_CASES, ids=[c[0] for c in _MLIR_NONCONTIG_CASES]
)
@pytest.mark.parametrize("cpp_dispatch", [False, True], ids=["python_dispatch", "cpp_dispatch"])
def test_sdpa_noncontig_mlir(name, q_cpu_fn, cpp_dispatch):
    """SDPA output strides match CPU for non-contiguous inputs on the MLIR path (D=512 > 128)."""
    _run_noncontig_sdpa(q_cpu_fn, cpp_dispatch)


@pytest.mark.parametrize(
    "name,q_cpu_fn", _NKI_NONCONTIG_CASES, ids=[c[0] for c in _NKI_NONCONTIG_CASES]
)
@pytest.mark.parametrize("cpp_dispatch", [False, True], ids=["python_dispatch", "cpp_dispatch"])
def test_sdpa_noncontig_nki(name, q_cpu_fn, cpp_dispatch):
    """SDPA output strides match CPU for non-contiguous inputs on the NKI path (D=64, L%512==0)."""
    _run_noncontig_sdpa(q_cpu_fn, cpp_dispatch)


def test_scaled_dot_product_attention_fully_masked_rows():
    """Bool attn_mask with rows that are entirely False must produce finite output.

    Regression for fully-masked-row NaN in the MLIR softmax: a row of all-False
    bool mask is converted to additive -inf and makes max_logits = -inf, so the
    naive (logits - max_logits) subtraction yields NaN. The MLIR path now
    detects fully-masked rows and zeros their attention weights.
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(0)
    device = torch.device("neuron", 0)
    batch_size, num_heads, seq_len, d_head = 1, 4, 32, 32
    left_pad = 10

    q_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3
    k_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3
    v_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3

    # Bool mask: first `left_pad` query positions cannot attend to anything;
    # the active region is a (seq_len-left_pad)x(seq_len-left_pad) lower-triangular submatrix.
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    attn_mask_cpu = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    attn_mask_cpu[0, 0] = (i >= left_pad) & (j >= left_pad) & (j <= i)

    sdpa_kwargs = {"dropout_p": 0.0, "scale": 1.0 / math.sqrt(d_head), "is_causal": False}

    with torch.no_grad():
        output_cpu = F.scaled_dot_product_attention(
            q_cpu, k_cpu, v_cpu, attn_mask=attn_mask_cpu, **sdpa_kwargs
        )

    with torch.no_grad(), track_neuron_ops():
        output_neuron = F.scaled_dot_product_attention(
            q_cpu.to(device),
            k_cpu.to(device),
            v_cpu.to(device),
            attn_mask=attn_mask_cpu.to(device),
            **sdpa_kwargs,
        )
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")

    output_neuron_cpu = output_neuron.cpu()
    assert torch.isfinite(output_neuron_cpu).all(), (
        f"Output contains non-finite values: "
        f"nan={int(torch.isnan(output_neuron_cpu).sum())}, "
        f"inf={int(torch.isinf(output_neuron_cpu).sum())}"
    )
    # The first `left_pad` query rows are fully masked → output is 0.
    assert torch.equal(
        output_neuron_cpu[:, :, :left_pad, :],
        torch.zeros(batch_size, num_heads, left_pad, d_head),
    )
    # The remaining rows should match CPU SDPA semantics on the active submatrix.
    torch.testing.assert_close(
        output_neuron_cpu[:, :, left_pad:, :],
        output_cpu[:, :, left_pad:, :],
        rtol=1e-2,
        atol=1e-2,
    )


def test_scaled_dot_product_attention_fully_masked_rows_backward():
    """Bool attn_mask with fully-False rows must produce finite gradients.

    Regression for fully-masked-row NaN in the MLIR backward: lse is -inf and
    logits is -inf for those rows, so exp(logits - lse) would be NaN and
    cascade into grad_q / grad_k / grad_v. The backward path now detects
    fully-masked rows and zeros their probs (and thus their grads).
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(0)
    device = torch.device("neuron", 0)
    batch_size, num_heads, seq_len, d_head = 1, 4, 32, 32
    left_pad = 10

    q_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3
    k_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3
    v_cpu = torch.rand(batch_size, num_heads, seq_len, d_head, dtype=torch.float32) * 3

    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    attn_mask_cpu = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    attn_mask_cpu[0, 0] = (i >= left_pad) & (j >= left_pad) & (j <= i)

    sdpa_kwargs = {"dropout_p": 0.0, "scale": 1.0 / math.sqrt(d_head), "is_causal": False}

    grad_output = torch.randn(batch_size, num_heads, seq_len, d_head, dtype=torch.float32)

    q_neuron = q_cpu.detach().clone().to(device).requires_grad_(True)
    k_neuron = k_cpu.detach().clone().to(device).requires_grad_(True)
    v_neuron = v_cpu.detach().clone().to(device).requires_grad_(True)

    with track_neuron_ops():
        output_neuron = F.scaled_dot_product_attention(
            q_neuron, k_neuron, v_neuron, attn_mask=attn_mask_cpu.to(device), **sdpa_kwargs
        )
        output_neuron.backward(grad_output.to(device))
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    grad_q = q_neuron.grad.cpu()
    grad_k = k_neuron.grad.cpu()
    grad_v = v_neuron.grad.cpu()

    for name, grad in [("grad_q", grad_q), ("grad_k", grad_k), ("grad_v", grad_v)]:
        assert torch.isfinite(grad).all(), (
            f"{name} contains non-finite values: "
            f"nan={int(torch.isnan(grad).sum())}, inf={int(torch.isinf(grad).sum())}"
        )

    # The first `left_pad` query rows are fully masked → grad_q is 0 on those rows.
    assert torch.equal(
        grad_q[:, :, :left_pad, :], torch.zeros(batch_size, num_heads, left_pad, d_head)
    )


def _repeat_kv_expand_reshape(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Verbatim of transformers.integrations.sdpa_attention.repeat_kv.

    The expand+reshape produces a view with stride=0 on broadcast dims.
    """
    b, kv, s, d = t.shape
    return t[:, :, None, :, :].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


@pytest.mark.parametrize(
    "q_heads,kv_heads,seq_len,d_head,dtype",
    [
        # MLIR backward path (D > 128).
        (8, 1, 1280, 256, torch.bfloat16),
        # NKI backward path (D <= 128, L % 512 == 0).
        (8, 1, 512, 64, torch.bfloat16),
    ],
    ids=["mlir_path_d256", "nki_path_d64"],
)
def test_sdpa_backward_stride0_kv_from_expand_reshape(q_heads, kv_heads, seq_len, d_head, dtype):
    """Regression: SDPA backward must not crash on stride=0 K/V from repeat_kv.

    Numerical accuracy for GQA backward is covered by
    test_scaled_dot_product_attention_accuracy.
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(0)
    device = torch.device("neuron", 0)
    batch_size = 1
    n_rep = q_heads // kv_heads

    q = torch.randn(
        batch_size, q_heads, seq_len, d_head, dtype=dtype, device=device
    ).requires_grad_()
    k = torch.randn(
        batch_size, kv_heads, seq_len, d_head, dtype=dtype, device=device
    ).requires_grad_()
    v = torch.randn(
        batch_size, kv_heads, seq_len, d_head, dtype=dtype, device=device
    ).requires_grad_()

    k_in = _repeat_kv_expand_reshape(k, n_rep)
    v_in = _repeat_kv_expand_reshape(v, n_rep)
    assert k_in.stride()[:2] == (0, 0), f"expected stride=(0,0,...), got {k_in.stride()}"
    assert v_in.stride()[:2] == (0, 0), f"expected stride=(0,0,...), got {v_in.stride()}"

    grad_output = torch.randn(batch_size, q_heads, seq_len, d_head, dtype=dtype, device=device)

    with track_neuron_ops():
        out = F.scaled_dot_product_attention(
            q, k_in, v_in, attn_mask=None, dropout_p=0.0, is_causal=True, enable_gqa=False
        )
        out.backward(grad_output)  # used to raise overlap-write here

        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    # autograd sum-reduces grads back to the original kv_heads shape
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape

    for name, g in [("q.grad", q.grad), ("k.grad", k.grad), ("v.grad", v.grad)]:
        assert torch.isfinite(g).all(), f"{name} contains non-finite values"

    # Check leaf grad strides match CPU reference
    cpu = torch.device("cpu")
    q_cpu = torch.randn(
        batch_size, q_heads, seq_len, d_head, dtype=dtype, device=cpu
    ).requires_grad_()
    k_cpu = torch.randn(
        batch_size, kv_heads, seq_len, d_head, dtype=dtype, device=cpu
    ).requires_grad_()
    v_cpu = torch.randn(
        batch_size, kv_heads, seq_len, d_head, dtype=dtype, device=cpu
    ).requires_grad_()
    assert q_cpu.device.type == "cpu"
    assert k_cpu.device.type == "cpu"
    assert v_cpu.device.type == "cpu"
    k_in_cpu = _repeat_kv_expand_reshape(k_cpu, n_rep)
    v_in_cpu = _repeat_kv_expand_reshape(v_cpu, n_rep)
    grad_output_cpu = torch.randn(batch_size, q_heads, seq_len, d_head, dtype=dtype, device=cpu)

    out_cpu = F.scaled_dot_product_attention(
        q_cpu,
        k_in_cpu,
        v_in_cpu,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=False,
    )
    out_cpu.backward(grad_output_cpu)

    assert q.grad.stride() == q_cpu.grad.stride(), (q.grad.stride(), q_cpu.grad.stride())
    assert k.grad.stride() == k_cpu.grad.stride(), (k.grad.stride(), k_cpu.grad.stride())
    assert v.grad.stride() == v_cpu.grad.stride(), (v.grad.stride(), v_cpu.grad.stride())


def test_scaled_dot_product_attention_zero_qk_head_dim():
    """Zero-size query/key head dim (embedding dim == 0).

    The fused NKI/MLIR kernels behind the 'overrideable' backend cannot handle
    empty inputs, so `_fused_sdp_choice_neuron` routes this to the math backend
    (matmul + softmax), which handles empty tensors like CPU/CUDA. Regression
    test for the "Expected Tensor but got InvalidTag" corruption on head_dim=0.
    """
    torch.manual_seed(0)
    device = torch.device("neuron", 0)
    for value_dim in (1, 8):
        for scale in (1.0, None):
            q = torch.empty((1, 16, 16, 0), device=device)
            k = torch.empty((1, 16, 16, 0), device=device)
            v = torch.randn((1, 16, 16, value_dim), device=device)

            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
            ref = F.scaled_dot_product_attention(
                q.cpu(), k.cpu(), v.cpu(), dropout_p=0.0, scale=scale
            )

            assert out.shape == (1, 16, 16, value_dim)
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), ref)


def test_scaled_dot_product_attention_empty_value():
    """Empty value tensor: head_dim_v == 0 (Q/K non-empty) and seq_k == 0.

    These are the empty-V edge cases. head_dim_v == 0 is caught only by the
    `value.numel() == 0` branch of `_fused_sdp_choice_neuron` (Q and K are
    non-empty there), so this guards that specific routing. The math backend
    produces the CPU-matching empty / all-zero output; the fused NKI/MLIR
    kernels reject empty inputs.
    """
    torch.manual_seed(0)
    device = torch.device("neuron", 0)

    # (label, q_shape, k_shape, v_shape) — expected out shape is q_shape[:-1] + (v_shape[-1],)
    cases = [
        ("head_dim_v=0", (1, 4, 10, 8), (1, 4, 10, 8), (1, 4, 10, 0)),
        ("seq_k=0", (1, 4, 10, 8), (1, 4, 0, 8), (1, 4, 0, 8)),
    ]
    for label, q_shape, k_shape, v_shape in cases:
        q = torch.randn(*q_shape, device=device, requires_grad=True)
        k = torch.randn(*k_shape, device=device, requires_grad=True)
        v = torch.randn(*v_shape, device=device, requires_grad=True)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        expected_shape = (*q_shape[:-1], v_shape[-1])
        assert out.shape == expected_shape, label
        assert out.device.type == "neuron"

        ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), dropout_p=0.0)
        torch.testing.assert_close(out.cpu(), ref)

        # Backward must produce input-shaped grads (matches upstream test_sdpa_zero_output).
        out.sum().backward()
        assert q.grad.shape == q.shape, label
        assert k.grad.shape == k.shape, label
        assert v.grad.shape == v.shape, label


def test_scaled_dot_product_attention_float64_downcasts():
    """A float64 query computes in float32 and returns float64.

    Neuron computes SDPA in float32, and the platform casts float64 to float32
    for every other op (add, matmul, softmax all accept float64). SDPA used to
    reject Double outright because it skips DowncastArgs.
    """
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(42)
    device = torch.device("neuron", 0)
    cpu_q = torch.randn(1, 2, 8, 16, dtype=torch.float64)

    with torch.no_grad():
        actual = F.scaled_dot_product_attention(
            cpu_q.to(device), cpu_q.to(device), cpu_q.to(device)
        )
        expected = F.scaled_dot_product_attention(cpu_q, cpu_q, cpu_q)

    assert actual.dtype == torch.float64
    assert actual.shape == expected.shape
    # Compared at float32 precision: the values are computed in float32 on Neuron.
    torch.testing.assert_close(
        actual.cpu().to(torch.float32), expected.to(torch.float32), rtol=1e-2, atol=1e-2
    )


def test_scaled_dot_product_attention_float64_with_mask():
    """A float64 attn_mask is downcast alongside query, key and value."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    torch.manual_seed(42)
    device = torch.device("neuron", 0)
    cpu_q = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    cpu_mask = torch.zeros(8, 8, dtype=torch.float64)

    with torch.no_grad():
        actual = F.scaled_dot_product_attention(
            cpu_q.to(device), cpu_q.to(device), cpu_q.to(device), cpu_mask.to(device)
        )
        expected = F.scaled_dot_product_attention(cpu_q, cpu_q, cpu_q, cpu_mask)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(
        actual.cpu().to(torch.float32), expected.to(torch.float32), rtol=1e-2, atol=1e-2
    )
