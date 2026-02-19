"""Tests for scaled_dot_product_attention operation"""

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
        (torch.float32, 8, 8),  # MHA float32
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

    torch.manual_seed(42)
    device = torch.device("neuron", 0)
    batch_size, seq_len, d_head = 1, 512, 64

    # Generate tensors and clamp to reasonable range
    scale = d_head**-0.5

    q_cpu = torch.randn(batch_size, q_heads, seq_len, d_head, dtype=dtype).clamp(-2, 2) * scale
    k_cpu = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype).clamp(-2, 2) * scale
    v_cpu = torch.randn(batch_size, kv_heads, seq_len, d_head, dtype=dtype).clamp(-2, 2) * scale

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
        output_neuron.backward(grad_output.to(device))

        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    # Compare results with dtype-appropriate tolerances
    rtol, atol = (1e-2, 2e-3) if dtype == torch.float32 else (5e-2, 5e-3)

    torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(q_neuron.grad.cpu(), q_grad_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_neuron.grad.cpu(), k_grad_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(v_neuron.grad.cpu(), v_grad_cpu, rtol=rtol, atol=atol)


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
        (2, 2, 64, 256, 0.1, torch.float32, False, True),
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
    "torch_neuronx.python_ops.torch_mlir.kernel.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_correctness(mock_compiler):
    """Test that IR cache is properly incremented and hit on subsequent calls."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

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
    "torch_neuronx.python_ops.torch_mlir.kernel.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_different_inputs_same_shape(mock_compiler):
    """Test IR cache hit with different inputs, same shapes."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

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
    "torch_neuronx.python_ops.torch_mlir.kernel.convert_fx_to_stablehlo",
    wraps=convert_fx_to_stablehlo,
)
def test_sdpa_ir_cache_different_shapes(mock_compiler):
    """Test that IR cache creates separate entries for different shapes and different inputs."""
    if not torch.neuron.is_available():
        pytest.skip("Neuron device not available")

    device = torch.device("neuron", 0)

    initial_count = mock_compiler.call_count

    # First shape: batch_size=2, num_heads=4, seq_len=128, head_dim=64
    torch.manual_seed(42)
    q1 = torch.randn(2, 4, 512, 128, device=device)
    k1 = torch.randn(2, 4, 512, 128, device=device)
    v1 = torch.randn(2, 4, 512, 128, device=device)

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
    q3 = torch.randn(2, 4, 512, 128, device=device)
    k3 = torch.randn(2, 4, 512, 128, device=device)
    v3 = torch.randn(2, 4, 512, 128, device=device)

    with torch.no_grad():
        output3 = F.scaled_dot_product_attention(q3, k3, v3, dropout_p=0.0, is_causal=False)

    assert mock_compiler.call_count == traced_call_count, "Compiler was not called as expected"

    # Verify output shapes
    assert output1.shape == (2, 4, 512, 128)
    assert output2.shape == (1, 8, 512, 128)
    assert output3.shape == (2, 4, 512, 128)


def test_scaled_dot_product_attention_uses_mlir_with_dropout():
    """Test that MLIR implementation is used when dropout is non-zero."""

    from torch_neuronx.python_ops.scaled_dot_product_fused_attention import (
        ScaledDotProductFusedAttentionNKIImpl,
    )
    from torch_neuronx.python_ops.torch_mlir.ops.scaled_dot_product_attention import (
        ScaledDotProductAttnMLIRImpl,
    )

    nki_called = {"count": 0}
    mlir_called = {"count": 0}

    # Wrap NKI implementation to track calls
    orig_nki_execute = ScaledDotProductFusedAttentionNKIImpl._execute_impl

    def wrapped_nki_execute(self, *args, **kwargs):
        nki_called["count"] += 1
        return orig_nki_execute(self, *args, **kwargs)

    # Wrap MLIR implementation to track calls
    orig_mlir_execute = ScaledDotProductAttnMLIRImpl._execute_impl

    def wrapped_mlir_execute(self, *args, **kwargs):
        mlir_called["count"] += 1
        return orig_mlir_execute(self, *args, **kwargs)

    patches = [
        patch.object(ScaledDotProductFusedAttentionNKIImpl, "_execute_impl", wrapped_nki_execute),
        patch.object(ScaledDotProductAttnMLIRImpl, "_execute_impl", wrapped_mlir_execute),
    ]

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        device = "neuron"
        batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        with torch.no_grad():
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.device.type == "neuron"
        mlir_calls = mlir_called["count"]
        # Verify MLIR implementation was called and NKI was not
        assert mlir_calls >= 1, (
            f"Expected MLIR implementation to be called with dropout; " f"mlir calls: {mlir_calls}"
        )
        assert nki_called["count"] == 0, "NKI implementation should not be called with dropout"
