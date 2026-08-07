"""Tests for C++ NeuronDispatcher registration path.

These tests verify that when TORCH_NEURONX_CPP_DISPATCH=1, torch-mlir and @neuron_op ops
are registered through the C++ NeuronDispatcher (boxed handler) and produce
correct results.

Tests are structured in two groups:
1. Registration tests — verify _register_neuron_op bindings and state
2. LoweringResult (C++ bound class) tests
"""

import pytest
import torch

import torch_neuronx
import torch_neuronx._C as _C
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)

# ============================================================================
# 1. Registration Tests — verify _C bindings work
# ============================================================================


class TestCppDispatchRegistration:
    """Test that _register_neuron_op and _get_neuron_dispatcher_ops bindings exist."""

    def test_register_neuron_op_binding_exists(self):
        """Verify _C._register_neuron_op is callable."""
        assert hasattr(_C, "_register_neuron_op"), (
            "_C._register_neuron_op binding not found — " "NeuronDispatcher not compiled into _C.so"
        )
        assert callable(_C._register_neuron_op)

    def test_get_neuron_dispatcher_ops_binding_exists(self):
        """Verify _C._get_neuron_dispatcher_ops is callable."""
        assert hasattr(
            _C, "_get_neuron_dispatcher_ops"
        ), "_C._get_neuron_dispatcher_ops binding not found"
        assert callable(_C._get_neuron_dispatcher_ops)

    def test_get_neuron_dispatcher_ops_returns_list(self):
        """Verify _get_neuron_dispatcher_ops returns a list."""
        ops = _C._get_neuron_dispatcher_ops()
        assert isinstance(ops, list)

    def test_register_neuron_op_api_reachable(self):
        """Verify the binding API is reachable without crashing."""
        ops_before = set(_C._get_neuron_dispatcher_ops())
        assert isinstance(ops_before, set)

    def test_cross_entropy_skip_dispatch_registration(self):
        """cross_entropy_loss uses @neuron_op so it must not be in _PENDING_OPERATIONS."""
        from torch_neuronx.python_ops.torch_mlir.operation_registry import _PENDING_OPERATIONS

        ce_ops = [op for op in _PENDING_OPERATIONS if "cross_entropy" in op["aten_name"]]
        assert (
            ce_ops == []
        ), f"cross_entropy_loss should be skipped from dispatch registration, found: {ce_ops}"

    def test_preprocessing_validation_uses_original_inputs_first(self):
        """Validation sees logical dtypes before preprocessing sees downcasted inputs."""
        from torch_neuronx.python_ops.torch_mlir.operation_registry import (
            _make_preprocessing_lower,
        )

        events = []

        def validate(tensors):
            events.append(("validate", tensors[0].dtype))

        def preprocess(tensors):
            events.append(("preprocess", tensors[0].dtype))
            return lambda: None, (), {}

        lower = _make_preprocessing_lower(
            preprocess,
            "aten::_test_preprocessing_validation",
            output_params=None,
            static_argnums=None,
            static_argnames=None,
            validate_fn=validate,
        )
        lower(
            [torch.ones(1, dtype=torch.int32)],
            original_args=([torch.ones(1, dtype=torch.int8)],),
            original_kwargs={},
        )

        assert events == [
            ("validate", torch.int8),
            ("preprocess", torch.int32),
        ]


# ============================================================================
# 2. LoweringResult (C++ bound class) tests
# ============================================================================


class TestLoweringResult:
    """Test the LoweringResult C++ bound class."""

    def test_default_construction(self):
        """Test LoweringResult can be constructed with defaults."""
        from torch_neuronx.python_ops.lowering_result import LoweringResult

        lr = LoweringResult()
        assert lr.stablehlo_bytes == b""
        assert lr.cache_key == ""
        assert lr.kept_input_indices == []
        assert lr.kept_output_indices == []
        assert lr.output_specs == []
        assert lr.passthrough_mappings == {}
        assert lr.op_name == ""
        assert lr.is_inplace is False
        assert lr.has_collectives is False

    def test_construction_with_values(self):
        """Test LoweringResult with actual values."""
        from torch_neuronx.python_ops.lowering_result import LoweringResult

        lr = LoweringResult(
            stablehlo_bytes=b"\x00\x01\x02",
            cache_key="relu_f32_16x16",
            kept_input_indices=[0],
            kept_output_indices=[0],
            output_specs=[((16, 16), 6)],
            op_name="aten::relu",
            is_inplace=False,
            has_collectives=False,
        )
        assert lr.stablehlo_bytes == b"\x00\x01\x02"
        assert lr.cache_key == "relu_f32_16x16"
        assert lr.kept_input_indices == [0]
        assert lr.output_specs == [((16, 16), 6)]
        assert lr.op_name == "aten::relu"

    def test_lowering_result_has_expected_fields(self):
        """Verify LoweringResult (C++ bound class) has expected attributes."""
        from torch_neuronx.python_ops.lowering_result import LoweringResult

        lr = LoweringResult()
        for field in [
            "stablehlo_bytes",
            "cache_key",
            "kept_input_indices",
            "kept_output_indices",
            "output_specs",
            "original_output_specs",
            "passthrough_mappings",
            "input_specs",
            "kw_input_specs",
            "op_name",
            "is_inplace",
            "has_collectives",
            "is_empty_tensor",
            "input_output_aliases",
            "error",
        ]:
            assert hasattr(lr, field), f"LoweringResult missing field: {field}"


# ============================================================================
# 3. NKI Dispatch Tests — verify DispatchNkiImpl handles NKI-backed ops
# ============================================================================


class TestNkiDispatchRegistration:
    """Test that NKI-backed ops are registered through the unified dispatch path."""

    def test_nki_eager_dispatch_available(self):
        """_nki_dispatch_eager pybind11 entry point should be available."""
        assert hasattr(_C, "_nki_dispatch_eager")

    def test_sdpa_registered_in_cpp_dispatcher(self):
        """Verify SDPA ops are in the C++ dispatcher (dedicated handlers)."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "_scaled_dot_product_fused_attention_overrideable" in ops
        assert "_scaled_dot_product_fused_attention_overrideable_backward" in ops

    def test_gather_registered_in_cpp_dispatcher(self):
        """Verify gather ops are in the C++ dispatcher via @neuron_op."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "gather" in ops
        assert "gather.out" in ops

    def test_scatter_add_registered_in_cpp_dispatcher(self):
        """Verify scatter_add ops are in the C++ dispatcher via @neuron_op."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "scatter_add" in ops
        assert "scatter_add_" in ops
        assert "scatter_add.out" in ops


class TestNkiDispatchSdpaForward:
    """Test SDPA forward through DispatchNkiImpl."""

    def test_sdpa_forward_shape(self):
        """Verify SDPA forward produces correct output shape."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 8, 4096, 128)

    def test_sdpa_forward_no_nan(self):
        """Verify SDPA forward output has no NaN values."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert not out.isnan().any()

    def test_sdpa_forward_stride_contiguous(self):
        """Verify output strides match query strides for contiguous input."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.stride() == q.stride()

    def test_sdpa_forward_stride_noncontiguous(self):
        """Verify output strides match query strides for non-contiguous (transposed) input."""
        q = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron").transpose(1, 2)
        k = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron").transpose(1, 2)
        v = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron").transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.stride() == q.stride()

    def test_sdpa_forward_runs_on_neuron(self):
        """Verify SDPA forward executes on Neuron device."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        with track_neuron_ops():
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert_op_runs_on_neuron("_scaled_dot_product_fused_attention_overrideable")


class TestNkiDispatchSdpaBackward:
    """Test SDPA backward through DispatchNkiImpl."""

    def test_sdpa_backward_shapes(self):
        """Verify SDPA backward produces correct gradient shapes."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape

    def test_sdpa_backward_no_nan(self):
        """Verify SDPA backward gradients have no NaN values."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron", requires_grad=True)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()
        assert not q.grad.isnan().any()
        assert not k.grad.isnan().any()
        assert not v.grad.isnan().any()

    def test_sdpa_backward_stride_noncontiguous(self):
        """Verify backward gradient strides match input strides for non-contiguous input."""
        q = (
            torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron")
            .transpose(1, 2)
            .requires_grad_(True)
        )
        k = (
            torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron")
            .transpose(1, 2)
            .requires_grad_(True)
        )
        v = (
            torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="neuron")
            .transpose(1, 2)
            .requires_grad_(True)
        )
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()
        assert q.grad.stride() == q.stride()
        assert k.grad.stride() == k.stride()
        assert v.grad.stride() == v.stride()


class TestNkiDispatchSdpaRepeatedCalls:
    """Test that repeated SDPA calls with same shapes produce valid output."""

    def test_sdpa_repeated_calls_valid(self):
        """Verify repeated SDPA calls with same shapes produce valid output."""
        q = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        out1 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        q2 = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        k2 = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        v2 = torch.randn(1, 8, 4096, 128, dtype=torch.bfloat16, device="neuron")
        out2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
        assert out2.shape == out1.shape
        assert not out2.isnan().any()
        # Outputs should differ (different random inputs)
        assert not torch.equal(out1, out2)


class TestNkiDispatchGather:
    """Test gather through DispatchNkiImpl."""

    def test_gather_correctness(self):
        """Verify gather produces correct results against CPU reference."""
        input_cpu = torch.randn(64, 128, dtype=torch.bfloat16)
        index_cpu = torch.arange(64).unsqueeze(1).expand(64, 128)
        expected = torch.gather(input_cpu, 0, index_cpu)
        out = torch.gather(input_cpu.to("neuron"), 0, index_cpu.to("neuron"))
        assert torch.equal(out.cpu(), expected)

    def test_gather_runs_on_neuron(self):
        """Verify gather executes on Neuron device."""
        input_t = torch.randn(64, 128, dtype=torch.bfloat16, device="neuron")
        index = torch.zeros(64, 128, dtype=torch.int64, device="neuron")
        index[:, :] = torch.arange(64, device="neuron").unsqueeze(1)
        with track_neuron_ops():
            torch.gather(input_t, 0, index)
        assert_op_runs_on_neuron("gather")


class TestNkiDispatchScatterAdd:
    """Test scatter_add through DispatchNkiImpl."""

    def test_scatter_add_correctness(self):
        """Verify scatter_add produces correct results against CPU reference."""
        input_cpu = torch.zeros(64, 128, dtype=torch.bfloat16)
        src_cpu = torch.ones(64, 128, dtype=torch.bfloat16)
        index_cpu = torch.arange(64).unsqueeze(1).expand(64, 128)
        expected = torch.scatter_add(input_cpu, 0, index_cpu, src_cpu)
        out = torch.scatter_add(
            input_cpu.to("neuron"), 0, index_cpu.to("neuron"), src_cpu.to("neuron")
        )
        assert torch.equal(out.cpu(), expected)

    def test_scatter_add_runs_on_neuron(self):
        """Verify scatter_add executes on Neuron device."""
        input_t = torch.zeros(64, 128, dtype=torch.bfloat16, device="neuron")
        src = torch.ones(64, 128, dtype=torch.bfloat16, device="neuron")
        index = torch.zeros(64, 128, dtype=torch.int64, device="neuron")
        index[:, :] = torch.arange(64, device="neuron").unsqueeze(1)
        with track_neuron_ops():
            torch.scatter_add(input_t, 0, index, src)
        assert_op_runs_on_neuron("scatter_add")


class TestNkiCallScalarCacheKey:
    """Test that different scalar args produce different cache entries."""

    def test_gather_different_dim_produces_correct_results(self):
        """gather(dim=0) and gather(dim=1) must not share cache entries."""
        input_t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="neuron")
        # Same index shape for both calls — only dim differs
        index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="neuron")

        result_dim0 = torch.gather(input_t, 0, index)
        result_dim1 = torch.gather(input_t, 1, index)
        torch.neuron.synchronize()

        expected_dim0 = torch.gather(input_t.cpu(), 0, index.cpu())
        expected_dim1 = torch.gather(input_t.cpu(), 1, index.cpu())

        assert torch.allclose(result_dim0.cpu(), expected_dim0)
        assert torch.allclose(result_dim1.cpu(), expected_dim1)

    def test_sdpa_different_is_causal_produces_correct_results(self):
        """SDPA with is_causal=True vs False must not share cache entries."""
        q = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="neuron")
        k = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="neuron")
        v = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="neuron")

        with torch.no_grad():
            out_causal = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )
            out_no_causal = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
            )
        torch.neuron.synchronize()

        # Causal and non-causal should produce different results
        assert not torch.allclose(out_causal.cpu(), out_no_causal.cpu())


class TestRmsNormCppDispatch:
    """Test that rms_norm uses C++ dispatch when available."""

    def test_rms_norm_registered_in_dispatcher(self):
        """rms_norm should be registered in the C++ dispatcher."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "rms_norm" in ops

    def test_rms_norm_produces_correct_output(self):
        """rms_norm should produce results through C++ dispatch without error."""
        x = torch.randn(2, 128, dtype=torch.bfloat16, device="neuron")
        weight = torch.ones(128, dtype=torch.bfloat16, device="neuron")
        result = torch.ops.aten.rms_norm(x, [128], weight, eps=1e-5)
        torch.neuron.synchronize()
        out = result[0] if isinstance(result, (tuple | list)) else result
        assert out.device.type == "neuron"


class TestNonNkiCppDispatch:
    """Test registration of non-NKI ops via C++ dispatch."""

    def test_nonzero_registered_in_dispatcher(self):
        """nonzero should be registered in the C++ dispatcher."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "nonzero" in ops

    def test_index_registered_in_dispatcher(self):
        """index.Tensor should be registered in the C++ dispatcher."""
        ops = set(_C._get_neuron_dispatcher_ops())
        assert "index.Tensor" in ops
