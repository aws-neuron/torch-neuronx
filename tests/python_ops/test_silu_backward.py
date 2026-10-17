import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

TEST_CONFIGS = [
    # Test different tensor shapes with normal values
    pytest.param(
        (1000,),
        "normal",
        id="1d_normal",
    ),
    pytest.param(
        (32, 128),
        "normal",
        id="2d_normal",
    ),
    pytest.param(
        (8, 64, 32),
        "normal",
        id="3d_normal",
    ),
    pytest.param(
        (4, 16, 32, 32),
        "normal",
        id="4d_normal",
    ),
    # Test with negative values
    pytest.param(
        (100,),
        "negative",
        id="1d_negative",
    ),
    pytest.param(
        (16, 64),
        "negative",
        id="2d_negative",
    ),
    pytest.param(
        (4, 32, 16),
        "negative",
        id="3d_negative",
    ),
    # Test with extreme positive values
    pytest.param(
        (50,),
        "extreme_positive",
        id="1d_extreme_positive",
    ),
    pytest.param(
        (8, 32),
        "extreme_positive",
        id="2d_extreme_positive",
    ),
    pytest.param(
        (2, 16, 8),
        "extreme_positive",
        id="3d_extreme_positive",
    ),
    # Test with extreme negative values
    pytest.param(
        (50,),
        "extreme_negative",
        id="1d_extreme_negative",
    ),
    pytest.param(
        (8, 32),
        "extreme_negative",
        id="2d_extreme_negative",
    ),
    pytest.param(
        (2, 16, 8),
        "extreme_negative",
        id="3d_extreme_negative",
    ),
    # Test with mixed positive and negative values
    pytest.param(
        (200,),
        "mixed",
        id="1d_mixed",
    ),
    pytest.param(
        (16, 32),
        "mixed",
        id="2d_mixed",
    ),
    pytest.param(
        (4, 8, 16),
        "mixed",
        id="3d_mixed",
    ),
    # Test with values close to zero
    pytest.param(
        (100,),
        "near_zero",
        id="1d_near_zero",
    ),
    pytest.param(
        (8, 16),
        "near_zero",
        id="2d_near_zero",
    ),
    # Test with zero values
    pytest.param(
        (50,),
        "zeros",
        id="1d_zeros",
    ),
    pytest.param(
        (4, 8),
        "zeros",
        id="2d_zeros",
    ),
    # Test with very large tensors and extreme values
    pytest.param(
        (10000,),
        "extreme_positive",
        id="large_1d_extreme_positive",
    ),
    pytest.param(
        (10000,),
        "extreme_negative",
        id="large_1d_extreme_negative",
    ),
]

"""
Setting loose thresholds for low precision because fp32 passes. Input distribution
is very varied so global thresholds would not work. Need a mechanism to calculate
thresholds based on the data distribution. To be included in vertical testing.
"""
DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-5, 1.3e-6, id="float32"),
    pytest.param(torch.float16, 1e-1, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 1e-1, 1e-2, id="bfloat16"),
]


def generate_test_input(shape, value_type, dtype=torch.float32):
    """Generate test input tensor based on shape and value type"""
    if value_type == "normal":
        return torch.randn(shape, dtype=dtype, requires_grad=True)
    elif value_type == "negative":
        return -torch.abs(torch.randn(shape, dtype=dtype, requires_grad=True))
    elif value_type == "extreme_positive":
        return (
            torch.randn(shape, dtype=dtype, requires_grad=True) * 50 + 100
        )  # Values around 100 ± 50
    elif value_type == "extreme_negative":
        return -(
            torch.randn(shape, dtype=dtype, requires_grad=True) * 50 + 100
        )  # Values around -100 ± 50
    elif value_type == "mixed":
        tensor = torch.randn(shape, dtype=dtype, requires_grad=True) * 10
        # Ensure we have both positive and negative values
        mask = torch.rand(shape) > 0.5
        tensor_data = tensor.detach()  # Create a new tensor without grad to modify values
        tensor_data[mask] = torch.abs(tensor_data[mask])
        tensor_data[~mask] = -torch.abs(tensor_data[~mask])
        return torch.tensor(tensor_data, dtype=dtype, requires_grad=True)
    elif value_type == "near_zero":
        return (
            torch.randn(shape, dtype=dtype, requires_grad=True) * 0.01
        )  # Small values around zero
    elif value_type == "zeros":
        return torch.zeros(shape, dtype=dtype, requires_grad=True)
    else:
        raise ValueError(f"Unknown value_type: {value_type}")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSiluRegistration:
    """Test cases for silu operation"""

    @pytest.mark.parametrize("input_shape, value_type", TEST_CONFIGS)
    @pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_silu_run_on_neuron(self, input_shape, value_type, dtype, atol, rtol):
        """
        Test if the op runs on neuron and output matches CPU for varying
        dtypes, tensor shapes, sizes, and value ranges.
        """

        torch.manual_seed(0)
        x_cpu = generate_test_input(input_shape, value_type, dtype=dtype).detach()
        grad_output_cpu = torch.randn_like(x_cpu)

        # CPU reference
        x_ref = x_cpu.clone().requires_grad_(True)
        cpu_out = torch.nn.functional.silu(x_ref)
        cpu_out.backward(grad_output_cpu)
        cpu_grad = x_ref.grad

        # Neuron
        x_neuron = x_cpu.clone().to("neuron").requires_grad_(True)
        with track_neuron_ops():
            neuron_out = torch.nn.functional.silu(x_neuron)
            neuron_out.backward(grad_output_cpu.to("neuron"))
        neuron_grad = x_neuron.grad

        assert neuron_out.dtype == dtype
        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad, atol=atol, rtol=rtol)
        assert_op_runs_on_neuron("aten::silu_backward")

    @pytest.mark.parametrize("shape", [(32,), (8, 16), (4, 8, 16)])
    def test_silu_backward_grad_input_variant(self, shape):
        """Test aten::silu_backward.grad_input dispatches to Neuron."""
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.silu_backward.grad_input(grad_output, x, grad_input=grad_input)
        assert_op_runs_on_neuron("aten::silu_backward")

        expected = torch.ops.aten.silu_backward(grad_output, x)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())


# ============================================================================
# Precision regression test for silu_backward fp32 upcast
# Uses the same input distribution from isolate_silu_bwd.py that originally
# exposed the precision issue. If someone removes the fp32 upcast, this test
# will fail because Neuron bf16 error will exceed CPU bf16 error vs fp32.
# ============================================================================


def _fractional_ratio(target, baseline, reference):
    """Compute error ratio: target_error / baseline_error relative to reference.

    Returns how much worse target is compared to baseline, both measured
    against reference (fp32 ground truth). Ratio <= 1.1 means target is
    no worse than baseline.
    """
    err_baseline = (baseline.float() - reference.float()).norm().item()
    err_target = (target.float() - reference.float()).norm().item()
    return err_target / err_baseline if err_baseline > 0 else float("inf")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSiluBackwardPrecision:
    """Regression tests ensuring silu_backward computes in fp32 internally.

    These tests use the exact input distributions from the original accuracy
    investigation (isolate_silu_bwd.py). The key insight: with bf16 silu_backward,
    Neuron produces significantly worse gradients than CPU bf16 relative to fp32
    ground truth. The fp32 upcast fix makes Neuron match CPU bf16 accuracy.
    """

    B, S, D, H = 1, 2048, 4096, 12288
    MAX_RATIO = 1.1  # Neuron bf16 error must be <= 1.1x CPU bf16 error

    @pytest.fixture(autouse=True)
    def setup_weights(self):
        """Fixed seed weights matching isolate_silu_bwd.py distribution."""
        torch.manual_seed(42)
        self.x_base = torch.normal(1e-6, 1e-6, (self.B, self.S, self.D))
        self.w1_base = torch.normal(-0.81, 0.47, (self.H, self.D))
        self.w3_base = torch.normal(-0.81, 0.47, (self.H, self.D))
        self.w2_base = torch.normal(-0.81, 0.47, (self.D, self.H))

    def _run_on(self, model_fn, dt, device):
        """Run forward+backward, return weight grads and input grad."""
        m, x = model_fn(dt, device)
        x = x.detach().requires_grad_(True)
        out = m(x)
        out.sum().backward()
        grads = {
            n: p.grad.detach().cpu().float() for n, p in m.named_parameters() if p.grad is not None
        }
        grads["input"] = x.grad.detach().cpu().float()
        return grads

    def _assert_parity(self, model_fn, grad_names):
        """Assert Neuron bf16 is no worse than CPU bf16 vs fp32 ground truth."""
        ref = self._run_on(model_fn, torch.float32, "cpu")
        cpu_bf16 = self._run_on(model_fn, torch.bfloat16, "cpu")
        trn_bf16 = self._run_on(model_fn, torch.bfloat16, "neuron")

        for name in grad_names:
            ratio = _fractional_ratio(trn_bf16[name], cpu_bf16[name], ref[name])
            assert ratio <= self.MAX_RATIO, (
                f"g({name}): Neuron bf16 error is {ratio:.4f}x CPU bf16 error "
                f"(max allowed {self.MAX_RATIO}x). "
                f"silu_backward may not be upcasting to fp32."
            )

    def test_silu_w1_backward_precision(self):
        """silu(w1(x)) — the core pattern in LLM MLP layers."""

        def model_fn(dt, device):
            m = nn.Sequential(nn.Linear(self.D, self.H, bias=False), nn.SiLU()).to(dt).to(device)
            m[0].weight.data.copy_(self.w1_base.to(dt))
            return m, self.x_base.to(dt).to(device)

        self._assert_parity(model_fn, ["0.weight", "input"])

    def test_gate_up_backward_precision(self):
        """silu(w1(x)) * w3(x) — SwiGLU gate pattern."""

        class GateUp(nn.Module):
            def __init__(self, hidden_dim, intermediate_dim):
                super().__init__()
                self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

            def forward(self, x):
                return functional.silu(self.w1(x)) * self.w3(x)

        def model_fn(dt, device):
            m = GateUp(self.D, self.H).to(dt).to(device)
            m.w1.weight.data.copy_(self.w1_base.to(dt))
            m.w3.weight.data.copy_(self.w3_base.to(dt))
            return m, self.x_base.to(dt).to(device)

        self._assert_parity(model_fn, ["w1.weight", "w3.weight", "input"])

    def test_full_ff_backward_precision(self):
        """w2(silu(w1(x)) * w3(x)) — full LLM feed-forward block."""

        class FF(nn.Module):
            def __init__(self, hidden_dim, intermediate_dim):
                super().__init__()
                self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
                self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

            def forward(self, x):
                return self.w2(functional.silu(self.w1(x)) * self.w3(x))

        def model_fn(dt, device):
            m = FF(self.D, self.H).to(dt).to(device)
            m.w1.weight.data.copy_(self.w1_base.to(dt))
            m.w2.weight.data.copy_(self.w2_base.to(dt))
            m.w3.weight.data.copy_(self.w3_base.to(dt))
            return m, self.x_base.to(dt).to(device)

        self._assert_parity(model_fn, ["w1.weight", "w2.weight", "w3.weight", "input"])
