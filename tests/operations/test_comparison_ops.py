import numpy as np
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestComparisonOps:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("neuron", 0)
        self.cpu_device = torch.device("cpu")

    def _test_scalar_comparison(self, op_name, scalar_value, expected_fn):
        """Helper to test scalar comparison operations"""
        # Test different dtypes
        for dtype in [torch.float32, torch.int32, torch.int64]:
            # Create test tensor
            cpu_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=dtype)
            neuron_tensor = cpu_tensor.to(self.device)

            # Create output tensors
            cpu_out = torch.empty_like(cpu_tensor, dtype=torch.bool)
            neuron_out = torch.empty_like(neuron_tensor, dtype=torch.bool)

            # Perform operation
            op_fn = getattr(torch, op_name)
            cpu_result = op_fn(cpu_tensor, scalar_value, out=cpu_out)
            neuron_result = op_fn(neuron_tensor, scalar_value, out=neuron_out)

            # Verify results
            expected = expected_fn(cpu_tensor, scalar_value)
            assert torch.equal(cpu_result.to("cpu"), expected)
            assert torch.equal(neuron_result.to("cpu"), expected)
            assert torch.equal(cpu_out, expected)
            assert torch.equal(neuron_out.to("cpu"), expected)

    def test_eq_runs_on_neuron(self):
        """Test that eq runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.eq(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::eq")

    def test_eq_scalar_out(self):
        """Test eq.Scalar_out operation"""
        self._test_scalar_comparison("eq", 3, lambda t, s: t == s)

    def test_ne_runs_on_neuron(self):
        """Test that ne runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.ne(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::ne")

    def test_ne_scalar_out(self):
        """Test ne.Scalar_out operation"""
        self._test_scalar_comparison("ne", 3, lambda t, s: t != s)

    def test_lt_runs_on_neuron(self):
        """Test that lt runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.lt(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::lt")

    def test_lt_scalar_out(self):
        """Test lt.Scalar_out operation"""
        self._test_scalar_comparison("lt", 3, lambda t, s: t < s)

    def test_le_runs_on_neuron(self):
        """Test that le runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.le(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::le")

    def test_le_scalar_out(self):
        """Test le.Scalar_out operation"""
        self._test_scalar_comparison("le", 3, lambda t, s: t <= s)

    def test_gt_runs_on_neuron(self):
        """Test that gt runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.gt(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::gt")

    def test_gt_scalar_out(self):
        """Test gt.Scalar_out operation"""
        self._test_scalar_comparison("gt", 3, lambda t, s: t > s)

    def test_ge_runs_on_neuron(self):
        """Test that ge runs on Neuron"""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        with track_neuron_ops():
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
            result = torch.ge(input_tensor, 2.0)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::ge")

    def test_ge_scalar_out(self):
        """Test ge.Scalar_out operation"""
        self._test_scalar_comparison("ge", 3, lambda t, s: t >= s)

    def test_float_comparisons(self):
        """Test comparison with float values including special cases"""
        cpu_tensor = torch.tensor(
            [1.0, 2.5, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
        )
        neuron_tensor = cpu_tensor.to(self.device)

        # Test with regular float
        cpu_out = torch.empty_like(cpu_tensor, dtype=torch.bool)
        neuron_out = torch.empty_like(neuron_tensor, dtype=torch.bool)

        torch.eq(cpu_tensor, 2.5, out=cpu_out)
        torch.eq(neuron_tensor, 2.5, out=neuron_out)

        assert torch.equal(cpu_out, neuron_out.to("cpu"))

        # Test with inf
        torch.eq(cpu_tensor, float("inf"), out=cpu_out)
        torch.eq(neuron_tensor, float("inf"), out=neuron_out)

        assert torch.equal(cpu_out, neuron_out.to("cpu"))

    def test_empty_tensor(self):
        """Test comparison with empty tensors"""
        cpu_tensor = torch.empty(0, dtype=torch.float32)
        neuron_tensor = cpu_tensor.to(self.device)

        cpu_out = torch.empty_like(cpu_tensor, dtype=torch.bool)
        neuron_out = torch.empty_like(neuron_tensor, dtype=torch.bool)

        torch.eq(cpu_tensor, 0, out=cpu_out)
        torch.eq(neuron_tensor, 0, out=neuron_out)

        assert cpu_out.shape == (0,)
        assert neuron_out.shape == (0,)

    def test_multidimensional(self):
        """Test comparison with multi-dimensional tensors"""
        cpu_tensor = torch.randn(3, 4, 5)
        neuron_tensor = cpu_tensor.to(self.device)

        cpu_out = torch.empty_like(cpu_tensor, dtype=torch.bool)
        neuron_out = torch.empty_like(neuron_tensor, dtype=torch.bool)

        torch.gt(cpu_tensor, 0.0, out=cpu_out)
        torch.gt(neuron_tensor, 0.0, out=neuron_out)

        assert torch.equal(cpu_out, neuron_out.to("cpu"))

    def test_type_promotion(self):
        """Test that type promotion works correctly"""
        # Int tensor compared with float scalar
        cpu_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        neuron_tensor = cpu_tensor.to(self.device)

        cpu_out = torch.empty_like(cpu_tensor, dtype=torch.bool)
        neuron_out = torch.empty_like(neuron_tensor, dtype=torch.bool)

        torch.eq(cpu_tensor, 2.0, out=cpu_out)
        torch.eq(neuron_tensor, 2.0, out=neuron_out)

        assert torch.equal(cpu_out, neuron_out.to("cpu"))

    def _test_tensor_comparison(self, op_name, expected_fn):
        """Helper to test tensor comparison operations"""
        # Create test tensors
        cpu_a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        cpu_b = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)

        neuron_a = cpu_a.to(self.device)
        neuron_b = cpu_b.to(self.device)

        # Create output tensors
        cpu_out = torch.empty_like(cpu_a, dtype=torch.bool)
        neuron_out = torch.empty_like(neuron_a, dtype=torch.bool)

        # Perform operation
        op_fn = getattr(torch, op_name)
        cpu_result = op_fn(cpu_a, cpu_b, out=cpu_out)
        neuron_result = op_fn(neuron_a, neuron_b, out=neuron_out)

        # Verify results
        expected = expected_fn(cpu_a, cpu_b)
        assert torch.equal(cpu_result, expected)
        assert torch.equal(neuron_result.to("cpu"), expected)

    def test_eq_tensor_out(self):
        """Test eq.Tensor_out operation"""
        self._test_tensor_comparison("eq", lambda a, b: a == b)

    def test_ne_tensor_out(self):
        """Test ne.Tensor_out operation"""
        self._test_tensor_comparison("ne", lambda a, b: a != b)

    def test_lt_tensor_out(self):
        """Test lt.Tensor_out operation"""
        self._test_tensor_comparison("lt", lambda a, b: a < b)

    def test_le_tensor_out(self):
        """Test le.Tensor_out operation"""
        self._test_tensor_comparison("le", lambda a, b: a <= b)

    def test_gt_tensor_out(self):
        """Test gt.Tensor_out operation"""
        self._test_tensor_comparison("gt", lambda a, b: a > b)

    def test_ge_tensor_out(self):
        """Test ge.Tensor_out operation"""
        self._test_tensor_comparison("ge", lambda a, b: a >= b)

    def test_broadcasting(self):
        """Test broadcasting in tensor comparisons"""
        cpu_a = torch.randn(3, 1, 5)
        cpu_b = torch.randn(1, 4, 5)

        neuron_a = cpu_a.to(self.device)
        neuron_b = cpu_b.to(self.device)

        # Expected shape after broadcasting: (3, 4, 5)
        cpu_out = torch.empty(3, 4, 5, dtype=torch.bool)
        neuron_out = torch.empty(3, 4, 5, dtype=torch.bool, device=self.device)

        torch.gt(cpu_a, cpu_b, out=cpu_out)
        torch.gt(neuron_a, neuron_b, out=neuron_out)

        assert torch.equal(cpu_out, neuron_out.to("cpu"))
