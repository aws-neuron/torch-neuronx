"""Tests for aten::isin (membership test) and its out variants on Neuron.

torch.isin(elements, test_elements) returns a bool tensor of elements.shape,
True where the element appears in test_elements. Covers the Tensor_Tensor,
Tensor_Scalar, and Scalar_Tensor overloads, plus invert and the out= variant.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import get_executed_op_list, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIsin:
    """Test suite for aten::isin.{Tensor_Tensor,Tensor_Scalar,Scalar_Tensor}[_out]."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_isin_basic(self, dtype):
        """Basic membership across all Neuron-supported isin dtypes.

        Covers the integer (incl. int8/int16/uint8) and float (incl. fp16/bf16)
        families. torch.isin rejects bool and complex upstream, and Neuron does
        not support float64, so those are out of scope.
        """
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            test_elements = torch.tensor([2, 5, 9], dtype=dtype)
            expected = torch.isin(elements, test_elements)
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"))
            assert result.dtype == torch.bool
            assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_bool_rejected(self):
        """isin rejects bool inputs on Neuron, matching CPU torch.isin."""
        a = torch.tensor([True, False, True]).to("neuron")
        b = torch.tensor([True]).to("neuron")
        with pytest.raises(RuntimeError, match="Bool"):
            torch.isin(a, b)

    def test_isin_invert(self):
        """invert=True flips membership."""
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            test_elements = torch.tensor([2, 5, 9], dtype=torch.int64)
            expected = torch.isin(elements, test_elements, invert=True)
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"), invert=True)
            assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_empty_test_elements(self):
        """Empty test_elements => all-False (all-True when inverted).

        The empty (0-element) test_elements case is handled in the C++ dispatcher
        (TryHandleEmptyTensors), which fills the result with the invert value
        rather than the generic zeros, so both invert values are correct.
        """
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            empty = torch.tensor([], dtype=torch.int64)
            for invert in (False, True):
                expected = torch.isin(elements, empty, invert=invert)
                result = torch.isin(elements.to("neuron"), empty.to("neuron"), invert=invert)
                assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_empty_elements(self):
        """Empty elements => empty bool result of matching shape."""
        with track_neuron_ops():
            elements = torch.tensor([], dtype=torch.int64)
            test_elements = torch.tensor([2, 5, 9], dtype=torch.int64)
            expected = torch.isin(elements, test_elements)
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"))
            assert result.shape == expected.shape
            assert result.dtype == torch.bool
            assert torch.equal(result.cpu(), expected)

    def test_isin_tensor_scalar(self):
        """Tensor_Scalar overload (scalar test element)."""
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            for invert in (False, True):
                expected = torch.isin(elements, 5, invert=invert)
                result = torch.isin(elements.to("neuron"), 5, invert=invert)
                assert result.dtype == torch.bool
                assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_scalar_tensor(self):
        """Scalar_Tensor overload (scalar element) => 0-dim bool result."""
        with track_neuron_ops():
            test_elements = torch.tensor([2, 5, 9], dtype=torch.int64)
            for invert in (False, True):
                expected = torch.isin(5, test_elements, invert=invert)
                result = torch.isin(5, test_elements.to("neuron"), invert=invert)
                assert result.shape == expected.shape
                assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_mixed_dtypes(self):
        """int32 elements vs int64 test_elements promote to a common dtype like CPU."""
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
            test_elements = torch.tensor([2, 5, 9], dtype=torch.int64)
            expected = torch.isin(elements, test_elements)
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"))
            assert result.dtype == torch.bool
            assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_non_contiguous(self):
        """Non-contiguous (transposed, strided) elements."""
        with track_neuron_ops():
            base = torch.arange(24, dtype=torch.int64).reshape(4, 6)
            expected = torch.isin(base.t()[::2], torch.tensor([0, 7, 13, 23]))
            elements_n = base.to("neuron").t()[::2]
            result = torch.isin(elements_n, torch.tensor([0, 7, 13, 23]).to("neuron"))
            assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_out_variant(self):
        """The out= variant writes into the pre-allocated bool tensor."""
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            test_elements = torch.tensor([2, 5, 9], dtype=torch.int64)
            expected = torch.isin(elements, test_elements)
            out = torch.empty(elements.shape, dtype=torch.bool, device="neuron")
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"), out=out)
            assert out.data_ptr() == result.data_ptr()
            assert torch.equal(out.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"

    def test_isin_large_test_elements(self):
        """Large test set (broadcast-compare materializes elements x test bools)."""
        with track_neuron_ops():
            elements = torch.tensor([[1, 2, 3], [4, 500, 1500]], dtype=torch.int64)
            test_elements = torch.arange(1000, dtype=torch.int64)
            expected = torch.isin(elements, test_elements)
            result = torch.isin(elements.to("neuron"), test_elements.to("neuron"))
            assert torch.equal(result.cpu(), expected)
            assert not get_executed_op_list()["fallback"], "isin fell back to CPU"
