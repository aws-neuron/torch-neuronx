"""Tests for nki kernel integration."""

from unittest.mock import MagicMock, patch

import neuronxcc.nki.typing as nt
import pytest
import torch
from neuronxcc import nki

from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx import wrap_nki


class TestBasicExecution:
    """Test basic kernel execution functionality."""

    def test_simple_dma_copy(self):
        """Test a simple DMA copy kernel."""

        @nki.jit
        def dma_copy(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Create input and output tensors
        x = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        # Execute kernel
        wrap_nki(dma_copy)(x, y)

        # Verify output matches input
        assert torch.allclose(x, y)

    def test_tensor_scale(self):
        """Test scaling a tensor by a constant."""

        @nki.jit
        def tensor_scale(x, y: nt.mutable_tensor, scale):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile * scale
            nl.store(y[0:128], value=y_tile)
            return y

        x = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")
        scale = 2.5

        wrap_nki(tensor_scale)(x, y, scale)

        expected = x * scale
        assert torch.allclose(y, expected)

    def test_element_wise_add(self):
        """Test element-wise addition of two tensors."""

        @nki.jit
        def tensor_add(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        wrap_nki(tensor_add)(x1, x2, y)

        expected = x1 + x2
        assert torch.allclose(y, expected)

    def test_different_shapes(self):
        """Test kernels with different tensor shapes."""

        @nki.jit
        def copy_2d(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Test various 2D shapes
        shapes = [(10, 10), (16, 32), (128, 64)]

        for shape in shapes:
            x = torch.randn(*shape, device="neuron")
            y = torch.empty(*shape, device="neuron")
            wrap_nki(copy_2d)(x, y)
            assert torch.allclose(x, y)

    def test_different_dtypes(self):
        """Test kernels with different data types."""

        @nki.jit
        def copy_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Test various dtypes
        dtypes = [torch.float32, torch.float16, torch.int32]

        for dtype in dtypes:
            if dtype in [torch.int32]:
                x = torch.randint(0, 100, (64,), device="neuron", dtype=dtype)
            else:
                x = torch.randn(64, device="neuron", dtype=dtype)
            y = torch.empty(64, device="neuron", dtype=dtype)

            wrap_nki(copy_kernel)(x, y)
            assert torch.equal(x, y)
