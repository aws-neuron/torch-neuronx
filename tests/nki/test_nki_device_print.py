import os

import neuronxcc.nki.language as nl
import pytest
import torch
from neuronxcc import nki

from torch_neuronx import nki_op, wrap_nki

# Use the test name as a namespace to avoid conflicts
my_namespace = "TestNKIDevicePrint"


@nki.jit
def nki_tensor_add_print(a_input, b_input):
    """NKI kernel to compute element-wise addition"""
    assert a_input.shape == b_input.shape
    assert a_input.shape[0] <= nl.tile_size.pmax

    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    c_tile = nl.add(a_tile, b_tile)

    nl.device_print("print_c_tile", c_tile)

    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    nl.store(c_output, value=c_tile)
    return c_output


@nki_op(f"{my_namespace}::add_print", mutates_args={})
def nki_add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """NKI add operation wrapper"""
    return wrap_nki(nki_tensor_add_print)(a, b)


def nki_func(a, b):
    """Function using NKI kernel with device print"""
    return nki_add_op(a, b)


def ref_func(a, b):
    """Reference function for comparison"""
    return a + b


class TestNKIDevicePrint:
    @pytest.mark.xfail(reason="Eager runtime does not fully integrate NKI runtime yet.")
    def test_nki_device_print(self):
        """Test NKI kernel with device print using neuron backend"""
        debug_dir = "/tmp/debug_dir"
        os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"] = debug_dir

        a = torch.randn(128, dtype=torch.float32)
        b = torch.randn(128, dtype=torch.float32)

        result = nki_func(a.to("neuron"), b.to("neuron")).to("cpu")
        printed_file = os.path.join(debug_dir, "print_c_tile/core_0/0/tensor.pt")
        printed_tensor = torch.load(printed_file)
        torch.testing.assert_close(result.flatten(), printed_tensor.flatten(), rtol=1e-2, atol=1e-3)

    @pytest.mark.xfail(reason="Eager runtime does not fully integrate NKI runtime yet.")
    def test_nki_device_print_compiled(self):
        """Test NKI kernel with device print using neuron backend"""
        debug_dir = "/tmp/debug_dir"
        os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"] = debug_dir

        a = torch.randn(128, dtype=torch.float32)
        b = torch.randn(128, dtype=torch.float32)

        compiled_func = torch.compile(nki_func, backend="neuron", fullgraph=True)

        result = compiled_func(a.to("neuron"), b.to("neuron")).to("cpu")
        printed_file = os.path.join(debug_dir, "print_c_tile/core_0/0/tensor.pt")
        printed_tensor = torch.load(printed_file)
        torch.testing.assert_close(result.flatten(), printed_tensor.flatten(), rtol=1e-2, atol=1e-3)
