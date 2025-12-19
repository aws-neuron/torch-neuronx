import logging
import os

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import pytest
import torch

from torch_neuronx import nki, nki_op

# Use the test name as a namespace to avoid conflicts
my_namespace = "TestNKIDyamoIntegration"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
    """NKI kernel for tensor addition"""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512

    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]

    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])

    c_tile = a_tile + b_tile

    nl.store(c_output[ix, iy], value=c_tile)

    return c_output


@nki.jit
def nki_add_alias_mul(a_input: nt.mutable_tensor, b_input):
    """NKI kernel that computes multiplication and in_place add"""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512

    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]

    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])

    c_tile = a_tile + b_tile
    c_tile_mul = a_tile * b_tile
    nl.store(a_input[ix, iy], value=c_tile)
    nl.store(c_output[ix, iy], value=c_tile_mul)
    return c_output, a_input


@nki.jit
def nki_scaled_add_inplace(a_input, scale, b_output: nt.mutable_tensor):
    """NKI kernel that computes scaled in_place add"""
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512

    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]

    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_output[ix, iy])

    c_tile = a_tile * scale + b_tile
    nl.store(b_output[ix, iy], value=c_tile)
    return b_output


@nki_op(f"{my_namespace}::add_op", mutates_args={})
def nki_add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    grid = [a.shape[0] // 128, b.shape[1] // 512]
    return nki_tensor_add_kernel_[grid](a, b)


@nki_op(f"{my_namespace}::add_inplace_and_mul", mutates_args={"a"})
def nki_add_inplace_and_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    grid = [a.shape[0] // 128, b.shape[1] // 512]
    e, _ = nki_add_alias_mul[grid](a, b)
    return e


@nki_op("nki::scaled_add_inplace", mutates_args={"b"})
def nki_scaled_add_inplace_op(a: torch.Tensor, scale: float, b: torch.Tensor) -> torch.Tensor:
    grid = [a.shape[0] // 128, b.shape[1] // 512]
    return nki_scaled_add_inplace[grid](a, scale, b)


def nki_func(a, b):
    """Function using wrapped NKI kernel"""
    c = a * 2
    d = b + 3
    e = nki_add_op(c, d)
    return e * c


def ref_func(a, b):
    """Reference CPU function"""
    c = a * 2
    d = b + 3
    e = c + d
    return e * c


def nki_aliased_func(a, b):
    """Function using wrapped NKI kernel"""
    c = a * 2
    d = b + 3
    e = nki_add_inplace_and_mul(c, d)
    return e + c


def ref_aliased_func(a, b):
    """Reference CPU function"""
    c = a * 2
    d = b + 3
    return (c + d) + c * d


def nki_scaled_add_func(a, b, scale):
    """Function using nki_scaled_add_inplace"""
    c = a * 2
    d = b.clone()
    nki_scaled_add_inplace_op(c, scale, d)
    return d


def ref_scaled_add_func(a, b, scale):
    """Reference function for scaled add inplace"""
    c = a * 2
    d = b.clone()
    return c * scale + d


class TestNKIDyamoIntegration:
    def test_nki_func_compiled(self):
        """Test nki_func with neuron backend"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)
        compiled_func = torch.compile(nki_func, backend="neuron", fullgraph=True)
        ref = ref_func(a, b)
        result = compiled_func(a.to("neuron"), b.to("neuron")).to("cpu")
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_aliased_func_compiled(self):
        """Test aliased_func with neuron backend"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)
        compiled_alias = torch.compile(nki_aliased_func, backend="neuron", fullgraph=True)
        ref = ref_aliased_func(a, b)
        result = compiled_alias(a.to("neuron"), b.to("neuron")).to("cpu")
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_nki_scaled_add_inplace(self):
        """Test nki_scaled_add_inplace with neuron backend"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)
        scale = 2.5

        compiled_func = torch.compile(nki_scaled_add_func, backend="neuron", fullgraph=True)
        ref = ref_scaled_add_func(a, b, scale)
        result = compiled_func(a.to("neuron"), b.to("neuron"), scale).to("cpu")

        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)
