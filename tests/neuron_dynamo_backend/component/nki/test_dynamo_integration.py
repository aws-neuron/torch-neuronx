import logging
import os
from typing import Optional

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import pytest
import torch
import torch.nn as nn

from torch_neuronx import nki_op, wrap_nki

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
    return wrap_nki(nki_tensor_add_kernel_)[grid](a, b)


@nki_op(f"{my_namespace}::add_inplace_and_mul", mutates_args={"a"})
def nki_add_inplace_and_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    grid = [a.shape[0] // 128, b.shape[1] // 512]
    e, _ = wrap_nki(nki_add_alias_mul)[grid](a, b)
    return e


@nki_op(f"{my_namespace}::scaled_add_inplace", mutates_args={"b"})
def nki_scaled_add_inplace_op(a: torch.Tensor, scale: float, b: torch.Tensor) -> torch.Tensor:
    grid = [a.shape[0] // 128, b.shape[1] // 512]
    return wrap_nki(nki_scaled_add_inplace)[grid](a, scale, b)


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


def ref_aliased_with_constant_func(
    scale: float, out: torch.Tensor, mode: str | None = None, input: torch.Tensor | None = None
) -> torch.Tensor:
    """Reference function for nki_aliased_kernel_with_constant"""

    scaled = out * scale if mode == "multiply" else out + scale

    if input is not None:
        return input + scaled
    else:
        return scaled


@nki.jit
def nki_aliased_kernel_with_constant(
    scale: float, out: nt.mutable_tensor, mode: str | None = None, input: torch.Tensor | None = None
):
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512

    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]

    out_tile = nl.load(out[ix, iy])
    scaled_tile = out_tile * scale

    scaled_tile = out_tile * scale if mode == "multiply" else out_tile + scale

    if input is not None:
        input_tile = nl.load(input[ix, iy])
        result_tile = input_tile + scaled_tile
    else:
        result_tile = scaled_tile

    nl.store(out[ix, iy], value=result_tile)
    return out


@nki_op(f"{my_namespace}::aliased_with_constant", mutates_args={"out"})
def nki_aliased_with_constant_op(
    scale: float, out: torch.Tensor, mode: str | None, input: torch.Tensor | None = None
) -> torch.Tensor:
    grid = [out.shape[0] // 128, out.shape[1] // 512]
    wrap_nki(nki_aliased_kernel_with_constant)[grid](scale, out, mode, input)


class TestNKIOPIntegration:
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

    @pytest.mark.parametrize(
        "mode,has_input",
        [
            ("multiply", True),
            ("multiply", False),
            (None, True),
            (None, False),
        ],
    )
    def test_nki_aliased_with_constant(self, mode, has_input):
        """Test nki_aliased_kernel_with_constant with different optional parameter combinations"""
        out = torch.ones((256, 1024), dtype=torch.float32)
        input_tensor = torch.rand((256, 1024), dtype=torch.float32) if has_input else None
        out_neuron = out.clone().to("neuron")
        input_neuron = input_tensor.clone().to("neuron") if has_input else None
        scale = 2.5

        # Reference
        ref = ref_aliased_with_constant_func(scale, out, mode, input_tensor)

        # NKI op
        compiled_func = torch.compile(
            lambda *args: nki_aliased_with_constant_op(*args), backend="neuron", fullgraph=True
        )
        compiled_func(scale, out_neuron, mode, input_neuron)

        assert torch.allclose(out_neuron.cpu(), ref, rtol=1e-5, atol=1e-5)

    def test_nki_multiout(self):
        """Test multi-output NKI kernel"""

        @nki.jit
        def nki_multiout_add_kernel(
            a_input: nl.ndarray,
            b_input: nl.ndarray,
            c_input: nl.ndarray,
        ) -> tuple[nl.ndarray, nl.ndarray]:
            assert a_input.shape == b_input.shape
            assert a_input.shape == c_input.shape
            assert a_input.shape[0] <= nl.tile_size.pmax

            a_tile = nl.load(a_input)
            b_tile = nl.load(b_input)
            c_tile = nl.load(c_input)

            ab_tile = nl.add(a_tile, b_tile)
            ac_tile = nl.add(a_tile, c_tile)

            ab_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
            ac_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

            nl.store(ab_output, value=ab_tile)
            nl.store(ac_output, value=ac_tile)

            return ab_output, ac_output

        @nki_op("test_multi_return_kernel::nki_tensor_add_kernel_3")
        def nki_tensor_add_kernel_3(
            a_input: torch.Tensor,
            b_input: torch.Tensor,
            c_input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return wrap_nki(nki_multiout_add_kernel)(a_input, b_input, c_input)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                return nki_tensor_add_kernel_3(a, b, c)

        model = SimpleModel()
        model = model.to("neuron:0")
        model = torch.compile(model, backend="neuron", fullgraph=True)

        a = torch.ones((4, 3), dtype=torch.bfloat16)
        b = torch.ones((4, 3), dtype=torch.bfloat16)
        c = torch.ones((4, 3), dtype=torch.bfloat16)

        output = model(a.to("neuron:0"), b.to("neuron:0"), c.to("neuron:0"))

        assert torch.allclose(output[0].to("cpu"), a + b)
        assert torch.allclose(output[1].to("cpu"), b + c)

    def test_model_with_cache_return_on_neuron(self):
        """Test KV cache update with mutable tensors"""

        @nki.jit
        def write_kv_cache_at_batch_kernel(
            k: nt.tensor,
            v: nt.tensor,
            k_prior: nt.mutable_tensor,
            v_prior: nt.mutable_tensor,
            batch_idx: nt.tensor,
        ):
            batch_idx = nl.load(batch_idx)

            _, h, s, d = k.shape
            _, v_h, v_s, v_d = v.shape

            _, i_h, i_s, i_d = nl.mgrid[:1, :h, :s, :d]
            _, v_i_h, v_i_s, v_i_d = nl.mgrid[:1, :v_h, :v_s, :v_d]

            nisa.dma_copy(src=k, dst=k_prior[batch_idx, i_h, i_s, i_d], oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(
                src=v, dst=v_prior[batch_idx, v_i_h, v_i_s, v_i_d], oob_mode=nisa.oob_mode.skip
            )

            return k_prior, v_prior

        @nki_op(
            "test_cache_update_kernel::write_kv_cache_at_batch_kernel",
            mutates_args={"k_prior", "v_prior"},
        )
        def write_kv_cache_at_batch(
            k: torch.Tensor,
            v: torch.Tensor,
            k_prior: torch.Tensor,
            v_prior: torch.Tensor,
            batch_index: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return wrap_nki(write_kv_cache_at_batch_kernel)(k, v, k_prior, v_prior, batch_index)

        class ModelWithCacheReturn(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k_prior", torch.zeros((1, 1, 256, 128), dtype=torch.bfloat16))
                self.register_buffer("v_prior", torch.zeros((1, 1, 256, 128), dtype=torch.bfloat16))

            def forward(self, k, v, batch_idx):
                updated_k, updated_v = write_kv_cache_at_batch(
                    k, v, self.k_prior, self.v_prior, batch_idx
                )
                return updated_k, updated_v

        model = ModelWithCacheReturn()
        model = model.to("neuron:0")
        model = torch.compile(model, backend="neuron", fullgraph=True)

        k = torch.ones((1, 1, 256, 128), dtype=torch.bfloat16)
        v = torch.ones((1, 1, 256, 128), dtype=torch.bfloat16)

        batch_idx = torch.tensor([0], dtype=torch.int32)

        model(k.to("neuron:0"), v.to("neuron:0"), batch_idx.to("neuron:0"))

        batch_idx = torch.tensor([1], dtype=torch.int32)

        result = model(k.to("neuron:0"), v.to("neuron:0"), batch_idx.to("neuron:0"))

        k, v = result[0].to("cpu"), result[1].to("cpu")

        assert torch.all(k == 1)
        assert torch.all(v == 1)


class TestNKINativeIntegration:
    def test_nki_func_compiled(self):
        """Test nki_func with neuron backend using native kernel calls"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)

        def nki_native_func(a, b):
            c = a * 2
            d = b + 3
            grid = [a.shape[0] // 128, b.shape[1] // 512]
            e = wrap_nki(nki_tensor_add_kernel_)[grid](c, d)
            return e * c

        compiled_func = torch.compile(nki_native_func, backend="neuron", fullgraph=True)
        ref = ref_func(a, b)
        result = compiled_func(a.to("neuron"), b.to("neuron")).to("cpu")
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_aliased_func_compiled(self):
        """Test aliased_func with neuron backend using native kernel calls"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)

        def nki_native_aliased_func(a, b):
            c = a * 2
            d = b + 3
            grid = [a.shape[0] // 128, b.shape[1] // 512]
            e, _ = wrap_nki(nki_add_alias_mul)[grid](c, d)
            return e + c

        compiled_alias = torch.compile(nki_native_aliased_func, backend="neuron", fullgraph=True)
        ref = ref_aliased_func(a, b)
        result = compiled_alias(a.to("neuron"), b.to("neuron")).to("cpu")
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_nki_scaled_add_inplace(self):
        """Test nki_scaled_add_inplace with neuron backend using native kernel calls"""
        a = torch.rand((256, 1024), dtype=torch.float32)
        b = torch.rand((256, 1024), dtype=torch.float32)
        scale = 2.5

        def nki_native_scaled_add_func(a, b, scale):
            c = a * 2
            d = b.clone()
            grid = [a.shape[0] // 128, b.shape[1] // 512]
            wrap_nki(nki_scaled_add_inplace)[grid](c, scale, d)
            return d

        compiled_func = torch.compile(nki_native_scaled_add_func, backend="neuron", fullgraph=True)
        ref = ref_scaled_add_func(a, b, scale)
        result = compiled_func(a.to("neuron"), b.to("neuron"), scale).to("cpu")
        assert result.shape == ref.shape
        assert result.dtype == ref.dtype
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "mode,has_input",
        [
            ("multiply", True),
            ("multiply", False),
            (None, True),
            (None, False),
        ],
    )
    def test_nki_aliased_with_constant(self, mode, has_input):
        """
        Test nki_aliased_kernel_with_constant with different optional parameter
        combinations using native kernel calls
        """
        out = torch.ones((256, 1024), dtype=torch.float32)
        input_tensor = torch.rand((256, 1024), dtype=torch.float32) if has_input else None
        out_neuron = out.clone().to("neuron")
        input_neuron = input_tensor.clone().to("neuron") if has_input else None
        scale = 2.5

        ref = ref_aliased_with_constant_func(scale, out, mode, input_tensor)

        def nki_native_aliased_with_constant(scale, out, mode, input):
            grid = [out.shape[0] // 128, out.shape[1] // 512]
            wrap_nki(nki_aliased_kernel_with_constant)[grid](scale, out, mode, input)

        compiled_func = torch.compile(
            lambda *args: nki_native_aliased_with_constant(*args), backend="neuron", fullgraph=True
        )
        compiled_func(scale, out_neuron, mode, input_neuron)
        assert torch.allclose(out_neuron.cpu(), ref, rtol=1e-5, atol=1e-5)
