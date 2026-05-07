import logging

import jinja2
import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import and_masks, create_block_mask

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIArray
from torch_neuronx.neuron_dynamo_backend._nki.subgraph_lowering import NISAPointwiseSubgraphLowering
from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
    generate_mod_template,
    save_and_import_kernel,
)

logger = logging.getLogger(__name__)

template_mask_module = """
import nki
import nki.language as nl
import nki.isa as nisa
def mask_mod(b_tile, h_tile, q_idx_tile, kv_idx_tile):
    {{ render_mask_function(
            b="b_tile",
            h="h_tile",
            q_idx="q_idx_tile",
            kv_idx="kv_idx_tile",
            prefix="mask_tmp",
        ) | indent(width=4)}}
@nki.jit
def mask_kernel(b, h, q_idx, kv_idx):
    M, N = b.shape

    result = nl.ndarray((M, N), dtype=b.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // 128):
        for n in nl.affine_range(N // 128):
            b_tile = nl.ndarray((128, 128), dtype=b.dtype, buffer=nl.sbuf)
            h_tile = nl.ndarray((128, 128), dtype=h.dtype, buffer=nl.sbuf)
            q_idx_tile = nl.ndarray((128, 128), dtype=q_idx.dtype, buffer=nl.sbuf)
            kv_idx_tile = nl.ndarray((128, 128), dtype=kv_idx.dtype, buffer=nl.sbuf)

            nisa.dma_copy(b_tile, b[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(h_tile, h[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(q_idx_tile, q_idx[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(kv_idx_tile, kv_idx[m*128:(m+1)*128, n*128:(n+1)*128])
            mask_tile = mask_mod(b_tile, h_tile, q_idx_tile, kv_idx_tile)
            nisa.dma_copy(result[m*128:(m+1)*128, n*128:(n+1)*128], mask_tile)

    return result
"""


def example_inputs():
    score = torch.ones((1024, 1024), dtype=torch.float32, device="cpu")
    b = torch.ones((1024, 1024), dtype=torch.int32, device="cpu") * 2
    h = torch.ones((1024, 1024), dtype=torch.int32, device="cpu") * 3
    q_idx = torch.arange(1024, dtype=torch.int32, device="cpu").unsqueeze(1).expand(-1, 1024)
    kv_idx = torch.arange(1024, dtype=torch.int32, device="cpu").unsqueeze(0).expand(1024, -1)
    return score, b, h, q_idx, kv_idx


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx / 2


def sliding_window_mod(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx - kv_idx < 128


sliding_window_causal = and_masks(causal, sliding_window_mod)


class TestNKILowering:
    def test_mask_mod_lowering(self):
        block_mask = create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=2048, KV_LEN=2048, device="cpu"
        )
        mask_mod = block_mask.mask_mod

        score, b, h, q_idx, kv_idx = example_inputs()

        gm = make_fx(mask_mod)(b, h, q_idx, kv_idx)
        logger.debug(gm.code)
        input_spec = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        mask_template = generate_mod_template(gm, input_spec, (128, 128))

        def render_mask_function(b, h, q_idx, kv_idx, prefix):
            return jinja2.Template(mask_template).render(
                b=b,
                h=h,
                q_idx=q_idx,
                kv_idx=kv_idx,
                prefix=prefix,
            )

        logger.debug(
            "#GENERATED KERNEL:\n"
            f"{render_mask_function('b_tile', 'h_tile', 'q_tile', 'kv_tile', 'mask_tmp')}"
        )

        template = jinja2.Template(template_mask_module)
        rendered_kernel = template.render(render_mask_function=render_mask_function)
        logger.debug(rendered_kernel)

        temp_module = save_and_import_kernel(rendered_kernel, "temp_mask_mod")
        mask_kernel = temp_module.mask_kernel

        cpu_result = mask_mod(b, h, q_idx, kv_idx)

        @torch.compile(backend="neuron")
        def mask_nki(b, h, q_idx, kv_idx):
            return mask_kernel(b, h, q_idx, kv_idx)

        nki_result = mask_nki(
            b.to("neuron"), h.to("neuron"), q_idx.to("neuron"), kv_idx.to("neuron")
        )
        torch.testing.assert_close(nki_result.to(torch.int32).cpu(), cpu_result.to(torch.int32))
        logger.debug("SUCCESS!")


class TestFlexAttentionLowering:
    """End-to-end tests for FlexAttentionNKILowering pass."""

    @pytest.mark.parametrize(
        "b,h,seq_len,s,e",
        [
            pytest.param(1, 4, 1024, 2048, 32, id="small"),
        ],
    )
    def test_flex_attention_nki_lowering(self, b, h, seq_len, s, e):
        """Test that FlexAttentionNKILowering pass processes flex_attention nodes."""
        from functorch.compile import aot_module
        from torch.fx.passes.pass_manager import PassManager
        from torch.nn.attention.flex_attention import flex_attention

        from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
            FlexAttentionNKILowering,
        )

        query = torch.randn(b, h, seq_len, e)
        key = torch.randn(b, h, s, e)
        value = torch.randn(b, h, s, e)

        captured_gm_holder = []

        def capture_compiler(gm, example_inputs):
            captured_gm_holder.append(gm)
            return gm

        def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
            return score / 2.0

        block_mask = create_block_mask(
            sliding_window_causal,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=s,
            BLOCK_SIZE=(128, 512),
            device="cpu",
        )
        # Test reference: flex_attention with both score_mod and block_mask
        with torch.no_grad():
            _ = flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask)

        class CombinedFlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        model = CombinedFlexAttnModule()

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)
            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0, "Graph not captured"
        captured_gm = captured_gm_holder[0]

        # Apply NKI lowering pass
        nki_pass = FlexAttentionNKILowering()
        pm = PassManager(passes=[nki_pass])
        result = pm(captured_gm)

        # Verify pass executed without errors
        assert result.modified
