# ruff: noqa: N806

import logging
from typing import ClassVar

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


# Float-score counterpart of template_mask_module: applies a traced score_mod
# to one (128, 128) tile, for exercising scalar-op lowerings end-to-end.
template_score_module = """
import nki
import nki.language as nl
import nki.isa as nisa
def score_mod(score_tile, b_tile, h_tile, q_idx_tile, kv_idx_tile):
    {{ render_score_function(
            score="score_tile",
            b="b_tile",
            h="h_tile",
            q_idx="q_idx_tile",
            kv_idx="kv_idx_tile",
            prefix="score_tmp",
        ) | indent(width=4)}}
@nki.jit
def score_kernel(score, b, h, q_idx, kv_idx):
    M, N = score.shape
    result = nl.ndarray((M, N), dtype=score.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // 128):
        for n in nl.affine_range(N // 128):
            score_tile = nl.ndarray((128, 128), dtype=score.dtype, buffer=nl.sbuf)
            b_tile = nl.ndarray((128, 128), dtype=b.dtype, buffer=nl.sbuf)
            h_tile = nl.ndarray((128, 128), dtype=h.dtype, buffer=nl.sbuf)
            q_idx_tile = nl.ndarray((128, 128), dtype=q_idx.dtype, buffer=nl.sbuf)
            kv_idx_tile = nl.ndarray((128, 128), dtype=kv_idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(score_tile, score[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(b_tile, b[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(h_tile, h[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(q_idx_tile, q_idx[m*128:(m+1)*128, n*128:(n+1)*128])
            nisa.dma_copy(kv_idx_tile, kv_idx[m*128:(m+1)*128, n*128:(n+1)*128])
            out_tile = score_mod(score_tile, b_tile, h_tile, q_idx_tile, kv_idx_tile)
            nisa.dma_copy(result[m*128:(m+1)*128, n*128:(n+1)*128], out_tile)
    return result
"""


def _run_score_mod_on_neuron(score_mod_fn, score, b, h, q_idx, kv_idx, kernel_name):
    """Trace score_mod_fn, lower it to NKI, compile on neuron, return the tile."""
    gm = make_fx(score_mod_fn)(score, b, h, q_idx, kv_idx)
    logger.debug(gm.code)
    input_spec = [
        ("score", torch.float32),
        ("b", torch.int32),
        ("h", torch.int32),
        ("q_idx", torch.int32),
        ("kv_idx", torch.int32),
    ]
    score_template = generate_mod_template(gm, input_spec, (128, 128))

    def render_score_function(score, b, h, q_idx, kv_idx, prefix):
        return jinja2.Template(score_template).render(
            score=score, b=b, h=h, q_idx=q_idx, kv_idx=kv_idx, prefix=prefix
        )

    rendered_kernel = jinja2.Template(template_score_module).render(
        render_score_function=render_score_function
    )
    logger.debug(rendered_kernel)

    temp_module = save_and_import_kernel(rendered_kernel, kernel_name)
    score_kernel = temp_module.score_kernel

    @torch.compile(backend="neuron")
    def score_nki(score, b, h, q_idx, kv_idx):
        return score_kernel(score, b, h, q_idx, kv_idx)

    return score_nki(
        score.to("neuron"),
        b.to("neuron"),
        h.to("neuron"),
        q_idx.to("neuron"),
        kv_idx.to("neuron"),
    )


class TestScalarOpLowerings:
    """Per-op end-to-end validation: trace a score_mod, run on neuron, compare to CPU."""

    CASES: ClassVar = [
        pytest.param(lambda s, b, h, q, kv: -s, 1e-5, id="neg"),
        pytest.param(lambda s, b, h, q, kv: torch.exp2(s), 1e-3, id="exp2"),
        pytest.param(lambda s, b, h, q, kv: 2.0 - s, 1e-5, id="rsub"),
        pytest.param(lambda s, b, h, q, kv: torch.floor(s * 4.0), 1e-5, id="floor"),
        pytest.param(
            lambda s, b, h, q, kv: s + (q // 4).to(torch.float32), 1e-5, id="floor_divide_pow2"
        ),
        pytest.param(
            lambda s, b, h, q, kv: s + (q // 3).to(torch.float32), 1e-5, id="floor_divide_nonpow2"
        ),
        pytest.param(
            lambda s, b, h, q, kv: s + (q % 4).to(torch.float32), 1e-5, id="remainder_pow2"
        ),
        pytest.param(
            lambda s, b, h, q, kv: s + (q % 3).to(torch.float32), 1e-5, id="remainder_nonpow2"
        ),
        pytest.param(lambda s, b, h, q, kv: torch.clamp(s, -1.0, 1.0), 1e-5, id="clamp_both"),
        pytest.param(lambda s, b, h, q, kv: torch.clamp(s, min=0.0), 1e-5, id="clamp_min"),
        pytest.param(lambda s, b, h, q, kv: torch.clamp(s, max=0.5), 1e-5, id="clamp_max"),
    ]

    @pytest.mark.parametrize("score_mod_fn,atol", CASES)
    def test_scalar_op_lowering(self, score_mod_fn, atol, request):
        torch.manual_seed(0)
        score = torch.randn((128, 128), dtype=torch.float32)
        b = torch.zeros((128, 128), dtype=torch.int32)
        h = torch.zeros((128, 128), dtype=torch.int32)
        q_idx = torch.arange(128, dtype=torch.int32).unsqueeze(1).expand(128, 128).contiguous()
        kv_idx = torch.arange(128, dtype=torch.int32).unsqueeze(0).expand(128, 128).contiguous()

        cpu = score_mod_fn(score, b, h, q_idx, kv_idx)
        nki = _run_score_mod_on_neuron(
            score_mod_fn,
            score,
            b,
            h,
            q_idx,
            kv_idx,
            kernel_name=f"score_{request.node.callspec.id}",
        )
        torch.testing.assert_close(nki.cpu().float(), cpu.float(), atol=atol, rtol=1e-3)

    def test_full_lowering_codegen(self):
        """full((), c) -> a (1, 1) constant memset tile (emitted by clamp_decomp)."""
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import full_lowering

        ctx = NKIContext()
        result = full_lowering(ctx, [], 5, dtype=torch.int32)
        assert "nisa.memset" in result.codegen()
        assert result.dst_arr.meta_tensor.shape == (1, 1)

    def test_floor_divide_pow2_codegen(self):
        """Power-of-2 floor_divide lowers to an exact right_shift."""
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import floor_divide_lowering

        ctx = NKIContext()
        src = NKIArray(ctx, "q_idx_tile", torch.int32, (128, 128))
        result = floor_divide_lowering(ctx, src, 8)  # 8 == 2^3
        assert "nl.right_shift" in result.codegen() and ", 3)" in result.codegen()

    def test_floor_divide_nonpow2_codegen(self):
        """Non-power-of-2 floor_divide lowers to floor(x * (1/N)) (no right_shift)."""
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import floor_divide_lowering

        ctx = NKIContext()
        src = NKIArray(ctx, "q_idx_tile", torch.int32, (128, 128))
        floor_divide_lowering(ctx, src, 3)
        code = "\n".join(s.codegen() for s in ctx.code_body)
        assert "nl.floor" in code and "nl.right_shift" not in code, code

    @pytest.mark.parametrize("divisor", [0, -2])
    def test_floor_divide_invalid_divisor_raises(self, divisor):
        """Non-positive divisors are unsupported (-> graph fallback), never miscompile."""
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import floor_divide_lowering

        ctx = NKIContext()
        src = NKIArray(ctx, "q_idx_tile", torch.int32, (128, 128))
        with pytest.raises(NotImplementedError):
            floor_divide_lowering(ctx, src, divisor)


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
        "b,h,seq_len,s,qk_dim,v_dim",
        [
            pytest.param(1, 4, 1024, 2048, 32, 32, id="small"),
            pytest.param(1, 4, 1024, 2048, 192, 128, id="qk192-v128"),
            pytest.param(1, 4, 1024, 2048, 512, 512, id="max-head-dim"),
        ],
    )
    def test_flex_attention_nki_lowering(self, b, h, seq_len, s, qk_dim, v_dim):
        """Test that FlexAttentionNKILowering pass processes flex_attention nodes."""
        from functorch.compile import aot_module
        from torch.fx.passes.pass_manager import PassManager
        from torch.nn.attention.flex_attention import flex_attention

        from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
            FlexAttentionNKILowering,
        )
        from torch_neuronx.nki_hop import nki_kernel_wrapper

        query = torch.randn(b, h, seq_len, qk_dim)
        key = torch.randn(b, h, s, qk_dim)
        value = torch.randn(b, h, s, v_dim)

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
        assert any(node.target is nki_kernel_wrapper for node in captured_gm.graph.nodes)


class TestFreeVariableIndexLowering:
    """Tests for aten.index lowering to NKI nl.gather_flattened (free variable support)."""

    def test_index_lowering_1d_codegen(self):
        """1-D source: broadcast the pre-loaded (1, N) row to (P, N), then gather.

        The template pre-loads the capture, so index_lowering must not re-load.
        """
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import index_lowering

        ctx = NKIContext()
        shape = (128, 128)

        source = NKIArray(ctx, "doc_id", torch.int32, (1, 512))
        idx = NKIArray(ctx, "q_idx_tile", torch.int32, shape)

        result = index_lowering(ctx, source, [idx])

        codegen_strs = [s.codegen() for s in ctx.code_body]
        all_code = "\n".join(codegen_strs) + "\n" + result.codegen()

        # No second nl.load — the template already loaded the (1, N) row.
        assert not any("nl.load" in s for s in codegen_strs), f"Unexpected nl.load:\n{all_code}"
        # Broadcast full vector to all partitions, then per-partition gather.
        assert any(
            "nl.broadcast_to(doc_id" in s for s in codegen_strs
        ), f"Expected broadcast_to(doc_id):\n{all_code}"
        assert "nl.gather_flattened" in result.codegen(), f"Expected gather_flattened:\n{all_code}"
        assert "q_idx_tile" in all_code, f"Expected q_idx_tile:\n{all_code}"

    def test_index_lowering_2d_codegen(self):
        """2-D source pre-sliced to (1, num_cols): broadcast to (P, num_cols) + gather."""
        from torch_neuronx.neuron_dynamo_backend._nki.ast import NKIContext
        from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import index_lowering

        ctx = NKIContext()
        shape = (128, 128)
        source_shape = (1, 1024)

        source = NKIArray(ctx, "seq_indices", torch.int32, source_shape)
        row_idx = NKIArray(ctx, "b_tile", torch.int32, shape)
        col_idx = NKIArray(ctx, "q_idx_tile", torch.int32, shape)

        result = index_lowering(ctx, source, [row_idx, col_idx])

        codegen_strs = [s.codegen() for s in ctx.code_body]
        all_code = "\n".join(codegen_strs) + "\n" + result.codegen()

        assert any(
            "nl.broadcast_to" in s for s in codegen_strs
        ), f"Expected broadcast_to:\n{all_code}"
        assert "nl.gather_flattened" in result.codegen(), f"Expected gather_flattened:\n{all_code}"
        assert "q_idx_tile" in all_code, f"Expected q_idx_tile:\n{all_code}"

    @pytest.mark.xfail(
        reason="make_fx cannot trace data-dependent indexing (buf[idx]); "
        "flex_attention uses reenter_make_fx internally which handles this"
    )
    def test_index_lowering_via_subgraph(self):
        """Test free variable indexing through the full subgraph lowering pipeline."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
            generate_body_template,
            generate_graph_lowering,
        )

        doc_id = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32)

        def mask_mod_with_free_var(b, h, q_idx, kv_idx, doc_id_buf):
            return doc_id_buf[q_idx] == doc_id_buf[kv_idx]

        b = torch.zeros((), dtype=torch.int32)
        h = torch.zeros((), dtype=torch.int32)
        q_idx = torch.zeros((), dtype=torch.int32)
        kv_idx = torch.zeros((), dtype=torch.int32)

        gm = make_fx(mask_mod_with_free_var, tracing_mode="symbolic")(b, h, q_idx, kv_idx, doc_id)

        input_spec = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
            ("other_buf_0", torch.int32),
        ]
        tile_shape = (128, 128)

        graph_lowering = generate_graph_lowering(gm, input_spec, tile_shape)
        body = generate_body_template(graph_lowering)

        assert "nl.load" in body, f"Expected nl.load in generated body:\n{body}"


class TestFreeVariableTemplateRendering:
    """Tests for Jinja template rendering with free variable extra args."""

    @pytest.mark.xfail(
        reason="make_fx cannot trace data-dependent indexing (buf[idx]); "
        "flex_attention uses reenter_make_fx internally which handles this"
    )
    def test_fwd_template_includes_extra_args_in_signature(self):
        """Test that rendered forward kernel includes extra buffer args in function signatures."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
            generate_body_template,
            generate_graph_lowering,
        )

        # Simple score_mod with free variable: bias[h]
        bias = torch.randn(4)

        def score_mod(score, b, h, q_idx, kv_idx, bias_buf):
            return score + bias_buf[h]

        gm = make_fx(score_mod, tracing_mode="symbolic")(
            torch.zeros((), dtype=torch.float32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            bias,
        )

        input_spec = [
            ("score", torch.float32),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
            ("other_buf_0", torch.float32),
        ]
        tile_shape = (128, 128)

        graph_lowering = generate_graph_lowering(gm, input_spec, tile_shape)
        body = generate_body_template(graph_lowering)

        # Body should reference the free variable buffer
        assert "other_buf_0" in body, f"Expected other_buf_0 in body:\n{body}"
        assert "nl.load" in body, f"Expected nl.load for free variable access:\n{body}"

    def test_render_mask_with_extra_names(self):
        """Test that Jinja rendering passes extra names to mask_mod/score_mod signatures."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
            generate_body_template,
            generate_graph_lowering,
        )

        # Simple causal mask (no free vars) — verify extra_names plumbing works
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        gm = make_fx(causal_mask)(
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
        )

        input_mask = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        tile_shape = (128, 128)

        mask_graph_lowering = generate_graph_lowering(gm, input_mask, tile_shape)
        body_template = generate_body_template(mask_graph_lowering)

        # Render the body template with the extra free-variable names injected.
        render_kwargs = {
            "b": "b_tile",
            "h": "h_tile",
            "q_idx": "q_idx_tile",
            "kv_idx": "kv_idx_tile",
            "prefix": "bufs.mask_tmp",
        }
        # Add a fake extra name to verify it gets passed through
        render_kwargs["other_buf_0"] = "other_buf_0"

        rendered = (
            jinja2.Environment(autoescape=True).from_string(body_template).render(**render_kwargs)
        )

        # Should render without errors and contain the expected NKI ops
        assert (
            "greater_equal" in rendered or "ge" in rendered.lower()
        ), f"Expected comparison op in rendered mask:\n{rendered}"


class TestNKILoweringPassFreeVariables:
    """Tests for the FlexAttentionNKILowering pass with free variable buffers.

    Verify the pass runs to completion on flex_attention with score_mod /
    mask_mod closures that capture tensors (graph-fallback path included).
    """

    @pytest.mark.xfail(
        reason="1D scalar indexing (head_bias[h]) not yet supported in NKI lowering; "
        "NKI requires 2D SBUF tensors"
    )
    def test_nki_lowering_with_score_mod_free_variable(self):
        """Test NKI lowering pass handles flex_attention with score_mod free variables."""
        from functorch.compile import aot_module
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.pass_manager import PassManager
        from torch.nn.attention.flex_attention import flex_attention

        from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
            FlexAttentionNKILowering,
        )

        B, H, L, S, E = 1, 4, 1024, 2048, 32
        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        head_bias = torch.randn(H)

        def biased_score_mod(score, b, h, q_idx, kv_idx):
            return score + head_bias[h]

        block_mask = create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=L,
            KV_LEN=S,
            BLOCK_SIZE=(128, 512),
            device="cpu",
        )

        captured_gm_holder = []

        def capture_compiler(gm, example_inputs):
            captured_gm_holder.append(gm)
            return gm

        class FreeVarFlexAttnModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, score_mod=biased_score_mod, block_mask=block_mask)

        model = FreeVarFlexAttnModule()

        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)
            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0, "Graph not captured"
        captured_gm = captured_gm_holder[0]

        # Verify the captured graph has flex_attention with other_buffers
        flex_nodes = [
            n
            for n in captured_gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.flex_attention
        ]
        assert len(flex_nodes) > 0, "No flex_attention nodes found in captured graph"

        # Check that score_mod_other_buffers is non-empty (args[7])
        flex_node = flex_nodes[0]
        if len(flex_node.args) > 7:
            score_mod_other_buffers = flex_node.args[7]
            assert (
                len(score_mod_other_buffers) > 0
            ), "Expected non-empty score_mod_other_buffers for free variable score_mod"

        # Apply NKI lowering pass
        nki_pass = FlexAttentionNKILowering()
        pm = PassManager(passes=[nki_pass])
        result = pm(captured_gm)

        assert result.modified, "NKI lowering pass should have modified the graph"

    def test_nki_lowering_with_mask_mod_free_variable(self):
        """Test NKI lowering pass handles flex_attention with mask_mod free variables."""
        from functorch.compile import aot_module
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.pass_manager import PassManager
        from torch.nn.attention.flex_attention import flex_attention

        from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
            FlexAttentionNKILowering,
        )

        B, H, L, S, E = 2, 2, 1024, 1024, 32
        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        prefix_length = torch.tensor([512, 768], dtype=torch.int32)

        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx <= prefix_length[b]

        block_mask = create_block_mask(
            prefix_mask,
            B=B,
            H=H,
            Q_LEN=L,
            KV_LEN=S,
            BLOCK_SIZE=(128, 512),
            device="cpu",
        )

        captured_gm_holder = []

        def capture_compiler(gm, example_inputs):
            captured_gm_holder.append(gm)
            return gm

        class FreeVarMaskModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, block_mask=block_mask)

        model = FreeVarMaskModule()

        with FakeTensorMode(allow_non_fake_inputs=True):
            aot_model = aot_module(model, fw_compiler=capture_compiler)
            with torch.no_grad():
                _ = aot_model(query, key, value)

        assert len(captured_gm_holder) > 0, "Graph not captured"
        captured_gm = captured_gm_holder[0]

        # Apply NKI lowering pass.
        nki_pass = FlexAttentionNKILowering()
        pm = PassManager(passes=[nki_pass])
        result = pm(captured_gm)

        assert result is not None


class TestScoreModDtype:
    """score_mod / joint_mod must be lowered on the same dtype the kernels use for the score tile.

    torch traces score_mod with a placeholder of the query dtype, but the kernels hand it an fp32
    tile. If the lowering trusts the traced dtype, every score_mod intermediate is sized in bf16 for
    bf16 models, and a large additive bias such as `score + (q_idx - kv_idx)` gets rounded to bf16.
    The kernel templates and the lowering both take the dtype from
    kernel_rendering.SCORE_TILE_DTYPE.
    """

    @staticmethod
    def _render(kind, monkeypatch):
        """Lower a bf16 `rel` score_mod through the pass; return the rendered kernel source."""
        from torch.nn.attention.flex_attention import flex_attention

        from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
            FlexAttentionNKILowering,
        )

        B, H, L, E = 1, 2, 512, 64

        def rel(score, b, h, q_idx, kv_idx):
            return score + (q_idx - kv_idx)

        block_mask = create_block_mask(
            causal, B=None, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=(128, 128), device="cpu"
        )

        class RelModule(torch.nn.Module):
            def forward(self, q, k, v):
                return flex_attention(q, k, v, score_mod=rel, block_mask=block_mask)

        # Spy on the rendered kernel source (rendering happens before any kernel tracing).
        rendered = {}
        for which in ("fwd", "bwd"):
            orig = getattr(FlexAttentionNKILowering, f"_render_{which}_kernel_template")

            def spy(self, *args, _orig=orig, _which=which, **kwargs):
                rendered[_which] = _orig(self, *args, **kwargs)
                return rendered[_which]

            monkeypatch.setattr(FlexAttentionNKILowering, f"_render_{which}_kernel_template", spy)

        graphs = {}

        def capture(which):
            def compiler(gm, example_inputs):
                graphs[which] = gm
                return gm

            return compiler

        q, k, v = (
            torch.randn(B, H, L, E, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        )
        # Real (tiny) CPU tensors through dynamo + aot_autograd, like the neuron backend does; the
        # bwd graph is only compiled once backward() actually runs.
        backend = aot_autograd(fw_compiler=capture("fwd"), bw_compiler=capture("bwd"))
        torch.compile(RelModule(), backend=backend)(q, k, v).sum().backward()

        assert kind in graphs, f"{kind} graph not captured"
        FlexAttentionNKILowering()(graphs[kind])
        assert kind in rendered, f"{kind} kernel was not rendered"
        return rendered[kind]

    @staticmethod
    def _score_tile_dtype(kind, src):
        import re

        pattern = {
            "fwd": (
                r"bufs\.mm1_copy_sb = allocator\.alloc_sbuf_tensor\("
                r"\s*shape=\([^)]*\),\s*dtype=(nl\.\w+)"
            ),
            "bwd": r"pre_mod_scores = ndarray\(.*,\s*(nl\.\w+)\)",
        }[kind]
        found = re.findall(pattern, src)
        assert len(found) == 1, f"expected exactly one score tile allocation in {kind}: {found}"
        return found[0]

    @staticmethod
    def _score_mod_temp_dtypes(src):
        import re

        return re.findall(r"score_tmp_\d+ = .*?dtype=nl\.(\w+)", src)

    @pytest.mark.parametrize("kind", ["fwd", "bwd"])
    def test_score_tile_and_intermediates_are_fp32_for_bf16_query(
        self, kind, monkeypatch, disable_flex_attention_device_check
    ):
        src = self._render(kind, monkeypatch)
        assert self._score_tile_dtype(kind, src) == "nl.float32"
        dtypes = self._score_mod_temp_dtypes(src)
        assert dtypes, f"no score_mod intermediates found in rendered {kind} kernel"
        assert "bfloat16" not in dtypes, f"score_mod intermediates lowered in bf16: {dtypes}"
        assert "float32" in dtypes, f"expected fp32 score_mod intermediates: {dtypes}"

    @pytest.mark.parametrize("kind", ["fwd", "bwd"])
    def test_kernel_and_lowering_follow_the_same_constant(
        self, kind, monkeypatch, disable_flex_attention_device_check
    ):
        """Changing SCORE_TILE_DTYPE moves the kernel tile and the lowered temporaries together."""
        from torch_neuronx.neuron_dynamo_backend.flex_attention import kernel_rendering

        monkeypatch.setattr(kernel_rendering, "SCORE_TILE_DTYPE", torch.bfloat16)
        src = self._render(kind, monkeypatch)
        assert self._score_tile_dtype(kind, src) == "nl.bfloat16"
        assert "float32" not in self._score_mod_temp_dtypes(src)
        assert "bfloat16" in self._score_mod_temp_dtypes(src)
