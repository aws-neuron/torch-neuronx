"""Shape-matrix coverage for flex_attention on both Neuron lowering paths.

Every cell uses seq_len=8192, 8 query heads and bfloat16. The matrix crosses four
mask and score patterns, two batch sizes, two key/value head counts (GQA 8:2 and MHA 8:8) and
two head dimensions.

A cell runs in the default configuration, so the compile budget picks its lowering the
way it would for a real caller. At this sequence length that covers both paths: the
sparser patterns at batch 1 stay on the NKI kernel, and the rest exceed the budget and
reach the decomposition. The cell records which pass consumed the operator and requires
one pass to take all of them, rather than requiring a particular pass.

One cell can need an hour to compile, so the suite is opt-in. Set
TORCH_NEURONX_RUN_FLEX_SHAPE_MATRIX=1 to run it. The cells whose compile time was
measured above 300 s are skipped even then.
"""

import contextlib
import time
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, noop_mask

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_legalization import (
    FlexAttentionLegalization,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    FlexAttentionNKILowering,
)
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

_SEQ_LEN = 8192
_Q_HEADS = 8
_DTYPE = torch.bfloat16
_WINDOW = 2048
# The default block size is a sentinel that the lowering rejects, so set it.
_BLOCK_SIZE = (128, 128)
# Float32 scores in one reference block. A whole score matrix does not fit in host memory.
_REF_BLOCK_SCORES = 2**26

_RUN_ENV = "TORCH_NEURONX_RUN_FLEX_SHAPE_MATRIX"

# Both lowerings matmul in bfloat16, so the relative L2 error against a float32 reference
# sits near 8e-3. The limit leaves room above that and still rejects a dropped mask or a
# missing score_mod derivative, which move the error to the order of 0.1 or more.
_L2_LIMIT = 2e-2

# Seconds one direction of one cell may take to compile. The budget is per direction, not
# per cell, so a forward and a backward each get the whole of it.
_COMPILE_BUDGET_S = 300.0

# Cells whose backward does not fit the budget, measured at 304 to 330 s. A sliding window
# holds a score-shaped tensor per live block, which the decomposition does not tile, so every
# batch-4 cell of that pattern is over.
_BACKWARD_OVER_BUDGET = frozenset(
    {
        ("sliding_window", 4, 2, 64),
        ("sliding_window", 4, 2, 128),
        ("sliding_window", 4, 8, 64),
        ("sliding_window", 4, 8, 128),
    }
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _getenv_bool(_RUN_ENV, False),
        reason=f"one cell can need an hour to compile. Set {_RUN_ENV}=1 to run the matrix.",
    ),
    pytest.mark.usefixtures("disable_flex_attention_device_check"),
]


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= _WINDOW)


def alibi_score(score, b, h, q_idx, kv_idx):
    return score + (kv_idx - q_idx) * torch.exp2(-((h + 1) * 8.0 / _Q_HEADS))


# pattern -> (mask_mod, score_mod)
_PATTERNS = {
    "noop": (noop_mask, None),
    "causal": (causal, None),
    "alibi": (causal, alibi_score),
    "sliding_window": (sliding_window, None),
}

_PASSES = (FlexAttentionNKILowering, FlexAttentionLegalization)
_DIRECTIONS = ("forward", "backward")


@contextlib.contextmanager
def _record_lowerings():
    """Record what each lowering returned for each flex_attention operator.

    A pass legalizes the forward and the backward separately, so a fallback in one
    direction is invisible in the other. Record every pass and direction.
    """
    recorded = {key: [] for key in ((p, d) for p in _PASSES for d in _DIRECTIONS)}

    def recorder(pass_cls, direction):
        method = f"_legalize_flex_{direction}"
        results = recorded[(pass_cls, direction)]
        original = getattr(pass_cls, method)

        def wrapper(self, graph, gm, node):
            lowered = original(self, graph, gm, node)
            results.append(lowered)
            return lowered

        return patch.object(pass_cls, method, wrapper)

    with contextlib.ExitStack() as stack:
        for pass_cls in _PASSES:
            for direction in _DIRECTIONS:
                stack.enter_context(recorder(pass_cls, direction))
        yield recorded


def _assert_lowered(recorded, directions):
    """Check that exactly one lowering consumed every operator, whichever the budget picked.

    The compile budget decides the path, so the cell does not require a particular one. It
    does require a clean choice: one pass took every operator of a direction and none of
    them fell back.
    """
    for direction in directions:
        nki = recorded[(FlexAttentionNKILowering, direction)]
        decomposed = recorded[(FlexAttentionLegalization, direction)]
        used = [lowered for lowered in nki + decomposed if lowered]
        assert used, f"no {direction} lowering consumed the operator"
        assert bool(nki and all(nki)) != bool(
            decomposed and all(decomposed)
        ), f"the {direction} operators split across lowerings: nki={nki}, decomposed={decomposed}"


def _block_mask(mask_mod):
    """Build the block mask on the host, then move it to the device.

    B=1 and H=1 broadcast over batch and heads. No pattern here reads b or h, so
    one plane is enough and the host evaluates the index grid once.
    """
    return create_block_mask(
        mask_mod,
        B=1,
        H=1,
        Q_LEN=_SEQ_LEN,
        KV_LEN=_SEQ_LEN,
        BLOCK_SIZE=_BLOCK_SIZE,
        device="cpu",
    ).to("neuron")


def _index_grids(batch, q_heads, q_start, q_len, kv_len):
    """Broadcastable b, h, q_idx and kv_idx grids for one block of query rows."""
    return (
        torch.arange(batch).view(-1, 1, 1, 1),
        torch.arange(q_heads).view(1, -1, 1, 1),
        torch.arange(q_start, q_start + q_len).view(1, 1, -1, 1),
        torch.arange(kv_len).view(1, 1, 1, -1),
    )


def _attention_block(q_rows, key, value, mask_mod, score_mod, grids):
    """Dense float32 attention for one block of query rows."""
    b, h, q_idx, kv_idx = grids
    scores = q_rows @ key.transpose(-1, -2) / (q_rows.shape[-1] ** 0.5)
    if score_mod is not None:
        scores = score_mod(scores, b, h, q_idx, kv_idx)
    scores = scores.masked_fill(~mask_mod(b, h, q_idx, kv_idx), float("-inf"))
    return torch.softmax(scores, dim=-1).nan_to_num() @ value


def _expand_kv(key, value, q_heads):
    """Repeat the key/value heads so the reference computes one head per query head."""
    repeats = q_heads // key.shape[1]
    return key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)


def _q_block_size(batch, q_heads, kv_len):
    return max(_BLOCK_SIZE[0], _REF_BLOCK_SCORES // (batch * q_heads * kv_len))


def _reference(query, key, value, mask_mod, score_mod):
    """Attention in float32 over blocks of query rows."""
    batch, q_heads, q_len, _ = query.shape
    kv_len = key.shape[-2]
    key, value = _expand_kv(key.float(), value.float(), q_heads)
    q_block = _q_block_size(batch, q_heads, kv_len)

    blocks = []
    for start in range(0, q_len, q_block):
        q_rows = query[:, :, start : start + q_block].float()
        grids = _index_grids(batch, q_heads, start, q_rows.shape[-2], kv_len)
        blocks.append(_attention_block(q_rows, key, value, mask_mod, score_mod, grids))
    return torch.cat(blocks, dim=-2)


def _reference_grads(query, key, value, mask_mod, score_mod, grad_out):
    """Input gradients in float32, accumulated over blocks of query rows.

    Each block builds its own autograd graph, so the host holds one block of
    scores at a time. grad_q is local to a block; grad_k and grad_v accumulate.
    The key/value expansion is inside the graph, so autograd reduces the GQA
    gradients back to the key/value head count.
    """
    batch, q_heads, q_len, _ = query.shape
    kv_len = key.shape[-2]
    q_block = _q_block_size(batch, q_heads, kv_len)

    grad_q = torch.empty(query.shape, dtype=torch.float32)
    grad_k = torch.zeros(key.shape, dtype=torch.float32)
    grad_v = torch.zeros(value.shape, dtype=torch.float32)

    for start in range(0, q_len, q_block):
        stop = min(start + q_block, q_len)
        q_leaf = query[:, :, start:stop].float().detach().requires_grad_(True)
        k_leaf = key.float().detach().requires_grad_(True)
        v_leaf = value.float().detach().requires_grad_(True)
        k_wide, v_wide = _expand_kv(k_leaf, v_leaf, q_heads)
        grids = _index_grids(batch, q_heads, start, stop - start, kv_len)
        out = _attention_block(q_leaf, k_wide, v_wide, mask_mod, score_mod, grids)
        dq, dk, dv = torch.autograd.grad(
            out, (q_leaf, k_leaf, v_leaf), grad_out[:, :, start:stop].float()
        )
        grad_q[:, :, start:stop] = dq
        grad_k += dk
        grad_v += dv
    return grad_q, grad_k, grad_v


def _inputs(batch, kv_heads, head_dim):
    torch.manual_seed(42)
    return [
        torch.randn(batch, heads, _SEQ_LEN, head_dim, dtype=_DTYPE)
        for heads in (_Q_HEADS, kv_heads, kv_heads)
    ]


def _assert_close_in_norm(actual, expected, name):
    """Compare in relative L2 rather than element by element.

    An output element is a sum of thousands of terms of order one that nearly cancel, so the
    bfloat16 accumulation error is set by the scale of the terms and not by the size of the
    result. An elementwise bound therefore rejects an element that cancels to near zero while
    the tensor as a whole is accurate, and the effect grows with a concentrating score_mod
    such as ALiBi. The norm states the accuracy a caller depends on.
    """
    assert torch.isfinite(actual).all(), f"{name}: {(~torch.isfinite(actual)).sum()} non-finite"
    error = (
        torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    ).item()
    assert error <= _L2_LIMIT, f"{name}: relative L2 error {error:.3e} is above {_L2_LIMIT:.0e}"


@contextlib.contextmanager
def _timed(direction, recorded):
    """Record what `direction` costs, into `recorded`.

    Execution is deferred, so the block only queues work and the synchronize is what waits for
    the compile.
    """
    start = time.perf_counter()
    yield
    torch_neuronx.synchronize()
    recorded.append((direction, time.perf_counter() - start))


def _assert_within_compile_budget(recorded):
    """Fail the cell for a direction over budget.

    Called after the accuracy comparisons, so a slow cell still reports whether its values
    were right.
    """
    over = [f"{d}: {s:.1f} s" for d, s in recorded if s > _COMPILE_BUDGET_S]
    assert (
        not over
    ), f"above the {_COMPILE_BUDGET_S:.0f} s budget for one direction: {', '.join(over)}"


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("pattern", ["noop", "causal", "alibi", "sliding_window"])
class TestFlexAttentionShapeMatrix:
    """One cell of the shape matrix, on whichever lowering the compile budget picks."""

    @staticmethod
    def _compiled(score_mod, kv_heads):
        @torch.compile(backend="neuron")
        def flex_attn_neuron(q, k, v, mask):
            return flex_attention(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=mask,
                enable_gqa=kv_heads != _Q_HEADS,
            )

        return flex_attn_neuron

    def test_forward_matches_reference(self, pattern, batch, kv_heads, head_dim):
        mask_mod, score_mod = _PATTERNS[pattern]
        query, key, value = _inputs(batch, kv_heads, head_dim)
        flex_attn_neuron = self._compiled(score_mod, kv_heads)

        timings = []
        with _record_lowerings() as recorded, torch.no_grad(), _timed("forward", timings):
            neuron_output = flex_attn_neuron(
                query.to("neuron"),
                key.to("neuron"),
                value.to("neuron"),
                _block_mask(mask_mod),
            )

        _assert_lowered(recorded, ("forward",))
        expected = _reference(query, key, value, mask_mod, score_mod)
        _assert_close_in_norm(neuron_output.cpu().float(), expected, "output")
        _assert_within_compile_budget(timings)

    def test_backward_matches_reference(self, pattern, batch, kv_heads, head_dim):
        if (pattern, batch, kv_heads, head_dim) in _BACKWARD_OVER_BUDGET:
            pytest.skip(
                f"the backward of this cell compiles above the {_COMPILE_BUDGET_S:.0f} s budget"
            )
        mask_mod, score_mod = _PATTERNS[pattern]
        inputs = _inputs(batch, kv_heads, head_dim)
        grad_out = torch.randn(batch, _Q_HEADS, _SEQ_LEN, head_dim, dtype=_DTYPE)
        neuron_inputs = [t.detach().clone().to("neuron").requires_grad_(True) for t in inputs]
        flex_attn_neuron = self._compiled(score_mod, kv_heads)

        timings = []
        with _record_lowerings() as recorded:
            with _timed("forward", timings):
                neuron_output = flex_attn_neuron(*neuron_inputs, _block_mask(mask_mod))
            with _timed("backward", timings):
                neuron_output.backward(grad_out.to("neuron"))

        _assert_lowered(recorded, _DIRECTIONS)
        expected = _reference_grads(*inputs, mask_mod, score_mod, grad_out)
        for neuron_input, expected_grad, name in zip(neuron_inputs, expected, "qkv", strict=True):
            _assert_close_in_norm(neuron_input.grad.cpu().float(), expected_grad, f"grad {name}")
        _assert_within_compile_budget(timings)
