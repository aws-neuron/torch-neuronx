"""Tests for the ATen output layout that the Neuron dispatcher keeps.

ATen propagates the layout of the inputs to the output. An op with a transposed
input returns a transposed output. The dispatcher recovers that layout on a
cache miss and stores it in the cache entry, and the epilogue merges the
transpose into the NEFF.

Each test pins three contracts:

1. The output layout agrees with CPU on every dim of size > 1. Upstream
   compares exactly these dims (`_prims_common._check_strides_helper` with
   `significant_only=True`).
2. The op runs on Neuron. A CPU fallback would keep the layout for free and
   hide a defect in the dispatcher.
3. The relayout stays in the NEFF. The epilogue reports a host round trip as
   `<op>[epilogue_cpu_roundtrip]`, so that name must be absent.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)

HOST_BOUNCE_SUFFIX = "[epilogue_cpu_roundtrip]"


def _assert_no_host_bounce():
    """Assert that no output relayout took the host path."""
    torch.neuron.synchronize()
    bounced = [op for op in torch_neuronx.get_fallback_ops() if HOST_BOUNCE_SUFFIX in op]
    assert not bounced, f"Output relayout took the host path: {bounced}"


def _significant_strides(t):
    """Return the strides of the dims with size > 1.

    ATen ignores the stride of a size-1 dim, and so does the epilogue.
    """
    return tuple(s for s, size in zip(t.stride(), t.shape, strict=True) if size > 1)


def _assert_same_layout_and_values(neuron_out, cpu_out, rtol=1e-5, atol=1e-5):
    _assert_no_host_bounce()
    assert _significant_strides(neuron_out) == _significant_strides(cpu_out), (
        f"neuron strides {neuron_out.stride()} do not match cpu {cpu_out.stride()} "
        f"for shape {tuple(neuron_out.shape)}"
    )
    torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)


BINARY_OPS = [
    pytest.param(torch.add, "aten::add", id="add"),
    pytest.param(torch.mul, "aten::mul", id="mul"),
    pytest.param(torch.sub, "aten::sub", id="sub"),
]

UNARY_OPS = [
    pytest.param(torch.abs, "aten::abs", id="abs"),
    pytest.param(torch.isnan, "aten::isnan", id="isnan"),
]


@pytest.mark.parametrize("op, op_name", BINARY_OPS)
def test_binary_op_keeps_transposed_output_layout(op, op_name):
    """A binary op with two transposed inputs returns a transposed output."""
    a_cpu = torch.randn(4, 5)
    b_cpu = torch.randn(4, 5)
    expected = op(a_cpu.t(), b_cpu.t())

    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    with track_neuron_ops():
        actual = op(a.t(), b.t())
        assert_op_runs_on_neuron(op_name)
    _assert_same_layout_and_values(actual, expected)


@pytest.mark.parametrize("op, op_name", UNARY_OPS)
def test_unary_op_keeps_transposed_output_layout(op, op_name):
    """A unary op with a transposed input returns a transposed output."""
    a_cpu = torch.randn(3, 5)
    expected = op(a_cpu.t())

    a = a_cpu.to("neuron")
    with track_neuron_ops():
        actual = op(a.t())
        assert_op_runs_on_neuron(op_name)
    _assert_same_layout_and_values(actual, expected)


def test_permuted_output_is_not_a_view_on_the_prim_path():
    """The prim path and the direct path agree that the output is not a view.

    A torch._refs prim runs below Autograd only, so the ADInplaceOrView kernel
    sees the as_strided call that builds the permuted output. Without a guard
    it marks the output as a view, and the direct path does not. The upstream
    test_python_ref tests compare _is_view() between the two paths.
    """
    a = torch.randn(3, 5, device="neuron").t()
    direct = torch.abs(a)
    prim = torch.ops.prims.abs(a)
    assert _significant_strides(prim) == _significant_strides(direct)
    assert not direct._is_view()
    assert not prim._is_view()


def test_int32_neg_with_a_lone_epilogue_transpose():
    """An input that is not a dense permutation can still give a permuted output.

    ATen orders the output dims by the input strides and ignores a stride-0 dim,
    so this upstream sample gives strides (10, 1, 5). PreprocessArgs makes the
    input contiguous, so the NEFF holds the transpose epilogue alone. neuronx-cc
    fails to compile an int32 negate in that form. The lowering avoids it.
    """
    base = torch.arange(500, dtype=torch.int32)
    a_cpu = base.as_strided((9, 5, 2), (0, 1, 7), 3)
    expected = torch.neg(a_cpu)

    a = base.to("neuron").as_strided((9, 5, 2), (0, 1, 7), 3)
    with track_neuron_ops():
        actual = torch.neg(a)
        assert_op_runs_on_neuron("aten::neg")
    _assert_same_layout_and_values(actual, expected)


def test_scalar_mul_keeps_transposed_output_layout():
    """A scalar operand does not change the layout rule."""
    a_cpu = torch.randn(3, 5)
    expected = a_cpu.t() * 2

    a = a_cpu.to("neuron")
    with track_neuron_ops():
        actual = a.t() * 2
        assert_op_runs_on_neuron("aten::mul")
    _assert_same_layout_and_values(actual, expected)


@pytest.mark.parametrize(
    "shape, perm",
    [
        pytest.param((2, 3, 1), (2, 1, 0), id="trailing_singleton"),
        pytest.param((2, 1, 3), (2, 1, 0), id="middle_singleton"),
        pytest.param((1, 2, 3), (0, 2, 1), id="leading_singleton"),
    ],
)
def test_singleton_dim_permutation_keeps_output_layout(shape, perm):
    """A size-1 dim does not push the relayout to the host.

    The epilogue ignores a size-1 dim, so only the significant strides are
    compared. A raw stride tuple can differ from CPU and still be the same
    layout.
    """
    a_cpu = torch.randn(*shape)
    expected = torch.abs(a_cpu.permute(perm))

    a = a_cpu.to("neuron")
    with track_neuron_ops():
        actual = torch.abs(a.permute(perm))
        assert_op_runs_on_neuron("aten::abs")
    _assert_same_layout_and_values(actual, expected)


def test_cat_keeps_channels_last_output_layout():
    """cat of channels-last inputs returns a channels-last output."""
    a_cpu = torch.randn(2, 2, 2, 2).to(memory_format=torch.channels_last)
    b_cpu = torch.randn(2, 2, 2, 2).to(memory_format=torch.channels_last)
    expected = torch.cat([a_cpu, b_cpu], dim=1)

    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    with track_neuron_ops():
        actual = torch.cat([a, b], dim=1)
        assert_op_runs_on_neuron("aten::cat")
    _assert_same_layout_and_values(actual, expected)


def test_contiguous_input_keeps_contiguous_output():
    """Contiguous inputs keep the contiguous output. No behavior change."""
    a_cpu = torch.randn(4, 5)
    b_cpu = torch.randn(4, 5)
    expected = a_cpu + b_cpu

    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    with track_neuron_ops():
        actual = a + b
        assert_op_runs_on_neuron("aten::add")
    assert actual.is_contiguous()
    _assert_same_layout_and_values(actual, expected)


def test_layout_survives_a_cache_hit():
    """The layout comes from the cache entry, so every later call keeps it.

    The dispatch cache key holds the strides of a non-contiguous input, so one
    entry serves one input layout. This test fails if the layout is recovered
    on the miss only.
    """
    a_cpu = torch.randn(6, 7)
    b_cpu = torch.randn(6, 7)
    expected = a_cpu.t() + b_cpu.t()

    a = a_cpu.to("neuron")
    b = b_cpu.to("neuron")
    with track_neuron_ops():
        for _ in range(3):
            actual = a.t() + b.t()
            _assert_same_layout_and_values(actual, expected)
        assert_op_runs_on_neuron("aten::add")


def test_permutation_propagates_through_two_ops():
    """The layout of one op is the input layout of the next op."""
    a_cpu = torch.randn(4, 5)
    expected = torch.abs(a_cpu.t()) * 3

    a = a_cpu.to("neuron")
    with track_neuron_ops():
        actual = torch.abs(a.t()) * 3
        assert_op_runs_on_neuron("aten::abs")
        assert_op_runs_on_neuron("aten::mul")
    _assert_same_layout_and_values(actual, expected)


@pytest.mark.parametrize(
    "shape, make_view",
    [
        pytest.param((4, 5), lambda t: t.t(), id="dense_permuted"),
        pytest.param((4, 6), lambda t: t[:, :3].t(), id="strided_view"),
    ],
)
def test_to_copy_keeps_the_preserved_layout(shape, make_view):
    """A dtype cast with MemoryFormat::Preserve keeps the ATen dim order.

    at::infer_dense_strides is the ATen rule for Preserve. It replays the strides
    of a dense source. It densifies the dim order of a strided view of a bigger
    buffer, so the second case must not replay the strides of the source.
    """
    base_cpu = torch.randn(*shape)
    expected = make_view(base_cpu).to(torch.float16)

    base = base_cpu.to("neuron")
    with track_neuron_ops():
        actual = make_view(base).to(torch.float16)
        assert_op_runs_on_neuron("aten::_to_copy")
    _assert_same_layout_and_values(actual, expected, rtol=1e-3, atol=1e-3)


def test_to_copy_preserve_skips_the_second_phase_copy():
    """A dtype cast on a dense permuted input needs no relayout copy.

    _to_copy converts the dtype first and applies the memory_format second. The
    dtype conversion already returns the ATen layout, so a second copy into a
    fresh buffer costs one device copy for each call and gives the same result.
    A relayout copy dispatches aten::copy_, so the tracker must not report it.
    """
    a_cpu = torch.randn(4, 5)
    expected = a_cpu.t().to(torch.float16)

    a = a_cpu.to("neuron").t()
    torch.neuron.synchronize()
    with track_neuron_ops():
        actual = a.to(torch.float16)
        assert_op_runs_on_neuron("aten::_to_copy")
        assert_op_did_not_run_on_neuron("aten::copy_")
    _assert_same_layout_and_values(actual, expected, rtol=1e-3, atol=1e-3)
