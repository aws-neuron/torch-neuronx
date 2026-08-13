"""Identity-op passthrough: ops whose traced graph reduces to `output = input`.

`torch.cat([t])` (single-element list) is eliminated during FX tracing, leaving
`placeholder -> output`. Nothing computes that output, so the lowering emits no
HLO for it and records a passthrough mapping (output 0 <- input 0) instead;
DispatchCore detects the mapping, skips execution entirely, and satisfies the op
by cloning the mapped input.

The regression these guard against is that slot being served from a device
buffer no kernel ever wrote - allocator blocks are recycled, so
the op returned whatever tensor last occupied that memory.

Detection rests on never expecting a value that anything else in the process
wrote, since a recycled block then cannot coincidentally match. Repeating a call
with an unchanged input is what made the bug look self-healing (the stale block
held the previous call's data, which was also the expected data), so every value
below is unique within the module.
"""

import subprocess
import sys
import textwrap

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

import torch_neuronx
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

_DTYPE_CASES = [
    (torch.long, 163),
    (torch.int32, 9161),
    (torch.float32, 3.5),
    (torch.bfloat16, 2.5),
]


def _trace(fn, fake_inputs, compilation_kwargs=None):
    """Trace fn the way _generate_hlo does, returning (gm, original_output_args).

    _generate_fx_graph feeds make_fx the pytree leaves of
    (fake_inputs, compilation_kwargs), so placeholders track every leaf,
    non-tensors included.
    """
    flat_args, _ = pytree.tree_flatten((fake_inputs, compilation_kwargs or {}))
    gm = make_fx(fn)(*flat_args)
    output_args = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    if not isinstance(output_args, tuple):
        output_args = (output_args,)
    return gm, output_args


class TestPassthroughMappingIndexSpace:
    """_detect_passthrough_mappings must emit tensor-only input indices.

    Both consumers of the mapping index a tensor-only list but placeholders
    also cover non-tensor pytree leaves. The two orders coincide only when
    every tensor precedes every non-tensor.
    """

    def test_tensor_first_signature_maps_to_index_zero(self):
        """cat([t], dim=0): tensor leads, so tensor index == placeholder index."""
        gm, outs = _trace(lambda t, _dim: (t,), [[torch.empty(2, device="meta")], 0])
        mappings = TorchMlirKernel._detect_passthrough_mappings(None, gm, outs)
        assert mappings == {0: 0}

    def test_scalar_before_tensor_skips_non_tensor_leaf(self):
        """A leading scalar must not shift the mapped index.

        The tensor sits at placeholder index 1 but tensor index 0. Returning 1
        would index past the end of a single-tensor input list (or select the
        wrong tensor in a longer one).
        """
        gm, outs = _trace(lambda _s, t: (t,), [1.0, torch.empty(2, device="meta")])
        mappings = TorchMlirKernel._detect_passthrough_mappings(None, gm, outs)
        assert mappings == {0: 0}

    def test_mapped_index_resolves_against_flattened_inputs(self):
        """The mapping must index the real input list without going out of range."""
        gm, outs = _trace(lambda _s, t: (t,), [1.0, torch.empty(2, device="meta")])
        mappings = TorchMlirKernel._detect_passthrough_mappings(None, gm, outs)

        real_inputs = [1.0, torch.arange(2)]
        flat = TorchMlirKernel._flatten_tensor_inputs(real_inputs)
        idx = mappings[0]
        assert idx < len(flat), f"index {idx} out of range for {len(flat)} tensors"
        assert torch.equal(flat[idx], real_inputs[1])

    def test_interleaved_scalars_between_tensors(self):
        """Only tensors advance the index, wherever the non-tensors sit."""
        gm, outs = _trace(
            lambda _a, _t0, _b, t1: (t1,),
            [0, torch.empty(2, device="meta"), 5, torch.empty(3, device="meta")],
        )
        mappings = TorchMlirKernel._detect_passthrough_mappings(None, gm, outs)
        # Second tensor -> tensor index 1 (placeholder index would be 3).
        assert mappings == {0: 1}

        real = [0, torch.zeros(2), 5, torch.ones(3)]
        flat = TorchMlirKernel._flatten_tensor_inputs(real)
        assert torch.equal(flat[mappings[0]], real[3])


class TestFlattenTensorInputs:
    """_flatten_tensor_inputs must match C++ ExtractTensorInputs ordering."""

    def test_recurses_into_lists_and_drops_non_tensors(self):
        t0, t1 = torch.empty(1), torch.empty(2)
        flat = TorchMlirKernel._flatten_tensor_inputs([[t0, t1], 0, None])
        assert flat == [t0, t1]

    def test_tensor_kwargs_appended_after_positional(self):
        t0, out = torch.empty(1), torch.empty(2)
        flat = TorchMlirKernel._flatten_tensor_inputs([t0], {"out": out, "dim": 0})
        assert flat == [t0, out]


class TestIsPureInputPassthrough:
    """Python predicate must agree with C++ IsPureInputPassthrough.

    Partial coverage is the important case: a graph where some outputs are
    passthroughs but others still need a NEFF is not a pure passthrough, and
    short-circuiting it would return input clones for real computed outputs.
    """

    def test_full_coverage_single_output(self):
        assert TorchMlirKernel._is_pure_input_passthrough({0: 0}, [], torch.empty(2))

    def test_full_coverage_multiple_outputs(self):
        outs = (torch.empty(2), torch.empty(3))
        assert TorchMlirKernel._is_pure_input_passthrough({0: 0, 1: 1}, [], outs)

    def test_partial_coverage_rejected_with_none_output(self):
        """Output 0 passthrough, output 1 a masked backward grad (None).

        None isn't an FX Node, so it survives capture without a name to match
        and never lands in kept_output_indices — leaving kept_output_indices
        empty and coverage partial.
        """
        outs = (torch.empty(2), None)
        assert not TorchMlirKernel._is_pure_input_passthrough({0: 0}, [], outs)

    def test_partial_coverage_rejected_with_two_tensors(self):
        outs = (torch.empty(2), torch.empty(3))
        assert not TorchMlirKernel._is_pure_input_passthrough({0: 0}, [], outs)

    def test_surviving_kept_outputs_rejected(self):
        assert not TorchMlirKernel._is_pure_input_passthrough({0: 0}, [0], torch.empty(2))

    def test_no_mappings_rejected(self):
        assert not TorchMlirKernel._is_pure_input_passthrough({}, [], torch.empty(2))


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIdentityOpPassthrough:
    """Tests for identity ops with passthrough (op node is removed from graph)."""

    @pytest.mark.parametrize(
        "dtype,value", _DTYPE_CASES, ids=[str(d).removeprefix("torch.") for d, _ in _DTYPE_CASES]
    )
    def test_single_tensor_cat_returns_input_data(self, dtype, value):
        """cat([t]) returns t's data rather than an unwritten device buffer."""
        t = torch.tensor([value], dtype=dtype, device="neuron")
        torch.testing.assert_close(
            torch.cat([t]).cpu(), torch.tensor([value], dtype=dtype), rtol=0, atol=0
        )

    def test_single_tensor_cat_does_not_return_its_input(self):
        """The result must be a copy, not the input tensor itself.
        Mutating the result would otherwise also write through to the argument.
        """
        t = torch.tensor([8317], dtype=torch.long, device="neuron")
        out = torch.cat([t])

        assert out.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()
        out += 1
        assert t.cpu().item() == 8317, "writing to the result mutated the input"

    def test_inplace_identity_op_returns_self(self):
        """An in-place identity op must still return the input tensor itself.

        torch_floor returns bare self for non-floating dtypes, so floor_ on an int
        tensor is a pure passthrough with a schema whose return aliases self.
        """
        t = torch.tensor([7, 9], dtype=torch.long, device="neuron")
        out = t.floor_()

        assert out.data_ptr() == t.data_ptr(), "floor_ did not return its input"
        torch.testing.assert_close(out.cpu(), torch.tensor([7, 9]), rtol=0, atol=0)

    def test_out_variant_identity_op_writes_out_tensor(self):
        """An out= identity op must write the destination and return it."""
        t = torch.tensor([11, 13], dtype=torch.long, device="neuron")
        dest = torch.zeros_like(t)
        ret = torch.floor(t, out=dest)

        assert ret.data_ptr() == dest.data_ptr(), "floor.out did not return its out tensor"
        torch.testing.assert_close(dest.cpu(), torch.tensor([11, 13]), rtol=0, atol=0)

    def test_single_tensor_cat_tracks_changing_input(self):
        """Each cat([t]) reflects its own input, not the previous call's."""
        for value in (4231, 5077, 6143, 7219):
            t = torch.tensor([value], dtype=torch.long, device="neuron")
            got = torch.cat([t]).cpu().item()
            assert got == value, f"cat([{value}]) returned {got}"

    def test_size_tensor_cat_result_is_usable_as_allocation_size(self):
        """Reproduces send_object_list's size-exchange shape without collectives (PP).

        _object_to_tensor builds a 1-element int64 size tensor per object and
        send_object_list concatenates them. The value becomes an allocation size,
        so a stale one raises instead of failing a tolerance check.
        """
        payload_len = 10357
        size_list = [torch.tensor([payload_len], dtype=torch.long, device="neuron")]
        sizes = torch.cat(size_list)

        total = int(torch.sum(sizes).item())
        assert total == payload_len, f"expected {payload_len}, got {total}"

        buf = torch.empty(total, dtype=torch.uint8, device="neuron")
        assert buf.numel() == payload_len

    def test_single_tensor_cat_is_first_op_in_process(self):
        """cat([t]) is correct as the very first device op in a new process.

        Mirrors the PP metadata exchange, which is a one-shot first call on a fresh rank.
        """
        script = textwrap.dedent(
            """
            import torch
            import torch_neuronx

            t = torch.tensor([12289], dtype=torch.long, device="neuron")
            got = torch.cat([t]).cpu().item()
            assert got == 12_289, f"first-call cat returned {got}"
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
