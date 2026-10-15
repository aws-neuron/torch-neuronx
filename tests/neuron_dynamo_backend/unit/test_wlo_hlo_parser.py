"""
Unit tests for wlo_hlo_parser.parse_transforms.

parse_transforms reads a wrapped_neff.hlo protobuf (HloModuleProto) and extracts
the WLO weight-layout transforms: for each computation named "*_transpose", it
recovers the weight index (from a `call` in the entry computation carrying the
`transposable_weight_idx` frontend attribute) and the reshape dims / transpose perm.

The parser hard-imports neuronxcc's hlo_pb2 (ships with the neuronx-cc compiler,
present on Neuron pods / CI, absent on plain dev hosts), so these tests are guarded
with importorskip. We build synthetic HloModuleProtos with that same hlo_pb2, so
they are byte-compatible with what neuronx-cc emits.
"""

import tempfile
from pathlib import Path

import pytest

# Skip the whole module unless the neuronx-cc protobufs are importable.
pytest.importorskip("neuronxcc")

from neuronxcc.thirdparty_libs.xla.service import hlo_pb2  # noqa: E402

from torch_neuronx.neuron_dynamo_backend.wlo_hlo_parser import parse_transforms  # noqa: E402


def _build_module(
    *,
    with_transpose=True,
    with_weight_idx=True,
    weight_idx=0,
    reshape=(4, 2, 3),
    transpose=(0, 2, 1),
    reshape_back_dims=None,
    num_transposes=1,
):
    """Construct an HloModuleProto with an entry computation and (optionally) a
    matching `*_transpose` computation the parser should extract a transform from.

    reshape_back_dims models the real neuronx-cc structure (reshape -> transpose ->
    reshape-back-to-original); num_transposes lets a test emit a malformed count."""
    m = hlo_pb2.HloModuleProto()
    m.entry_computation_name = "entry"

    transpose_comp_id = 2

    entry = m.computations.add()
    entry.name = "entry"
    entry.id = 1
    if with_transpose:
        call = entry.instructions.add()
        call.opcode = "call"
        call.called_computation_ids.append(transpose_comp_id)
        if with_weight_idx:
            call.frontend_attributes.map["transposable_weight_idx"] = str(weight_idx)

        tcomp = m.computations.add()
        tcomp.name = "w_transpose"  # must end with "_transpose"
        tcomp.id = transpose_comp_id
        rs = tcomp.instructions.add()
        rs.opcode = "reshape"
        rs.shape.dimensions.extend(reshape)
        for _ in range(num_transposes):
            tp = tcomp.instructions.add()
            tp.opcode = "transpose"
            tp.dimensions.extend(transpose)
        if reshape_back_dims is not None:
            rb = tcomp.instructions.add()
            rb.opcode = "reshape"
            rb.shape.dimensions.extend(reshape_back_dims)

    return m


def _write(module) -> str:
    with tempfile.NamedTemporaryFile(suffix=".hlo", delete=False) as f:
        f.write(module.SerializeToString())
        return f.name


class TestParseTransforms:
    def test_extracts_transform(self):
        """A well-formed module yields one transform with the recovered index,
        reshape dims, and transpose perm."""
        path = _write(_build_module(weight_idx=3, reshape=(4, 2, 3), transpose=(0, 2, 1)))
        try:
            transforms = parse_transforms(path)
        finally:
            Path(path).unlink(missing_ok=True)

        assert transforms == [{"input_index": 3, "reshape": [4, 2, 3], "transpose": [0, 2, 1]}]

    def test_no_transpose_computation_returns_empty(self):
        """A module with no `*_transpose` computation has no transforms."""
        path = _write(_build_module(with_transpose=False))
        try:
            assert parse_transforms(path) == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_transpose_without_weight_idx_is_skipped(self):
        """A `*_transpose` computation whose call carries no transposable_weight_idx
        cannot be attributed to a weight, so it is skipped (not an error)."""
        path = _write(_build_module(with_weight_idx=False))
        try:
            assert parse_transforms(path) == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_entry_returns_empty(self):
        """If the entry computation name doesn't resolve, parse_transforms returns []."""
        m = _build_module()
        m.entry_computation_name = "does_not_exist"
        path = _write(m)
        try:
            assert parse_transforms(path) == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extracts_first_reshape_with_reshape_back(self):
        """Real neuronx-cc structure: reshape(target) -> transpose -> reshape(original).
        The parser extracts the FIRST (target) reshape and ignores the reshape-back."""
        path = _write(
            _build_module(weight_idx=1, reshape=(8, 4), transpose=(1, 0), reshape_back_dims=(32,))
        )
        try:
            assert parse_transforms(path) == [
                {"input_index": 1, "reshape": [8, 4], "transpose": [1, 0]}
            ]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_multiple_relaid_weights_each_take_target_reshape(self):
        """The real multi-weight case: several `weightN_transpose` computations, each with
        the full reshape(target) -> transpose -> reshape(back) structure and distinct dims.
        Every computation is attributed to the right weight index (via its call's
        transposable_weight_idx) and yields its FIRST (target) reshape, not the reshape-back.

        Mirrors neuronx-cc's emit (HloNeffWrapper.cc createTransposeComputation): the target
        reshape is the transpose's operand and the reshape-back is the computation root, so it
        is always listed first in the topologically-serialized proto."""
        m = hlo_pb2.HloModuleProto()
        m.entry_computation_name = "entry"

        entry = m.computations.add()
        entry.name = "entry"
        entry.id = 1

        # (weight_idx, comp_id, target_reshape, transpose_perm, reshape_back_dims)
        specs = [
            (0, 2, (8, 4), (1, 0), (32,)),
            (5, 3, (2, 3, 4), (0, 2, 1), (24,)),
        ]
        for weight_idx, comp_id, reshape, transpose, reshape_back in specs:
            call = entry.instructions.add()
            call.opcode = "call"
            call.called_computation_ids.append(comp_id)
            call.frontend_attributes.map["transposable_weight_idx"] = str(weight_idx)

            tcomp = m.computations.add()
            tcomp.name = f"weight{weight_idx}_transpose"
            tcomp.id = comp_id
            rs = tcomp.instructions.add()
            rs.opcode = "reshape"
            rs.shape.dimensions.extend(reshape)
            tp = tcomp.instructions.add()
            tp.opcode = "transpose"
            tp.dimensions.extend(transpose)
            rb = tcomp.instructions.add()
            rb.opcode = "reshape"
            rb.shape.dimensions.extend(reshape_back)

        path = _write(m)
        try:
            transforms = parse_transforms(path)
        finally:
            Path(path).unlink(missing_ok=True)

        assert transforms == [
            {"input_index": 0, "reshape": [8, 4], "transpose": [1, 0]},
            {"input_index": 5, "reshape": [2, 3, 4], "transpose": [0, 2, 1]},
        ]

    def test_multiple_transposes_fail_closed(self):
        """More than one transpose is a contract violation -> raise, not silently pick one."""
        path = _write(_build_module(num_transposes=2))
        try:
            with pytest.raises(RuntimeError, match="unexpected transform structure"):
                parse_transforms(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_transpose_fail_closed(self):
        """A weight-attributed transform with no transpose is malformed -> raise, not drop."""
        path = _write(_build_module(num_transposes=0))
        try:
            with pytest.raises(RuntimeError, match="unexpected transform structure"):
                parse_transforms(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_transpose_not_a_permutation_fails_closed(self):
        """A transpose that isn't a permutation of the reshape rank violates the
        wlo_contract -> the parser fails closed rather than emitting a bad rule."""
        path = _write(_build_module(reshape=(4, 3), transpose=(0, 0)))
        try:
            with pytest.raises(RuntimeError, match="permutation"):
                parse_transforms(path)
        finally:
            Path(path).unlink(missing_ok=True)
