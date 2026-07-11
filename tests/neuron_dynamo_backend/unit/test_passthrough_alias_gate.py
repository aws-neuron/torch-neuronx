"""Unit tests for ``_set_module_alias`` pass-through aliasing (added in #1161).

#1161 (``8ffb9543``) added ``_detect_passthrough_aliases_from_module`` and the
bounds-validated alias merge to ``fx_transform._set_module_alias`` without unit
coverage. These tests lock that behavior, plus the eager gate that fixes the
integer ``ceil``/``trunc`` "returns 0" regression:

* A function output that is a bare block argument (an input returned unchanged) is
  detected as a pass-through and, in the AOT/compile context (``aliasing_info``
  provided), emitted as an ``mhlo.input_output_alias``.
* In the eager per-op path (``aliasing_info is None``) NO alias is emitted --
  eager uses a separate output buffer, so donating an identity/pass-through output
  would leave it unwritten (uninitialized), which is the ceil/trunc bug.
* An explicitly-provided alias whose indices do not map onto the real lowered MLIR
  IO is skipped rather than raising ``IndexError`` (the crux of #1161's PT2.12
  ``forward_pre_hook`` placeholder-reindexing fix).

These are hardware-free: they build a tiny StableHLO module and inspect the MLIR
attribute, with no Neuron device or execution required.
"""

from torch_neuronx.neuron_dynamo_backend.fx import fx_transform as fxt
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

# output 0 is a bare block-arg pass-through of input 0; output 1 is computed.
_PASSTHROUGH_MLIR = """
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    return %arg0, %0 : tensor<4xf32>, tensor<4xf32>
  }
}
"""

# no pass-through: the single output is computed.
_COMPUTED_MLIR = """
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""


def _parse_module(src: str):
    from torch_mlir import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ir.Module.parse(src, ctx)


def _has_alias_attr(module) -> bool:
    return "mhlo.input_output_alias" in module.operation.attributes


class TestPassthroughAliasDetectionAndGate:
    def test_passthrough_detected_from_module(self):
        """A returned-unchanged input (bare block arg) is detected as output->input."""
        mod = _parse_module(_PASSTHROUGH_MLIR)
        with mod.context:
            passthroughs = fxt._detect_passthrough_aliases_from_module(mod)
        assert passthroughs == {0: 0}, f"expected {{0: 0}}, got {passthroughs}"

    def test_passthrough_aliased_when_aliasing_info_present(self):
        """Compile/AOT context (aliasing_info provided): pass-through is emitted as
        an mhlo.input_output_alias so the executor can reuse the input buffer."""
        mod = _parse_module(_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, AliasingInfo())
        assert _has_alias_attr(
            mod
        ), "pass-through alias should be emitted when aliasing_info is provided"

    def test_no_alias_in_eager_gate_when_aliasing_info_none(self):
        """Eager per-op path (aliasing_info is None): NO alias may be emitted.

        Regression guard: the eager runtime uses a separate output buffer, so
        emitting input_output_alias for an identity/pass-through output donates a
        buffer the graph never writes -> uninitialized result (integer ceil/trunc
        returned 0). This assertion fails on the pre-fix code.
        """
        mod = _parse_module(_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, None)
        assert not _has_alias_attr(
            mod
        ), "eager path (aliasing_info=None) must not emit an input_output_alias"

    def test_out_of_range_alias_is_skipped_not_raised(self):
        """An explicit alias whose indices exceed the real MLIR IO is skipped, not
        raised (the #1161 fix for stale FX-space indices under PT2.12)."""
        mod = _parse_module(_COMPUTED_MLIR)
        stale = AliasingInfo()
        stale.add(parameter_number=99, parameter_index=[], output_index=99)
        fxt._set_module_alias(mod, stale)  # must not raise
        assert not _has_alias_attr(mod), "out-of-range alias must be skipped, not emitted"


# Same input (arg0) claimed by BOTH a bare pass-through (arg0 returned unchanged
# at output 0) AND an in-place mutation (its computed value at output 2). The
# pass-through is deliberately at the LOWER output index so that a naive
# lowest-index dedup would keep the wrong (pass-through) alias; the mutation must
# win regardless of output order.
_MUTATION_VS_PASSTHROUGH_MLIR = """
module {
  func.func @main(
      %arg0: tensor<4xf32>, %arg1: tensor<4xf32>
  ) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %arg0, %arg1, %0 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
"""

# The same input (arg0) returned as a bare pass-through at TWO outputs (0 and 1).
_DOUBLE_PASSTHROUGH_MLIR = """
module {
  func.func @main(
      %arg0: tensor<4xf32>, %arg1: tensor<4xf32>
  ) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    return %arg0, %arg0, %0 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
"""


def _emitted_aliases(module):
    """Return the emitted aliases as a list of (parameter_number, output_index).

    output_index is an int for tuple outputs, or None for the empty (single
    tensor output) form.
    """
    import re

    txt = str(module)
    pairs = re.findall(
        r"parameter_number = (\d+) : si64\}, output_index = array<i64:?\s*(\d*)>", txt
    )
    return [(int(p), (int(o) if o else None)) for p, o in pairs]


class TestAliasParameterDedup:
    """An input parameter may be donated to at most ONE output (XLA
    param_has_seen RET_CHECK). These lock the dedup and the mutation-preference."""

    def test_duplicate_passthrough_param_deduped_to_single_alias(self):
        """One input returned bare at two outputs -> exactly one alias emitted
        for it (pre-fix emits two -> XLA param_has_seen failure)."""
        mod = _parse_module(_DOUBLE_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, AliasingInfo())
        aliases = _emitted_aliases(mod)
        params = [p for p, _ in aliases]
        assert params.count(0) == 1, f"input 0 must be donated exactly once, got {aliases}"

    def test_mutation_preferred_over_passthrough_for_same_param(self):
        """When an input is claimed by both a mutation and a pass-through, the
        MUTATION alias must survive (donating the input to the pass-through while
        the mutation copy-back overwrites it would corrupt the pass-through).

        The pass-through is at output 0 and the mutation at output 2, so a naive
        lowest-index dedup would keep the pass-through -- this asserts the
        mutation wins regardless of output order."""
        mod = _parse_module(_MUTATION_VS_PASSTHROUGH_MLIR)
        info = AliasingInfo()
        info.add(parameter_number=0, parameter_index=[], output_index=2)  # mutation -> out 2
        fxt._set_module_alias(mod, info)

        aliases = _emitted_aliases(mod)
        params = [p for p, _ in aliases]
        assert params.count(0) == 1, f"input 0 must be donated exactly once, got {aliases}"
        out_for_param0 = dict(aliases)[0]
        assert out_for_param0 == 2, (
            "the mutation alias (output 2) must win over the pass-through (output 0); "
            f"got {aliases}"
        )
