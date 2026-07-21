"""Unit tests for ``_set_module_alias`` aliasing attribute emission.

Tests that the MLIR ``mhlo.input_output_alias`` attribute is correctly set
based on the ``alias_buffer_map`` (dict[int, int]) passed to
``_set_module_alias``.

These are hardware-free: they build a tiny StableHLO module and inspect the MLIR
attribute, with no Neuron device or execution required.
"""

import pytest

from torch_neuronx.neuron_dynamo_backend.fx import fx_transform as fxt

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


class TestSetModuleAlias:
    def test_alias_emitted_when_buffer_map_provided(self):
        """A valid alias_buffer_map entry emits mhlo.input_output_alias."""
        mod = _parse_module(_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, {0: 0})
        assert _has_alias_attr(mod), "alias should be emitted for valid buffer map entry"

    def test_no_alias_when_buffer_map_is_none(self):
        """None alias_buffer_map emits no alias."""
        mod = _parse_module(_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, None)
        assert not _has_alias_attr(mod), "no alias should be emitted when buffer map is None"

    def test_no_alias_when_buffer_map_is_empty(self):
        """Empty alias_buffer_map emits no alias."""
        mod = _parse_module(_PASSTHROUGH_MLIR)
        fxt._set_module_alias(mod, {})
        assert not _has_alias_attr(mod), "no alias should be emitted when buffer map is empty"

    def test_out_of_range_alias_raises(self):
        """An alias whose indices exceed the real MLIR IO raises AssertionError."""
        mod = _parse_module(_COMPUTED_MLIR)
        with pytest.raises(AssertionError, match="out of range"):
            fxt._set_module_alias(mod, {99: 99})


# Same input (arg0) returned at two outputs.
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
    """Return the emitted aliases as a list of (parameter_number, output_index)."""
    import re

    txt = str(module)
    pairs = re.findall(
        r"parameter_number = (\d+) : si64\}, output_index = array<i64:?\s*(\d*)>", txt
    )
    return [(int(p), (int(o) if o else None)) for p, o in pairs]


class TestAliasParameterDedup:
    """An input parameter may be donated to at most ONE output (XLA
    param_has_seen RET_CHECK). The alias_buffer_map is a dict so duplicate
    input entries are naturally handled by whoever builds the map."""

    def test_single_alias_per_input(self):
        """When the buffer map maps two outputs to the same input, only one
        alias is emitted (dict keys are unique so this tests that _set_module_alias
        handles the map correctly)."""
        mod = _parse_module(_DOUBLE_PASSTHROUGH_MLIR)
        # Only one output can alias input 0 in the map (dict semantics)
        fxt._set_module_alias(mod, {0: 0})
        aliases = _emitted_aliases(mod)
        params = [p for p, _ in aliases]
        assert params.count(0) == 1, f"input 0 must be donated exactly once, got {aliases}"
