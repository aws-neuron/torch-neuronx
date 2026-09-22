"""Unit tests for the dtype_conversion MLIR pass — complex128 demotion.

Trainium has no 64-bit float, so the pass must demote complex128->complex64 as
well as f64->f32. Historically it only handled f64 (and half-converted
complex<f64>), which produced an invalid StableHLO module (NCC_EMOD021). These
tests lower a complex128 graph device-free via ``fx_to_stablehlo`` and assert
the emitted StableHLO carries no ``complex<f64>`` / ``f64`` left behind.
"""

import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_TARGET,
    requires_compiler,
)
from tests.neuron_dynamo_backend.unit.utils.test_utils import CapturedGraphs
from torch_neuronx.compiler import CompileOptions, compile_only, fx_to_stablehlo


def _capture_post_aot_forward(model, x):
    """The post-AOT forward GraphModule the neuron backend would lower."""
    from torch._dynamo.backends.common import aot_autograd

    from torch_neuronx.neuron_dynamo_backend.decompositions import (
        get_compile_decomposition_table,
    )

    captured = CapturedGraphs()

    def fw_compiler(gm, example_inputs):
        captured.post_aot_forward_graph = gm
        return gm.forward

    aot_backend = aot_autograd(
        fw_compiler=fw_compiler,
        decompositions=get_compile_decomposition_table(),
    )

    def capture_backend(gm, example_inputs):
        captured.pre_aot_graph = gm
        return aot_backend(gm, example_inputs)

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(model, backend=capture_backend)(x)
    assert captured.post_aot_forward_graph is not None, "failed to capture post-AOT graph"
    return captured.post_aot_forward_graph


class _ComplexRope(nn.Module):
    """Returns a complex128 tensor (view_as_complex on an fp64 upcast, times
    itself) so complex survives into the StableHLO signature — a bare
    view_as_complex/view_as_real round-trip decomposes to real arithmetic and
    emits no complex op to demote."""

    def forward(self, x):
        xf = x.to(torch.float64).reshape(x.shape[0], -1, 2)
        xc = torch.view_as_complex(xf)
        return xc * xc  # complex128 output


@requires_compiler
class TestDtypeConversionComplex:
    def test_stablehlo_has_no_complex_f64(self):
        opts = CompileOptions(target=COMPILE_ONLY_TARGET)
        x = torch.randn(2, 8)

        with compile_only(opts):
            gm = _capture_post_aot_forward(_ComplexRope(), x)
            artifact = fx_to_stablehlo(gm, [x], opts)

        # The complex128 output must be demoted to complex64 (Trainium has no
        # complex128). Before the fix the pass left it complex<f64>, producing an
        # invalid StableHLO module. Assert on io_spec (parsed from the post-pass
        # module) rather than the serialized bytecode.
        outs = artifact.io_spec.outputs
        assert outs, "fx_to_stablehlo produced no outputs"
        assert outs[0].dtype == torch.complex64, f"expected complex64, got {outs[0].dtype}"
        assert all(o.dtype != torch.complex128 for o in outs)
