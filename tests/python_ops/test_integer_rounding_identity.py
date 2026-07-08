"""Regression: eager integer ceil/trunc/floor must be correct on Neuron.

Guards the ``fx_transform`` pass-through-alias fix. ``#1138``'s
``_detect_passthrough_aliases_from_module`` emitted ``mhlo.input_output_alias``
for identity-lowered outputs even in the pure *eager* per-op lowering path
(``aliasing_info is None``). In eager the output uses a SEPARATE device buffer,
so donating the input buffer meant the compiled graph never wrote the output ->
it read uninitialized memory and the op returned zeros/garbage.

Integer ``ceil``/``trunc`` lower to identity (``return %arg``), so they were
~99.9% wrong on-device (e.g. int32 input -956 -> 0); float ops were fine because
they compute a new value (not a bare pass-through). The fix gates the pass-through
detection behind ``aliasing_info is not None``. This test fails if that regression
returns.

(``floor`` on integers is not covered here: it is elided rather than dispatched as
``aten::floor`` on the device, so it does not exercise this code path.)
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# ceil/trunc of an integer is the identity, so neuron must equal CPU.
_INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
_IDENTITY_OPS = ["ceil", "trunc"]


def _int_input(dtype: torch.dtype) -> torch.Tensor:
    """Deterministic integer tensor spanning negatives, zero and positives."""
    torch.manual_seed(0)
    lo, hi = (0, 200) if dtype == torch.uint8 else (-100, 100)
    return torch.randint(lo, hi, (128, 256), dtype=dtype)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIntegerRoundingIdentityEager:
    """Eager integer ceil/trunc/floor identity correctness on Neuron."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("dtype", _INT_DTYPES, ids=lambda d: str(d).split(".")[-1])
    @pytest.mark.parametrize("op_name", _IDENTITY_OPS)
    def test_integer_rounding_identity_matches_cpu(self, device, dtype, op_name):
        op = getattr(torch, op_name)
        x = _int_input(dtype)
        expected = op(x)  # identity for integer dtypes

        with track_neuron_ops():
            actual = op(x.to(device)).cpu()
            # Must exercise the on-device path (where the pass-through-alias bug lived),
            # not silently fall back to CPU.
            assert_op_runs_on_neuron(f"aten::{op_name}")

        # Correctness: neuron result must equal the CPU (identity) result.
        torch.testing.assert_close(actual, expected)

        # Targeted guard for the specific failure mode: the donated-but-unwritten
        # output buffer came back all-zeros/garbage. A non-trivial input must not
        # produce an all-zero output.
        if int(x.abs().sum()) > 0:
            assert int(actual.abs().sum()) > 0, (
                f"{op_name}({dtype}) returned an all-zero tensor on Neuron -- the "
                "pass-through input_output_alias donated an output buffer the graph "
                "never wrote (regression of the fx_transform eager-aliasing fix)."
            )
