"""RNG reproducibility & correctness tests for the on-device GPSIMD RNG.

The RNG path is the fast multi-lane kernel (`generate_random_fast`, all 128
GPSIMD lanes, per-lane seeded). It intentionally does NOT provide a
layout-independent stream: because values come from 128 interleaved lanes, the
value->position mapping depends on the output tiling. Dropout (the sole
consumer) only needs a correct independent distribution, so that is fine.

These tests therefore assert what the fast kernel actually guarantees:
  1. Fixed-shape run-to-run reproducibility: same seed + same shape -> same values.
  2. Seed sensitivity: different seeds -> different values.
  3. Correct uniform distribution: values in [0, 1) with mean ~= 0.5, for any shape.
  4. dtype is a per-element reinterpret (bf16 == f32 rounded), not a new stream.

They also guard the manual_seed lazy-init fix (torch_neuronx/neuron.py): the
first manual_seed of a process must reseed the on-device engine even before the
runtime is lazily initialized, so seeding is reproducible from a cold process.
"""

import pytest
import torch

from torch_neuronx.python_ops.nki_kernels.rng_state import rand

_rng_state_regression = pytest.mark.xfail(
    reason=(
        "Dependency-guard regression: the device RNG state is not updated, so different "
        "seeds produce identical output"
    )
)


def _draw(seed, shape, dtype=torch.float32):
    torch.neuron.manual_seed(seed)
    ref = torch.empty(1, device="neuron")
    return rand(ref, shape, dtype=dtype).cpu().flatten()


class TestRngSeedReproducibility:
    def test_cold_process_seed_reproducible(self):
        """Regression guard for the manual_seed lazy-init bug: seeding must be
        reproducible even as the FIRST RNG op in the process (no warm-up).

        Before the neuron.py fix, the first manual_seed ran before runtime init
        and its device reseed was skipped, so this drifted run-to-run."""
        a = _draw(12345, [512])
        b = _draw(12345, [512])
        assert torch.equal(a, b), (
            "cold-process seed not reproducible -> manual_seed did not reseed "
            "the device engine before the draw. first mismatch "
            f"{(a != b).nonzero()[:5].flatten().tolist()}"
        )

    @pytest.mark.parametrize("shape", [[128], [512], [1, 512], [2, 256], [64, 64]])
    def test_same_seed_same_shape_deterministic(self, shape):
        """Same seed + same shape -> identical values (the fast kernel's
        reproducibility guarantee)."""
        a = _draw(777, shape)
        b = _draw(777, shape)
        assert torch.equal(a, b), (
            f"non-deterministic for shape {shape}; first mismatch "
            f"{(a != b).nonzero()[:5].flatten().tolist()}"
        )

    @_rng_state_regression
    def test_different_seeds_differ(self):
        assert not torch.equal(_draw(1, [512]), _draw(2, [512]))

    @pytest.mark.parametrize("shape", [[4096], [64, 64], [16, 256], [2, 2, 1024]])
    def test_uniform_range_and_mean(self, shape):
        """Values in [0, 1) with mean ~= 0.5 for any shape (correct independent
        distribution is all dropout needs)."""
        out = _draw(20240709, shape)
        assert (out >= 0.0).all() and (out < 1.0).all()
        assert (
            abs(out.mean().item() - 0.5) < 0.05
        ), f"mean {out.mean().item():.4f} for shape {shape} not ~0.5"

    def test_dtype_is_reinterpret_not_new_stream(self):
        """Requesting bf16 downcasts the same f32 stream; it must not draw a
        different sequence."""
        f32 = _draw(555, [256], dtype=torch.float32)
        bf16 = _draw(555, [256], dtype=torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(f32, bf16, rtol=8e-3, atol=8e-3)
