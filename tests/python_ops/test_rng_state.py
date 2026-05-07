"""Tests for RNG state kernel ops: rand, set/get rng state, manual_seed."""

import pytest
import torch

from torch_neuronx.python_ops.nki_kernels.rng_state import (
    NUM_RNG_SEEDS,
    eager_rand,
    get_rng_state,
    rand,
    set_rng_state,
)


class TestRand:
    """Tests for the rand kernel op."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)

    def test_output_shape(self):
        ref = torch.empty(1, device="neuron")
        out = rand(ref, [4, 8])
        assert out.shape == (4, 8)

    def test_output_range(self):
        """All values should be in [0, 1)."""
        ref = torch.empty(1, device="neuron")
        out = rand(ref, [256, 16])
        out_cpu = out.cpu()
        assert (out_cpu >= 0.0).all()
        assert (out_cpu < 1.0).all()

    def test_output_dtype(self):
        ref = torch.empty(1, device="neuron")
        out_f32 = rand(ref, [64], dtype=torch.float32)
        assert out_f32.dtype == torch.float32
        out_bf16 = rand(ref, [64], dtype=torch.bfloat16)
        assert out_bf16.dtype == torch.bfloat16

    def test_stochasticity(self):
        """Two calls should produce different values."""
        ref = torch.empty(1, device="neuron")
        out1 = rand(ref, [128])
        out2 = rand(ref, [128])
        assert not torch.equal(out1, out2)

    def test_uniform_distribution(self):
        """Mean should be approximately 0.5."""
        ref = torch.empty(1, device="neuron")
        out = rand(ref, [4096])
        mean = out.cpu().float().mean().item()
        assert abs(mean - 0.5) < 0.05, f"Expected mean ~0.5, got {mean}"


class TestEagerRand:
    """Tests for the eager_rand nki_op wrapper."""

    def test_matches_rand(self):
        """eager_rand should produce valid [0, 1) floats."""
        torch.manual_seed(0)
        ref = torch.empty(1, device="neuron")
        out = eager_rand(ref, [256])
        out_cpu = out.cpu()
        assert (out_cpu >= 0.0).all()
        assert (out_cpu < 1.0).all()


class TestRngState:
    """Tests for get/set RNG state kernel ops."""

    def test_get_rng_state_shape(self):
        state = get_rng_state()
        assert state.shape == (NUM_RNG_SEEDS,)

    def test_set_then_get_rng_state(self):
        """Setting state and getting it back should be consistent."""
        seed_val = 12345
        state = torch.zeros(1, NUM_RNG_SEEDS, dtype=torch.uint32, device="cpu")
        state[0][0] = seed_val
        state_neuron = state.view(torch.int32).to("neuron")
        set_rng_state(state_neuron)

        retrieved = get_rng_state()
        retrieved_cpu = retrieved.cpu().view(torch.uint32)
        assert retrieved_cpu[0].item() == seed_val

    def test_set_state_affects_rand_output(self):
        """Same RNG state should produce same rand output."""
        state = torch.zeros(1, NUM_RNG_SEEDS, dtype=torch.uint32, device="cpu")
        state[0][0] = 999
        state_neuron = state.view(torch.int32).to("neuron")

        ref = torch.empty(1, device="neuron")

        set_rng_state(state_neuron)
        out1 = rand(ref, [128])

        set_rng_state(state_neuron)
        out2 = rand(ref, [128])

        torch.testing.assert_close(out1, out2)


class TestManualSeed:
    """Tests for torch.neuron.manual_seed."""

    def test_manual_seed_determinism(self):
        """Same seed should produce same rand output."""
        ref = torch.empty(1, device="neuron")

        torch.neuron.manual_seed(42)
        out1 = rand(ref, [128])

        torch.neuron.manual_seed(42)
        out2 = rand(ref, [128])

        torch.testing.assert_close(out1, out2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different rand output."""
        ref = torch.empty(1, device="neuron")

        torch.neuron.manual_seed(1)
        out1 = rand(ref, [128])

        torch.neuron.manual_seed(2)
        out2 = rand(ref, [128])

        assert not torch.equal(out1, out2)
