"""E2E: RNG ops must be reproducible under torch.manual_seed through torch.compile.

Eager RNG ops are reproducible under ``manual_seed``, but an RNG op lowered into
a compiled NEFF is only reproducible if it goes through the seeded on-device
GPSIMD RNG (``nki_kernels.rand``). Ops that fall through to torch-mlir's inline
``stablehlo.rng`` are NOT threaded with the ``manual_seed`` GPSIMD state and
diverge run-to-run once compiled.

The Neuron decompositions in ``neuron_dynamo_backend/decompositions.py`` route
the whole ``rand`` / ``randn`` / ``uniform`` / ``bernoulli`` / ``normal`` family
onto the seeded path (the same one ``native_dropout`` uses), so these tests
assert that every one of them reproduces under a fixed seed when compiled.

Companion unit tests (distribution/structure, CPU-fallback branch) live in
``tests/neuron_dynamo_backend/unit/test_rng_decompositions.py``.
"""

import pytest
import torch

import torch_neuronx

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."

SHAPE = (2, 4, 8, 8)

# name -> callable(x) drawing randomness. ``x`` is only a shape/device reference
# (needed by the ``*_like`` variants); the RNG output does NOT depend on ``x``'s
# values — the op is still captured into the compiled NEFF, so no artificial
# input dependency is required to exercise the compiled RNG.
_RNG_OPS = {
    "rand": lambda x: torch.rand(x.shape, device=x.device),
    "rand_like": lambda x: torch.rand_like(x),
    "randn": lambda x: torch.randn(x.shape, device=x.device),
    "randn_like": lambda x: torch.randn_like(x),
    "bernoulli": lambda x: torch.bernoulli(torch.full_like(x, 0.5)),
    "normal": lambda x: torch.normal(torch.zeros_like(x), torch.ones_like(x)),
    "uniform_": lambda x: torch.empty_like(x).uniform_(),
}


def _warm():
    """Initialize runtime + warm the on-device RNG so cold-start seeding (a
    separate effect) doesn't confound the compiled-RNG reproducibility check."""
    if not torch_neuronx.is_neuron_runtime_initialized():
        torch_neuronx._lazy_init()
    _ = torch.bernoulli(torch.full(SHAPE, 0.5, device="neuron")).cpu()


def _mismatch(a, b):
    return f"{(a != b).sum().item()}/{a.numel()} elements differ"


def _run_compiled_twice(fn):
    _warm()

    @torch.compile(backend="neuron", dynamic=False)
    def f(x):
        return fn(x)

    x = torch.zeros(SHAPE, device="neuron")
    _ = f(x).cpu()  # trigger compilation

    def draw():
        torch.manual_seed(42)
        return f(x).cpu()

    return draw(), draw()


class TestRngSeedReproducibilityE2E:
    @pytest.mark.parametrize("op_name", list(_RNG_OPS))
    def test_eager_rng_reproducible(self, op_name):
        """Baseline: eager RNG ops reproduce under manual_seed."""
        _warm()
        fn = _RNG_OPS[op_name]
        x = torch.zeros(SHAPE, device="neuron")

        def draw():
            torch.manual_seed(42)
            return fn(x).cpu()

        a, b = draw(), draw()
        assert torch.equal(a, b), f"eager {op_name}: " + _mismatch(a, b)

    @pytest.mark.parametrize("op_name", list(_RNG_OPS))
    def test_compiled_rng_reproducible(self, op_name):
        """Compiled RNG ops must reproduce under manual_seed (seeded decomposition)."""
        a, b = _run_compiled_twice(_RNG_OPS[op_name])
        assert torch.equal(a, b), f"compiled {op_name}: " + _mismatch(a, b)

    def test_compiled_native_dropout_reproducible(self):
        """Control: the seeded on-device RNG path native_dropout uses is
        reproducible when compiled."""

        def nd(x):
            _, mask = torch.ops.aten.native_dropout.default(torch.ones_like(x), 0.5, True)
            return mask.to(x.dtype)

        a, b = _run_compiled_twice(nd)
        assert torch.equal(a, b), "compiled native_dropout: " + _mismatch(a, b)


class TestRngEmptyShapeE2E:
    """Zero-element RNG draws must compile and run without hitting the NKI
    kernel's positive-dimension assertion.

    Mirrors the upstream failure in
    ``test/dynamo/test_misc.py::MiscTests::test_load_fast_and_clear_graph_break``:
    ``torch.cat([torch.randn(r, 5) for r in range(3)])`` includes
    ``torch.randn(0, 5)``, which routed through the seeded on-device RNG and
    allocated a ``(1, 0)`` buffer, tripping
    ``AssertionError: shape must be a tuple of positive integers``.
    """

    def test_compiled_cat_of_randn_with_empty(self):
        _warm()

        @torch.compile(backend="neuron", dynamic=False)
        def fn():
            return torch.cat([torch.randn(r, 5, device="neuron") for r in range(3)])

        out = fn().cpu()
        assert out.shape == (3, 5), out.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
