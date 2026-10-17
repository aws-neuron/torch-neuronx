"""E2E tests for the on-device NKI random ops through torch.compile(backend="neuron").

Covers the four ops PR #916 registers -- ``aten::uniform_``, ``aten::normal_``,
``aten::random_`` and ``aten::random_.from``/``.to`` -- confirming they work under
``torch.compile`` as well as in eager. ``uniform_``/``normal_`` are dynamo-traceable
and share one implementation with eager (``uniform_fill``/``normal_fill`` via the
seeded ``rng_state.rand`` wrapper), so they run under ``fullgraph=True``. The in-place
integer ``random_`` family is NOT dynamo-traceable (``Tensor.random_`` graph-breaks),
so it is exercised through the graceful (non-fullgraph) eager fallback instead.

RNG output cannot be compared to a CPU oracle, so assertions check support bounds,
finiteness and (for ``normal_``) generous statistics over ``N`` draws.
"""

import pytest
import torch

N = 1 << 14
FLOATS = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _fullgraph(fn):
    return torch.compile(fn, backend="neuron", fullgraph=True)


def _graceful(fn):
    return torch.compile(fn, backend="neuron")


def _run_float(fn, dtype, shape=(N,)):
    with torch.no_grad():
        return fn(torch.empty(shape, dtype=dtype, device="neuron")).float().cpu()


def _run_int(fn, dtype=torch.int32):
    with torch.no_grad():
        return fn(torch.empty(N, dtype=dtype, device="neuron")).cpu()


class TestUniformCompile:
    """aten::uniform_ under fullgraph=True (shared uniform_fill)."""

    @pytest.mark.parametrize("dtype", FLOATS)
    def test_default_range_bounds(self, dtype):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_()), dtype)
        assert not torch.isnan(out).any()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_custom_range_bounds(self, dtype):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_(-2.0, 3.0)), dtype)
        assert not torch.isnan(out).any()
        assert out.min().item() >= -2.0
        assert out.max().item() <= 3.0

    def test_negative_range(self):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_(-10.0, -5.0)), torch.float32)
        assert out.min().item() >= -10.0
        assert out.max().item() <= -5.0

    def test_zero_width_range(self):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_(2.0, 2.0)), torch.float32)
        torch.testing.assert_close(out, torch.full((N,), 2.0))

    def test_spans_range(self):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_(-1.0, 1.0)), torch.float32)
        assert out.min().item() < 0.0
        assert out.max().item() > 0.0

    def test_shape_preserved(self):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().uniform_()), torch.float32, shape=(3, 5, 7))
        assert out.shape == (3, 5, 7)


class TestNormalCompile:
    """aten::normal_ under fullgraph=True (shared normal_fill; no bf16 NaN)."""

    @pytest.mark.parametrize("dtype", FLOATS)
    def test_default_params(self, dtype):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().normal_()), dtype)
        assert not torch.isnan(out).any()
        assert abs(out.mean().item()) < 0.1
        assert 0.8 < out.std().item() < 1.2

    def test_custom_mean_std(self):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: x.clone().normal_(5.0, 2.0)), torch.float32)
        assert not torch.isnan(out).any()
        assert abs(out.mean().item() - 5.0) < 0.2
        assert 1.6 < out.std().item() < 2.4


class TestRandomCompileGraceful:
    """aten::random_ family: dynamo-untraceable, exercised via graceful fallback."""

    def test_random_default_nonneg(self):
        torch.manual_seed(0)
        out = _run_int(_graceful(lambda x: x.clone().random_()))
        assert out.min().item() >= 0

    def test_random_float_tensor(self):
        torch.manual_seed(0)
        out = _run_int(_graceful(lambda x: x.clone().random_()), dtype=torch.float32)
        assert out.min().item() >= 0

    def test_random_to(self):
        torch.manual_seed(0)
        out = _run_int(_graceful(lambda x: x.clone().random_(10)))
        assert out.min().item() >= 0
        assert out.max().item() < 10

    def test_random_from_to(self):
        torch.manual_seed(0)
        out = _run_int(_graceful(lambda x: x.clone().random_(-5, 5)))
        assert out.min().item() >= -5
        assert out.max().item() < 5


class TestRandomDecomposition:
    """The compile-path ``aten.random`` decompositions.

    ``Tensor.random_`` graph-breaks in dynamo (``Unsupported: Tensor.random_``), so
    ``torch.compile`` eager-falls-back and never reaches these decompositions -- they
    exist for the functionalization/export paths that DO capture ``aten.random`` into
    a graph. They share ``random_fill`` with eager, so calling the registered
    decomposition functions directly validates the compile-path math (bounds, dtype,
    and ATen's non-negative ``from=0`` default) without a torch.compile route.
    """

    @staticmethod
    def _dev(dtype):
        return torch.empty(8192, dtype=dtype, device="neuron")

    def test_registered_in_backend(self):
        from torch_neuronx.neuron_dynamo_backend.decompositions import neuron_decompositions

        assert torch.ops.aten.random.default in neuron_decompositions
        assert torch.ops.aten.random.to in neuron_decompositions

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.int8, torch.uint8, torch.float32]
    )
    def test_full_range_nonneg(self, dtype):
        from torch_neuronx.neuron_dynamo_backend.decompositions import random_functional

        out = random_functional(self._dev(dtype)).cpu()
        assert out.dtype == dtype
        # ATen's random_() defaults from=0, so signed dtypes stay non-negative.
        assert out.min().item() >= 0

    def test_full_range_bool(self):
        from torch_neuronx.neuron_dynamo_backend.decompositions import random_functional

        out = random_functional(self._dev(torch.bool)).cpu()
        assert out.dtype == torch.bool

    def test_to_bounds(self):
        from torch_neuronx.neuron_dynamo_backend.decompositions import random_functional_to

        out = random_functional_to(self._dev(torch.int32), 10).cpu()
        assert out.min().item() >= 0
        assert out.max().item() < 10

    def test_inplace_returns_self_and_bounds(self):
        from torch_neuronx.neuron_dynamo_backend.decompositions import random_inplace_to

        x = self._dev(torch.int32)
        ret = random_inplace_to(x, 5)
        assert ret is x
        c = x.cpu()
        assert c.min().item() >= 0
        assert c.max().item() < 5


class TestFactoryRoutesThroughSharedFill:
    """torch.rand / torch.randn are pre-existing factory decompositions,
    but this PR routes them through the shared uniform_fill / normal_fill. These tests
    guard that coupling: the bf16 correctness the shared leaf brings -- uniform's
    exclusive-upper clamp (never reaches 1.0) and normal's Box-Muller (no bf16 NaN) --
    must hold for the factory ops under compile, so a future edit to the shared leaf
    can't silently regress torch.rand / torch.randn.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rand_exclusive_upper(self, dtype):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: torch.rand(N, dtype=dtype, device=x.device)), dtype)
        assert not torch.isnan(out).any()
        assert out.min().item() >= 0.0
        # shared uniform_fill caps below the open upper bound, even in bf16.
        assert out.max().item() < 1.0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_randn_finite_and_stats(self, dtype):
        torch.manual_seed(0)
        out = _run_float(_fullgraph(lambda x: torch.randn(N, dtype=dtype, device=x.device)), dtype)
        # shared normal_fill uses Box-Muller -> no bf16 erfinv NaN.
        assert not torch.isnan(out).any()
        assert abs(out.mean().item()) < 0.1
        assert 0.8 < out.std().item() < 1.2
