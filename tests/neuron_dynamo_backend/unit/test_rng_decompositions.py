# ruff: noqa: N806

"""
Unit tests for normal.* and random_.* decompositions.
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# ---------------------------------------------------------------------------
# normal.* — sample from a Normal distribution using the on-device RNG kernel
# ---------------------------------------------------------------------------


class TestNormalDecomposition:
    """Unit tests for the ``normal.*`` decomposition family.

    The decomposition is stochastic (uses the same on-device RNG as
    ``native_dropout``), so we can't bit-compare against CPU. Instead we
    check structure (shape, dtype), broadcasting, and — for large samples —
    that the empirical mean/std match the requested distribution to a
    tolerance that's tight for N=100k iid Normal draws (std_error ≈ 3e-3).

    ``_has_nki_rng`` is forced off so these tests exercise the pure-Python
    CPU fallback branch of ``_standard_normal`` / ``_random_int32_raw``;
    the on-device NKI branch is covered by the E2E class below.
    """

    @pytest.fixture(autouse=True)
    def _force_cpu_rng(self, monkeypatch):
        monkeypatch.setattr(decompositions, "_has_nki_rng", False)

    # Enough samples that the 4-sigma bound below is a stable check.
    _N = 100_000
    _MEAN_TOL = 3e-2  # 4sigma ≈ 4/sqrt(N) * sigma  →  ~0.013 for sigma=1, cushion x2 for bf16
    _STD_TOL = 3e-2

    def test_all_variants_registered(self):
        for op in [
            torch.ops.aten.normal.Tensor_Tensor,
            torch.ops.aten.normal.Tensor_Tensor_out,
            torch.ops.aten.normal.Tensor_float,
            torch.ops.aten.normal.Tensor_float_out,
            torch.ops.aten.normal.float_Tensor,
            torch.ops.aten.normal.float_float,
            torch.ops.aten.normal.float_float_out,
        ]:
            assert (
                op in decompositions.neuron_decompositions
            ), f"{op} missing from neuron_decompositions"

    def test_standard_normal_moments(self):
        """``_standard_normal`` on a large sample: mean≈0, std≈1."""
        ref = torch.zeros(1)
        z = decompositions._standard_normal(ref, (self._N,), torch.float32)
        assert z.shape == (self._N,)
        assert z.dtype == torch.float32
        assert abs(z.mean().item()) < self._MEAN_TOL
        assert abs(z.std().item() - 1.0) < self._STD_TOL

    def test_normal_tensor_tensor_moments_and_shape(self):
        mean_val, std_val = 3.0, 0.5
        mean = torch.full((self._N,), mean_val)
        std = torch.full((self._N,), std_val)
        r = decompositions.normal_tensor_tensor(mean, std)
        assert r.shape == mean.shape
        assert r.dtype == mean.dtype
        assert abs(r.mean().item() - mean_val) < self._MEAN_TOL * max(1.0, std_val)
        assert abs(r.std().item() - std_val) < self._STD_TOL * max(1.0, std_val)

    def test_normal_tensor_tensor_broadcast_shape(self):
        """Output shape follows broadcast(mean.shape, std.shape)."""
        mean = torch.zeros(4, 1)
        std = torch.ones(1, 5)
        r = decompositions.normal_tensor_tensor(mean, std)
        assert r.shape == (4, 5)

    def test_normal_tensor_float_shape_and_moments(self):
        mean = torch.full((self._N,), -2.0)
        r = decompositions.normal_tensor_float(mean, 1.5)
        assert r.shape == mean.shape
        assert abs(r.mean().item() - (-2.0)) < self._MEAN_TOL * 1.5
        assert abs(r.std().item() - 1.5) < self._STD_TOL * 1.5

    def test_normal_float_tensor_shape_and_moments(self):
        std = torch.full((self._N,), 2.0)
        r = decompositions.normal_float_tensor(4.0, std)
        assert r.shape == std.shape
        assert abs(r.mean().item() - 4.0) < self._MEAN_TOL * 2.0
        assert abs(r.std().item() - 2.0) < self._STD_TOL * 2.0

    def test_normal_float_float_shape_and_moments(self):
        r = decompositions.normal_float_float(0.0, 1.0, [self._N], device=torch.device("cpu"))
        assert r.shape == (self._N,)
        assert abs(r.mean().item()) < self._MEAN_TOL
        assert abs(r.std().item() - 1.0) < self._STD_TOL

    def test_out_variants_write_and_return_out(self):
        """The .out overloads copy the result into ``out`` and return it."""
        mean = torch.full((256,), 1.0)
        std = torch.full((256,), 0.25)

        # Tensor_Tensor_out
        out = torch.empty(256)
        r = decompositions.normal_tensor_tensor_out(mean, std, out=out)
        assert r.data_ptr() == out.data_ptr()
        assert out.shape == (256,) and out.dtype == torch.float32

        # Tensor_float_out
        out2 = torch.empty(256)
        r2 = decompositions.normal_tensor_float_out(mean, 0.25, out=out2)
        assert r2.data_ptr() == out2.data_ptr()

        # float_float_out
        out3 = torch.empty(256)
        r3 = decompositions.normal_float_float_out(0.0, 1.0, [256], out=out3)
        assert r3.data_ptr() == out3.data_ptr()

    def test_generator_kwarg_returns_notimplemented(self):
        """A user-supplied generator falls through to PyTorch's kernel."""
        gen = torch.Generator()
        mean = torch.zeros(4)
        std = torch.ones(4)
        # All overloads should return NotImplemented for a non-None generator.
        assert decompositions.normal_tensor_tensor(mean, std, generator=gen) is NotImplemented
        assert decompositions.normal_tensor_float(mean, 1.0, generator=gen) is NotImplemented
        assert decompositions.normal_float_tensor(0.0, std, generator=gen) is NotImplemented
        assert (
            decompositions.normal_float_float(
                0.0, 1.0, [4], device=torch.device("cpu"), generator=gen
            )
            is NotImplemented
        )


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNormalDecompositionE2E:
    """E2E: torch.compile(backend='neuron') traces the normal.* ops via our
    decompositions, executes them on-device, and returns tensors whose
    empirical mean/std match the requested distribution."""

    _N = 50_000
    _MEAN_TOL = 5e-2
    _STD_TOL = 5e-2

    @pytest.mark.parametrize("mean_val,std_val", [(0.0, 1.0), (2.0, 0.5), (-1.0, 3.0)])
    def test_tensor_tensor_e2e(self, mean_val, std_val):
        mean = torch.full((self._N,), mean_val, device="neuron")
        std = torch.full((self._N,), std_val, device="neuron")

        def fn(m, s):
            return torch.ops.aten.normal.Tensor_Tensor(m, s)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(mean, std).cpu()
        assert r.shape == (self._N,)
        assert abs(r.mean().item() - mean_val) < self._MEAN_TOL * max(1.0, std_val)
        assert abs(r.std().item() - std_val) < self._STD_TOL * max(1.0, std_val)

    def test_tensor_float_e2e(self):
        mean = torch.full((self._N,), 3.0, device="neuron")

        def fn(m):
            return torch.ops.aten.normal.Tensor_float(m, 0.5)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(mean).cpu()
        assert abs(r.mean().item() - 3.0) < self._MEAN_TOL
        assert abs(r.std().item() - 0.5) < self._STD_TOL

    def test_float_tensor_e2e(self):
        std = torch.full((self._N,), 2.0, device="neuron")

        def fn(s):
            return torch.ops.aten.normal.float_Tensor(1.0, s)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(std).cpu()
        assert abs(r.mean().item() - 1.0) < self._MEAN_TOL * 2.0
        assert abs(r.std().item() - 2.0) < self._STD_TOL * 2.0

    def test_float_float_e2e(self):
        def fn():
            return torch.ops.aten.normal.float_float(0.0, 1.0, [self._N], device="neuron")

        r = torch.compile(fn, backend="neuron", fullgraph=True)().cpu()
        assert abs(r.mean().item()) < self._MEAN_TOL
        assert abs(r.std().item() - 1.0) < self._STD_TOL

    def test_out_variant_e2e(self):
        """.out variant writes into the caller-provided buffer."""
        mean = torch.full((self._N,), 5.0, device="neuron")
        std = torch.full((self._N,), 0.5, device="neuron")

        def fn(m, s, o):
            return torch.ops.aten.normal.Tensor_Tensor_out(m, s, out=o)

        out = torch.empty(self._N, device="neuron")
        r = torch.compile(fn, backend="neuron", fullgraph=True)(mean, std, out).cpu()
        assert abs(r.mean().item() - 5.0) < self._MEAN_TOL
        assert abs(r.std().item() - 0.5) < self._STD_TOL


# ---------------------------------------------------------------------------
# random_ / random_.to — in-place fill with random values
# ---------------------------------------------------------------------------


class TestRandomInplaceDecomposition:
    """Unit tests for ``random_`` and ``random_.to``.

    Both ops are in-place; the decompositions preserve mutation on ``self``.
    We also register the functional counterparts (``aten.random.default`` and
    ``aten.random.to``) since AOT autograd's functionalization pass rewrites
    the in-place calls into those inside the compiled FX graph.

    ``_has_nki_rng`` is forced off so unit tests take the CPU fallback
    branch; the on-device path is covered by the E2E class below.
    """

    @pytest.fixture(autouse=True)
    def _force_cpu_rng(self, monkeypatch):
        monkeypatch.setattr(decompositions, "_has_nki_rng", False)

    def test_all_variants_registered(self):
        # In-place ops (user-facing)
        assert torch.ops.aten.random_.default in decompositions.neuron_decompositions
        assert torch.ops.aten.random_.to in decompositions.neuron_decompositions
        # Functional counterparts (functionalization target)
        assert torch.ops.aten.random.default in decompositions.neuron_decompositions
        assert torch.ops.aten.random.to in decompositions.neuron_decompositions

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.int16, torch.int8, torch.float32]
    )
    def test_random_inplace_shape_and_dtype(self, dtype):
        """``random_`` preserves shape and dtype of ``self``."""
        x = torch.empty(64, dtype=dtype)
        r = decompositions.random_inplace(x)
        assert r is x  # returns self
        assert r.shape == (64,)
        assert r.dtype == dtype

    def test_random_inplace_bool(self):
        """Bool dtype: values ∈ {False, True} with roughly balanced draws."""
        x = torch.empty(10_000, dtype=torch.bool)
        r = decompositions.random_inplace(x)
        assert r.dtype == torch.bool
        # Expect ~50% True, allow generous slack.
        frac_true = r.to(torch.float32).mean().item()
        assert 0.4 < frac_true < 0.6

    def test_random_inplace_int_range(self):
        """Integer dtypes: values fall within the dtype's range."""
        for dtype in (torch.int8, torch.int16, torch.int32):
            x = torch.empty(1000, dtype=dtype)
            r = decompositions.random_inplace(x)
            info = torch.iinfo(dtype)
            r64 = r.to(torch.int64)
            assert r64.min().item() >= info.min
            assert r64.max().item() <= info.max

    def test_random_inplace_float_range(self):
        """Float dtype: values fall within ``[0, 2**24)`` (int-representable window)."""
        x = torch.empty(1000, dtype=torch.float32)
        r = decompositions.random_inplace(x)
        assert r.dtype == torch.float32
        assert r.min().item() >= 0.0
        assert r.max().item() < (1 << 24)

    @pytest.mark.parametrize("upper", [2, 10, 100, 1000])
    def test_random_inplace_to_range(self, upper):
        """``random_.to(self, upper)``: values ∈ [0, upper)."""
        x = torch.empty(2000, dtype=torch.int64)
        r = decompositions.random_inplace_to(x, upper)
        assert r is x
        assert r.min().item() >= 0
        assert r.max().item() < upper

    def test_random_inplace_to_uniform_distribution(self):
        """Empirically test the distribution is roughly uniform.

        For a large-enough sample the empirical proportion in each bucket
        should be within ~3sigma of the true 1/upper (Poisson approximation).
        """
        upper = 4
        n = 20_000
        x = torch.empty(n, dtype=torch.int64)
        r = decompositions.random_inplace_to(x, upper)
        counts = torch.bincount(r, minlength=upper).float() / n
        # Under a uniform distribution each bucket ≈ 1/upper with
        # std ≈ sqrt((1/u)*(1-1/u)/n). Use 5sigma as the tolerance.
        expected = 1.0 / upper
        stderr = (expected * (1 - expected) / n) ** 0.5
        assert torch.allclose(counts, torch.full_like(counts, expected), atol=5 * stderr)

    def test_generator_kwarg_returns_notimplemented(self):
        gen = torch.Generator()
        x = torch.empty(4, dtype=torch.int32)
        assert decompositions.random_inplace(x, generator=gen) is NotImplemented
        assert decompositions.random_inplace_to(x, 10, generator=gen) is NotImplemented


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRandomInplaceDecompositionE2E:
    """E2E: ``torch.compile(backend='neuron')`` compiles ``random_`` (and its
    functional counterpart after AOT autograd's functionalization pass) into
    a graph that executes on-device."""

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
    def test_random_inplace_e2e(self, dtype):
        def fn(x):
            return torch.ops.aten.random_(x)

        x = torch.empty(64, dtype=dtype, device="neuron")
        r = torch.compile(fn, backend="neuron", fullgraph=True)(x).cpu()
        assert r.shape == (64,)
        assert r.dtype == dtype

    @pytest.mark.parametrize("upper", [10, 256, 1024])
    def test_random_inplace_to_e2e(self, upper):
        def fn(x):
            return torch.ops.aten.random_.to(x, upper)

        x = torch.empty(1000, dtype=torch.int64, device="neuron")
        r = torch.compile(fn, backend="neuron", fullgraph=True)(x).cpu()
        assert r.min().item() >= 0
        assert r.max().item() < upper


class TestRandGeneratorDecomposition:
    """Unit tests for the ``rand``/``randint`` ``.generator`` overloads.

    Currently expects real Generator inputs to assert NotImplemented
    since that path is still not supported.
    """

    @pytest.fixture(autouse=True)
    def _force_cpu_rng(self, monkeypatch):
        monkeypatch.setattr(decompositions, "_has_nki_rng", False)

    def test_all_variants_registered(self):
        for op in [
            torch.ops.aten.rand.generator,
            torch.ops.aten.randint.generator,
            torch.ops.aten.randint.generator_out,
            torch.ops.aten.randint.low_generator,
            torch.ops.aten.randint.low_generator_out,
        ]:
            assert (
                op in decompositions.neuron_decompositions
            ), f"{op} missing from neuron_decompositions"

    def test_rand_generator_shape_dtype_range(self):
        r = decompositions.rand_generator_decomp([8, 8])
        assert r.shape == (8, 8)
        assert r.dtype == torch.float32
        assert r.min().item() >= 0.0 and r.max().item() < 1.0

    def test_randint_generator_shape_dtype_range(self):
        r = decompositions.randint_generator_decomp(10, [8, 8])
        assert r.shape == (8, 8)
        assert r.dtype == torch.long
        assert int(r.min().item()) >= 0 and int(r.max().item()) < 10

    def test_randint_low_generator_shape_dtype_range(self):
        r = decompositions.randint_low_generator_decomp(2, 10, [8, 8])
        assert r.shape == (8, 8)
        assert r.dtype == torch.long
        assert int(r.min().item()) >= 2 and int(r.max().item()) < 10

    def test_out_variants_write_and_return_out(self):
        out = torch.empty(8, 8, dtype=torch.long)
        r = decompositions.randint_generator_out_decomp(10, [8, 8], out=out)
        assert r.data_ptr() == out.data_ptr()
        assert int(out.min().item()) >= 0 and int(out.max().item()) < 10

        out2 = torch.empty(8, 8, dtype=torch.long)
        r2 = decompositions.randint_low_generator_out_decomp(2, 10, [8, 8], out=out2)
        assert r2.data_ptr() == out2.data_ptr()
        assert int(out2.min().item()) >= 2 and int(out2.max().item()) < 10

    def test_generator_kwarg_returns_notimplemented(self):
        """A user-supplied generator falls through to PyTorch's kernel."""
        gen = torch.Generator()
        out = torch.empty(4, dtype=torch.long)
        assert decompositions.rand_generator_decomp([4], generator=gen) is NotImplemented
        assert decompositions.randint_generator_decomp(10, [4], generator=gen) is NotImplemented
        assert (
            decompositions.randint_low_generator_decomp(2, 10, [4], generator=gen) is NotImplemented
        )
        assert (
            decompositions.randint_generator_out_decomp(10, [4], generator=gen, out=out)
            is NotImplemented
        )
        assert (
            decompositions.randint_low_generator_out_decomp(2, 10, [4], generator=gen, out=out)
            is NotImplemented
        )


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRandGeneratorDecompositionE2E:
    """E2E: ``torch.compile(backend='neuron')`` traces the ``.generator``
    overloads through our decompositions and executes them on-device.
    """

    def test_rand_generator_e2e(self):
        def fn():
            return torch.ops.aten.rand.generator([8, 8], generator=None, device="neuron")

        r = torch.compile(fn, backend="neuron", fullgraph=True)().cpu()
        assert r.shape == (8, 8)
        assert r.min().item() >= 0.0 and r.max().item() < 1.0

    def test_randint_generator_e2e(self):
        def fn():
            return torch.ops.aten.randint.generator(10, [8, 8], generator=None, device="neuron")

        r = torch.compile(fn, backend="neuron", fullgraph=True)().cpu()
        assert r.shape == (8, 8)
        assert int(r.min().item()) >= 0 and int(r.max().item()) < 10

    def test_randint_low_generator_e2e(self):
        def fn():
            return torch.ops.aten.randint.low_generator(
                2, 10, [8, 8], generator=None, device="neuron"
            )

        r = torch.compile(fn, backend="neuron", fullgraph=True)().cpu()
        assert r.shape == (8, 8)
        assert int(r.min().item()) >= 2 and int(r.max().item()) < 10

    def test_randint_generator_out_e2e(self):
        def fn(o):
            return torch.ops.aten.randint.generator_out(10, [8, 8], generator=None, out=o)

        out = torch.empty(8, 8, dtype=torch.long, device="neuron")
        r = torch.compile(fn, backend="neuron", fullgraph=True)(out).cpu()
        assert r.shape == (8, 8)
        assert int(r.min().item()) >= 0 and int(r.max().item()) < 10
