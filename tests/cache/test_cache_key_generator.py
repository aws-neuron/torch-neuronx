"""Tests for CacheKeyGenerator Python path (GenerateFromPython via _C._generate_cache_key).

These tests exercise the pybind entry point that calls CacheKeyGenerator::GenerateFromPython.
They cover: type inference, static/non-static scalars, list tag inference, dtype auto-static
detection, device filtering, kwarg ordering, and tensor key parity with the C++ IValue path.

Companion to the C++ tests in tests/csrc/core/cache/CacheKeyGeneratorTest.cpp which test
the IValue path (Generate) directly.
"""

import pytest
import torch

from torch_neuronx import _C


def _gen(op_name, args, kwargs=None, static_argnums=None, static_argnames=None):
    """Helper to call the C++ CacheKeyGenerator via pybind."""
    return _C._generate_cache_key(
        op_name,
        tuple(args),
        kwargs or {},
        static_argnums=list(static_argnums or []),
        static_argnames=list(static_argnames or []),
    )


class TestBasicTypes:
    def test_tensor_shape_in_key(self):
        key = _gen("op", [torch.zeros(2, 3)])
        assert "[2,3]" in key

    def test_tensor_dtype_uses_tostring(self):
        key = _gen("op", [torch.zeros(2, dtype=torch.float32)])
        assert "Float" in key

    def test_tensor_strides_in_key(self):
        key = _gen("op", [torch.zeros(2, 3)])
        assert "S(C)" in key

    def test_different_shapes_different_keys(self):
        k1 = _gen("op", [torch.zeros(2, 3)])
        k2 = _gen("op", [torch.zeros(4, 5)])
        assert k1 != k2

    def test_same_shapes_same_key(self):
        k1 = _gen("op", [torch.zeros(2, 3)])
        k2 = _gen("op", [torch.ones(2, 3)])
        assert k1 == k2

    def test_different_dtypes_different_keys(self):
        k1 = _gen("op", [torch.zeros(2, dtype=torch.float32)])
        k2 = _gen("op", [torch.zeros(2, dtype=torch.float64)])
        assert k1 != k2

    def test_different_strides_different_keys(self):
        k1 = _gen("op", [torch.zeros(2, 3)])
        k2 = _gen("op", [torch.zeros(3, 2).t()])
        assert k1 != k2

    def test_bool_value_aware(self):
        k1 = _gen("op", [torch.zeros(2), True])
        k2 = _gen("op", [torch.zeros(2), False])
        assert k1 != k2

    def test_string_value_aware(self):
        k1 = _gen("op", [torch.zeros(2), "abc"])
        k2 = _gen("op", [torch.zeros(2), "xyz"])
        assert k1 != k2

    def test_none_in_key(self):
        key = _gen("op", [torch.zeros(2), None])
        assert "None" in key


class TestStaticNonStatic:
    def test_non_static_int_same_key(self):
        k1 = _gen("op", [torch.zeros(2), 1])
        k2 = _gen("op", [torch.zeros(2), 2])
        assert k1 == k2

    def test_static_int_different_keys(self):
        k1 = _gen("op", [torch.zeros(2), 1], static_argnums=[1])
        k2 = _gen("op", [torch.zeros(2), 2], static_argnums=[1])
        assert k1 != k2

    def test_non_static_float_same_key(self):
        k1 = _gen("op", [torch.zeros(2), 1.5])
        k2 = _gen("op", [torch.zeros(2), 2.5])
        assert k1 == k2

    def test_static_float_different_keys(self):
        k1 = _gen("op", [torch.zeros(2), 1.5], static_argnums=[1])
        k2 = _gen("op", [torch.zeros(2), 2.5], static_argnums=[1])
        assert k1 != k2

    def test_static_float_keeps_precision_beyond_6_digits(self):
        """A float that differs past the 6th significant digit needs its own key.

        The key used the default ostream format, which keeps 6 significant digits,
        so arange(0.0, -8.000001, -4.0) and arange(0.0, -8.0, -4.0) shared one
        entry. Their output lengths differ (3 vs 2), so the first to compile
        returned its length for both.
        """
        k1 = _gen("op", [torch.zeros(2), -8.000001], static_argnums=[1])
        k2 = _gen("op", [torch.zeros(2), -8.0], static_argnums=[1])
        assert k1 != k2

    def test_static_float_distinguishes_signed_zero(self):
        k1 = _gen("op", [torch.zeros(2), 0.0], static_argnums=[1])
        k2 = _gen("op", [torch.zeros(2), -0.0], static_argnums=[1])
        assert k1 != k2


class TestListInference:
    def test_int_list_tag(self):
        key = _gen("op", [torch.zeros(2), [1, 2, 3]])
        assert "IL[" in key

    def test_bool_list_tag(self):
        key = _gen("op", [torch.zeros(2), [True, False]])
        assert "BL[" in key

    def test_float_list_tag(self):
        key = _gen("op", [torch.zeros(2), [1.0, 2.0]])
        assert "DL[" in key

    def test_empty_list_tag(self):
        key = _gen("op", [torch.zeros(2), []])
        assert "GL[]" in key

    def test_tensor_list_tag(self):
        key = _gen("op", [torch.zeros(2), [torch.zeros(3), torch.ones(4)]])
        assert "TL[" in key


class TestDtypeAutoStatic:
    def test_dtype_kwarg_auto_detected_as_static(self):
        k1 = _gen("op", [torch.zeros(2)], kwargs={"dtype": torch.float32})
        k2 = _gen("op", [torch.zeros(2)], kwargs={"dtype": torch.float16})
        assert k1 != k2

    def test_dtype_none_same_key(self):
        k1 = _gen("op", [torch.zeros(2)], kwargs={"dtype": None})
        k2 = _gen("op", [torch.zeros(2)], kwargs={"dtype": None})
        assert k1 == k2


class TestKwargFiltering:
    def test_device_kwarg_filtered_by_default(self):
        k_no_dev = _gen("op", [torch.zeros(2)])
        k_with_dev = _gen("op", [torch.zeros(2)], kwargs={"device": torch.device("cpu")})
        # "device" is in FilteredKwargs — should be excluded, so keys match
        assert k_no_dev == k_with_dev

    def test_device_kwarg_included_when_static(self):
        key = _gen(
            "op",
            [torch.zeros(2)],
            kwargs={"device": torch.device("cpu")},
            static_argnames=["device"],
        )
        assert "dev:" in key

    def test_kwargs_sorted_deterministic(self):
        k1 = _gen(
            "op",
            [torch.zeros(2)],
            kwargs={"alpha": 1, "beta": 2},
            static_argnames=["alpha", "beta"],
        )
        k2 = _gen(
            "op",
            [torch.zeros(2)],
            kwargs={"beta": 2, "alpha": 1},
            static_argnames=["alpha", "beta"],
        )
        assert k1 == k2


class TestTensorParity:
    def test_tensor_fragment_matches_cpp_format(self):
        """Verify the Python path produces the same tensor encoding as the C++ path.

        Both paths use the shared AppendTensor formatter, so the tensor fragment
        should be identical: [shape]:DtypeName:S(strides).
        """
        t = torch.zeros(4, 8, dtype=torch.float32)
        key = _gen("op", [t])
        assert "[4,8]:Float:S(C)" in key

    def test_op_name_in_key(self):
        key = _gen("my_custom_op", [torch.zeros(2)])
        assert "my_custom_op" in key
