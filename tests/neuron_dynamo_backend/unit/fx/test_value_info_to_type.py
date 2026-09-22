"""
Unit tests for ContextCache.value_info_to_type error handling.

Regression coverage for NCF-1607: when the FX importer encounters a value
whose type it cannot deduce (e.g. MoE models with sparse expert routing that
produce nodes outside the supported tensor/scalar set), the final fallback in
``value_info_to_type`` must raise a clean ``NotImplementedError``. Previously
the error message interpolated an undefined ``sparsity`` variable, so building
the message raised ``NameError`` instead, masking the real "unsupported value"
signal with a confusing crash.
"""

import pytest

from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronContextCache


@pytest.fixture
def cache():
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d

    context = ir.Context()
    torch_d.register_dialect(context)
    return NeuronContextCache(context, py_attr_tracker=None)


class _Undeducible:
    """An object that is neither a fake tensor nor a supported scalar type."""


class TestValueInfoToTypeFallback:
    """Tests for the un-deducible-value fallback in value_info_to_type."""

    @pytest.mark.parametrize(
        "val",
        [
            _Undeducible(),
            [1, 2, 3],  # list of non-tensors
            ("a", "b"),  # tuple is not a supported scalar type
            {"k": "v"},  # dict is not a supported scalar type
        ],
    )
    def test_undeducible_value_raises_notimplementederror(self, cache, val):
        """A value with no deducible type must raise NotImplementedError, not NameError."""
        with cache._c, pytest.raises(NotImplementedError) as exc_info:
            cache.value_info_to_type(val)

        # The message must be the intended "could not deduce" diagnostic and must
        # not reference the removed ``sparsity`` variable.
        message = str(exc_info.value)
        assert "Could not deduce type from value info" in message
        assert "sparsity" not in message

    def test_undeducible_value_does_not_raise_nameerror(self, cache):
        """Guards specifically against the NCF-1607 regression (NameError from the f-string)."""
        with cache._c:
            try:
                cache.value_info_to_type(_Undeducible())
            except NotImplementedError:
                pass  # expected
            except NameError as e:  # pragma: no cover - this is the bug we fixed
                pytest.fail(
                    f"value_info_to_type raised NameError instead of NotImplementedError: {e}"
                )

    def test_supported_scalar_still_resolves(self, cache):
        """Sanity check: supported scalar values still resolve without hitting the fallback."""
        with cache._c:
            # int and None are in SCALAR_TYPE_TO_TORCH_MLIR_TYPE and should resolve.
            # Compare against the cache's own canonical types rather than printed
            # strings: MLIR types are uniqued within a Context, so equality holds
            # by interning and does not depend on the printer's formatting.
            assert cache.value_info_to_type(5) == cache.torch_int_type
            assert cache.value_info_to_type(None) == cache.torch_none_type
