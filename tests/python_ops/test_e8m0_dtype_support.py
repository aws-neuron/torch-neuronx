"""Unit tests for float8_e8m0fnu (e8m0) dtype support in torch_neuronx."""

import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu not available in this PyTorch build",
)
class TestE8M0DtypeSupport:
    """Tests for e8m0 dtype registration across torch_neuronx mapping layers."""

    def test_nki_kernel_dtype_mapping(self):
        """e8m0 is registered in _TORCH_TO_NKI_DTYPE."""
        from torch_neuronx.nki_kernel import _TORCH_TO_NKI_DTYPE

        assert torch.float8_e8m0fnu in _TORCH_TO_NKI_DTYPE
        assert _TORCH_TO_NKI_DTYPE[torch.float8_e8m0fnu] == "float8_e8m0fnu"

    def test_nki_kernel_dtype_conversion(self):
        """_torch_to_nki_dtype converts e8m0 without error."""
        from torch_neuronx.nki_kernel import _torch_to_nki_dtype

        result = _torch_to_nki_dtype(torch.float8_e8m0fnu)
        assert result == "float8_e8m0fnu"

    def test_stablehlo_utils_dtype_mapping(self):
        """e8m0 is registered in stablehlo_utils MLIR_TO_TORCH_DTYPE."""
        from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
            MLIR_TO_TORCH_DTYPE,
        )

        assert "f8E8M0FNU" in MLIR_TO_TORCH_DTYPE
        assert MLIR_TO_TORCH_DTYPE["f8E8M0FNU"] == torch.float8_e8m0fnu

    def test_map_external_dtype_to_torch_e8m0(self):
        """map_external_dtype_to_torch handles e8m0 starfish dtype."""
        from torch_neuronx.utils import is_float8_e8m0fnu_supported, map_external_dtype_to_torch

        if not is_float8_e8m0fnu_supported():
            pytest.skip("e8m0 not supported on this platform")

        try:
            from neuronxcc.starfish.support import dtype as _ndt

            e8m0_ndt = getattr(_ndt, "float8_e8m0fnu", None)
            if e8m0_ndt is None:
                pytest.skip("neuronxcc does not expose float8_e8m0fnu dtype")
            result = map_external_dtype_to_torch(e8m0_ndt)
            assert result == torch.float8_e8m0fnu
        except ImportError:
            pytest.skip("neuronxcc not available")

    def test_map_external_dtype_rejects_on_unsupported_platform(self):
        """map_external_dtype_to_torch raises on unsupported platform for e8m0."""
        from torch_neuronx.utils import is_float8_e8m0fnu_supported, map_external_dtype_to_torch

        if is_float8_e8m0fnu_supported():
            pytest.skip("e8m0 is supported on this platform — cannot test rejection")

        try:
            from neuronxcc.starfish.support import dtype as _ndt

            e8m0_ndt = getattr(_ndt, "float8_e8m0fnu", None)
            if e8m0_ndt is None:
                pytest.skip("neuronxcc does not expose float8_e8m0fnu dtype")
            with pytest.raises(RuntimeError, match="Unsupported dtype"):
                map_external_dtype_to_torch(e8m0_ndt)
        except ImportError:
            pytest.skip("neuronxcc not available")

    def test_e8m0_not_in_general_supported_dtypes(self):
        """e8m0 should NOT be in general AMP/autocast supported dtypes."""
        from torch_neuronx.neuron import get_amp_supported_dtype

        supported = get_amp_supported_dtype()
        assert torch.float8_e8m0fnu not in supported
