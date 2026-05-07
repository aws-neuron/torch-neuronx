"""
Unit tests for custom FxImporter hooks.
"""

import ast
import re

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.fx.fx_hooks import CustomFxImporterHooks


@pytest.fixture
def create_test_data():
    def _create_test_data(dtype, shape):
        """Create test data for given dtype and shape."""
        import numpy as np

        if not shape:  # scalar
            if dtype in [torch.float16, torch.float32, torch.float64]:
                return 1.5
            elif dtype in [torch.int32, torch.int64]:
                return 123
            else:
                return True

        # Generate data based on total size
        total_size = np.prod(shape)

        if dtype in [torch.float16, torch.float32, torch.float64]:
            flat_data = np.arange(1, total_size + 1, dtype=np.float32)
        elif dtype in [torch.int32, torch.int64]:
            flat_data = np.arange(1, total_size + 1, dtype=np.int32)
        else:
            flat_data = np.arange(total_size) % 2 == 0

        # Reshape and convert to nested Python lists
        return flat_data.reshape(shape).tolist()

    return _create_test_data


@pytest.fixture
def hooks():
    return CustomFxImporterHooks()


def _extract_dense_data(module_str):
    """Extract data from dense<...> patterns in MLIR string."""
    pattern = r"dense<([^>]+)>"
    matches = re.findall(pattern, module_str)
    extracted_data = []
    for match in matches:
        try:
            # Replace 'true'/'false' with Python booleans
            match = match.replace("true", "True").replace("false", "False")
            data = ast.literal_eval(match)
            extracted_data.append(data)
        except (ValueError, SyntaxError):
            continue
    return extracted_data


class TestCustomFxImporterHooks:
    """Tests the use of CustomFxImporterHooks in importing the FX graph"""

    @pytest.mark.parametrize(
        "dtype", [torch.float64, torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
    )
    @pytest.mark.parametrize("shape", [(), (4,), (2, 3), (1, 4, 4)])
    def test_custom_fx_importer_hooks_literal_tensors(self, create_test_data, hooks, dtype, shape):
        """Test CustomFxImporterHooks with constant tensors of various dtypes and shapes."""

        data = create_test_data(dtype, shape)

        # Create a simple model that returns a constant tensor
        class ConstantModel(nn.Module):
            def forward(self, x):
                return x + torch.tensor(data, dtype=dtype)

        model = ConstantModel()
        input_tensor = torch.ones(shape, dtype=torch.float32)

        # Get graph
        gm = get_aot_graphs(model, input_tensor).post_aot_forward_graph

        # Import with custom hooks
        from torch_mlir import fx

        # Should not raise an exception
        mlir_module = fx.stateless_fx_import(gm, hooks=hooks)

        # Verify the module was created successfully
        assert mlir_module is not None
        module_str = str(mlir_module)
        assert "torch.vtensor.literal" in module_str
        assert "dense_resource" not in module_str

        # Check that tensor data appears in the MLIR
        extracted_data = _extract_dense_data(module_str)
        assert len(extracted_data) > 0
        assert data in extracted_data

    @pytest.mark.parametrize(
        "dtype", [torch.float64, torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
    )
    @pytest.mark.parametrize("shape", [(), (6,), (2, 3), (1, 4, 4)])
    def test_custom_fx_importer_hooks_multiple_constants(
        self, create_test_data, hooks, dtype, shape
    ):
        """Test CustomFxImporterHooks with multiple constant tensors."""

        const1_data = create_test_data(dtype, shape)
        const2_data = create_test_data(torch.int32, shape)

        class MultiConstantModel(nn.Module):
            def forward(self, x):
                return (
                    x
                    + torch.tensor(const1_data, dtype=dtype).sum()
                    + torch.tensor(const2_data, dtype=torch.int32).float().mean()
                )

        model = MultiConstantModel()
        input_tensor = torch.ones(shape, dtype=torch.float32)

        gm = get_aot_graphs(model, input_tensor).post_aot_forward_graph

        from torch_mlir import fx

        mlir_module = fx.stateless_fx_import(gm, hooks=hooks)

        assert mlir_module is not None
        module_str = str(mlir_module)
        assert "torch.vtensor.literal" in module_str
        assert "dense_resource" not in module_str

        # Check that tensor data appears in the MLIR
        extracted_data = _extract_dense_data(module_str)
        assert len(extracted_data) >= 2
        assert const1_data in extracted_data
        assert const2_data in extracted_data
