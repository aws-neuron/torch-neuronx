import pytest
import torch

import torch_neuronx  # Registers neuron backend and ops
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestIndicesDtypeIntegration:
    def test_max_dim_indices_types_and_values_neuron(self):
        """
        End-to-end: ensure indices are int64 on return and values/indices
        match CPU results when running on neuron.
        """
        device = "neuron:0"
        x = torch.randn(4, 5, dtype=torch.float32, device=device)

        with track_neuron_ops():
            values, indices = torch.max(x, dim=1)

            # Dtype expectations
            assert values.dtype == torch.float32
            assert indices.dtype == torch.int64

            # CPU parity
            v_cpu, i_cpu = torch.max(x.cpu(), dim=1)
            torch.testing.assert_close(values.cpu(), v_cpu)
            torch.testing.assert_close(indices.cpu(), i_cpu)

            # Op tracking (accepts either full or base name via helper)
            assert_op_runs_on_neuron("aten::max.dim")

    def test_max_dim_indices_out_tuple_resize_and_castback(self):
        """
        Provide out tensors with wrong shapes; expect resize warning and proper
        cast-back to int64 for indices. Values/indices must match CPU results.
        """
        device = "neuron:0"
        x = torch.randn(3, 7, dtype=torch.float32, device=device)

        # Prepare outs with mismatched shapes
        v_out = torch.empty((1,), dtype=torch.float32, device=device)
        i_out = torch.empty((1,), dtype=torch.int64, device=device)

        with track_neuron_ops():
            values, indices = torch.max(x, dim=1, out=(v_out, i_out))

            # Returned tensors should be the same objects
            assert values.data_ptr() == v_out.data_ptr()
            assert indices.data_ptr() == i_out.data_ptr()

            # Shapes resized to expected
            assert values.shape == (x.shape[0],)
            assert indices.shape == (x.shape[0],)

            # Dtypes (indices must be int64)
            assert values.dtype == torch.float32
            assert indices.dtype == torch.int64

            # CPU parity
            v_cpu, i_cpu = torch.max(x.cpu(), dim=1)
            torch.testing.assert_close(values.cpu(), v_cpu)
            torch.testing.assert_close(indices.cpu(), i_cpu)

            assert_op_runs_on_neuron("aten::max.dim")
