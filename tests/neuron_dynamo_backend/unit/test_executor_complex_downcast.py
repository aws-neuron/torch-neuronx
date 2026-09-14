"""Executor complex128<->complex64 IO downcast.

Trainium has no native complex128, so once the graph is demoted to complex64
(DemoteFp64Complex FX pass) the executor must feed the NEFF complex64 inputs and
restore complex128 outputs, mirroring its existing float64 handling. Without it
the raw complex128 buffer is bound to a complex64 slot and reads as garbage
(NaN). These tests drive the dtype conversion device-free via ``meta`` tensors.
"""

import torch

from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec


def _executor(inputs=None, outputs=None, cast_spec=None):
    io = FunctionIO(inputs=inputs or [], outputs=outputs or [])
    return Executor("g", "k", io, cast_spec or [])


class TestPrepareInputsDowncast:
    def test_complex128_input_downcast_to_complex64(self):
        ex = _executor(inputs=[TensorSpec(dtype=torch.complex64, shape=[4])])
        inp = torch.empty(4, dtype=torch.complex128, device="meta")
        out = list(ex._prepare_inputs([inp], 0))
        assert out[0].dtype == torch.complex64

    def test_float64_input_still_downcast_to_float32(self):
        # regression guard: the pre-existing float64 branch keeps working.
        ex = _executor(inputs=[TensorSpec(dtype=torch.float32, shape=[4])])
        inp = torch.empty(4, dtype=torch.float64, device="meta")
        out = list(ex._prepare_inputs([inp], 0))
        assert out[0].dtype == torch.float32

    def test_complex64_input_unchanged(self):
        ex = _executor(inputs=[TensorSpec(dtype=torch.complex64, shape=[4])])
        inp = torch.empty(4, dtype=torch.complex64, device="meta")
        out = list(ex._prepare_inputs([inp], 0))
        assert out[0].dtype == torch.complex64


class TestProcessOutputsUpcast:
    def test_complex128_output_restored(self):
        ex = _executor(cast_spec=[TensorSpec(dtype=torch.complex128, shape=[4])])
        neff_out = torch.empty(4, dtype=torch.complex64, device="meta")
        res = ex._process_outputs([neff_out])
        assert res[0].dtype == torch.complex128

    def test_complex64_output_unchanged(self):
        ex = _executor(cast_spec=[TensorSpec(dtype=torch.complex64, shape=[4])])
        neff_out = torch.empty(4, dtype=torch.complex64, device="meta")
        res = ex._process_outputs([neff_out])
        assert res[0].dtype == torch.complex64
