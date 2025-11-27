import os

import pytest
import torch

from tests.utils.neuron_test_utils import count_cpu_bounce_calls
from torch_neuronx.kernels.type_converter import TypeConverter


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0",
    reason="Bindings are only applicable to the legacy logic",
)
class TestTypeConverter:
    """Test type converter for various implicit downcasts/upcasts"""

    def test_cpu_int64_downcasts_without_nrt_calls(self):
        x64 = torch.ones(8, dtype=torch.int64, device="cpu")

        def run():
            y = TypeConverter.convert_for_neuron(x64)
            assert y.device.type == "cpu"
            assert y.dtype == torch.int32

        counts = count_cpu_bounce_calls(run)
        assert counts["reads"] == 0 and counts["writes"] == 0

    def test_cpu_float64_downcasts_without_nrt_calls(self):
        x64 = torch.ones(8, dtype=torch.float64, device="cpu")

        def run():
            y = TypeConverter.convert_for_neuron(x64)
            assert y.device.type == "cpu"
            assert y.dtype == torch.float32

        counts = count_cpu_bounce_calls(run)
        assert counts["reads"] == 0 and counts["writes"] == 0

    def test_neuron_int64_downcasts_with_single_bounce(self):
        x64 = torch.ones(8, dtype=torch.int64, device="neuron")

        def run():
            y = TypeConverter.convert_for_neuron(x64)
            assert y.device.type == "neuron"
            assert y.dtype == torch.int32

        counts = count_cpu_bounce_calls(run)
        assert counts["reads"] == 1 and counts["writes"] == 1

    def test_neuron_float64_downcasts_with_single_bounce(self):
        x64 = torch.ones(8, dtype=torch.float64, device="neuron")

        def run():
            y = TypeConverter.convert_for_neuron(x64)
            assert y.device.type == "neuron"
            assert y.dtype == torch.float32

        counts = count_cpu_bounce_calls(run)
        assert counts["reads"] == 1 and counts["writes"] == 1

    def test_neuron_32bit_types_return_unchanged(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int32):
            x = (
                torch.randn(4, dtype=dtype, device="neuron")
                if dtype.is_floating_point
                else torch.ones(4, dtype=dtype, device="neuron")
            )

            def run(x=x, dtype=dtype):
                y = TypeConverter.convert_for_neuron(x)
                assert y is x
                assert y.dtype == dtype and y.device.type == "neuron"

            counts = count_cpu_bounce_calls(run)
            assert counts["reads"] == 0 and counts["writes"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
