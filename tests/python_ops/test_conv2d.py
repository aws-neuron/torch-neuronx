import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

TEST_CONFIGS = [
    # Test bias
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "bias": False,
        },
        (2, 3, 32, 32),
        "convolution",
        id="bias_False",
    ),
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
        },
        (2, 3, 32, 32),
        "convolution",
        id="bias_True",
    ),
    # Test different kernel sizes
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0},
        (2, 3, 32, 32),
        "convolution",
        id="kernel_size_1",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
        (2, 3, 32, 32),
        "convolution",
        id="kernel_size_5",
    ),
    # Test different strides
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1},
        (2, 3, 32, 32),
        "convolution",
        id="stride_2",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": (2, 1), "padding": 1},
        (2, 3, 32, 32),
        "convolution",
        id="stride_2_1",
    ),
    # Test different padding
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0},
        (2, 3, 32, 32),
        "convolution",
        id="padding_0",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": (1, 2)},
        (2, 3, 32, 32),
        "convolution",
        id="padding_1_2",
    ),
    # Test with "valid"/"same" padding
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "valid"},
        (2, 3, 32, 32),
        "convolution",
        id="padding_valid",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "same"},
        (2, 3, 32, 32),
        "convolution",
        id="padding_same",
    ),
    # Test groups
    pytest.param(
        {
            "in_channels": 4,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "groups": 2,
        },
        (2, 4, 32, 32),
        "convolution",
        id="groups_2",
    ),
    pytest.param(
        {
            "in_channels": 6,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "groups": 3,
        },
        (2, 6, 32, 32),
        "convolution",
        id="groups_3",
    ),
    # Test dilation
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 2,
        },
        (2, 3, 32, 32),
        "convolution",
        id="dilation_2",
    ),
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": (2, 1),
        },
        (2, 3, 32, 32),
        "convolution",
        id="dilation_2_1",
    ),
]

DTYPES = [
    pytest.param(torch.float16, id="float16"),
    pytest.param(torch.bfloat16, id="bfloat16"),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestConv2dRegistration:
    """Test cases for conv2d operation"""

    @pytest.mark.parametrize("config,input_shape,op_name", TEST_CONFIGS)
    def test_conv2d_run_on_neuron(self, config, input_shape, op_name):
        # Test if the op runs on neuron

        input = torch.randn(input_shape, dtype=torch.float32).to("neuron")
        conv2d = torch.nn.Conv2d(**config).to("neuron")

        with track_neuron_ops():
            output = conv2d(input)

        assert output.device.type == "neuron"
        assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("config,input_shape,op_name", TEST_CONFIGS)
    def test_conv2d_cpu_match(self, config, input_shape, op_name):
        """
        Test if conv2d runs accurately with different configurations of kernel,
        stride, padding, groups, dilation, and bias
        """

        input = torch.randn(input_shape, dtype=torch.float32).to("neuron")
        conv2d = torch.nn.Conv2d(**config).to("neuron")

        output = conv2d(input)
        assert output.dtype == torch.float32

        # Test on CPU for comparison
        input_cpu = input.cpu()
        conv2d_cpu = conv2d.cpu()
        output_cpu = conv2d_cpu(input_cpu)

        # Compare results
        torch.testing.assert_close(output.cpu(), output_cpu)
        torch_neuronx.clear_op_tracking()

    @pytest.mark.parametrize("config,input_shape,op_name", TEST_CONFIGS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_conv2d_different_dtypes(self, config, input_shape, op_name, dtype):
        # Test if conv2d runs accurately with different dtypes

        input = torch.randn(input_shape, dtype=dtype).to("neuron")
        conv2d = torch.nn.Conv2d(**config).to(dtype).to("neuron")

        output = conv2d(input)
        assert output.dtype == dtype

        # Test on CPU for comparison
        input_cpu = input.cpu()
        conv2d_cpu = conv2d.cpu()
        output_cpu = conv2d_cpu(input_cpu)

        # Compare results
        torch.testing.assert_close(
            output.cpu(),
            output_cpu,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Tensor-likes are not close for dtype: {dtype}!",
        )
        torch_neuronx.clear_op_tracking()
