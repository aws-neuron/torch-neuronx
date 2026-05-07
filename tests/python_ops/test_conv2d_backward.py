import time

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.utils import use_mlir_aten_ops

TEST_CONFIGS = [
    # Test bias
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "bias": False,
        },
        (2, 3, 10, 10),
        "convolution_backward",
        id="bias_False",
    ),
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
        },
        (2, 3, 10, 10),
        "convolution_backward",
        id="bias_True",
    ),
    # Test valid padding with asymmetric spatial dimensions
    # for tuple expansion bug
    pytest.param(
        {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": (1, 4),
            "stride": 1,
            "padding": 0,
            "bias": False,
        },
        (1, 1, 1, 10),
        "convolution_backward",
        id="valid_padding_asymmetric_spatial",
    ),
    # Test different kernel sizes
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 1, "stride": 1, "padding": 0},
        (2, 3, 10, 10),
        "convolution_backward",
        id="kernel_size_1",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 5, "stride": 1, "padding": 2},
        (2, 3, 10, 10),
        "convolution_backward",
        id="kernel_size_5",
    ),
    # Test different strides
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1},
        (2, 3, 10, 10),
        "convolution_backward",
        id="stride_2",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "stride": 3, "padding": 1},
        (2, 3, 15, 15),
        "convolution_backward",
        id="stride_3",
    ),
    # Test different padding
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 0},
        (2, 3, 10, 10),
        "convolution_backward",
        id="padding_0",
    ),
    pytest.param(
        {"in_channels": 3, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 2},
        (2, 3, 10, 10),
        "convolution_backward",
        id="padding_2",
        marks=pytest.mark.xfail(
            reason="invalid offset in SB - memory layout issue in compiled kernel"
        ),
    ),
    # Test different channel configurations
    pytest.param(
        {"in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        (2, 1, 10, 10),
        "convolution_backward",
        id="channels_1_to_4",
    ),
    pytest.param(
        {"in_channels": 4, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1},
        (2, 4, 10, 10),
        "convolution_backward",
        id="channels_4_to_1",
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
        (2, 4, 10, 10),
        "convolution_backward",
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
        (2, 6, 10, 10),
        "convolution_backward",
        id="groups_3",
    ),
    # Test dilation
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 2,
        },
        (2, 3, 10, 10),
        "convolution_backward",
        id="dilation_2",
        marks=pytest.mark.xfail(
            condition=not use_mlir_aten_ops(),
            reason="dilation=2 not supported without torch-mlir aten ops",
        ),
    ),
    pytest.param(
        {
            "in_channels": 3,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 3,
        },
        (2, 3, 14, 14),
        "convolution_backward",
        id="dilation_3",
    ),
]

DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-4, 1e-4, id="float32"),
    pytest.param(torch.float16, 1e-2, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bfloat16"),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestConv2dBackwardRegistration:
    """Test cases for conv2d backward operation"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        yield
        torch_neuronx.clear_op_tracking()

    @pytest.mark.parametrize("config,input_shape,op_name", TEST_CONFIGS)
    def test_conv2d_run_on_neuron(self, config, input_shape, op_name):
        # Test if the op runs on neuron

        input = torch.randn(input_shape, dtype=torch.float32, requires_grad=True).to("neuron")
        kernel_size = config["kernel_size"]
        # Handle tuple or int kernel_size
        if isinstance(kernel_size, int):
            kernel_h, kernel_w = kernel_size, kernel_size
        else:
            kernel_h, kernel_w = kernel_size
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        groups = config.get("groups", 1)
        weight = torch.randn(
            out_channels, in_channels // groups, kernel_h, kernel_w, dtype=torch.float32
        ).to("neuron")
        bias = (
            torch.randn(out_channels, dtype=torch.float32).to("neuron")
            if config.get("bias", True)
            else None
        )
        padding = config.get("padding", 0)
        stride = config.get("stride", 1)
        dilation = config.get("dilation", 1)

        # Calculate output shape and create grad_output
        h_out = ((input_shape[-2] + 2 * padding - dilation * (kernel_h - 1) - 1) // stride) + 1
        w_out = ((input_shape[-1] + 2 * padding - dilation * (kernel_w - 1) - 1) // stride) + 1
        grad_output = torch.randn(
            input_shape[0], out_channels, h_out, w_out, dtype=torch.float32
        ).to("neuron")

        with track_neuron_ops():
            grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                bias_sizes=bias.shape if bias is not None else None,
                stride=(stride, stride),
                padding=(padding, padding),
                dilation=(dilation, dilation),
                transposed=False,
                output_padding=(0, 0),
                groups=groups,
                output_mask=[True, True, bias is not None],
            )

        assert_op_runs_on_neuron(op_name)

    def test_conv2d_fwd_bwd_basic(self):
        """Test forward-backward pass using loss.backward()"""

        input = torch.randn(2, 3, 10, 10, dtype=torch.float32, requires_grad=True).to("neuron")
        input.retain_grad()
        conv = torch.nn.Conv2d(3, 2, 3, padding=1).to("neuron")
        target = torch.randn(2, 2, 10, 10, dtype=torch.float32).to("neuron")

        with track_neuron_ops():
            output = conv(input)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()

        assert_op_runs_on_neuron("convolution_backward")
        assert input.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    @pytest.mark.parametrize("config,input_shape,op_name", TEST_CONFIGS)
    @pytest.mark.parametrize("dtype,atol,rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_conv2d_cpu_match(self, config, input_shape, op_name, dtype, atol, rtol):
        # Test if gradients match between cpu and neuron

        def run_conv2d_backward(device):
            torch.manual_seed(0)
            input = torch.randn(input_shape, dtype=dtype, requires_grad=True)
            kernel_size = config["kernel_size"]
            # Handle tuple or int kernel_size
            if isinstance(kernel_size, int):
                kernel_h, kernel_w = kernel_size, kernel_size
            else:
                kernel_h, kernel_w = kernel_size
            in_channels = config["in_channels"]
            out_channels = config["out_channels"]
            groups = config.get("groups", 1)
            weight = torch.randn(
                out_channels, in_channels // groups, kernel_h, kernel_w, dtype=dtype
            )
            bias = torch.randn(out_channels, dtype=dtype) if config.get("bias", True) else None
            padding = config.get("padding", 0)
            stride = config.get("stride", 1)
            dilation = config.get("dilation", 1)

            h_out = ((input_shape[-2] + 2 * padding - dilation * (kernel_h - 1) - 1) // stride) + 1
            w_out = ((input_shape[-1] + 2 * padding - dilation * (kernel_w - 1) - 1) // stride) + 1
            grad_output = torch.randn(input_shape[0], out_channels, h_out, w_out, dtype=dtype)

            if device == "neuron":
                input = input.to(device)
                weight = weight.to(device)
                grad_output = grad_output.to(device)
                if bias is not None:
                    bias = bias.to(device)

            grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                bias_sizes=bias.shape if bias is not None else None,
                stride=(stride, stride),
                padding=(padding, padding),
                dilation=(dilation, dilation),
                transposed=False,
                output_padding=(0, 0),
                groups=groups,
                output_mask=[True, True, bias is not None],
            )

            return grad_input, grad_weight, grad_bias

        # Run on both devices
        with track_neuron_ops():
            neuron_grad_input, neuron_grad_weight, neuron_grad_bias = run_conv2d_backward("neuron")
            assert_op_runs_on_neuron("convolution_backward")
        neuron_grad_input, neuron_grad_weight, neuron_grad_bias = run_conv2d_backward("neuron")
        cpu_grad_input, cpu_grad_weight, cpu_grad_bias = run_conv2d_backward("cpu")

        # Compare results
        torch.testing.assert_close(neuron_grad_input.cpu(), cpu_grad_input, atol=atol, rtol=rtol)
        torch.testing.assert_close(neuron_grad_weight.cpu(), cpu_grad_weight, atol=atol, rtol=rtol)
        if neuron_grad_bias is not None:
            torch.testing.assert_close(neuron_grad_bias.cpu(), cpu_grad_bias, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "output_mask",
        [
            [True, True, True],
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
    )
    def test_conv2d_output_mask(self, output_mask):
        """Test convolution_backward with different output_mask combinations"""

        def run_with_mask(device):
            torch.manual_seed(0)
            input = torch.randn((2, 3, 10, 10), dtype=torch.float32, requires_grad=True)
            weight = torch.randn(2, 3, 3, 3, dtype=torch.float32)
            bias = torch.randn(2, dtype=torch.float32)
            grad_output = torch.randn(2, 2, 10, 10, dtype=torch.float32)

            if device == "neuron":
                input = input.to(device)
                weight = weight.to(device)
                bias = bias.to(device)
                grad_output = grad_output.to(device)

            return torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                bias_sizes=bias.shape,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
                output_mask=output_mask,
            )

        # Run on both devices
        with track_neuron_ops():
            neuron_results = run_with_mask("neuron")
            assert_op_runs_on_neuron("convolution_backward")
        cpu_results = run_with_mask("cpu")

        # Compare results
        for i, (neuron_grad, cpu_grad) in enumerate(zip(neuron_results, cpu_results, strict=False)):
            if output_mask[i]:
                assert neuron_grad is not None and cpu_grad is not None
                torch.testing.assert_close(neuron_grad.cpu(), cpu_grad, atol=1e-5, rtol=1e-5)
            else:
                assert neuron_grad is None

    @pytest.mark.parametrize(
        "error_type,args,expected_error,match_pattern",
        [
            (
                "negative_padding",
                {
                    "grad_output": (2, 2, 10, 10),
                    "input": (2, 3, 10, 10),
                    "weight": (2, 3, 3, 3),
                    "stride": (1, 1),
                    "padding": (-1, -1),
                    "dilation": (1, 1),
                    "groups": 1,
                    "transposed": False,
                },
                RuntimeError,
                "negative padding is not supported",
            ),
            (
                "non_positive_stride",
                {
                    "grad_output": (2, 2, 10, 10),
                    "input": (2, 3, 10, 10),
                    "weight": (2, 3, 3, 3),
                    "stride": (0, 0),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "transposed": False,
                },
                RuntimeError,
                "non-positive stride is not supported",
            ),
            (
                "weight_groups_mismatch",
                {
                    "grad_output": (2, 1, 10, 10),
                    "input": (2, 4, 10, 10),
                    "weight": (1, 2, 3, 3),
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 2,
                    "transposed": False,
                },
                RuntimeError,
                "Given groups=2, expected weight to be at least 2.*but got weight of size.*1",
            ),
            (
                "weight_not_divisible_by_groups",
                {
                    "grad_output": (2, 3, 10, 10),
                    "input": (2, 4, 10, 10),
                    "weight": (3, 2, 3, 3),
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 2,
                    "transposed": False,
                },
                RuntimeError,
                "Given groups=2, expected weight to be divisible by 2.*but got weight of "
                "size.*3",
            ),
            (
                "grad_output_input_dimension_mismatch",
                {
                    "grad_output": (2, 2, 10),
                    "input": (2, 3, 10, 10),
                    "weight": (2, 3, 3, 3),
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "transposed": False,
                },
                RuntimeError,
                "Expected input and grad_output to have the same number of dimensions.*"
                "got.*4.*3",
            ),
            (
                "input_channels_mismatch",
                {
                    "grad_output": (2, 2, 10, 10),
                    "input": (2, 5, 10, 10),
                    "weight": (2, 2, 3, 3),
                    "stride": (1, 1),
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "groups": 1,
                    "transposed": False,
                },
                RuntimeError,
                "Given groups=1.*expected input.*to have 2 channels.*but got 5 channels",
            ),
        ],
    )
    def test_conv2d_backward_validation_errors(
        self, error_type, args, expected_error, match_pattern
    ):
        """Test validation errors in convolution backward"""
        # Create tensors from shape tuples
        grad_output = torch.randn(args["grad_output"]).to("neuron")
        input = torch.randn(args["input"]).to("neuron")
        weight = torch.randn(args["weight"]).to("neuron")

        stride = args.get("stride", (1, 1))
        padding = args.get("padding", (0, 0))
        dilation = args.get("dilation", (1, 1))

        with pytest.raises(expected_error, match=match_pattern):
            torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                None,
                stride,
                padding,
                dilation,
                args["transposed"],
                (0, 0),
                args["groups"],
                [True, True, False],
            )
