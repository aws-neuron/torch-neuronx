"""Pooling operations registered via register_aten for eager dispatch."""

import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type_name

from ..operation_registry import Preprocessing, register_aten
from ..output_layout import channels_last_strides, suggest_output_memory_format


def _validate_adaptive_avg_pool2d(x, output_size, out=None):
    """Reject non-floating-point input, matching CPU's adaptive_avg_pool2d."""
    if not x.is_floating_point():
        raise RuntimeError(f"\"adaptive_avg_pool2d\" not implemented for '{x.dtype}'")


@register_aten(
    ["aten::_adaptive_avg_pool2d", "aten::_adaptive_avg_pool2d.out"],
    output_params=("out",),
    static_argnums=(1,),
    validate_static_inputs=_validate_adaptive_avg_pool2d,
)
def torch_adaptive_avg_pool2d(x, output_size, out=None):
    # An empty output dimension yields an empty result; short-circuit before the
    # upstream decomposition, which divides input shape by output_size (% 0 crash).
    """Compute two-dimensional adaptive average pooling.

    Args:
        x (torch.Tensor): Input tensor.
        output_size (list or tuple): Requested output height and width.
        out (torch.Tensor, optional): Destination written by the out variant. Defaults to ``None``.

    Returns:
        torch.Tensor: Adaptively pooled tensor, including empty requested dimensions.

    Raises:
        RuntimeError: If ``x`` is not floating point.
    """
    if 0 in tuple(output_size):
        return x.new_empty((*x.shape[:-2], *output_size))
    return torch.ops.aten._adaptive_avg_pool2d(x, output_size)


def _validate_adaptive_avg_pool2d_backward(grad_output, self, out=None):
    """Reject grad_output whose rank doesn't match the input, matching CPU."""
    if grad_output.ndim != self.ndim:
        raise RuntimeError(
            f"Expected dimensions {self.ndim} for grad_output but got dimensions "
            f"{grad_output.ndim}"
        )


@register_aten(
    ["aten::_adaptive_avg_pool2d_backward", "aten::_adaptive_avg_pool2d_backward.out"],
    output_params=("out",),
    validate_static_inputs=_validate_adaptive_avg_pool2d_backward,
)
def torch_adaptive_avg_pool2d_backward(grad_output, self, out=None):
    """Compute the adaptive average-pooling input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the pooled output.
        self (torch.Tensor): Input tensor from the forward pass.
        out (torch.Tensor, optional): Destination written by the out variant. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.

    Raises:
        RuntimeError: If ``grad_output`` and ``self`` have different ranks.
    """
    return torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, self)


def _pair(value):
    """Return a length-2 tuple from an int or a length-1/2 sequence."""
    if isinstance(value, list | tuple):
        return (
            (int(value[0]), int(value[0]))
            if len(value) == 1
            else (
                int(value[0]),
                int(value[1]),
            )
        )
    return int(value), int(value)


def _iter(value):
    """Iterate a scalar or sequence uniformly."""
    return value if isinstance(value, list | tuple) else (value,)


# CPU avg_pool2d supports floating-point dtypes and int64, but rejects the
# smaller integer/bool dtypes (int8/int16/int32/uint8/bool).
_AVG_POOL2D_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int64,
)

INT32_MAX = 2**31 - 1


def _validate_avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    out=None,
):
    """Validate avg_pool2d inputs, matching CPU's error messages verbatim.

    The Neuron dispatch path skips PyTorch's structured-kernel validation, so we
    reproduce every CPU check (and its exact wording) here rather than in the op
    body. The validator receives the full op signature (it is called with the
    same args as the op), so out= and divisor_override are validated here too.
    """
    if x.dtype not in _AVG_POOL2D_SUPPORTED_DTYPES:
        name = torch_dtype_to_scalar_type_name(x.dtype) or str(x.dtype)
        raise NotImplementedError(f"\"avg_pool2d\" not implemented for '{name}'")
    # kernel_size/stride/padding must fit int32; oversized values overflow the
    # output-shape computation (issue #145077). CPU raises "integer out of range".
    for seq in (kernel_size, stride, padding):
        for v in _iter(seq) if seq is not None else ():
            if abs(int(v)) > INT32_MAX:
                raise RuntimeError("integer out of range")
    kh, kw = _pair(kernel_size)
    if kh <= 0 or kw <= 0:
        raise RuntimeError("integer out of range")
    if stride is not None and len(_iter(stride)) > 0:
        sh, sw = _pair(stride)
        if sh == 0 or sw == 0:
            raise RuntimeError("stride should not be zero")
        # Negative stride is invalid; upstream (issue #145077) surfaces it as an
        # integer-range error rather than the zero-stride message.
        if sh < 0 or sw < 0:
            raise RuntimeError("integer out of range")
    ph, pw = _pair(padding)
    if ph < 0 or pw < 0:
        bad = ph if ph < 0 else pw
        raise RuntimeError(f"pad must be non-negative, but got pad: {bad}")
    if ph > kh // 2 or pw > kw // 2:
        raise RuntimeError(
            "pad should be at most half of effective kernel size, but got "
            f"pad={ph if ph > kh // 2 else pw}, "
            f"kernel_size={kh if ph > kh // 2 else kw} and dilation=1"
        )
    if divisor_override is not None and divisor_override == 0:
        raise RuntimeError("divisor must be not zero")
    # out= must preserve the input dtype (avg_pool2d does not promote); reject a
    # mismatched out= rather than silently unsafe-casting, matching CPU.
    if out is not None and out.dtype != x.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {x.dtype}, but got {out.dtype} instead"
        )


def _request_channels_last_epilogue(lowering_result, tensor):
    """Set channels-last output strides so the C++ dispatch path returns the
    input's memory format. reduce_window always yields a contiguous result and a
    fused ``.to(memory_format)`` is dropped by lowering, so the layout is applied
    as a fused output epilogue (CPU/CUDA AvgPool2d propagate the input format)."""
    if not lowering_result.output_specs or not isinstance(tensor, torch.Tensor):
        return
    memory_format = suggest_output_memory_format(tensor)
    if memory_format not in (torch.channels_last, torch.channels_last_3d):
        return
    spec = lowering_result.output_specs[0]
    if not isinstance(spec, tuple) or len(spec) < 2:
        return
    target_strides = channels_last_strides(spec[0], memory_format)
    if target_strides is not None:
        lowering_result.output_specs[0] = (spec[0], spec[1], target_strides)


def _set_avg_pool2d_output_strides(lowering_result, logical_args, logical_kwargs):
    """avg_pool2d output follows the input (arg 0)."""
    if logical_args:
        _request_channels_last_epilogue(lowering_result, logical_args[0])


def _set_avg_pool2d_backward_output_strides(lowering_result, logical_args, logical_kwargs):
    """avg_pool2d_backward grad_input follows ``self`` (arg 1)."""
    if len(logical_args) >= 2:
        _request_channels_last_epilogue(lowering_result, logical_args[1])


@register_aten(
    ["aten::avg_pool2d", "aten::avg_pool2d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4, 5, 6),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_avg_pool2d,
    # CPU/CUDA AvgPool2d propagate the input's memory format; reduce_window yields
    # a contiguous result, so request a fused channels-last output epilogue on the
    # C++ dispatch path (a fused ``.to(memory_format)`` would be dropped by
    # lowering). The module memory-format tests assert this layout.
    lowering_result_transform=_set_avg_pool2d_output_strides,
)
def torch_avg_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    out=None,
):
    """Compute two-dimensional average pooling through Torch-MLIR.

    Args:
        x (torch.Tensor): Input tensor.
        kernel_size (int, list, or tuple): Pooling-window dimensions.
        stride (int, list, or tuple, optional): Window stride. Defaults to ``None``;
            uses ``kernel_size``.
        padding (int, list, or tuple, optional): Implicit padding. Defaults to ``0``.
        ceil_mode (bool, optional): Use ceiling for output shape. Defaults to ``False``.
        count_include_pad (bool, optional): Include padding in divisors. Defaults to ``True``.
        divisor_override (int, optional): Replacement divisor. Defaults to ``None``.
        out (torch.Tensor, optional): Destination written by the out variant. Defaults to ``None``.

    Returns:
        tuple: Compute callable that converts ``int64`` through ``float32``, its
            positional arguments, and the output-keyword mapping.

    Raises:
        NotImplementedError: If ``x`` has an unsupported dtype.
        RuntimeError: If pooling dimensions, padding, divisor, or ``out`` dtype are invalid.
    """
    # PyTorch uses kernel_size as the stride when stride is omitted/empty.
    if stride is None or (isinstance(stride, list | tuple) and len(stride) == 0):
        stride = kernel_size

    # CPU/CUDA AvgPool2d propagate the input's memory format to the output; the
    # Neuron lowering always yields a contiguous result, so a channels_last 4D
    # input needs its layout restored afterward. Only for the non-out= path — the
    # out= tensor's layout is the caller's to choose (and restoring would break
    # the out aliasing).
    def compute(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out=None
    ):
        """Compute average pooling for the traced graph.

        Args:
            x (torch.Tensor): Input tensor.
            kernel_size (int, list, or tuple): Pooling-window dimensions.
            stride (int, list, or tuple): Window stride.
            padding (int, list, or tuple): Implicit padding dimensions.
            ceil_mode (bool): Whether to use ceiling for output shape.
            count_include_pad (bool): Whether padding contributes to each divisor.
            divisor_override (int or None): Optional replacement divisor.
            out (torch.Tensor, optional): Unused out-variant placeholder.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Pooled tensor in the lowering's contiguous layout.
        """
        if x.dtype == torch.int64:
            return torch.ops.aten.avg_pool2d(
                x.to(torch.float32),
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            ).to(torch.int64)
        return torch.ops.aten.avg_pool2d(
            x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )

    return (
        compute,
        (x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override),
        {"out": out},
    )


def _validate_avg_pool2d_backward(
    grad_output,
    self,
    kernel_size=None,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    grad_input=None,
):
    """Validate avg_pool2d_backward inputs, matching CPU's error messages."""
    if divisor_override is not None and divisor_override == 0:
        raise RuntimeError("divisor must be not zero")


@register_aten(
    ["aten::avg_pool2d_backward", "aten::avg_pool2d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(2, 3, 4, 5, 6, 7),
    preprocessing=Preprocessing.SHAPE_ONLY,
    validate_static_inputs=_validate_avg_pool2d_backward,
    # grad_input follows self's memory format (CPU/CUDA propagate it); reduce_window
    # yields a contiguous result, so request a fused channels-last output epilogue on
    # the C++ dispatch path. Same pattern as torch_avg_pool2d; the module
    # memory-format tests assert the grad layout.
    lowering_result_transform=_set_avg_pool2d_backward_output_strides,
)
def torch_avg_pool2d_backward(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
    grad_input=None,
):
    """Compute the two-dimensional average-pooling input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the pooled output.
        self (torch.Tensor): Input tensor from the forward pass.
        kernel_size (int, list, or tuple): Pooling-window dimensions.
        stride (int, list, or tuple): Window stride.
        padding (int, list, or tuple): Implicit padding dimensions.
        ceil_mode (bool): Whether forward output shape used ceiling.
        count_include_pad (bool): Whether padding contributed to each divisor.
        divisor_override (int, optional): Replacement divisor. Defaults to ``None``.
        grad_input (torch.Tensor, optional): Destination written by the out variant.
            Defaults to ``None``.

    Returns:
        tuple: Compute callable supporting overlapping windows and excluded padding,
            its positional arguments, and the output-keyword mapping.

    Raises:
        RuntimeError: If ``divisor_override`` is zero.
    """

    # grad_input follows self's memory format (CPU/CUDA behavior). Skip the
    # grad_input= (out=) path, whose layout the caller owns.
    def compute(
        grad_output,
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=None,
        grad_input=None,
    ):
        """Compute the average-pooling gradient for the traced graph.

        Args:
            grad_output (torch.Tensor): Gradient of the pooled output.
            self (torch.Tensor): Input tensor from the forward pass.
            kernel_size (int, list, or tuple): Pooling-window dimensions.
            stride (int, list, or tuple): Window stride.
            padding (int, list, or tuple): Implicit padding dimensions.
            ceil_mode (bool): Whether forward output shape used ceiling.
            count_include_pad (bool): Whether padding contributed to each divisor.
            divisor_override (int, optional): Replacement divisor. Defaults to ``None``.
            grad_input (torch.Tensor, optional): Unused placeholder. Defaults to ``None``.

        Returns:
            torch.Tensor: Gradient with respect to ``self``.
        """
        return torch.ops.aten.avg_pool2d_backward(
            grad_output,
            self,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    return (
        compute,
        (
            grad_output,
            self,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        ),
        {"grad_input": grad_input},
    )
