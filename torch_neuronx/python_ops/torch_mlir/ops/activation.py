"""Activation function operations."""

import torch

from ..operation_registry import register_aten
from ..output_layout import channels_last_strides, suggest_output_memory_format
from ..validation import aten_scalar_type_name
from .unary import _validate_float_promoting_out


def _preserve_channels_last(input_index=0):
    """Mirror a channels-last input layout, which the eager kernel would drop.

    The kernel returns a contiguous output, but aten keeps the input layout. Take
    the format from the input, not the traced output. Skip a user out=/grad_input.
    """

    def transform(lowering_result, logical_args, logical_kwargs):
        """Preserve a channels-last stride specification when applicable.

        Args:
            lowering_result (LoweringResult): Result whose first output specification may change.
            logical_args (tuple): Logical positional operation arguments.
            logical_kwargs (dict): Logical keyword operation arguments.
        """
        if not lowering_result.output_specs:
            return
        if any(
            isinstance(value, torch.Tensor) and name in ("out", "grad_input")
            for name, value in logical_kwargs.items()
        ):
            return
        if len(logical_args) <= input_index:
            return
        inp = logical_args[input_index]
        if not isinstance(inp, torch.Tensor):
            return
        memory_format = suggest_output_memory_format(inp)
        spec = lowering_result.output_specs[0]
        if not isinstance(spec, tuple) or len(spec) < 2:
            return
        shape = spec[0]
        target_strides = channels_last_strides(shape, memory_format)
        if target_strides is None:
            return
        lowering_result.output_specs[0] = (shape, spec[1], target_strides)

    return transform


@register_aten(["aten::relu", "aten::relu.out", "aten::relu_"])
def torch_relu(x, out=None):
    """Apply the rectified linear unit activation.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise rectified values.

    Note:
        Registered ``aten::relu_`` dispatch aliases the result to ``x``.
    """
    return torch.relu(x)


@register_aten(["aten::gelu", "aten::gelu_", "aten::gelu.out"], static_argnames=("approximate",))
def torch_gelu(x, approximate="none", out=None):
    """Apply the Gaussian error linear unit activation.

    Args:
        x (torch.Tensor): Input tensor.
        approximate (str, optional): Approximation mode. Defaults to ``"none"``.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise GELU values.

    Note:
        Registered ``aten::gelu_`` dispatch aliases the result to ``x``.
    """
    return torch.nn.functional.gelu(x, approximate=approximate)


@register_aten(["aten::silu", "aten::silu.out"])
def torch_silu(x, out=None):
    """Apply the sigmoid linear unit activation.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise SiLU values.
    """
    return torch.nn.functional.silu(x)


@register_aten(
    ["aten::sigmoid", "aten::sigmoid.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sigmoid(x, out=None):
    """Apply the sigmoid activation.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise sigmoid values.

    Raises:
        RuntimeError: If floating-point ``x`` is paired with a non-floating ``out`` tensor.
    """
    return torch.sigmoid(x)


@register_aten(
    ["aten::tanh", "aten::tanh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_tanh(self, out=None):
    """Apply the hyperbolic tangent activation.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise hyperbolic tangent values.

    Raises:
        RuntimeError: If floating-point ``self`` is paired with a non-floating ``out`` tensor.
    """
    return torch.tanh(self)


@register_aten(["aten::softplus", "aten::softplus.out"], static_argnums=(1, 2))
def torch_softplus(self, beta=1, threshold=20, out=None):
    """Apply the softplus activation.

    Args:
        self (torch.Tensor): Input tensor.
        beta (float, optional): Smoothness parameter. Defaults to ``1``.
        threshold (float, optional): Linear-region threshold. Defaults to ``20``.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise softplus values.
    """
    return torch.nn.functional.softplus(self, beta=beta, threshold=threshold)


@register_aten(
    ["aten::sigmoid_backward", "aten::sigmoid_backward.grad_input"], output_params=("grad_input",)
)
def torch_sigmoid_backward(grad_output, output, grad_input=None):
    """Compute the sigmoid input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        output (torch.Tensor): Output produced by the sigmoid activation.
        grad_input (torch.Tensor, optional): Gradient output buffer accepted but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to the sigmoid input.
    """
    return torch.ops.aten.sigmoid_backward.default(grad_output, output)


@register_aten(
    ["aten::silu_backward", "aten::silu_backward.grad_input"], output_params=("grad_input",)
)
def torch_silu_backward(grad_output, x, grad_input=None):
    """Compute the SiLU input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        x (torch.Tensor): Input to the SiLU activation.
        grad_input (torch.Tensor, optional): Gradient output buffer accepted but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``x``.
    """
    return torch.ops.aten.silu_backward.default(grad_output, x)


@register_aten(
    ["aten::tanh_backward", "aten::tanh_backward.grad_input"], output_params=("grad_input",)
)
def torch_tanh_backward(grad_output, output, grad_input=None):
    """Compute the hyperbolic tangent input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        output (torch.Tensor): Output produced by the hyperbolic tangent activation.
        grad_input (torch.Tensor, optional): Gradient output buffer accepted but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to the activation input.
    """
    return torch.ops.aten.tanh_backward(grad_output, output)


@register_aten(
    ["aten::gelu_backward", "aten::gelu_backward.grad_input"],
    static_argnames=("approximate",),
    output_params=("grad_input",),
)
def torch_gelu_backward(grad_output, self, approximate="none", grad_input=None):
    """Compute the GELU input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        self (torch.Tensor): Input to the GELU activation.
        approximate (str, optional): Approximation mode. Defaults to ``"none"``.
        grad_input (torch.Tensor, optional): Gradient output buffer accepted but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.
    """
    return torch.ops.aten.gelu_backward(grad_output, self, approximate=approximate)


@register_aten(
    ["aten::softplus_backward", "aten::softplus_backward.grad_input"],
    static_argnums=(2, 3),
    output_params=("grad_input",),
)
def torch_softplus_backward(out_grad, x, beta, threshold, grad_input=None):
    """Compute the softplus input gradient.

    Args:
        out_grad (torch.Tensor): Gradient of the loss with respect to the output.
        x (torch.Tensor): Input to the softplus activation.
        beta (float): Smoothness parameter.
        threshold (float): Linear-region threshold.
        grad_input (torch.Tensor, optional): Gradient output buffer accepted but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``x``.
    """
    return torch.ops.aten.softplus_backward(out_grad, x, beta=beta, threshold=threshold)


@register_aten(
    ["aten::log_sigmoid_forward", "aten::log_sigmoid_forward.output"],
    output_params=("output", "buffer"),
)
def torch_log_sigmoid_forward(self, output=None, buffer=None):
    """Compute the log-sigmoid output and auxiliary buffer.

    Args:
        self (torch.Tensor): Input tensor.
        output (torch.Tensor, optional): Output buffer accepted but not forwarded by this
            implementation. Defaults to ``None``.
        buffer (torch.Tensor, optional): Auxiliary buffer accepted but not forwarded by this
            implementation. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Log-sigmoid values and backward buffer.
    """
    return torch.ops.aten.log_sigmoid_forward(self)


@register_aten(
    ["aten::log_sigmoid_backward", "aten::log_sigmoid_backward.grad_input"],
    output_params=("grad_input",),
)
def torch_log_sigmoid_backward(grad_output, self, buffer, grad_input=None):
    """Compute the log-sigmoid input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        self (torch.Tensor): Input to the log-sigmoid activation.
        buffer (torch.Tensor): Auxiliary buffer produced by the forward operation.
        grad_input (torch.Tensor, optional): Gradient destination populated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``; registered grad-input dispatch aliases
            ``grad_input`` when provided.
    """
    return torch.ops.aten.log_sigmoid_backward(grad_output, self, buffer)


@register_aten(
    ["aten::elu", "aten::elu.out"],
    static_argnums=(1, 2, 3),
    lowering_result_transform=_preserve_channels_last(0),
)
@register_aten(["aten::elu_"], static_argnums=(1, 2, 3))
def torch_elu(self, alpha=1, scale=1, input_scale=1, out=None):
    """Apply the exponential linear unit activation.

    Args:
        self (torch.Tensor): Input tensor.
        alpha (float, optional): Negative-region multiplier. Defaults to ``1``.
        scale (float, optional): Output multiplier. Defaults to ``1``.
        input_scale (float, optional): Input multiplier. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise ELU values; registered out dispatch aliases ``out`` when
            provided.

    Note:
        Registered ``aten::elu_`` dispatch aliases the result to ``self``.
    """
    return torch.ops.aten.elu(self, alpha, scale, input_scale)


def _validate_elu_backward(
    grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input=None
):
    # A negative alpha with is_result cannot recover the gradient from the output.
    if is_result and alpha < 0:
        raise RuntimeError(
            "In-place elu backward calculation is triggered with a negative slope which is not "
            "supported. This is caused by calling in-place forward function with a negative "
            "slope, please call out-of-place version instead."
        )


@register_aten(
    ["aten::elu_backward", "aten::elu_backward.grad_input"],
    static_argnums=(1, 2, 3, 4),
    output_params=("grad_input",),
    validate_static_inputs=_validate_elu_backward,
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_elu_backward(
    grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input=None
):
    """Compute the exponential linear unit input gradient.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the output.
        alpha (float): Negative-region multiplier.
        scale (float): Output multiplier.
        input_scale (float): Input multiplier.
        is_result (bool): Whether ``self_or_result`` contains the forward result.
        self_or_result (torch.Tensor): Forward input or result selected by ``is_result``.
        grad_input (torch.Tensor, optional): Gradient destination populated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: ELU input gradient; registered grad-input dispatch aliases ``grad_input``
            when provided.

    Raises:
        RuntimeError: If ``is_result`` is true and ``alpha`` is negative.
    """
    return torch.ops.aten.elu_backward.default(
        grad_output, alpha, scale, input_scale, is_result, self_or_result
    )


# selu and selu_ are CompositeImplicitAutograd and decompose to elu, so they run
# on Neuron with no dedicated kernel.


def _validate_hardshrink_out(self, lambd=0.5, out=None):
    """Reject a non-float input, which aten rejects but the decomposition runs."""
    if isinstance(self, torch.Tensor) and not self.is_floating_point():
        raise RuntimeError(
            f"\"hardshrink\" not implemented for '{aten_scalar_type_name(self.dtype)}'"
        )
    _validate_float_promoting_out(self, out=out)


@register_aten(
    ["aten::hardshrink", "aten::hardshrink.out"],
    static_argnums=(1,),
    static_argnames=("lambd",),
    validate_static_inputs=_validate_hardshrink_out,
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_hardshrink(self, lambd=0.5, out=None):
    """Apply the hard-shrinkage activation.

    Args:
        self (torch.Tensor): Floating-point input tensor.
        lambd (float, optional): Shrinkage threshold. Defaults to ``0.5``.
        out (torch.Tensor, optional): Output buffer accepted but not forwarded. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Values outside the threshold interval and zeros within it.

    Raises:
        RuntimeError: If ``self`` is not floating point or ``out`` cannot hold the
            floating-point result.
    """
    return torch.ops.aten.hardshrink(self, lambd)


@register_aten(
    ["aten::hardshrink_backward", "aten::hardshrink_backward.grad_input"],
    static_argnums=(2,),
    static_argnames=("lambd",),
    output_params=("grad_input",),
    lowering_result_transform=_preserve_channels_last(1),
)
def torch_hardshrink_backward(grad_out, self, lambd, grad_input=None):
    """Compute the hard-shrinkage input gradient.

    Args:
        grad_out (torch.Tensor): Gradient with respect to the output.
        self (torch.Tensor): Forward input tensor.
        lambd (float): Shrinkage threshold.
        grad_input (torch.Tensor, optional): Gradient destination populated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``; registered grad-input dispatch aliases
            ``grad_input`` when provided.
    """
    return torch.ops.aten.hardshrink_backward.default(grad_out, self, lambd)


# relu6 is CompositeImplicitAutograd and decomposes to hardtanh(0, 6), so it runs
# on Neuron with no dedicated kernel.
@register_aten(
    ["aten::hardtanh", "aten::hardtanh.out"],
    static_argnums=(1, 2),
    lowering_result_transform=_preserve_channels_last(0),
)
@register_aten(["aten::hardtanh_"], static_argnums=(1, 2))
def torch_hardtanh(self, min_val=-1, max_val=1, out=None):
    """Clamp input values to the hard-tanh interval.

    Args:
        self (torch.Tensor): Input tensor.
        min_val (float, optional): Lower bound. Defaults to ``-1``.
        max_val (float, optional): Upper bound. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Values clamped between ``min_val`` and ``max_val``; registered out dispatch
            aliases ``out`` when provided.

    Note:
        Registered ``aten::hardtanh_`` dispatch aliases the result to ``self``.
    """
    return torch.ops.aten.hardtanh(self, min_val, max_val)


@register_aten(
    ["aten::hardtanh_backward", "aten::hardtanh_backward.grad_input"],
    static_argnums=(2, 3),
    output_params=("grad_input",),
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_hardtanh_backward(grad_output, self, min_val, max_val, grad_input=None):
    """Compute the hard-tanh input gradient.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the output.
        self (torch.Tensor): Forward input tensor.
        min_val (float): Lower bound used by the forward operation.
        max_val (float): Upper bound used by the forward operation.
        grad_input (torch.Tensor, optional): Gradient destination populated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``; registered grad-input dispatch aliases
            ``grad_input`` when provided.
    """
    return torch.ops.aten.hardtanh_backward(grad_output, self, min_val, max_val)


def _validate_mish_float_dtype(self, *_operands, **_kwargs):
    """Reject non-floating inputs; ATen implements mish for float dtypes only."""
    if isinstance(self, torch.Tensor) and not self.is_floating_point():
        raise RuntimeError(f"\"mish\" not implemented for '{aten_scalar_type_name(self.dtype)}'")


@register_aten(
    ["aten::mish", "aten::mish.out"],
    validate_static_inputs=_validate_mish_float_dtype,
    lowering_result_transform=_preserve_channels_last(0),
)
@register_aten(["aten::mish_"], validate_static_inputs=_validate_mish_float_dtype)
def torch_mish(self, out=None):
    """Apply the Mish activation.

    Args:
        self (torch.Tensor): Floating-point input tensor.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise Mish values; registered out dispatch aliases ``out`` when
            provided.

    Raises:
        RuntimeError: If ``self`` is not floating point.

    Note:
        Registered ``aten::mish_`` dispatch aliases the result to ``self``.
    """
    return torch.nn.functional.mish(self)


@register_aten(["aten::mish_backward"], validate_static_inputs=_validate_mish_float_dtype)
def torch_mish_backward(grad_output, self):
    """Compute the Mish input gradient.

    Args:
        grad_output (torch.Tensor): Floating-point gradient with respect to the output.
        self (torch.Tensor): Floating-point forward input tensor.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.

    Raises:
        RuntimeError: If ``grad_output`` or ``self`` is not floating point.
    """
    return torch.ops.aten.mish_backward.default(grad_output, self)


def _validate_hardswish_float(self, *_args, **_kwargs):
    """Reject a non-float input, which hardswish does not support."""
    if isinstance(self, torch.Tensor) and not self.is_floating_point():
        raise RuntimeError(
            f"\"hardswish\" not implemented for '{aten_scalar_type_name(self.dtype)}'"
        )


@register_aten(
    ["aten::hardswish", "aten::hardswish.out"],
    validate_static_inputs=_validate_hardswish_float,
    lowering_result_transform=_preserve_channels_last(0),
)
@register_aten(
    ["aten::hardswish_"],
    validate_static_inputs=_validate_hardswish_float,
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_hardswish(self, out=None):
    """Apply the hard-swish activation.

    Args:
        self (torch.Tensor): Floating-point input tensor.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise hard-swish values; registered out dispatch aliases ``out`` when
            provided.

    Raises:
        RuntimeError: If ``self`` is not floating point.

    Note:
        Registered ``aten::hardswish_`` dispatch aliases the result to ``self``.
    """
    return torch.ops.aten.hardswish(self)


@register_aten(
    ["aten::hardswish_backward", "aten::hardswish_backward.out"],
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_hardswish_backward(grad_output, self, out=None):
    """Compute the hard-swish input gradient.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the output.
        self (torch.Tensor): Forward input tensor.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``; registered out dispatch aliases ``out``
            when provided.
    """
    return torch.ops.aten.hardswish_backward(grad_output, self)


def _validate_glu(self, dim=-1, out=None):
    """Guard glu like aten: float input, no 0-d tensor, even split dimension."""
    if not self.is_floating_point():
        raise RuntimeError(f"\"glu\" not implemented for '{aten_scalar_type_name(self.dtype)}'")
    if self.dim() == 0:
        raise RuntimeError("glu does not support 0-dimensional tensors")
    wrap_dim = dim if dim >= 0 else dim + self.dim()
    size = self.size(wrap_dim)
    if size % 2 != 0:
        raise RuntimeError(
            f"Halving dimension must be even, but dimension {wrap_dim} is size {size}"
        )
    _validate_float_promoting_out(self, out=out)


@register_aten(
    ["aten::glu", "aten::glu.out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_glu,
    lowering_result_transform=_preserve_channels_last(0),
)
def torch_glu(self, dim=-1, out=None):
    """Apply the gated linear unit activation.

    Args:
        self (torch.Tensor): Floating-point input tensor with an even split dimension.
        dim (int, optional): Dimension to split. Defaults to ``-1``.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: First half of ``self`` multiplied by the sigmoid of its second half;
            registered out dispatch aliases ``out`` when provided.

    Raises:
        RuntimeError: If the input is scalar or non-floating, the split dimension has odd size,
            or ``out`` cannot hold the floating-point result.
        IndexError: If ``dim`` is outside the input rank.
    """
    return torch.ops.aten.glu(self, dim)


def _validate_glu_backward(grad_output, self, dim, grad_input=None):
    # The glu_backward decomposition skips aten's even-dimension check; re-raise it.
    wrap_dim = dim if dim >= 0 else dim + self.dim()
    size = self.size(wrap_dim)
    if size % 2 != 0:
        raise RuntimeError(
            f"Halving dimension must be even, but dimension {wrap_dim} is size {size}"
        )


@register_aten(
    ["aten::glu_backward", "aten::glu_backward.grad_input"],
    static_argnums=(2,),
    output_params=("grad_input",),
    validate_static_inputs=_validate_glu_backward,
)
def torch_glu_backward(grad_output, self, dim, grad_input=None):
    """Compute the gated linear unit input gradient.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the output.
        self (torch.Tensor): Forward input tensor.
        dim (int): Dimension split by the forward operation.
        grad_input (torch.Tensor, optional): Output buffer accepted but not forwarded. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.

    Raises:
        RuntimeError: If the split dimension has odd size.
    """
    return torch.ops.aten.glu_backward.default(grad_output, self, dim)


@register_aten(["aten::celu", "aten::celu.out"], static_argnames=("alpha",))
def torch_celu(self, alpha=1.0, out=None):
    """Apply the continuously differentiable exponential linear unit activation.

    Args:
        self (torch.Tensor): Input tensor.
        alpha (float, optional): CELU alpha value. Defaults to ``1.0``.
        out (torch.Tensor, optional): Destination populated by registration but not forwarded by
            this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise CELU values; registered out dispatch aliases ``out`` when
            provided.

    Raises:
        RuntimeError: If ``alpha`` is zero.
    """
    return torch.ops.aten.celu(self, alpha)


@register_aten(["aten::leaky_relu", "aten::leaky_relu.out"], static_argnames=("negative_slope",))
def torch_leaky_relu(self, negative_slope=0.01, out=None):
    """Apply the leaky rectified linear unit activation.

    Args:
        self (torch.Tensor): Input tensor.
        negative_slope (float, optional): Negative-region slope. Defaults to ``0.01``.
        out (torch.Tensor, optional): Output tensor accepted but not forwarded by
        this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise leaky ReLU values.
    """
    return torch.ops.aten.leaky_relu(self, negative_slope)


@register_aten(["aten::leaky_relu_"], static_argnames=("negative_slope",))
def torch_leaky_relu_(self, negative_slope=0.01):
    """Apply leaky ReLU for the registered in-place operation.

    Args:
        self (torch.Tensor): Input tensor.
        negative_slope (float, optional): Negative-region slope. Defaults to ``0.01``.

    Returns:
        torch.Tensor: Leaky ReLU values.

    Note:
        The Python body computes out of place; registered dispatch aliases the result to
        ``self`` for ``aten::leaky_relu_``.
    """
    return torch.ops.aten.leaky_relu(self, negative_slope)


@register_aten(["aten::hardsigmoid", "aten::hardsigmoid.out"])
def torch_hardsigmoid(self, out=None):
    """Apply the hard sigmoid activation.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor that registration populates. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Elementwise hard sigmoid values; aliases ``out`` when provided.
    """
    return torch.ops.aten.hardsigmoid(self)


@register_aten(["aten::hardsigmoid_"])
def torch_hardsigmoid_(self):
    """Apply hard sigmoid for the registered in-place operation.

    Args:
        self (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Hard sigmoid values.

    Note:
        The Python body computes out of place; registered dispatch aliases the result to
        ``self`` for ``aten::hardsigmoid_``.
    """
    return torch.ops.aten.hardsigmoid(self)


@register_aten(
    ["aten::hardsigmoid_backward", "aten::hardsigmoid_backward.grad_input"],
    output_params=("grad_input",),
)
def torch_hardsigmoid_backward(grad_output, self, grad_input=None):
    """Compute the hard sigmoid input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        self (torch.Tensor): Input to the hard sigmoid activation.
        grad_input (torch.Tensor, optional): Gradient destination that registration populates.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``; aliases ``grad_input`` when provided.
    """
    return torch.ops.aten.hardsigmoid_backward.default(grad_output, self)
