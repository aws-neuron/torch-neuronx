"""Eager ``register_aten`` lowerings for the upsample/interpolate family."""

import torch

from ..operation_registry import register_aten
from ..validation import aten_scalar_type_name

aten = torch.ops.aten

# --- output layout ---


def _make_output_stride_transform(aten_op):
    """Tag the output spec with aten's own output strides, so a channels_last
    input is not silently returned contiguous.

    Reads the strides from the real ATen op on meta tensors, not from
    ``suggest_output_memory_format``: aten does not just propagate the input
    layout (bicubic_backward returns contiguous where the other 2-D backwards
    return channels_last). Leaves the spec alone when aten's output is contiguous.
    """

    def _transform(lowering_result, logical_args, logical_kwargs):
        specs = lowering_result.output_specs
        if not logical_args or not specs:
            return
        tensor = logical_args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.is_contiguous():
            return
        # Drop the positionally-passed out=/grad_input= tensor; every other arg is a
        # size list, bool or scale, so "not a tensor" isolates the static args.
        static_args = [a for a in logical_args[1:] if not isinstance(a, torch.Tensor)]
        static_kwargs = {
            k: v for k, v in (logical_kwargs or {}).items() if not isinstance(v, torch.Tensor)
        }
        meta = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="meta")
        reference = aten_op(meta, *static_args, **static_kwargs)
        if reference.is_contiguous():
            return
        spec = specs[0]
        if not isinstance(spec, tuple) or len(spec) < 2:
            return
        # A shape mismatch means the meta call did not model this overload; bail.
        if tuple(spec[0]) != tuple(reference.shape):
            return
        specs[0] = (spec[0], spec[1], tuple(int(s) for s in reference.stride()))

    return _transform


# --- dtype guards ---

_LINEAR_INTERP_KERNEL = "compute_indices_weights_linear"


def _make_validator(kernel_name, *, allow_uint8):
    """Reject the dtypes aten's own kernels do not dispatch over, so
    ``test_ops.py::test_dtypes`` matches the OpInfo table. Forwards take floating
    point plus uint8 (linear1d floating-only); every backward is floating-only.
    """

    def _validate(tensor, *_args, **_kwargs):
        if tensor.is_floating_point():
            return
        if allow_uint8 and tensor.dtype == torch.uint8:
            return
        raise NotImplementedError(
            f"\"{kernel_name}\" not implemented for '{aten_scalar_type_name(tensor.dtype)}'"
        )

    return _validate


@register_aten(
    ["aten::upsample_nearest1d", "aten::upsample_nearest1d.out"],
    output_params=("out",),
    static_argnums=(1, 2),
    validate_static_inputs=_make_validator("upsample_nearest1d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_nearest1d),
)
def torch_upsample_nearest1d(self, output_size, scales=None, out=None):
    return aten.upsample_nearest1d.default(self, output_size, scales)


@register_aten(
    ["aten::_upsample_nearest_exact1d", "aten::_upsample_nearest_exact1d.out"],
    output_params=("out",),
    static_argnums=(1, 2),
    validate_static_inputs=_make_validator("_upsample_nearest_exact1d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten._upsample_nearest_exact1d),
)
def torch_upsample_nearest_exact1d(self, output_size, scales=None, out=None):
    return aten._upsample_nearest_exact1d.default(self, output_size, scales)


@register_aten(
    ["aten::upsample_nearest2d", "aten::upsample_nearest2d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3),
    validate_static_inputs=_make_validator("upsample_nearest2d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_nearest2d),
)
def torch_upsample_nearest2d(self, output_size, scales_h=None, scales_w=None, out=None):
    return aten.upsample_nearest2d.default(self, output_size, scales_h, scales_w)


@register_aten(
    ["aten::_upsample_nearest_exact2d", "aten::_upsample_nearest_exact2d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3),
    validate_static_inputs=_make_validator("_upsample_nearest_exact2d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten._upsample_nearest_exact2d),
)
def torch_upsample_nearest_exact2d(self, output_size, scales_h=None, scales_w=None, out=None):
    return aten._upsample_nearest_exact2d.default(self, output_size, scales_h, scales_w)


@register_aten(
    ["aten::upsample_linear1d", "aten::upsample_linear1d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3),
    validate_static_inputs=_make_validator(_LINEAR_INTERP_KERNEL, allow_uint8=False),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_linear1d),
)
def torch_upsample_linear1d(self, output_size, align_corners, scales=None, out=None):
    return aten.upsample_linear1d.default(self, output_size, align_corners, scales)


@register_aten(
    ["aten::upsample_bilinear2d", "aten::upsample_bilinear2d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("upsample_bilinear2d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_bilinear2d),
)
def torch_upsample_bilinear2d(
    self, output_size, align_corners, scales_h=None, scales_w=None, out=None
):
    return aten.upsample_bilinear2d.default(self, output_size, align_corners, scales_h, scales_w)


@register_aten(
    ["aten::upsample_bicubic2d", "aten::upsample_bicubic2d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("upsample_bicubic2d", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_bicubic2d),
)
def torch_upsample_bicubic2d(
    self, output_size, align_corners, scales_h=None, scales_w=None, out=None
):
    return aten.upsample_bicubic2d.default(self, output_size, align_corners, scales_h, scales_w)


@register_aten(
    ["aten::upsample_nearest1d_backward", "aten::upsample_nearest1d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3),
    validate_static_inputs=_make_validator("upsample_nearest1d_backward", allow_uint8=False),
)
def torch_upsample_nearest1d_backward(
    grad_output, output_size, input_size, scales=None, grad_input=None
):
    return aten.upsample_nearest1d_backward.default(grad_output, output_size, input_size, scales)


@register_aten(
    [
        "aten::_upsample_nearest_exact1d_backward",
        "aten::_upsample_nearest_exact1d_backward.grad_input",
    ],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3),
    validate_static_inputs=_make_validator("_upsample_nearest_exact1d_backward", allow_uint8=False),
)
def torch_upsample_nearest_exact1d_backward(
    grad_output, output_size, input_size, scales=None, grad_input=None
):
    return aten._upsample_nearest_exact1d_backward.default(
        grad_output, output_size, input_size, scales
    )


@register_aten(
    ["aten::upsample_nearest2d_backward", "aten::upsample_nearest2d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("upsample_nearest2d_backward", allow_uint8=False),
)
def torch_upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None, grad_input=None
):
    return aten.upsample_nearest2d_backward.default(
        grad_output, output_size, input_size, scales_h, scales_w
    )


@register_aten(
    [
        "aten::_upsample_nearest_exact2d_backward",
        "aten::_upsample_nearest_exact2d_backward.grad_input",
    ],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("_upsample_nearest_exact2d_backward", allow_uint8=False),
)
def torch_upsample_nearest_exact2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None, grad_input=None
):
    return aten._upsample_nearest_exact2d_backward.default(
        grad_output, output_size, input_size, scales_h, scales_w
    )


@register_aten(
    ["aten::upsample_linear1d_backward", "aten::upsample_linear1d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("upsample_linear1d_backward", allow_uint8=False),
)
def torch_upsample_linear1d_backward(
    grad_output, output_size, input_size, align_corners, scales=None, grad_input=None
):
    return aten.upsample_linear1d_backward.default(
        grad_output, output_size, input_size, align_corners, scales
    )


@register_aten(
    ["aten::upsample_bilinear2d_backward", "aten::upsample_bilinear2d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4, 5),
    validate_static_inputs=_make_validator("upsample_bilinear2d_backward", allow_uint8=False),
)
def torch_upsample_bilinear2d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    return aten.upsample_bilinear2d_backward.default(
        grad_output, output_size, input_size, align_corners, scales_h, scales_w
    )


@register_aten(
    ["aten::upsample_bicubic2d_backward", "aten::upsample_bicubic2d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4, 5),
    validate_static_inputs=_make_validator("upsample_bicubic2d_backward", allow_uint8=False),
)
def torch_upsample_bicubic2d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    return aten.upsample_bicubic2d_backward.default(
        grad_output, output_size, input_size, align_corners, scales_h, scales_w
    )
