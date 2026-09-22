"""Eager ``register_aten`` lowerings for the upsample/interpolate and
grid-sample families."""

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
    ["aten::upsample_trilinear3d", "aten::upsample_trilinear3d.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4, 5),
    validate_static_inputs=_make_validator(_LINEAR_INTERP_KERNEL, allow_uint8=False),
    lowering_result_transform=_make_output_stride_transform(aten.upsample_trilinear3d),
)
def torch_upsample_trilinear3d(
    self, output_size, align_corners, scales_d=None, scales_h=None, scales_w=None, out=None
):
    return aten.upsample_trilinear3d.default(
        self, output_size, align_corners, scales_d, scales_h, scales_w
    )


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


@register_aten(
    ["aten::upsample_trilinear3d_backward", "aten::upsample_trilinear3d_backward.grad_input"],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4, 5, 6),
    validate_static_inputs=_make_validator("upsample_trilinear3d_backward", allow_uint8=False),
)
def torch_upsample_trilinear3d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    return aten.upsample_trilinear3d_backward.default(
        grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w
    )


# Anti-aliased bilinear. The bodies dispatch to .default; the decomposition in
# decompositions.py rewrites it under make_fx. Allowing bf16/fp16 (not only
# fp32/fp64/uint8 like aten's CPU kernel) is deliberate: the matmul accumulates
# in fp32, so Neuron gains the path the CPU kernel lacks.


@register_aten(
    ["aten::_upsample_bilinear2d_aa", "aten::_upsample_bilinear2d_aa.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_make_validator("_upsample_bilinear2d_aa", allow_uint8=True),
    lowering_result_transform=_make_output_stride_transform(aten._upsample_bilinear2d_aa),
)
def torch_upsample_bilinear2d_aa(
    self, output_size, align_corners, scales_h=None, scales_w=None, out=None
):
    return aten._upsample_bilinear2d_aa.default(
        self, output_size, align_corners, scales_h, scales_w
    )


@register_aten(
    [
        "aten::_upsample_bilinear2d_aa_backward",
        "aten::_upsample_bilinear2d_aa_backward.grad_input",
    ],
    output_params=("grad_input",),
    static_argnums=(1, 2, 3, 4, 5),
    validate_static_inputs=_make_validator("_upsample_bilinear2d_aa_backward", allow_uint8=False),
)
def torch_upsample_bilinear2d_aa_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scales_h=None,
    scales_w=None,
    grad_input=None,
):
    return aten._upsample_bilinear2d_aa_backward.default(
        grad_output, output_size, input_size, align_corners, scales_h, scales_w
    )


# --- grid sample ---


def _check_grid_sampler_common(input, grid):
    """Reproduce aten's check_grid_sampler_common with the exact CPU wording.

    The Neuron dispatch path skips PyTorch structured validation, so a bad shape
    reaches the body and the meta kernel raises the wrong error type. Check here
    first, in the CPU order: batch size, grid last dimension, non-empty spatial dims.
    """
    if input.size(0) != grid.size(0):
        raise RuntimeError(
            "grid_sampler(): expected grid and input to have same batch size, but got "
            f"input with sizes {list(input.shape)} and grid with sizes {list(grid.shape)}"
        )
    if grid.size(-1) != input.dim() - 2:
        raise RuntimeError(
            f"grid_sampler(): expected grid to have size {input.dim() - 2} in last "
            f"dimension, but got grid with sizes {list(grid.shape)}"
        )
    for i in range(2, input.dim()):
        if input.size(i) <= 0:
            raise RuntimeError(
                "grid_sampler(): expected input to have non-empty spatial dimensions, "
                f"but input has sizes {list(input.shape)} with dimension {i} being empty"
            )


def _check_grid_sampler_shapes(input, grid, expected_dim):
    """Reproduce aten's common checks, then the dimension check, with the CPU messages."""
    _check_grid_sampler_common(input, grid)
    if input.dim() != expected_dim or input.dim() != grid.dim():
        raise RuntimeError(
            f"grid_sampler(): expected {expected_dim}D input and grid with same number "
            f"of dimensions, but got input with sizes {list(input.shape)} and grid with "
            f"sizes {list(grid.shape)}"
        )


def _check_grid_sampler_backward_grad_output(input, grid, grad_output):
    """Reproduce aten's check_grid_sampler_backward for the grad_output shape.

    A mismatched grad_output shape causes out-of-bounds reads in the kernel. The
    Neuron dispatch path skips this structured check. Raise ValueError here, with
    the CPU wording, to match the aten TORCH_CHECK_VALUE error type.
    """
    expected = [input.size(0), input.size(1)]
    for i in range(1, grid.dim() - 1):
        expected.append(grid.size(i))
    if list(grad_output.shape) != expected:
        raise ValueError(
            "grid_sampler(): expected grad_output to have sizes "
            f"{expected} but got grad_output with sizes {list(grad_output.shape)}"
        )


def _make_grid_sampler_validator(kernel_name, expected_dim):
    """Validate the grid_sampler dtype, then shapes, with the CPU messages."""
    dtype_validate = _make_validator(kernel_name, allow_uint8=False)

    def _validate(input, grid, *_args, **_kwargs):
        dtype_validate(input)
        _check_grid_sampler_shapes(input, grid, expected_dim)

    return _validate


def _make_grid_sampler_backward_validator(kernel_name, expected_dim):
    """Validate the grid_sampler backward dtype, shapes, and grad_output shape."""
    dtype_validate = _make_validator(kernel_name, allow_uint8=False)

    def _validate(grad_output, input, grid, *_args, **_kwargs):
        dtype_validate(input)
        _check_grid_sampler_shapes(input, grid, expected_dim)
        _check_grid_sampler_backward_grad_output(input, grid, grad_output)

    return _validate


@register_aten(
    ["aten::grid_sampler_2d", "aten::grid_sampler_2d.out"],
    output_params=("out",),
    static_argnums=(2, 3, 4),
    validate_static_inputs=_make_grid_sampler_validator("grid_sampler_2d", 4),
)
def torch_grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners, out=None):
    return aten.grid_sampler_2d.default(
        input, grid, interpolation_mode, padding_mode, align_corners
    )


@register_aten(
    ["aten::grid_sampler_2d_backward", "aten::grid_sampler_2d_backward.out"],
    output_params=("out0", "out1"),
    static_argnums=(3, 4, 5, 6),
    validate_static_inputs=_make_grid_sampler_backward_validator("grid_sampler_2d_backward", 4),
)
def torch_grid_sampler_2d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
    out0=None,
    out1=None,
):
    return aten.grid_sampler_2d_backward.default(
        grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask
    )


@register_aten(
    ["aten::grid_sampler_3d", "aten::grid_sampler_3d.out"],
    output_params=("out",),
    static_argnums=(2, 3, 4),
    validate_static_inputs=_make_grid_sampler_validator("grid_sampler_3d", 5),
)
def torch_grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners, out=None):
    return aten.grid_sampler_3d.default(
        input, grid, interpolation_mode, padding_mode, align_corners
    )


@register_aten(
    ["aten::grid_sampler_3d_backward", "aten::grid_sampler_3d_backward.out"],
    output_params=("out0", "out1"),
    static_argnums=(3, 4, 5, 6),
    validate_static_inputs=_make_grid_sampler_backward_validator("grid_sampler_3d_backward", 5),
)
def torch_grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
    out0=None,
    out1=None,
):
    return aten.grid_sampler_3d_backward.default(
        grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask
    )
