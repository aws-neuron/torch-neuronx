"""Shared upsample/interpolate maps and the tensor ops that apply them.

Imported by both the eager (``@register_aten``) and compile
(``@register_decomposition``) paths so the two cannot drift.
"""

import struct

import torch

# --- scale resolution ---


def as_float32(value):
    """Round a value through an IEEE 754 single-precision representation.

    Args:
        value (float): Value to represent as a 32-bit float.

    Returns:
        float: Value after single-precision rounding.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def scales_value(in_size, out_size, scale):
    """Resolve the inverse scale used by interpolation index arithmetic.

    Args:
        in_size (int): Input extent along the interpolated dimension.
        out_size (int): Output extent along the interpolated dimension.
        scale (float or None): Explicit forward scale, or ``None`` to infer it.

    Returns:
        float: Reciprocal explicit scale when positive, otherwise ``in_size / out_size``.
    """
    if scale is not None and scale > 0:
        return 1.0 / scale
    return in_size / out_size


def area_pixel_scale(in_size, out_size, align_corners, scale):
    """Resolve the source-coordinate scale for area-pixel interpolation.

    Args:
        in_size (int): Input extent along the interpolated dimension.
        out_size (int): Output extent along the interpolated dimension.
        align_corners (bool): Whether to align the input and output corner pixels.
        scale (float or None): Explicit forward scale, or ``None`` to infer it.

    Returns:
        float: Source-coordinate scale, with zero for one aligned output pixel.
    """
    if align_corners:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    return scales_value(in_size, out_size, scale)


# --- index / weight maps (trace-time constants) ---


def _positions(out_size, device, dtype=torch.float32):
    # arange, not a Python-list constant, so the map stays functional under
    # AOTAutograd. The default float32 matches aten's index arithmetic for the
    # supported dtypes; callers pass the resample dtype so a float64 input maps
    # in float64, as aten does through its double opmath type.
    return torch.arange(out_size, dtype=dtype, device=device)


def _to_index(values, in_size):
    # The floor before the cast is load-bearing: a float->int convert fused into
    # the graph does not truncate on device.
    return torch.clamp(torch.floor(values).to(torch.int32), max=in_size - 1)


def nearest_indices(in_size, out_size, scale, *, exact, short_circuit, device):
    # short_circuit models aten's nearest_idx fast paths (taken by the 2-D forward
    # and every backward but not the 1-D forward).
    """Build nearest-neighbor source indices for one output dimension.

    Args:
        in_size (int): Input extent along the interpolated dimension.
        out_size (int): Output extent along the interpolated dimension.
        scale (float or None): Explicit forward scale, or ``None`` to infer it.
        exact (bool): Whether to apply the half-pixel offset of exact-nearest mode.
        short_circuit (bool): Whether to use identity and two-times-size fast paths from ``aten``.
        device (torch.device or str): Device on which to create the index tensor.

    Returns:
        torch.Tensor: ``torch.int32`` source index for each output position.
    """
    if short_circuit and not exact:
        if out_size == in_size:
            return torch.arange(out_size, dtype=torch.int32, device=device)
        if out_size == 2 * in_size:
            return torch.arange(out_size, dtype=torch.int32, device=device) // 2
    s = as_float32(scales_value(in_size, out_size, scale))
    offset = 0.5 if exact else 0.0
    return _to_index((_positions(out_size, device) + offset) * s, in_size)


def linear_taps(in_size, out_size, scale, align_corners, device, dtype=torch.float32):
    """Build linear interpolation indices and weights for one dimension.

    Args:
        in_size (int): Input extent along the interpolated dimension.
        out_size (int): Output extent along the interpolated dimension.
        scale (float or None): Explicit forward scale, or ``None`` to infer it.
        align_corners (bool): Whether to align the input and output corner pixels.
        device (torch.device or str): Device on which to create the tap tensors.
        dtype (torch.dtype, optional): Weight dtype. Defaults to ``torch.float32``.

    Returns:
        tuple: Pair of ``(low, high)`` indices and their corresponding weights.

    Note:
        Equal input and output extents return identity taps and ignore ``scale``.
    """
    if out_size == in_size:  # aten copies and ignores the scale
        identity = torch.arange(out_size, dtype=torch.int32, device=device)
        ones = torch.ones(out_size, dtype=dtype, device=device)
        return (identity, identity), (ones, torch.zeros_like(ones))
    s = area_pixel_scale(in_size, out_size, align_corners, scale)
    positions = _positions(out_size, device, dtype)
    # Spelled as aten's area_pixel_compute_source_index; an equivalent
    # rearrangement rounds differently in float32 and moves a tap.
    real = positions * s if align_corners else torch.clamp((positions + 0.5) * s - 0.5, min=0.0)
    low = _to_index(real, in_size)
    lam = real - low.to(dtype)
    high = torch.clamp(low + 1, max=in_size - 1)
    return (low, high), (1.0 - lam, lam)


_CUBIC_A = -0.75


def _cubic_centre_weight(t):
    return ((_CUBIC_A + 2) * t - (_CUBIC_A + 3)) * t * t + 1


def _cubic_outer_weight(t):
    return ((_CUBIC_A * t - 5 * _CUBIC_A) * t + 8 * _CUBIC_A) * t - 4 * _CUBIC_A


def bicubic_taps(in_size, out_size, scale, align_corners, device, dtype=torch.float32):
    """Build bicubic interpolation indices and weights for one dimension.

    Args:
        in_size (int): Input extent along the interpolated dimension.
        out_size (int): Output extent along the interpolated dimension.
        scale (float or None): Explicit forward scale, or ``None`` to infer it.
        align_corners (bool): Whether to align the input and output corner pixels.
        device (torch.device or str): Device on which to create the tap tensors.
        dtype (torch.dtype, optional): Weight dtype. Defaults to ``torch.float32``.

    Returns:
        tuple: Lists containing four clamped source indices and four cubic weights.
    """
    s = area_pixel_scale(in_size, out_size, align_corners, scale)
    positions = _positions(out_size, device, dtype)
    real = positions * s if align_corners else (positions + 0.5) * s - 0.5
    base = torch.floor(real)
    t = real - base
    base_index = base.to(torch.int32)
    taps = (
        _cubic_outer_weight(t + 1.0),
        _cubic_centre_weight(t),
        _cubic_centre_weight(1.0 - t),
        _cubic_outer_weight(2.0 - t),
    )
    indices = [torch.clamp(base_index - 1 + k, min=0, max=in_size - 1) for k in range(4)]
    return indices, list(taps)


# --- applying a map ---


def _shape_weight(weights, dim, ndim, dtype):
    shape = [1] * ndim
    shape[dim] = weights.shape[0]
    return weights.to(dtype).view(shape)


def resample_dtype(x):
    # fp32 accumulator (fp64 for fp64 input), matching aten's opmath.
    """Select the floating-point accumulator dtype used by ``aten`` for resampling.

    Args:
        x (torch.Tensor): Tensor whose dtype determines the accumulator dtype.

    Returns:
        torch.dtype: ``torch.float64`` for ``torch.float64`` input, otherwise ``torch.float32``.
    """
    return torch.float64 if x.dtype == torch.float64 else torch.float32


# Integral inputs reaching here are always a widened uint8, so the saturation
# range is uint8's (torch.iinfo(x.dtype) would read the widened int32 range).
_UINT8_MIN = 0.0
_UINT8_MAX = 255.0


def round_into_range(result, dtype):
    # floor(v + 0.5) (aten rounds halves up), not torch.round (to even), so uint8
    # output matches aten exactly.
    """Round an interpolation result into the supported integral output range.

    Args:
        result (torch.Tensor): Floating-point values to round and clamp.
        dtype (torch.dtype): Integral destination dtype used for validation.

    Returns:
        torch.Tensor: Values rounded half-up and clamped to the ``torch.uint8`` range.

    Raises:
        AssertionError: If ``dtype`` is floating-point.
    """
    if dtype.is_floating_point:
        raise AssertionError(f"round_into_range is for integral outputs, got {dtype}")
    return torch.clamp(torch.floor(result + 0.5), _UINT8_MIN, _UINT8_MAX)


def to_result_dtype(result, dtype):
    """Convert an interpolation result to its destination dtype.

    Args:
        result (torch.Tensor): Accumulated interpolation result.
        dtype (torch.dtype): Destination dtype.

    Returns:
        torch.Tensor: Original tensor when dtypes match, a cast floating tensor, or a
            half-up-rounded and ``torch.uint8``-range-clamped integral tensor.
    """
    if result.dtype == dtype:
        return result
    if dtype.is_floating_point:
        return result.to(dtype)
    return round_into_range(result, dtype).to(dtype)


def _is_identity_map(in_size, out_size, short_circuit, exact):
    # Decided from sizes alone; the index values are unavailable at trace time.
    return in_size == out_size and ((short_circuit and not exact) or in_size == 1)


def _gather_nearest(x, dim, indices):
    return torch.index_select(x, dim, indices)


def _blend_taps_gather(x, dim, indices, weights, dtype):
    result = None
    for tap_indices, tap_weights in zip(indices, weights, strict=True):
        gathered = torch.index_select(x, dim, tap_indices)
        contribution = gathered * _shape_weight(tap_weights, dim, x.ndim, dtype)
        result = contribution if result is None else result + contribution
    return result


def _resample_matrix(indices, weights, in_size, out_size, dtype, device):
    o = torch.arange(out_size, device=device)
    m = torch.zeros(out_size, in_size, dtype=dtype, device=device)
    for tap_indices, tap_weights in zip(indices, weights, strict=True):
        m = torch.index_put(m, (o, tap_indices), tap_weights.to(dtype), accumulate=True)
    return m


def _blend_taps_matmul(x, dim, indices, weights, dtype):
    in_size = x.shape[dim]
    out_size = weights[0].shape[0]
    m = _resample_matrix(indices, weights, in_size, out_size, dtype, x.device)
    moved = x.movedim(dim, -1)
    if moved.dtype != dtype:
        moved = moved.to(dtype)
    out = torch.matmul(moved, m.transpose(0, 1))
    return out.movedim(-1, dim)


def _index_add_off_minor(source, dim, out_size, indices, dtype):
    # Scattering into the minor axis is ~2000x slower on device, so move the
    # scattered axis to 0 and back.
    moved = source.movedim(dim, 0).contiguous()
    shape = [out_size, *(s for i, s in enumerate(source.shape) if i != dim)]
    zeros = torch.zeros(shape, dtype=dtype, device=source.device)
    return zeros.index_add(0, indices, moved).movedim(0, dim)


def _scatter_nearest(grad, dim, in_size, indices):
    return _index_add_off_minor(grad, dim, in_size, indices, grad.dtype)


def _scatter_taps(grad, dim, in_size, indices, weights, dtype):
    # One index_add over the concatenated taps so duplicate destinations
    # accumulate; one call per tap would drop all but the last.
    flat_indices = torch.cat(list(indices))
    contributions = [
        grad * _shape_weight(tap_weights, dim, grad.ndim, dtype) for tap_weights in weights
    ]
    source = torch.cat(contributions, dim=dim)
    return _index_add_off_minor(source, dim, in_size, flat_indices, dtype)


def _sizes(output_size):
    return [int(v) for v in output_size]


def _output_shape(x, spatial):
    return (*x.shape[: x.ndim - len(spatial)], *spatial)


def _output_numel(x, spatial):
    n = 1
    for s in _output_shape(x, spatial):
        n *= int(s)
    return n


# Output-element count above which the matmul blend beats gather on contiguous float.
_MATMUL_MIN_OUTPUT_ELEMS = 1 << 19  # 524,288


# --- forward / backward ---


def nearest_forward(x, output_size, scales, *, exact, short_circuit):
    """Apply separable nearest-neighbor interpolation to a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        output_size (Sequence[int]): Output extents for the spatial dimensions.
        scales (Sequence[float or None]): Per-dimension explicit scales or ``None`` values.
        exact (bool): Whether to use exact-nearest source coordinates.
        short_circuit (bool): Whether to use nearest-index fast paths from ``aten``.

    Returns:
        torch.Tensor: Resampled tensor with the requested spatial shape.

    Note:
        For nonempty ``x``, if every spatial dimension takes an identity path, this returns
        ``x`` itself.
    """
    spatial = _sizes(output_size)
    if x.numel() == 0:
        return x.new_empty(_output_shape(x, spatial))
    result = x
    for axis, out_size in enumerate(spatial):
        dim = x.ndim - len(spatial) + axis
        in_size = x.shape[dim]
        if _is_identity_map(in_size, out_size, short_circuit, exact):
            continue
        indices = nearest_indices(
            in_size,
            out_size,
            scales[axis],
            exact=exact,
            short_circuit=short_circuit,
            device=x.device,
        )
        result = _gather_nearest(result, dim, indices)
    return result


def bimode_forward(x, output_size, align_corners, scales, *, cubic):
    """Apply separable linear or bicubic interpolation to a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        output_size (Sequence[int]): Output extents for the spatial dimensions.
        align_corners (bool): Whether to align the input and output corner pixels.
        scales (Sequence[float or None]): Per-dimension explicit scales or ``None`` values.
        cubic (bool): Whether to use four-tap bicubic instead of two-tap linear blending.

    Returns:
        torch.Tensor: Resampled tensor converted back to the input dtype.

    Note:
        Integral inputs are rounded and clamped between passes to match ``aten``. Unchanged
        linear dimensions ignore their scales; without an accumulator cast, an entirely
        unchanged nonempty input returns the original ``x`` object.
    """
    spatial = _sizes(output_size)
    if x.numel() == 0:
        return x.new_empty(_output_shape(x, spatial))
    dtype = resample_dtype(x)
    # aten resamples each 1-D pass into scalar_t, so an integral dtype is rounded
    # and clamped between passes; reproducing that makes uint8 bit-exact.
    narrow_between_passes = not x.is_floating_point()
    # Gather is the default and only channels_last-safe blend. The matmul blend is
    # faster but confined to large contiguous float: it changes uint8 accumulation,
    # and fused with the channels_last transpose epilogue it crashes neuronx-cc.
    blend = _blend_taps_gather
    if (
        x.is_contiguous()
        and x.is_floating_point()
        and _output_numel(x, spatial) >= _MATMUL_MIN_OUTPUT_ELEMS
    ):
        blend = _blend_taps_matmul
    result = x if dtype == x.dtype else x.to(dtype)
    for axis in reversed(range(len(spatial))):  # innermost dimension first, as aten
        out_size = spatial[axis]
        dim = x.ndim - len(spatial) + axis
        in_size = x.shape[dim]
        if cubic:
            indices, weights = bicubic_taps(
                in_size, out_size, scales[axis], align_corners, x.device, dtype
            )
        else:
            if out_size == in_size:  # aten copies this dimension and ignores its scale
                continue
            indices, weights = linear_taps(
                in_size, out_size, scales[axis], align_corners, x.device, dtype
            )
        result = blend(result, dim, indices, weights, dtype)
        if narrow_between_passes and axis != 0:
            result = round_into_range(result, x.dtype)
    return to_result_dtype(result, x.dtype)


def nearest_backward(grad_output, output_size, input_size, scales, *, exact):
    """Scatter nearest-neighbor output gradients into the input shape.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the interpolated output.
        output_size (Sequence[int]): Spatial extents of the interpolated output.
        input_size (Sequence[int]): Full shape of the original input.
        scales (Sequence[float or None]): Per-dimension explicit scales or ``None`` values.
        exact (bool): Whether to use exact-nearest source coordinates.

    Returns:
        torch.Tensor: Gradient accumulated into the original input shape.

    Note:
        Plain-nearest backward uses the short-circuit map from ``aten`` at every supported
        rank. For nonempty ``grad_output``, if every spatial dimension takes an identity path,
        this returns ``grad_output`` itself.
    """
    spatial = _sizes(output_size)
    in_sizes = _sizes(input_size)
    in_spatial = in_sizes[-len(spatial) :]
    if grad_output.numel() == 0:
        return grad_output.new_zeros(in_sizes)
    result = grad_output
    for axis in range(len(spatial)):
        dim = grad_output.ndim - len(spatial) + axis
        in_size = in_spatial[axis]
        # Every nearest backward uses the short-circuiting map, at both ranks.
        if _is_identity_map(in_size, spatial[axis], not exact, exact):
            continue
        indices = nearest_indices(
            in_size,
            spatial[axis],
            scales[axis],
            exact=exact,
            short_circuit=not exact,
            device=grad_output.device,
        )
        result = _scatter_nearest(result, dim, in_size, indices)
    return result


def bimode_backward(grad_output, output_size, input_size, align_corners, scales, *, cubic):
    """Scatter linear or bicubic output gradients into the input shape.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the interpolated output.
        output_size (Sequence[int]): Spatial extents of the interpolated output.
        input_size (Sequence[int]): Full shape of the original input.
        align_corners (bool): Whether to align the input and output corner pixels.
        scales (Sequence[float or None]): Per-dimension explicit scales or ``None`` values.
        cubic (bool): Whether to use bicubic instead of linear tap weights.

    Returns:
        torch.Tensor: Gradient accumulated into the original input shape and dtype.

    Note:
        For nonempty ``grad_output``, bicubic mode clones the gradient when all spatial sizes
        are unchanged, ignoring ``scales``. Linear dimensions with unchanged sizes also ignore
        their scales; if every dimension is unchanged and no accumulator cast is needed, this
        returns ``grad_output`` itself.
    """
    spatial = _sizes(output_size)
    in_sizes = _sizes(input_size)
    in_spatial = in_sizes[-len(spatial) :]
    if grad_output.numel() == 0:
        return grad_output.new_zeros(in_sizes)
    if cubic and in_spatial == spatial:
        return grad_output.clone()  # aten copies when both sizes are unchanged
    dtype = resample_dtype(grad_output)
    result = grad_output if dtype == grad_output.dtype else grad_output.to(dtype)
    for axis in range(len(spatial)):
        dim = grad_output.ndim - len(spatial) + axis
        in_size = in_spatial[axis]
        if cubic:
            indices, weights = bicubic_taps(
                in_size, spatial[axis], scales[axis], align_corners, grad_output.device, dtype
            )
        else:
            if in_size == spatial[axis]:
                continue
            indices, weights = linear_taps(
                in_size, spatial[axis], scales[axis], align_corners, grad_output.device, dtype
            )
        result = _scatter_taps(result, dim, in_size, indices, weights, dtype)
    return to_result_dtype(result, grad_output.dtype)
