"""aten::im2col / aten::col2im (F.unfold / F.fold) eager lowerings.

Thin @register_aten wrappers: the bodies dispatch to .default; the decomposition
in decompositions.py (shared with the compile path) rewrites them under make_fx.
"""

import torch

from ..operation_registry import register_aten
from ..validation import aten_scalar_type_name

aten = torch.ops.aten


def _validate_fold_dtype(kernel_name):
    """Match aten: im2col/col2im take floating point and bool only.

    Reject integral dtypes with aten's wording so test_dtypes matches the OpInfo
    table.
    """

    def _validate(tensor, *_args, **_kwargs):
        if tensor.is_floating_point() or tensor.dtype == torch.bool:
            return
        raise NotImplementedError(
            f"\"{kernel_name}\" not implemented for '{aten_scalar_type_name(tensor.dtype)}'"
        )

    return _validate


def _validate_im2col(self, kernel_size, dilation, padding, stride, out=None):
    _validate_fold_dtype("im2col")(self)
    # torch's aten.im2col decomposition drives meta inference. It reports
    # kernel_size with different wording than aten's C++ kernel, and it skips
    # the padding check (it validates dilation twice). Re-check both with aten's
    # C++ shape-check wording so an invalid argument fails the same way here.
    if len(kernel_size) == 2 and min(kernel_size) <= 0:
        raise RuntimeError(
            "kernel size should be greater than zero, but got "
            f"kernel_height: {kernel_size[0]} kernel_width: {kernel_size[1]}"
        )
    if len(padding) == 2 and min(padding) < 0:
        raise RuntimeError(
            "padding should be non-negative, but got "
            f"pad_height: {padding[0]} pad_width: {padding[1]}"
        )


@register_aten(
    ["aten::im2col", "aten::im2col.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4),
    validate_static_inputs=_validate_im2col,
)
def torch_im2col(self, kernel_size, dilation, padding, stride, out=None):
    """Extract sliding image blocks through the registered ATen lowering.

    Args:
        self (torch.Tensor): Batched or unbatched image tensor.
        kernel_size (Sequence[int]): Height and width of each block.
        dilation (Sequence[int]): Spacing between kernel elements.
        padding (Sequence[int]): Implicit padding on each spatial dimension.
        stride (Sequence[int]): Step between adjacent blocks.
        out (torch.Tensor, optional): Output buffer handled by the registration wrapper rather
            than this function body. Defaults to ``None``.

    Returns:
        torch.Tensor: Columns containing the extracted image blocks.

    Raises:
        NotImplementedError: If ``self`` has an integral dtype.
        RuntimeError: If the kernel size is nonpositive or padding is negative.
    """
    return aten.im2col.default(self, kernel_size, dilation, padding, stride)


def _validate_col2im(self, output_size, kernel_size, dilation, padding, stride, out=None):
    _validate_fold_dtype("col2im")(self)
    # torch's aten.col2im decomposition uses Python ints, so it never detects the
    # uint64 kernel-product overflow that aten's C++ shape check reports. Re-raise
    # aten's wording so a huge kernel_size fails the same way here.
    if len(kernel_size) == 2 and kernel_size[0] * kernel_size[1] > 2**64 - 1:
        raise RuntimeError(
            f"Given kernel_width = {kernel_size[1]} and kernel_height = {kernel_size[0]} "
            "the product of kernel_width and kernel_height overflowed."
        )


@register_aten(
    ["aten::col2im", "aten::col2im.out"],
    output_params=("out",),
    static_argnums=(1, 2, 3, 4, 5),
    validate_static_inputs=_validate_col2im,
)
def torch_col2im(self, output_size, kernel_size, dilation, padding, stride, out=None):
    """Combine column blocks into an image through the registered ATen lowering.

    Args:
        self (torch.Tensor): Batched or unbatched tensor of column blocks.
        output_size (Sequence[int]): Height and width of the output image.
        kernel_size (Sequence[int]): Height and width of each block.
        dilation (Sequence[int]): Spacing between kernel elements.
        padding (Sequence[int]): Implicit padding on each spatial dimension.
        stride (Sequence[int]): Step between adjacent blocks.
        out (torch.Tensor, optional): Output buffer handled by the registration wrapper rather
            than this function body. Defaults to ``None``.

    Returns:
        torch.Tensor: Image formed by summing overlapping blocks.

    Raises:
        NotImplementedError: If ``self`` has an integral dtype.
        RuntimeError: If the kernel-size product overflows ``uint64``.
    """
    return aten.col2im.default(self, output_size, kernel_size, dilation, padding, stride)
