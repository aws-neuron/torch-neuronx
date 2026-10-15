"""Leaf math helpers shared by the eager and torch.compile lowerings.

Keep this module dependency-free (only ``torch`` and ``math``).
"""

import math

import torch

# log2(x) = ln(x) / ln(2). Neuron has no native exp2 lowering, so use
# exp(d * ln2) for 2**d.
_LN2 = math.log(2.0)
_INV_LN2 = 1.0 / _LN2


def logaddexp2_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the base-two logarithm of summed powers.

    Args:
        a (torch.Tensor): First exponent tensor.
        b (torch.Tensor): Second exponent tensor.

    Returns:
        torch.Tensor: Promoted result cast back from the float32 computation.

    Note:
        Uses a stable max-shift form, computes all inputs in float32, and preserves
        shared positive or negative infinity.
    """
    # Compute in float32 to match aten's float accumulate type for half/bfloat16.
    out_dtype = torch.promote_types(a.dtype, b.dtype)
    aa = a.float()
    bb = b.float()

    mask = aa >= bb
    max_ = torch.where(mask, aa, bb)
    min_ = torch.where(mask, bb, aa)
    # At a shared infinity the formula gives log1p(nan); return that infinity, as CPU does.
    inf_mask = torch.logical_and(torch.isinf(aa), aa == bb)
    result = max_ + torch.log1p(torch.exp((min_ - max_) * _LN2)) * _INV_LN2
    result = torch.where(inf_mask, aa, result)
    return result.to(out_dtype)
