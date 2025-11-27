from __future__ import annotations

import pytest

from torch_neuronx.python_ops.kernel_cache import (
    get_or_create_jax_kernel,
    kernel_cache_size,
)
from torch_neuronx.python_ops.processors.argument_processor import ArgumentProcessor


def _jax_copy_fn(x, out_dtype=None):
    import jax.numpy as jnp

    if out_dtype is None:
        return x
    return x.astype(out_dtype)


class _AP(ArgumentProcessor):
    pass


def test_kernel_cache_reuse():
    s0 = kernel_cache_size()
    k1 = get_or_create_jax_kernel(
        jax_fn=_jax_copy_fn,
        op_name="aten::copy_",
        static_argnames=("out_dtype",),
        argument_processor_cls=_AP,
    )
    k2 = get_or_create_jax_kernel(
        jax_fn=_jax_copy_fn,
        op_name="aten::copy_",
        static_argnames=("out_dtype",),
        argument_processor_cls=_AP,
    )
    assert k1 is k2
    assert kernel_cache_size() == s0 + 1
