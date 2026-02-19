import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten
from torch_neuronx.utils import get_gmm_align


@register_aten(["aten::_grouped_mm"])
def _aten_grouped_mm_2d_3d(a, b, offs):
    """Grouped matmul: a (t, d1), b (g, d1, d2), offs (g,) -> (t, d2)"""
    t, d1 = a.shape
    g, _, d2 = b.shape
    align = get_gmm_align()

    b_index = (offs[:, None] // align <= jnp.arange(t // align)).sum(0)
    b_index = jnp.clip(b_index, 0, g - 1)
    a_batched = a.reshape(t // align, align, d1)
    b_batched = b[b_index]
    r_batched = a_batched @ b_batched
    return r_batched.reshape(t, d2)
