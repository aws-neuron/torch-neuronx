"""Per-op dynamic NKI adapters.

One module per op (group), each exposing ``register()`` that adds its
DynamicNkiEntry(ies) to the registry. ``register_all()`` wires them all; the
package ``__init__`` calls it behind the feature flag.
"""


def register_all():
    from . import elementwise, gather, matmul, scatter, silu

    silu.register()
    elementwise.register()
    matmul.register()
    gather.register()
    scatter.register()
