"""Per-op dynamic NKI adapters.

Each module exposes ``register()`` that adds its DynamicNkiEntry(ies) to the
registry. ``elementwise`` covers silu/exp/sin/cos/add/mul via one factory;
matmul/gather/scatter stay bespoke. ``register_all()`` wires them all; the
package ``__init__`` calls it behind the feature flag.
"""


def register_all():
    from . import elementwise, gather, matmul, scatter

    elementwise.register()
    matmul.register()
    gather.register()
    scatter.register()
