"""Per-op dynamic NKI adapters.

Each module exposes ``register()`` that adds its DynamicNkiEntry(ies) to the
registry. ``elementwise`` covers silu/exp/sin/cos/add/mul via one factory;
matmul/gather/scatter stay bespoke. ``register_all()`` wires them all; the
package ``__init__`` calls it behind the feature flag.
"""


def register_all():
    from . import (
        bitwise,
        concat,
        elementwise,
        gather,
        matmul,
        matmul_k,
        reduction,
        scatter,
        silu_bwd,
        where,
    )

    silu_bwd.register()
    elementwise.register()
    matmul.register()
    matmul_k.register()
    gather.register()
    scatter.register()
    reduction.register()
    concat.register()
    where.register()
    bitwise.register()
