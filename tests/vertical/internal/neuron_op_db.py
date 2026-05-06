"""Build the unified Neuron op database.

Lookup Priority:
1. PyTorch op_db — for each op in op_db, if its base name matches any
   target op, creates a NeuronOpInfo for every target overload sharing
   that base name.
2. Schema fallback — for any target op NOT already in db from step 1,
   creates a NeuronOpInfo via from_schema.
"""

from __future__ import annotations

from .neuron_op_info import NeuronOpInfo


def build_neuron_op_db(
    target_ops: set[str] | None = None,
    target_op_groups: dict[str, str] | None = None,
) -> list[NeuronOpInfo]:
    """Build the complete Neuron op database.

    Args:
        target_ops: Set of aten op names (e.g., "aten::add.Tensor") to cover.
        target_op_groups: Mapping of op_name → group (elementwise, nn, etc.)
    """
    db: dict[str, NeuronOpInfo] = {}
    groups = target_op_groups or {}

    # 1. PyTorch op_db → create entry for EACH target overload sharing the base name
    if target_ops:
        base_to_overloads: dict[str, list[str]] = {}
        for op_name in target_ops:
            base = op_name.replace("aten::", "").split(".")[0]
            base_to_overloads.setdefault(base, []).append(op_name)

        try:
            from torch.testing._internal.common_methods_invocations import op_db
            from torch.testing._internal.opinfo.core import _dispatch_dtypes

            for info in op_db:
                if info.name not in base_to_overloads:
                    continue
                for op_name in base_to_overloads[info.name]:
                    if op_name not in db:
                        group = groups.get(op_name, "unknown")
                        kw = {}
                        for field in ("dtypes", "dtypesIfCUDA", "dtypesIfROCM", "dtypesIfXPU"):
                            val = getattr(info, field, None)
                            if val is not None:
                                kw[field] = (
                                    _dispatch_dtypes(val)
                                    if not isinstance(val, _dispatch_dtypes)
                                    else val
                                )
                        db[op_name] = NeuronOpInfo(
                            name=info.name,
                            op_name=op_name,
                            op_group=group,
                            _pytorch_sample_inputs_func=info.sample_inputs_func,
                            **kw,
                        )
        except ImportError:
            pass

    # 2. Auto-generate for remaining target ops
    if target_ops:
        for op_name in target_ops:
            if op_name not in db:
                group = groups.get(op_name, "unknown")
                db[op_name] = NeuronOpInfo.from_schema(op_name, op_group=group)

    return sorted(db.values(), key=lambda x: x.op_name)
