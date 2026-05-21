"""Neuron custom FX Importer."""

import torch.fx
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import ContextCache, FxImporter, FxImporterHooks
from torch_mlir.ir import Location


def _get_hierarchy_string(node: torch.fx.Node) -> str | None:
    """Build hierarchy string from FX node metadata.

    If 'call_hierarchy' metadata is present (function call and module stacks),
    uses it directly. Otherwise falls back to nn_module_stack (module-only).

    Output format: ClassName[attr][count]/func_name[count]/.../aten.op.overload[node_name]
    """
    # Preferred path: unified call hierarchy
    call_hierarchy = node.meta.get("call_hierarchy")
    if call_hierarchy:
        # TODO (pending PyTorch Upstream)
        pass

    # Fallback: module-only hierarchy from nn_module_stack
    # fwd_nn_module_stack is set by AOT autograd on backward nodes
    nn_module_stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")

    segments = []
    for key, (path, cls) in (nn_module_stack or {}).items():
        path = path.replace("L['self'].", "")

        if path == "L['self']":
            continue  # skip root module entry

        cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        attr = path.rsplit(".", 1)[-1] if "." in path else path
        call_count = int(str(key).split("@")[1]) if "@" in str(key) else 0
        segments.append(f"{cls_name}[{attr}][{call_count}]")

    if node.op == "call_function":
        segments.append(f"{node.target}[{node.name}]")

    return "/".join(segments) if segments else None


class NeuronContextCache(ContextCache):
    """Neuron custom context cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_hierarchy = False

    def get_node_location(self, node: torch.fx.Node) -> Location | None:
        """Get location with hierarchy string fused with source location."""
        source_loc = super().get_node_location(node)
        hierarchy = _get_hierarchy_string(node)

        if not hierarchy:
            return source_loc

        self.has_hierarchy = True
        hierarchy_loc = Location.name(hierarchy, context=self._c)
        return (
            Location.fused([hierarchy_loc, source_loc], context=self._c)
            if source_loc
            else hierarchy_loc
        )


class NeuronFxImporter(FxImporter):
    """Neuron custom FxImporter.

    Central extension point for torch-mlir FxImporter customizations.
    Add new parameters here for future hooks.

    Args:
        hooks: Optional FxImporterHooks for custom import behavior.
    """

    def __init__(self, *, hooks: FxImporterHooks | None = None, **kwargs):
        context = ir.Context()
        torch_d.register_dialect(context)
        super().__init__(context=context, hooks=hooks, **kwargs)

        self._cc = NeuronContextCache(self._c, py_attr_tracker=self._py_attr_tracker)

    @property
    def has_hierarchy(self) -> bool:
        """Returns True if any node had hierarchy metadata during import."""
        return self._cc.has_hierarchy
