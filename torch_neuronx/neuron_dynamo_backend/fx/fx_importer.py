"""Neuron custom FX Importer."""

import torch.fx
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import ContextCache, FxImporter, FxImporterHooks
from torch_mlir.ir import Location


def _get_hierarchy_string(node: torch.fx.Node) -> str | None:
    """Extract module hierarchy string from FX node metadata."""
    nn_module_stack = node.meta.get("nn_module_stack")
    if not nn_module_stack:
        return None

    parts = []
    for path, cls in nn_module_stack.values():
        path = path.replace("L['self'].", "")
        cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        parts.append(f"{path}|{cls_name}")

    return ">".join(parts) if parts else None


class NeuronContextCache(ContextCache):
    """Neuron custom context cache."""

    def get_node_location(self, node: torch.fx.Node) -> Location | None:
        """Get location with module hierarchy fused with source location."""
        source_loc = super().get_node_location(node)
        hierarchy = _get_hierarchy_string(node)

        if not hierarchy:
            return source_loc

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
