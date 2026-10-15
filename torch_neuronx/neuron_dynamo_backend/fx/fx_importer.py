"""Neuron custom FX Importer."""

import torch.fx
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import ContextCache, FxImporter, FxImporterHooks
from torch_mlir.ir import Location

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

ATEN_HIERARCHY_ENV_VAR = "TORCH_NEURONX_ATEN_HIERARCHY"


def enable_aten_hierarchy() -> bool:
    """Report whether node hierarchy strings include an ATen operation leaf.

    Returns:
        bool: ``True`` when ``TORCH_NEURONX_ATEN_HIERARCHY`` is enabled.

    Note:
        The setting is disabled by default and read from the process environment.
    """
    return bool(_getenv_bool(ATEN_HIERARCHY_ENV_VAR, False))


def _get_hierarchy_string(node: torch.fx.Node) -> str | None:
    """Build hierarchy string from FX node metadata.

    If 'call_hierarchy' metadata is present (function call and module stacks),
    uses it directly. Otherwise falls back to nn_module_stack (module-only).

    Output format (default): ClassName[attr][count]/func_name[count]/...

    When TORCH_NEURONX_ATEN_HIERARCHY=1, an aten-op leaf is appended:
        ClassName[attr][count]/.../aten.op.overload[node_name]
    """
    # Preferred path: unified call hierarchy
    call_hierarchy = node.meta.get("call_hierarchy")
    if call_hierarchy:
        # TODO (pending PyTorch Upstream)
        pass

    # Fallback: module-only hierarchy from nn_module_stack
    # fwd_nn_module_stack is set by AOT autograd on backward nodes
    nn_module_stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
    aten_hierarchy = enable_aten_hierarchy()

    # Legacy behavior when the flag is off: no module stack -> no hierarchy.
    # With the flag on, a call_function node still gets an aten-op leaf.
    if not nn_module_stack and not aten_hierarchy:
        return None

    segments = []
    for key, (path, cls) in (nn_module_stack or {}).items():
        path = path.replace("L['self'].", "")

        if path == "L['self']":
            continue  # skip root module entry

        cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        attr = path.rsplit(".", 1)[-1] if "." in path else path
        call_count = int(str(key).split("@")[1]) if "@" in str(key) else 0
        segments.append(f"{cls_name}[{attr}][{call_count}]")

    if aten_hierarchy and node.op == "call_function":
        segments.append(f"{node.target}[{node.name}]")

    return "/".join(segments) if segments else None


class NeuronContextCache(ContextCache):
    """Track hierarchy-aware source locations during FX import.

    Args:
        *args: Positional arguments forwarded to ``ContextCache``.
        **kwargs: Keyword arguments forwarded to ``ContextCache``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_hierarchy = False

    def get_node_location(self, node: torch.fx.Node) -> Location | None:
        """Fuse an FX node hierarchy with its source location.

        Args:
            node (torch.fx.Node): FX node whose import location is requested.

        Returns:
            Location | None: Fused hierarchy and source location, either location
            alone, or ``None`` when neither is available.

        Note:
            Finding a hierarchy sets ``has_hierarchy`` on this cache.
        """
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
    """Import FX graphs through the Neuron-customized torch-MLIR context.

    Args:
        hooks (FxImporterHooks | None, optional): Hooks for custom import behavior.
            Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to ``FxImporter``.

    Note:
        This class is the central extension point for Neuron FX importer
        customizations and installs ``NeuronContextCache``.
    """

    def __init__(self, *, hooks: FxImporterHooks | None = None, **kwargs):
        context = ir.Context()
        torch_d.register_dialect(context)
        super().__init__(context=context, hooks=hooks, **kwargs)

        self._cc = NeuronContextCache(self._c, py_attr_tracker=self._py_attr_tracker)

    @property
    def has_hierarchy(self) -> bool:
        """Report whether any imported node produced hierarchy metadata.

        Returns:
            bool: ``True`` after importing at least one hierarchy-aware node.
        """
        return self._cc.has_hierarchy
