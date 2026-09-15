"""Neuron custom FX Importer."""

import re

import torch.fx
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import ContextCache, FxImporter, FxImporterHooks
from torch_mlir.ir import Location

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

ATEN_HIERARCHY_ENV_VAR = "TORCH_NEURONX_ATEN_HIERARCHY"

_STACK_TRACE_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
_MODULE_CALL_WRAPPERS = frozenset({"_call_impl", "_wrapped_call_impl", "_slow_forward"})


def enable_aten_hierarchy() -> bool:
    """Whether to append an aten-op leaf to each node's hierarchy string.

    Off by default -- see ``_get_hierarchy_string`` for why this is opt-in
    """
    return bool(_getenv_bool(ATEN_HIERARCHY_ENV_VAR, False))


def _iter_stack_trace_frames(stack_trace: str):
    """Yield ``(file, function, line)`` in outermost-first traceback order."""
    for match in _STACK_TRACE_FRAME_RE.finditer(stack_trace):
        yield match.group("file"), match.group("func"), int(match.group("line"))


def _is_synthetic_frame(function: str) -> bool:
    """Return whether Python synthesized this frame rather than naming a call scope."""
    return function.startswith("<") and function.endswith(">")


def _render_module_segment(key, path: str, cls) -> str | None:
    """Render one module-stack entry, excluding the root module frame."""
    path = path.replace("L['self'].", "")
    if path == "L['self']" or not path:
        return None

    cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
    attr = path.rsplit(".", 1)[-1] if "." in path else path
    call_count = int(str(key).split("@")[1]) if "@" in str(key) else 0
    return f"{cls_name}[{attr}][{call_count}]"


def _render_function_segment(file: str, function: str) -> str:
    """Render a free-function frame as ``file_stem[function]``."""
    stem = file.rsplit("/", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    return f"{stem}[{function}]".translate(str.maketrans({":": "_", "<": "_", ">": "_"}))


def _get_hierarchy_string(node: torch.fx.Node) -> str | None:
    """Build a profiler-style hierarchy string from FX node metadata.

    ``nn_module_stack`` remains the source of module frames, with
    ``fwd_nn_module_stack`` retaining the existing backward-graph fallback.
    When ``stack_trace`` is present, its free-function frames are interleaved
    with module ``forward`` frames in call order. This preserves functional APIs
    without depending on the experimental ``call_hierarchy`` metadata.

    If module entries and ``forward`` frames cannot be aligned one-to-one, the
    function degrades safely to the module hierarchy followed by non-``forward``
    function frames. With no stack trace it reproduces module-only behavior.

    When TORCH_NEURONX_ATEN_HIERARCHY=1, an aten-op leaf is appended.
    """
    module_stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
    module_entries = list((module_stack or {}).items())
    stack_trace = node.meta.get("stack_trace", "")

    module_segments = [
        segment
        for key, (path, cls) in module_entries
        if (segment := _render_module_segment(key, path, cls))
    ]
    segments = module_segments

    if stack_trace:
        frames = [
            frame
            for frame in _iter_stack_trace_frames(stack_trace)
            if frame[1] not in _MODULE_CALL_WRAPPERS and not _is_synthetic_frame(frame[1])
        ]
        forward_frames = [frame for frame in frames if frame[1] == "forward"]

        if module_entries and len(forward_frames) == len(module_entries):
            segments = []
            module_index = 0
            for file, function, _line in frames:
                if function == "forward":
                    key, (path, cls) = module_entries[module_index]
                    module_index += 1
                    segment = _render_module_segment(key, path, cls)
                    if segment:
                        segments.append(segment)
                else:
                    segments.append(_render_function_segment(file, function))
        else:
            segments = module_segments + [
                _render_function_segment(file, function)
                for file, function, _line in frames
                if function != "forward"
            ]

    if enable_aten_hierarchy() and node.op == "call_function":
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
