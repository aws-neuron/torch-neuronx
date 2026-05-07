"""
Hierarchy string capture for eager-mode TorchMlirKernel.

At execution submission time the Python thread is synchronous and the full user
call stack is live. This module walks that stack and builds a hierarchy string
in the format:

    ClassName[attr][call_count]/ClassName[attr][call_count]/

Enable with: NEURON_HIERARCHY_DEBUG=1

No user script changes required. The hierarchy is built purely from the live
call stack — no monkey-patching of user code needed.

Forward pass: hierarchy from live nn.Module frames.
Backward pass: op_name (e.g. aten::linear_backward) used as a flat label since
               nn.Module frames and grad_fn objects are absent on the autograd
               C++ engine thread. Label format: op_label[backward][count]/
"""

import os
import sys
import threading

import torch

_FORWARD_NAMES = frozenset({"forward", "apply", "_call_impl", "_wrapped_call_impl"})
_THIS_FILE = os.path.normpath(__file__)

# Internal package prefixes — frames from these are not user hierarchy
_INTERNAL_PREFIXES = (
    os.path.join("torch", ""),
    os.path.join("torch_neuronx", ""),
    os.path.join("torch_mlir", ""),
    os.sep + "torch" + os.sep,
    os.sep + "torch_neuronx" + os.sep,
    os.sep + "torch_mlir" + os.sep,
    os.sep + "site-packages" + os.sep + "torch" + os.sep,
)

# Per-forward-pass invocation counters: maps (class_name, attr) -> count.
_invocation_counters: dict[tuple, int] = {}
_current_top_level_id: int = -1

# Per-backward-pass op counters stored in thread-local storage.
# Using threading.local() ensures automatic cleanup when threads die,
# preventing unbounded growth in long training runs with thread pools.
_backward_local = threading.local()

# Stack fingerprint cache: maps (fingerprint, tuple_of_obj_ids) -> hierarchy string.
# Multiple ops dispatched within the same nn.Module.forward() share the same
# frame chain and module instances — the fingerprint lets us skip the expensive
# f_locals walk + isinstance checks on cache hit.
_stack_cache: dict[tuple, str] = {}
_stack_cache_max_size: int = 256


def _is_internal_frame(filename: str) -> bool:
    """Return True if this frame is from torch/torch_neuronx internals."""
    norm = os.path.normpath(filename)
    return any(p in norm for p in _INTERNAL_PREFIXES)


def _backward_hierarchy_from_op(op_name: str) -> str:
    """Build a backward hierarchy label from the aten op_name.

    Stack dump investigation showed that during C++ autograd engine dispatch,
    the Python stack contains aten op wrapper frames (func_name like
    'linear_backward', 'threshold_backward') but no grad_fn objects with
    next_functions are accessible via frame.f_locals.get("self").

    The op_name is already known at call time, so we use it directly rather
    than re-walking the stack for something that isn't there.

    Label format mirrors the forward convention: op_label[backward][count]/
    e.g. 'aten::linear_backward' -> 'linear_backward[backward][1]/'
         'aten::threshold_backward' -> 'threshold_backward[backward][1]/'
    """
    if not op_name:
        return ""

    # Strip 'aten::' (or any other namespace) prefix
    label = op_name.split("::")[-1]

    counters = getattr(_backward_local, "counters", None)
    if counters is None:
        counters = {}
        _backward_local.counters = counters
    counters[label] = counters.get(label, 0) + 1

    return f"{label}[backward][{counters[label]}]/"


def reset_backward_counters() -> None:
    """Reset backward counters for the current thread.

    Call between backward passes to prevent unbounded counter growth
    during long training runs.
    """
    _backward_local.counters = {}


def _compute_stack_fingerprint(start_frame):
    """Compute a fingerprint from the frame chain.

    Uses id(f_code) for each frame — this identifies the code object (function)
    without touching f_locals. Two calls from the same call site within the same
    nn.Module.forward() will have identical fingerprints.

    For frames whose function name suggests an nn.Module dispatch (forward,
    _call_impl, etc.), also accesses f_locals to get id(self). This
    distinguishes different module instances at the same code location
    (e.g. two different Linear layers). Note: accessing f_locals on a live
    frame triggers a locals-to-dict copy in CPython, so this is only done
    for candidate frames, not all frames.

    Stops walking once internal frames are reached after user code, since
    deeper framework frames don't contribute to the hierarchy.
    """
    code_ids = []
    self_ids = []
    frame = start_frame
    seen_user_frame = False
    while frame is not None:
        filename = frame.f_code.co_filename
        if seen_user_frame and _is_internal_frame(filename):
            break
        if not _is_internal_frame(filename) and os.path.normpath(filename) != _THIS_FILE:
            seen_user_frame = True
        code_ids.append(id(frame.f_code))
        func_name = frame.f_code.co_name
        if (
            func_name in _FORWARD_NAMES
            and frame.f_code.co_varnames
            and frame.f_code.co_varnames[0] == "self"
        ):
            self_obj = frame.f_locals.get("self")
            if self_obj is not None:
                self_ids.append(id(self_obj))
        frame = frame.f_back
    return (tuple(code_ids), tuple(self_ids))


def get_hierarchy_from_stack(op_name: str = "") -> str:
    """Walk the live call stack and return a hierarchy string.

    Forward pass: builds hierarchy from live nn.Module frames.
    Backward pass: uses op_name directly as a flat label (stack investigation
                   confirmed grad_fn objects are not accessible on the C++
                   autograd engine thread).

    Uses stack fingerprint caching: if the frame chain + module instances
    match a previous call, returns the cached result without the expensive
    f_locals walk + isinstance checks.

    Args:
        op_name: The aten op name (e.g. 'aten::linear_backward'). Required for
                 backward ops to produce a meaningful label.

    Returns a '/'-delimited string or empty string if nothing found.
    """
    global _current_top_level_id

    frame = sys._getframe(1)

    # Try stack fingerprint cache first (cheap — only reads f_code pointers
    # and self ids for forward-name frames)
    fingerprint = _compute_stack_fingerprint(frame)
    cached = _stack_cache.get(fingerprint)
    if cached is not None:
        return cached

    # Cache miss — do the full expensive walk
    result = _walk_stack_full(frame, op_name)

    # Cache the result (only for forward path — backward labels have counters)
    if result and "[backward]" not in result:
        if len(_stack_cache) >= _stack_cache_max_size:
            _stack_cache.clear()
        _stack_cache[fingerprint] = result

    return result


def _walk_stack_full(frame, op_name: str) -> str:
    """Full stack walk — called on fingerprint cache miss."""
    global _current_top_level_id
    entries = []

    while frame is not None:
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name

        if os.path.normpath(filename) == _THIS_FILE:
            frame = frame.f_back
            continue

        self_obj = frame.f_locals.get("self")

        if self_obj is not None and isinstance(self_obj, torch.nn.Module):
            # Always process nn.Module frames regardless of _ prefix
            # (_call_impl, _wrapped_call_impl are the actual dispatch entry points)
            class_name = type(self_obj).__name__
            if func_name in _FORWARD_NAMES:
                attr = class_name.lower()
                parent = frame.f_back
                if parent is not None:
                    parent_self = parent.f_locals.get("self")
                    if parent_self is not None and isinstance(parent_self, torch.nn.Module):
                        for a, v in vars(parent_self).items():
                            if v is self_obj:
                                attr = a
                                break
                entries.append((id(self_obj), attr, class_name))
            frame = frame.f_back
            continue

        # For non-module frames, skip private methods and internal packages
        if func_name.startswith("_") or _is_internal_frame(filename):
            frame = frame.f_back
            continue

        frame = frame.f_back

    # ── Forward path ──────────────────────────────────────────────────────────
    if entries:
        entries.reverse()

        deduped = []
        for obj_id, attr, cls in entries:
            if not deduped or deduped[-1][0] != obj_id:
                deduped.append((obj_id, attr, cls))

        outermost_id = deduped[0][0]
        if outermost_id != _current_top_level_id:
            _invocation_counters.clear()
            _current_top_level_id = outermost_id

        segments = []
        for i, (obj_id, attr, cls) in enumerate(deduped):
            if i == 0 and attr == cls.lower():
                continue
            key = (cls, attr)
            # Only increment when we see a different instance for this key.
            # Multiple ops dispatched within the same module.forward() call
            # share the same obj_id — they should all get the same count.
            last_obj_key = ("_last_obj", cls, attr)
            if _invocation_counters.get(last_obj_key) != obj_id:
                _invocation_counters[key] = _invocation_counters.get(key, 0) + 1
                _invocation_counters[last_obj_key] = obj_id
            segments.append(f"{cls}[{attr}][{_invocation_counters[key]}]")

        return "/".join(segments) + "/" if segments else ""

    # ── Backward fallback: use op_name directly ───────────────────────────────
    return _backward_hierarchy_from_op(op_name)
