"""Per-op error handlers that turn a device fault into the exact op-specific
exception, using only the metadata already carried by the ``OperationContext``.

An aten op may register an error handler for one of its faults (via
``register_aten(error_handler=...)``). On a matching device fault (currently an
out-of-bounds index, NRT_EXEC_OOB), the C++ error path invokes the op's handler
via :func:`_invoke`, passing a lazy context (``_C.OpErrorContext``) over the
faulting op's inputs. The handler pulls only what it needs — ``ctx.shape(i)`` and
``ctx.scalar(i)`` come from metadata with no copy; ``ctx.tensor(i)`` /
``ctx.tensor_list(i)`` copy just that argument to CPU — and raises the precise
op-specific exception, WITHOUT re-running the op on the device or on CPU.

Handlers are registered once, at import, keyed by aten op name. There is no
per-call state and nothing captures/pins tensors, so this cannot leak op inputs
across calls. It is a general mechanism (not OOB-specific); today's users are
the index ops in torch_mlir/ops/indexing.py.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch_neuronx._C as _C

_lock = threading.Lock()
# aten op name -> handler(ctx). Populated at import.
_handlers: dict[str, Callable] = {}


def register(op_name: str, handler: Callable) -> None:
    """Register an ATen error handler for device faults.

    Args:
        op_name (str): ATen operation name used as the registry key.
        handler (Callable): Callback receiving the faulting operation's lazy context.

    Note:
        The callback must raise the precise exception only when its fault applies and must not
        mutate the context or inputs. Registration replaces any handler for the same name.
    """
    with _lock:
        _handlers[op_name] = handler


def _invoke(op_name: str, ctx) -> None:
    """Invoked from the C++ error path when ``op_name`` faults, with a lazy
    ``OpErrorContext`` over the op's inputs. Looks up and runs the handler; any
    exception it raises propagates as the op's precise error. No-ops if no
    handler is registered for ``op_name``."""
    with _lock:
        handler = _handlers.get(op_name)
    if handler is not None:
        handler(ctx)


# Register the invoker with the C++ error path once, at import. Guarded so
# importing against an older _C.so (without the binding) degrades gracefully.
_register_invoker = getattr(_C, "_register_error_handler_invoker", None)
if _register_invoker is not None:
    _register_invoker(_invoke)
