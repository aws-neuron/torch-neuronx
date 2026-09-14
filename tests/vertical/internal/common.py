"""Helpers shared by test_target_ops_coverage.py and
test_target_ops_compile_coverage.py.

Kept in the ``internal`` package (name not matching ``test_*``) so pytest
doesn't try to collect it as a test module. Extracting shared bits here lets
each test file build its own op_db without importing the other test module's
side effects at top level.

The op_db build is cached via ``functools.cache`` so the two test files
share a single build when both are collected in the same session.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
from functools import cache

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)


# ── Worst-case ranking when aggregating per-shape statuses into one CSV row ──

STATUS_RANK = {
    # Eager-path statuses
    "runs_on_neuron": 0,
    "cpu_fallback": 2,
    # Compile-path statuses
    "supported": 0,
    "graph_break": 2,
    # Shared (both paths)
    "no_input_sample": 1,
    # Sample is invalid
    "invalid_sample": 1,
    "unresolved": 2,
    "failed_lowering": 3,
    # Cold compile ran past the time limit (see compile_time_limit). Ranks as a
    # hard gap so one timed-out shape marks the whole op.
    "compile_timeout": 3,
}


# Per-shape cold-compile time limit in seconds. A single large op (for example a
# vocab-projection aten::gather) can take 30+ minutes to compile and stall an
# xdist worker for the rest of the run. The limit bounds one compile so the run
# continues. Set NEURON_COMPILE_TIMEOUT_S=0 to disable the limit.
COMPILE_TIMEOUT_S = int(os.environ.get("NEURON_COMPILE_TIMEOUT_S", "600"))


class CompileTimeoutError(Exception):
    """A single cold compile ran past COMPILE_TIMEOUT_S."""


@contextlib.contextmanager
def compile_time_limit(seconds: int = COMPILE_TIMEOUT_S):
    """Bound one cold compile to ``seconds``. Raise CompileTimeoutError on overrun.

    Two mechanisms run together, because the compile can run on different
    threads:

      - SIGALRM interrupts a compile that runs inline on the main thread, for
        example the eager coverage path.
      - A watchdog thread kills the neuronx-cc subprocess. The torch.compile
        backend submits the NEFF compile to a runtime background thread, so
        SIGALRM never reaches the Python handler on the main thread. A
        process-directed SIGALRM can also go to that background thread, where
        Python runs no handler, so the signal is lost. The watchdog does not
        depend on the compile thread. It kills the neuronx-cc subprocess, the
        wrapped call then raises, and this manager reports a CompileTimeoutError.

    A value of 0 or less disables the limit.
    """
    if seconds <= 0:
        yield
        return

    timed_out = threading.Event()
    done = threading.Event()

    def _watch():
        # Wait for the block to finish. On overrun, kill the compile subprocess.
        if not done.wait(seconds):
            timed_out.set()
            kill_compile_subprocesses()

    watcher = threading.Thread(target=_watch, name="neuron-compile-timeout", daemon=True)

    on_main = threading.current_thread() is threading.main_thread()
    previous = None
    if on_main:

        def _raise_timeout(signum, frame):
            raise CompileTimeoutError(f"compile exceeded {seconds}s")

        previous = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)

    watcher.start()
    try:
        yield
    except CompileTimeoutError:
        raise
    except Exception as exc:
        # The watchdog killed the compile subprocess, so the call raised.
        if timed_out.is_set():
            raise CompileTimeoutError(f"compile exceeded {seconds}s") from exc
        raise
    finally:
        done.set()
        if on_main:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        watcher.join(timeout=5)
    # The compile finished as the watchdog killed it. Report the overrun.
    if timed_out.is_set():
        raise CompileTimeoutError(f"compile exceeded {seconds}s")


def kill_compile_subprocesses():
    """Kill leftover neuronx-cc compile subprocesses of this worker.

    A timed-out compile leaves the neuronx-cc subprocess running, which keeps a
    NeuronCore busy. Walk the /proc process tree and kill every descendant so
    the worker frees the core. Dependency-free (no psutil).
    """
    my_pid = os.getpid()
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))

    stack = list(children.get(my_pid, []))
    while stack:
        pid = stack.pop()
        stack.extend(children.get(pid, []))
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


# ── Data movement ────────────────────────────────────────────────────────────


def allocate_to_device(data, device):
    """Deep-clone tensors/lists/dicts to ``device``. Non-tensors pass through.

    Broadcast (zero-stride) tensors are re-expanded on the target device rather
    than cloned. A plain ``.clone()`` materializes an expanded view as a dense
    contiguous tensor (expanded dims overlap in storage, so ``clone`` can't
    preserve the zero stride), which would silently change the input. This
    matters for ``aten::scatter_add_``: its index is a 1-D row index broadcast
    to ``(n, H)`` with ``stride(1)==0``, and the static NKI scatter_add only
    routes to its fast kernel when that stride is 0. A materialized contiguous
    index falls back to the slow GPSIMD path.
    """
    if isinstance(data, torch.Tensor):
        if data.dim() > 0 and 0 in data.stride():
            # Move the compact base (unique elements), then re-broadcast to the
            # original expanded shape so zero strides survive the device move.
            base_shape = [
                s if st != 0 else 1 for s, st in zip(data.shape, data.stride(), strict=True)
            ]
            base = data.detach()[tuple(slice(0, s) for s in base_shape)].contiguous()
            # Downcast int64 -> int32 for the device copy (the static NKI
            # scatter_add index is int32). The CPU tensor stays int64, which
            # torch's CPU scatter_add_ requires for the reference.
            if base.dtype == torch.int64:
                base = base.to(torch.int32)
            return base.to(device).expand(data.shape)
        return data.detach().clone().to(device)
    if isinstance(data, list | tuple):
        return type(data)(allocate_to_device(item, device) for item in data)
    if isinstance(data, dict):
        return {k: allocate_to_device(v, device) for k, v in data.items()}
    return data


# ── Dispatch-key classification (drives `supported_by` column) ───────────────


def _dispatch_has(op_name: str, key) -> bool:
    clean = op_name.replace("aten::", "")
    base, _, overload = clean.partition(".")
    overload = overload or "default"
    try:
        pkt = getattr(torch.ops.aten, base, None)
        if pkt is None:
            return False
        op = getattr(pkt, overload, None)
        if op is None:
            return False
        schema = op._schema
        q = f"{schema.name}.{schema.overload_name}" if schema.overload_name else str(schema.name)
        return torch._C._dispatch_has_kernel_for_dispatch_key(q, key)
    except Exception:
        return False


def has_pu1(op_name: str) -> bool:
    """True if Neuron (PrivateUse1) has a registered kernel for this op."""
    return _dispatch_has(op_name, torch._C.DispatchKey.PrivateUse1)


def is_composite(op_name: str) -> bool:
    """True if the op has any Composite*Autograd kernel — meaning it doesn't
    have a direct backend kernel but decomposes into primitives at dispatch
    time. Covers the three distinct composite keys PyTorch uses in practice:

      - CompositeImplicitAutograd: implicit decomposition (e.g. ``aten::linear``).
      - CompositeExplicitAutograd: explicit decomposition for ops that keep
        gradient math.
      - CompositeExplicitAutogradNonFunctional: decomposition for in-place
        and out-variant ops (e.g. ``aten::add_.Tensor`` → ``aten::add.out``,
        ``aten::as_strided_scatter``). Ops registered *only* under this key
        were previously mis-classified as ``unknown``.
    """
    return (
        _dispatch_has(op_name, torch._C.DispatchKey.CompositeImplicitAutograd)
        or _dispatch_has(op_name, torch._C.DispatchKey.CompositeExplicitAutograd)
        or _dispatch_has(op_name, torch._C.DispatchKey.CompositeExplicitAutogradNonFunctional)
    )


def classify_op(op_name: str) -> str:
    if has_pu1(op_name):
        return "registered"
    if is_composite(op_name):
        return "composite"
    return "unknown"


# ── Exception classification ────────────────────────────────────────────────


def classify_error(exc_str: str) -> str:
    """Classify a dispatcher/runtime error string into an error **cause**:

      - ``"unsupported"``      — no kernel on the target backend (dispatcher
                                 "Could not run ... backend", NYI, "only
                                 supports CPU", op-lookup failure).
      - ``"failed_lowering"``  — a real compile/lowering/runtime failure
                                 (CPU fallback disabled, Neuron compile error,
                                 registration bug).
      - ``"pending_registry"`` — the test supplied inputs incompatible with
                                 the op's schema (type mismatch, wrong arg
                                 count, etc.); the op itself may still work.

    The returned label is an error *cause*, not a test-path status. Callers
    map cause → status in their own vocabulary (e.g. eager path maps
    ``unsupported`` → ``cpu_fallback``; compile path maps it → ``missing``).
    """
    if "Could not run" in exc_str and "backend" in exc_str:
        return "unsupported"
    if "NYI" in exc_str or "not yet implemented" in exc_str.lower():
        return "unsupported"
    if "op not found" in exc_str:
        return "unsupported"
    if "CPU fallback disabled" in exc_str:
        return "failed_lowering"
    if "_copy_from_and_resize" in exc_str:
        return "failed_lowering"
    if "got an unexpected keyword argument" in exc_str:
        return "failed_lowering"
    if "Expected PythonKernelHolder" in exc_str:
        return "failed_lowering"
    if "Compilation error occurred on Neuron" in exc_str:
        return "failed_lowering"
    if "only supports CPU" in exc_str:
        return "unsupported"
    # Type mismatches from wrong sample inputs — not a real failure
    if "Expected a value of type" in exc_str and "but instead found type" in exc_str:
        return "pending_registry"
    if "is missing value for argument" in exc_str:
        return "pending_registry"
    if "expected at most" in exc_str and "argument(s) but received" in exc_str:
        return "pending_registry"
    if "scalar tensor expected to be 0 dim" in exc_str:
        return "pending_registry"
    if "Unknown keyword argument" in exc_str:
        return "pending_registry"
    if "takes" in exc_str and "positional argument(s) but" in exc_str:
        return "pending_registry"
    if "specified both as positional and keyword" in exc_str:
        return "pending_registry"
    if "Expected out tensor to have dtype" in exc_str:
        return "pending_registry"
    if "Expected result to be of dtype" in exc_str:
        return "pending_registry"
    if "Dimension specified as" in exc_str and "tensor has no dimensions" in exc_str:
        return "pending_registry"
    if "list assignment index out of range" in exc_str:
        return "pending_registry"
    return "failed_lowering"


# ── Output-shape probe (feeds compute_roofline) ─────────────────────────────


def flatten_tensors(obj):
    """Return the tensors inside ``obj``, dropping non-tensors and any tensors
    with symbolic (non-int) dimensions that FakeTensorMode can emit for
    data-dependent outputs such as ``nonzero``."""
    tensors = (
        [obj]
        if isinstance(obj, torch.Tensor)
        else list(obj)
        if isinstance(obj, list | tuple)
        else []
    )
    out = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        try:
            [int(d) for d in t.shape]
        except Exception:
            return []
        out.append(t)
    return out


def count_input_elems(sample) -> int:
    """Sum of ``numel()`` across every tensor input of a ``SampleInput``.

    Returns the total element count over ``sample.input`` and each positional
    arg (unwrapping ``List[Tensor]`` / ``Tuple[Tensor]`` inputs used by
    ``_foreach_*`` and ``cat``/``stack``). Non-tensor args are ignored.
    Kwarg tensors are not counted because most Neuron ops keep large-data
    args positional; kwargs are typically scalars, axes, modes, or
    write-alias ``out=`` slots already accounted for via the positional
    inputs.

    Returns ``0`` on any failure. The caller (test aggregator) treats 0
    as "unable to identify" and emits an empty CSV cell. This keeps the
    column empty for scalar-only samples and for ops whose sample builder
    produced no real tensors (e.g. pending_registry skips).
    """
    try:
        total = 0
        for t in (sample.input, *sample.args):
            if isinstance(t, torch.Tensor):
                total += int(t.numel())
            elif isinstance(t, list | tuple):
                for x in t:
                    if isinstance(x, torch.Tensor):
                        total += int(x.numel())
        return total
    except Exception as e:
        logger.debug("count_input_elems failed: %s", e)
        return 0


def sample_inputs_have_nan(sample) -> bool:
    """Return True if a positional float input tensor holds a NaN.

    Scan ``sample.input`` and each positional arg. This is the same scope as
    ``count_input_elems``. Do not scan kwargs. The write-alias ``out=`` slot is
    an uninitialized buffer, and its bytes can read as NaN. A NaN input makes
    the correctness compare not meaningful, so the caller skips that check.
    """
    try:
        for t in (sample.input, *sample.args):
            items = t if isinstance(t, list | tuple) else (t,)
            for x in items:
                if isinstance(x, torch.Tensor) and x.is_floating_point() and torch.isnan(x).any():
                    return True
    except Exception as e:
        logger.debug("sample_inputs_have_nan failed: %s", e)
    return False


def probe_output_meta(op, sample):
    """Infer output metadata (shape/dtype) via FakeTensorMode — no real kernel
    run, no storage allocated, no mutation. Empty list on failure; callers
    treat that as 'fall back to input-size estimate' in ``compute_roofline``.
    """
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            out = op(sample.input, *sample.args, **sample.kwargs)
        return flatten_tensors(out)
    except Exception as e:
        logger.debug("fake probe failed for %s: %s", op, e)
        return []


# ── Op database (cached; shared across test modules) ────────────────────────


@cache
def get_neuron_op_db():
    """Build the full target-ops database. Cached so that
    test_target_ops_coverage.py and test_target_ops_compile_coverage.py share
    a single ~1360-op DB build when collected in the same pytest session."""
    from .neuron_op_db import build_neuron_op_db
    from .target_ops import TARGET_ATEN_OP_GROUP, TARGET_ATEN_OPS

    return build_neuron_op_db(
        target_ops=TARGET_ATEN_OPS,
        target_op_groups=TARGET_ATEN_OP_GROUP,
    )


def build_params(op_db):
    """Return the pytest.param list for a given op_db."""
    return [pytest.param(op_info, id=f"{op_info.op_name}[{op_info.op_group}]") for op_info in op_db]


def reset_cache_dirs(*paths) -> None:
    """Empty the given cache dirs in place so the next compile is cold.

    Clears contents (rather than rmtree-ing the dir itself) to avoid the
    classic ``[Errno 17] File exists`` race when the path appears as a file
    or when recreation collides with concurrent state. Missing dirs are
    created. A stray file at a dir path is replaced. All OSErrors are
    swallowed to debug-log — a failed cache reset means this op's TTF may
    be warm, not that the test should crash.

    Call between ops (or between shapes) to ensure each ``time_to_first_ms``
    measurement really is a cold compile, not a reuse of a previously-cached
    primitive NEFF from the same worker.
    """
    import shutil
    from pathlib import Path

    for p in paths:
        p = Path(p)
        try:
            if p.is_dir():
                for child in p.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        with contextlib.suppress(OSError):
                            child.unlink()
            elif p.exists():
                # A stray file where we expect a dir — replace it.
                with contextlib.suppress(OSError):
                    p.unlink()
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug("cache reset failed for %s: %s", p, e)
