"""Shared helpers for compile-only (device-free) tests.

Compile-only mode compiles an FX graph to a NEFF without a Neuron device or
runtime. There are two ways to turn it on, and every compile-only test should
cover BOTH because they flip the same process-global flag through different
doors:

1. The ``torch_neuronx.compile.compile_only()`` context manager, which calls
   ``_C._set_compile_only(True)`` at runtime.
2. The ``TORCH_NEURONX_COMPILE_ONLY=1`` environment variable, which the C++
   ``is_compile_only()`` resolves lazily on first read (NeuronDevice.cpp).

The env-var path can only be exercised in a FRESH process: the C++ flag
(``g_compile_only``) is resolved once and cached for the lifetime of the
process, so setting the env var after import has no effect. ``run_in_subprocess``
below runs a snippet in a clean interpreter with the env applied, which is the
only faithful way to test that path.

These tests deliberately do NOT gate on ``torch.neuron.is_available()``: in
compile-only mode ``is_available()`` is False (the runtime is never started),
so gating on it would skip the tests exactly when they should run. They gate on
the compiler (``neuronxcc``) instead, since lowering to a NEFF needs it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

COMPILE_ONLY_TARGET = "trn2"

# NRT funtime instance for device-free nrt_init.
COMPILE_ONLY_FAKE_INSTANCE = "trn2.48xlarge"

# Logical NeuronCore config for the target. Needed at import: kernel registration
COMPILE_ONLY_LOGICAL_NC = "2"


def compile_only_platform_env() -> dict:
    """Env needed to compile device-free for COMPILE_ONLY_TARGET.

    - NEURON_PLATFORM_TARGET_OVERRIDE / NEURON_LOGICAL_NC_CONFIG: make the
      platform a compile input so no live instance is queried (needed even at
      import, for kernel registration).
    - TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS: the compilation_job context
      rmtree's its job dir on exit unless this is set (cache_utils.cleanup), so
      any test that asserts on on-disk StableHLO/NEFF files must set it. The NEFF
      bytes themselves live in the in-process cache regardless (neff_bytes()).
    """
    return {
        "NEURON_PLATFORM_TARGET_OVERRIDE": COMPILE_ONLY_TARGET,
        "NEURON_LOGICAL_NC_CONFIG": COMPILE_ONLY_LOGICAL_NC,
        "NEURON_RT_FAKE_INSTANCE_TYPE": COMPILE_ONLY_FAKE_INSTANCE,
        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
    }


# Skip if neuronx-cc is not installed.
requires_compiler = pytest.mark.skipif(
    importlib.util.find_spec("neuronxcc") is None,
    reason="neuronx-cc (neuronxcc) not installed; required to lower to a NEFF",
)


def count_artifacts(debug_dir: str) -> tuple[int, int]:
    """Return (num_stablehlo, num_neff) under a TORCH_NEURONX_DEBUG_DIR tree.

    StableHLO files are named ``{model}_{ts}.stablehlo.mlir`` (see
    cache_utils.CompilationJob.save_stablehlo); NEFFs end in ``.neff``.
    """
    num_stablehlo = 0
    num_neff = 0
    for _root, _dirs, files in os.walk(debug_dir):
        for f in files:
            if f.endswith(".stablehlo.mlir"):
                num_stablehlo += 1
            elif f.endswith(".neff"):
                num_neff += 1
    return num_stablehlo, num_neff


def neff_bytes_present(cache_key: str) -> bool:
    """True if the compilation cache holds resident NEFF bytes for cache_key."""
    import torch_neuronx

    info = torch_neuronx._C.get_neff_info(cache_key)
    return bool(info and info[1])


def stablehlo_op_digest(mlir_module) -> dict:
    """Return a structural digest of a StableHLO module: ``{op_name: count}``.

    Recursively walks every operation and tallies by name. This is a semantic
    fingerprint of the lowering — insensitive to cosmetic differences (SSA value
    numbering, whitespace, debug paths, attribute order) but sensitive to real
    divergence (a different op, a different count, a missing collective). Prefer
    this over hashing the serialized IR: robust, and a mismatch shows *which* ops
    differ rather than an opaque hash inequality.
    """
    from collections import Counter

    counts: Counter = Counter()

    def _walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    counts[inner.operation.name] += 1
                    _walk(inner.operation)

    _walk(mlir_module.operation)
    return dict(counts)


class _SubprocResult:
    """Minimal stand-in for CompletedProcess (stdout / stderr / returncode)."""

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_in_subprocess(body: str, *, env_overrides: dict | None = None, timeout: int = 600):
    """Run a Python snippet in a fresh interpreter with env applied.

    This is how the ``TORCH_NEURONX_COMPILE_ONLY`` env-var path is tested: the
    C++ compile-only flag is resolved once per process (NeuronDevice.cpp reads
    the env var on first ``is_compile_only()`` and caches it), so the env must
    be set before the interpreter starts. A fork (``pytest-forked``) would not
    work: it copies the already-resolved flag, and this module imports
    torch_neuronx at collection time. Only a genuinely new interpreter re-reads
    the env var. The snippet should print a final line ``RESULT:<...>`` that the
    caller parses; a non-zero exit fails the test with the child's output.

    Output is redirected to temp FILES, not pipes. ``torch.compile`` emits far
    more than a 64 KB pipe buffer holds; with ``capture_output=True`` the child
    blocks on write while the parent blocks on wait — a deadlock. A file has no
    such limit, so the child never blocks.

    Returns an object with ``.stdout`` / ``.stderr`` / ``.returncode``.
    """
    env = {
        **os.environ,
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",  # explicit import, don't autoload
    }
    if env_overrides:
        env.update(env_overrides)

    wrapped = (
        "import sys, traceback\n"
        "try:\n" + textwrap.indent(textwrap.dedent(body), "    ") + "\n"
        "    sys.stdout.flush(); sys.stderr.flush()\n"
        "except BaseException:\n"
        "    traceback.print_exc(); sys.stderr.flush()\n"
        "    sys.exit(1)\n"
    )

    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".out") as out_f:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                stdout=out_f,
                stderr=subprocess.STDOUT,  # merge; file has no buffer limit
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            out_f.seek(0)
            raise AssertionError(
                f"compile-only subprocess timed out after {timeout}s.\n"
                f"--- output ---\n{out_f.read()}"
            ) from e
        out_f.seek(0)
        output = out_f.read()

    if proc.returncode != 0:
        raise AssertionError(
            "compile-only subprocess failed "
            f"(returncode={proc.returncode}).\n--- output ---\n{output}"
        )
    return _SubprocResult(proc.returncode, output, output)


def parse_result_line(stdout: str) -> str:
    """Extract the value from the last ``RESULT:<value>`` line of subprocess stdout."""
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("RESULT:"):
            return line[len("RESULT:") :]
    raise AssertionError(f"no RESULT: line in subprocess stdout:\n{stdout}")
