"""
pytest configuration shared by tests under tests/python_ops/.

Provides the ``@pytest.mark.int64_mode("...")`` marker and an autouse fixture
that synchronises ``TORCH_NEURONX_INT64_MODE`` across the layers it lives in:

  - the env var read by ``get_int64_mode()`` and ``global_version_hash()``
  - the Python ``get_int64_mode`` / ``Int64Policy.get`` lru_caches
  - the C++ mode storage (``DispatchUtils.h`` GetInt64Mode), via
    ``_C._set_int64_mode_for_testing``

Without this fixture, switching modes in a single pytest process leaves the
C++ side stuck on whatever mode was active at module import — which causes
DowncastArgs to disagree with the Python-side cache key and load NEFFs
compiled for the wrong dtype.

Valid modes are the unified taxonomy: ``cpu_bounce``, ``native`` (default,
per-op cascade), ``packed``, ``native_strict``.

Mode selection precedence (highest to lowest):
    1. ``@pytest.mark.int64_mode("...")`` marker on the test/class/module.
    2. ``TORCH_NEURONX_INT64_MODE`` env var inherited from the shell.
    3. Default ``"native"``.
"""

import os

import pytest

_DEFAULT_INT64_MODE = "native"
_VALID_INT64_MODES = ("cpu_bounce", "native", "packed", "native_strict", "cpu_fallback")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "int64_mode(mode): pin TORCH_NEURONX_INT64_MODE for this test/class. "
        "Mode must be one of 'cpu_bounce', 'native', 'packed', 'native_strict', "
        "'cpu_fallback'. When unset, the autouse fixture inherits the value from "
        "the shell, falling back to 'native' if neither is set.",
    )


def _sync_cpp_mode(mode: str) -> None:
    """Override the C++ int64 mode + re-push op tables (best effort)."""
    from torch_neuronx.int64_policy import init_cpp_policy

    try:
        from torch_neuronx import _C

        if hasattr(_C, "_set_int64_mode_for_testing"):
            _C._set_int64_mode_for_testing(mode)
    except (ImportError, AttributeError):
        pass
    init_cpp_policy()


def _resync_cpp_mode_from_env() -> None:
    from torch_neuronx.int64_policy import init_cpp_policy

    try:
        from torch_neuronx import _C

        if hasattr(_C, "_reset_int64_mode_from_env"):
            _C._reset_int64_mode_from_env()
    except (ImportError, AttributeError):
        pass
    init_cpp_policy()


@pytest.fixture(autouse=True)
def _apply_int64_mode(request, monkeypatch):
    """Synchronise int64 mode across Python policy, hash cache, and C++ policy.

    Resolution order:
      1. ``@pytest.mark.int64_mode("...")`` marker — overrides everything.
      2. ``TORCH_NEURONX_INT64_MODE`` env var from the shell.
      3. ``"native"`` — final fallback.
    """
    from torch_neuronx.int64_policy import reset_int64_mode_cache
    from torch_neuronx.utils import global_version_hash

    marker = request.node.get_closest_marker("int64_mode")
    if marker is not None:
        mode = marker.args[0]
    else:
        mode = os.environ.get("TORCH_NEURONX_INT64_MODE", _DEFAULT_INT64_MODE)

    if mode not in _VALID_INT64_MODES:
        raise pytest.UsageError(
            f"Invalid int64_mode {mode!r}. Valid: {', '.join(_VALID_INT64_MODES)}."
        )

    monkeypatch.setenv("TORCH_NEURONX_INT64_MODE", mode)
    reset_int64_mode_cache()
    global_version_hash.cache_clear()
    _sync_cpp_mode(mode)

    yield

    # Teardown: monkeypatch reverts the env var; reset the Python + hash caches
    # and re-sync the C++ mode from the (reverted) env so the next test starts
    # from a clean baseline.
    reset_int64_mode_cache()
    global_version_hash.cache_clear()
    _resync_cpp_mode_from_env()
