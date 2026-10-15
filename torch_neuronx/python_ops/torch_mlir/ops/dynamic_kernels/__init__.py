"""Dynamic-shape NKI kernel dispatch infrastructure.

Provides the ``DynamicNkiEntry`` registry (see ``registry.py``) that the
TorchMlirKernel dispatch path (``kernel.py::_try_dynamic_nki_fallback``) consults
to route an ATen op to a pre-compiled dynamic NKI NEFF instead of the default
MLIR/PyTorch lowering. Per-op adapters register themselves here when the feature
is enabled; this package ships the framework, adapters build on it.

Enabled by ``TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS=1``. With the flag off
(default) nothing here changes behaviour — the registry stays empty and the
dispatch path is inert.
"""

from torch_neuronx.utils import use_dynamic_nki_kernels

from .registry import get_dynamic_nki_usage

# Register dynamic NKI adapters when the feature is enabled. Per-op adapters are
# optional: the dispatch infrastructure (registry + kernel.py fast path) ships
# standalone and adapters register on top of it when present. Presence is checked
# via find_spec so a genuine import error *inside* an adapter still surfaces
# (rather than being swallowed as "adapter not shipped").
if use_dynamic_nki_kernels():
    import importlib.util

    if importlib.util.find_spec(".adapters", __name__) is not None:
        from .adapters import register_all

        register_all()
