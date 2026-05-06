"""Tests for NKI kernel module imports and implementation routing."""

import importlib
import inspect
import pkgutil
import warnings

import pytest

import torch_neuronx

# ---------------------------------------------------------------------------
# NKI kernel module import validation
# ---------------------------------------------------------------------------


def _get_nki_kernel_module_names():
    """Discover all module names under torch_neuronx.python_ops.nki_kernels."""
    package = importlib.import_module("torch_neuronx.python_ops.nki_kernels")
    modules = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=package.__path__,
        prefix=package.__name__ + ".",
        onerror=lambda x: pytest.fail(f"Failed to traverse subpackage: {x}"),
    ):
        if "__pycache__" in modname or "test" in modname:
            continue
        modules.append(modname)
    return modules


# Collected at import time — required by pytest parametrize.
NKI_KERNEL_MODULES = _get_nki_kernel_module_names()


@pytest.mark.parametrize("module_name", NKI_KERNEL_MODULES)
def test_nki_kernel_module_imports(module_name):
    """Each NKI kernel module must import without raising.

    Any import failure is a hard fail — if a module can't import, the
    runtime silently falls back to slow MLIR compilation with no
    actionable error.
    """
    importlib.import_module(module_name)


def test_nki_kernel_modules_discovered():
    """Fail if no modules found — prevents parametrize from passing vacuously with 0 cases."""
    assert (
        NKI_KERNEL_MODULES
    ), "No nki_kernels modules discovered; check package path and walk_packages."


def test_discover_all_implementations_no_nki_warnings():
    """discover_all_implementations() must not emit NKI kernel import warnings."""
    from torch_neuronx.python_ops.auto_registration import discover_all_implementations

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        discover_all_implementations()

    nki_warnings = [w for w in caught if "nki_kernels" in str(w.message)]
    assert nki_warnings == [], "NKI kernel import warnings:\n" + "\n".join(
        str(w.message) for w in nki_warnings
    )


# ---------------------------------------------------------------------------
# Generic NKI implementation validation via registry
# ---------------------------------------------------------------------------


def _iter_subclasses(cls):
    """Recursively yield all concrete (non-abstract) subclasses of cls."""
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            yield subclass
        yield from _iter_subclasses(subclass)


def _find_all_nki_impls():
    """Find all NKI impl classes from both Operation subclasses and @neuron_op registry."""
    from torch_neuronx.python_ops.auto_registration import (
        discover_all_implementations,
        get_registry_info,
    )
    from torch_neuronx.python_ops.base import Operation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discover_all_implementations()

    results = []
    seen = set()

    # Source 1: Operation subclasses — walk full hierarchy, skip abstract bases
    for subclass in _iter_subclasses(Operation):
        instance = subclass()
        for impl in instance._implementations:
            cls_name = impl.__class__.__name__
            if "NKI" not in cls_name or cls_name in seen:
                continue
            seen.add(cls_name)
            results.append(pytest.param(cls_name, impl.__class__.__module__, id=cls_name))

    # Source 2: @neuron_op registry (grouped_mm, contiguous, etc.)
    for _aten_name, reg_info in get_registry_info().items():
        for impl_class in reg_info.get("implementations", []):
            cls_name = impl_class.__name__
            if "NKI" not in cls_name or cls_name in seen:
                continue
            seen.add(cls_name)
            results.append(pytest.param(cls_name, impl_class.__module__, id=cls_name))

    return results


# Collected at import time — required by pytest parametrize.
ALL_NKI_IMPLS = _find_all_nki_impls()


def test_nki_impls_discovered():
    """Guard against silent false-green when discovery finds nothing to parametrize."""
    assert ALL_NKI_IMPLS, (
        "0 NKI impls discovered — parametrized tests would silently pass. "
        "Check Operation subclass hierarchy and get_registry_info() API."
    )


@pytest.mark.parametrize("nki_impl_name,module_path", ALL_NKI_IMPLS)
def test_nki_impl_class_is_importable(nki_impl_name, module_path):
    """Every registered NKI implementation class must be importable."""
    mod = importlib.import_module(module_path)
    assert (
        getattr(mod, nki_impl_name, None) is not None
    ), f"{nki_impl_name} not found in {module_path}"


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize("nki_impl_name,module_path", ALL_NKI_IMPLS)
def test_nki_impl_can_be_instantiated(nki_impl_name, module_path):
    """Every registered NKI implementation must instantiate without error."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, nki_impl_name)
    cls()
