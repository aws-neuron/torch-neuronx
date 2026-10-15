"""Install and manage the module AutoCompile integration."""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
import threading
import warnings

import torch

_mac = None
_module_auto_compile_patch_lock = threading.RLock()
_module_auto_compile_patches_active = False
_module_auto_compile_integration_initialized = False
_coerce_tangent_patch_installed = False
_coerce_tangent_patch_owned = False
_coerce_tangent_patch_value = None


_original_module_compile = torch.nn.Module.compile
_original_module_to = torch.nn.Module.to
_original_module_to_empty = torch.nn.Module.to_empty
_original_module_neuron = getattr(torch.nn.Module, "neuron", None)
_original_module_getstate = torch.nn.Module.__getstate__
_optimize_ddp_mode_owner = None
_optimize_ddp_mode_attribute = None
_original_get_optimize_ddp_mode = None
_OptimizedModule = None
_original_optimized_module_init = None
_NullDecorator = None
_original_null_decorator_call = None


@functools.wraps(_original_module_compile)
def _patched_module_compile(self, *args, **kwargs):
    if _module_auto_compile_patches_active:
        with contextlib.suppress(Exception):
            _mac.record_explicit_module_compile(self)
    return _original_module_compile(self, *args, **kwargs)


def _patched_get_optimize_ddp_mode():
    if _mac._isolated_leaf_compile_active():
        return "no_optimization"
    return _original_get_optimize_ddp_mode()


# torch.compile(module) does not route through Module.compile: it wraps the module
# in an OptimizedModule. Both spellings are an explicit user compile, so both must
# relinquish. Relinquishing here, instead of on the first traced call, matters
# because our bootstrap hooks must be off the subtree before Dynamo guards on
# _forward_pre_hooks and _forward_hooks. Their presence keeps
# nn.Module._call_impl on its hook slow path. This reorders lifted placeholders
# and multiplies guard variants until fullgraph=True callers hit recompile_limit.
# Patching this constructor avoids patching global torch.compile or
# torch.compiler.compile.
def _patched_optimized_module_init(self, mod, *args, **kwargs):
    if _module_auto_compile_patches_active and isinstance(mod, torch.nn.Module):
        with contextlib.suppress(Exception):
            _mac.record_explicit_module_compile(mod)
    return _original_optimized_module_init(self, mod, *args, **kwargs)


def _patched_null_decorator_call(self, fn):
    if _module_auto_compile_patches_active and isinstance(fn, torch.nn.Module):
        with contextlib.suppress(Exception):
            _mac.record_explicit_module_compile(fn)
    return _original_null_decorator_call(self, fn)


def _parse_module_to_device(*args, **kwargs):
    try:
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
    except Exception:
        return None
    if device is None:
        return None
    return torch.device(device)


@functools.wraps(_original_module_to)
def _patched_module_to(self, *args, **kwargs):
    with _mac._internal_module_compile_scope():
        result = _original_module_to(self, *args, **kwargs)
    if (
        _module_auto_compile_patches_active
        and _mac.module_auto_compile_enabled()
        and not _mac._internal_module_compile_active()
    ):
        try:
            _mac.observe_module_tree_after_to(
                result if isinstance(result, torch.nn.Module) else self,
                target_device=_parse_module_to_device(*args, **kwargs),
            )
        except Exception as error:
            _mac._disable_feature(
                f"discovery failed in Module.to ({type(error).__name__}: {error})"
            )
    return result


@functools.wraps(_original_module_to_empty)
def _patched_module_to_empty(self, *args, **kwargs):
    with _mac._internal_module_compile_scope():
        result = _original_module_to_empty(self, *args, **kwargs)
    if (
        _module_auto_compile_patches_active
        and _mac.module_auto_compile_enabled()
        and not _mac._internal_module_compile_active()
    ):
        try:
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            _mac.observe_module_tree_after_to(
                result if isinstance(result, torch.nn.Module) else self,
                target_device=None if device is None else torch.device(device),
                recurse=kwargs.get("recurse", True),
            )
        except Exception as error:
            _mac._disable_feature(
                f"discovery failed in Module.to_empty ({type(error).__name__}: {error})"
            )
    return result


def _patched_module_neuron(self, *args, **kwargs):
    with _mac._internal_module_compile_scope():
        result = _original_module_neuron(self, *args, **kwargs)
    if (
        _module_auto_compile_patches_active
        and _mac.module_auto_compile_enabled()
        and not _mac._internal_module_compile_active()
    ):
        try:
            _mac.observe_module_tree_after_to(
                result if isinstance(result, torch.nn.Module) else self,
                target_device=torch.device("neuron"),
            )
        except Exception as error:
            _mac._disable_feature(
                f"discovery failed in Module.neuron ({type(error).__name__}: {error})"
            )
    return result


if _original_module_neuron is not None:
    _patched_module_neuron = functools.wraps(_original_module_neuron)(_patched_module_neuron)


@functools.wraps(_original_module_getstate)
def _patched_module_getstate(self):
    state = _original_module_getstate(self)
    return _mac._sanitize_module_getstate(self, state)


_MODULE_PATCHES = ()
_applied_module_patches = []


def _install_module_patches() -> None:
    if _applied_module_patches:
        return
    for owner, attribute, original, _replacement in _MODULE_PATCHES:
        if getattr(owner, attribute) is not original:
            owner_name = getattr(owner, "__name__", type(owner).__name__)
            owner_module = getattr(owner, "__module__", None)
            qualified_owner = (
                f"{owner_module}.{owner_name}" if owner_module is not None else owner_name
            )
            raise RuntimeError(f"{qualified_owner}.{attribute} was replaced by another patch")
    for owner, attribute, original, replacement in _MODULE_PATCHES:
        setattr(owner, attribute, replacement)
        _applied_module_patches.append((owner, attribute, original, replacement))


def _restore_module_patches() -> None:
    for owner, attribute, original, replacement in reversed(_applied_module_patches):
        if getattr(owner, attribute) is replacement:
            setattr(owner, attribute, original)
    _applied_module_patches.clear()


def _install_coerce_tangent_patch() -> None:
    """Let Neuron tensors coerce to AsyncCollectiveTensor in a compiled TP backward.

    AOT Autograd's backward expects AsyncCollectiveTensor tangents but gets plain
    torch.Tensor because the compiled backward strips the subclass. Return self
    because the asynchronous behavior is not relevant for gradients. Remove this
    compatibility shim when PyTorch preserves AsyncCollectiveTensor metadata
    through compiled backward. The hasattr guard preserves a stock implementation.
    """
    global _coerce_tangent_patch_installed, _coerce_tangent_patch_owned
    global _coerce_tangent_patch_value
    if _coerce_tangent_patch_installed:
        return
    if hasattr(torch.Tensor, "__coerce_same_metadata_as_tangent__"):
        _coerce_tangent_patch_installed = True
        return

    def _tensor_coerce_tangent(self, expected_metadata, expected_type=None):
        # Keep this process-wide method inactive outside Neuron AutoCompile.
        if not _mac.module_auto_compile_enabled() or self.device.type != "neuron":
            return None
        try:
            from torch.distributed._functional_collectives import (
                AsyncCollectiveTensor,
            )

            if expected_type is AsyncCollectiveTensor:
                return self
        except ImportError:
            # Some PyTorch builds omit this optional internal API.
            return None
        return None

    torch.Tensor.__coerce_same_metadata_as_tangent__ = _tensor_coerce_tangent
    _coerce_tangent_patch_value = _tensor_coerce_tangent
    _coerce_tangent_patch_owned = True
    _coerce_tangent_patch_installed = True


def _restore_coerce_tangent_patch() -> None:
    global _coerce_tangent_patch_installed, _coerce_tangent_patch_owned
    global _coerce_tangent_patch_value
    if not _coerce_tangent_patch_installed:
        return
    if _coerce_tangent_patch_owned and (
        getattr(torch.Tensor, "__coerce_same_metadata_as_tangent__", None)
        is _coerce_tangent_patch_value
    ):
        del torch.Tensor.__coerce_same_metadata_as_tangent__
    _coerce_tangent_patch_installed = False
    _coerce_tangent_patch_owned = False
    _coerce_tangent_patch_value = None


def _module_auto_compile_requested() -> bool:
    value = (os.environ.get("TORCH_NEURONX_MODULE_AUTO_COMPILE") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _load_module_auto_compile():
    global _mac
    if _mac is None:
        _mac = importlib.import_module("torch_neuronx.module_auto_compile")
    return _mac


def _initialize_module_auto_compile_integration():
    global _MODULE_PATCHES, _NullDecorator, _OptimizedModule
    global _module_auto_compile_integration_initialized
    global _optimize_ddp_mode_attribute, _optimize_ddp_mode_owner
    global _original_get_optimize_ddp_mode, _original_null_decorator_call
    global _original_optimized_module_init

    with _module_auto_compile_patch_lock:
        if _module_auto_compile_integration_initialized:
            return _mac

        mac = _load_module_auto_compile()
        dynamo_utils = getattr(torch._dynamo, "utils", None)
        if callable(getattr(dynamo_utils, "get_optimize_ddp_mode", None)):
            _optimize_ddp_mode_owner = dynamo_utils
            _optimize_ddp_mode_attribute = "get_optimize_ddp_mode"
        elif callable(getattr(torch._dynamo.config, "_get_optimize_ddp_mode", None)):
            _optimize_ddp_mode_owner = torch._dynamo.config
            _optimize_ddp_mode_attribute = "_get_optimize_ddp_mode"
        _original_get_optimize_ddp_mode = (
            getattr(_optimize_ddp_mode_owner, _optimize_ddp_mode_attribute)
            if _optimize_ddp_mode_owner is not None
            else None
        )

        try:
            from torch._dynamo.eval_frame import OptimizedModule
        except Exception:
            _OptimizedModule = None
            _original_optimized_module_init = None
        else:
            _OptimizedModule = OptimizedModule
            _original_optimized_module_init = OptimizedModule.__init__

        try:
            from torch._dynamo.eval_frame import _NullDecorator as NullDecorator
        except Exception:
            _NullDecorator = None
            _original_null_decorator_call = None
        else:
            _NullDecorator = NullDecorator
            _original_null_decorator_call = NullDecorator.__call__

        patches = [
            (torch.nn.Module, "compile", _original_module_compile, _patched_module_compile),
            (torch.nn.Module, "to", _original_module_to, _patched_module_to),
            (torch.nn.Module, "to_empty", _original_module_to_empty, _patched_module_to_empty),
            (
                torch.nn.Module,
                "__getstate__",
                _original_module_getstate,
                _patched_module_getstate,
            ),
        ]
        if _original_module_neuron is not None:
            patches.append(
                (
                    torch.nn.Module,
                    "neuron",
                    _original_module_neuron,
                    _patched_module_neuron,
                )
            )
        if _original_get_optimize_ddp_mode is not None:
            patches.append(
                (
                    _optimize_ddp_mode_owner,
                    _optimize_ddp_mode_attribute,
                    _original_get_optimize_ddp_mode,
                    _patched_get_optimize_ddp_mode,
                )
            )
        if _OptimizedModule is not None:
            patches.append(
                (
                    _OptimizedModule,
                    "__init__",
                    _original_optimized_module_init,
                    _patched_optimized_module_init,
                )
            )
        if _NullDecorator is not None:
            patches.append(
                (
                    _NullDecorator,
                    "__call__",
                    _original_null_decorator_call,
                    _patched_null_decorator_call,
                )
            )
        _MODULE_PATCHES = tuple(patches)
        _module_auto_compile_integration_initialized = True
        return mac


def install() -> None:
    """Install the module patches when the user enables AutoCompile.

    Raises:
        RuntimeError: If a required Dynamo hook is unavailable, another patch owns a target,
            or an active installation loses ownership of one of its patches.
    """
    global _module_auto_compile_patches_active
    if not _module_auto_compile_requested():
        return
    mac = _initialize_module_auto_compile_integration()
    if not mac._module_auto_compile_requested():
        return
    if _original_get_optimize_ddp_mode is None:
        raise RuntimeError("Dynamo optimize-DDP mode getter is unavailable")
    with _module_auto_compile_patch_lock:
        if _module_auto_compile_patches_active:
            owned = all(
                getattr(owner, attribute) is replacement
                for owner, attribute, _original, replacement in _MODULE_PATCHES
            ) and (
                not _coerce_tangent_patch_owned
                or getattr(torch.Tensor, "__coerce_same_metadata_as_tangent__", None)
                is _coerce_tangent_patch_value
            )
            if owned:
                _mac._set_installation_complete(True)
                return
            _mac._set_installation_complete(False)
            _module_auto_compile_patches_active = False
            _mac._disable_feature("lost ownership of an installed patch")
            _restore_coerce_tangent_patch()
            _restore_module_patches()
            raise RuntimeError("module auto-compile lost ownership of an installed patch")

        _mac._set_installation_complete(False)
        try:
            _install_module_patches()
            _install_coerce_tangent_patch()
        except Exception:
            _mac._set_installation_complete(False)
            _restore_coerce_tangent_patch()
            _restore_module_patches()
            raise
        _module_auto_compile_patches_active = True
        _mac._set_installation_complete(True)


def uninstall() -> None:
    """Remove the module patches for tests."""
    global _module_auto_compile_patches_active
    if _mac is None:
        _module_auto_compile_patches_active = False
        return
    with _module_auto_compile_patch_lock:
        _mac._set_installation_complete(False)
        _module_auto_compile_patches_active = False
        _mac._reset_state()
        _restore_module_patches()
        _restore_coerce_tangent_patch()
        _mac._reset_feature_disabled()


def handle_install_failure(error: Exception) -> None:
    """Restore the eager path after an installation failure.

    Args:
        error (Exception): Installation error included in the warning or disable reason.
    """
    _restore_coerce_tangent_patch()
    _restore_module_patches()
    if _mac is None:
        warnings.warn(
            "Module AutoCompile setup failed. "
            f"AutoCompile is disabled: {type(error).__name__}: {error}",
            stacklevel=2,
        )
        return
    _mac._set_installation_complete(False)
    _mac._disable_feature(f"patch installation failed ({type(error).__name__}: {error})")


def handle_child_after_fork(*, hard_disable: bool) -> None:
    """Reset AutoCompile state in a fork child.

    Args:
        hard_disable (bool): Whether to permanently disable AutoCompile in the child.
    """
    global _module_auto_compile_patch_lock, _module_auto_compile_patches_active
    if _mac is None:
        return

    # A fork copies locks and module records from the parent. Recreate and clear them.
    _module_auto_compile_patch_lock = threading.RLock()
    try:
        _mac._after_fork(hard_disable=hard_disable)
    except Exception as error:
        # A child must not execute an inherited compiled callable after reset fails.
        _mac._INSTALLATION_COMPLETE = False
        _mac._FEATURE_HARD_DISABLED = True
        _module_auto_compile_patches_active = False
        with contextlib.suppress(Exception):
            _mac._reset_state()
        with contextlib.suppress(Exception):
            _restore_coerce_tangent_patch()
        with contextlib.suppress(Exception):
            _restore_module_patches()
        warnings.warn(
            "Module AutoCompile child reset failed. "
            "AutoCompile is disabled in this child: "
            f"{type(error).__name__}: {error}",
            stacklevel=2,
        )
