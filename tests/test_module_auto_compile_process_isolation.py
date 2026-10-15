"""Process-level coverage for the module AutoCompile pytest fixture."""

import builtins
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

_ENABLE_ENV = "TORCH_NEURONX_MODULE_AUTO_COMPILE"
_DEBUG_ENV = "TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG"


def test_fixture_restores_environment_when_import_fails(
    monkeypatch, _module_auto_compile_patch_scope_factory
):
    original_import = builtins.__import__
    original_env = (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV))

    def fail_torch_neuronx_import(name, *args, **kwargs):
        if name == "torch_neuronx":
            raise ImportError("injected fixture import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_neuronx_import)
    with (
        pytest.raises(ImportError, match="injected fixture import failure"),
        _module_auto_compile_patch_scope_factory(),
    ):
        pass

    assert (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV)) == original_env


def test_fixture_restores_environment_when_uninstall_fails(
    monkeypatch, _module_auto_compile_patch_scope_factory
):
    fake_torch_neuronx = types.ModuleType("torch_neuronx")
    fake_torch_neuronx._module_auto_compile_patches_active = False
    fake_torch_neuronx._module_auto_compile_patches_are_active = (
        lambda: fake_torch_neuronx._module_auto_compile_patches_active
    )

    def install():
        fake_torch_neuronx._module_auto_compile_patches_active = True

    def uninstall():
        raise RuntimeError("injected fixture uninstall failure")

    fake_torch_neuronx._install_module_auto_compile_patches = install
    fake_torch_neuronx._uninstall_module_auto_compile_patches = uninstall
    monkeypatch.setitem(sys.modules, "torch_neuronx", fake_torch_neuronx)
    original_env = (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV))

    with (
        pytest.raises(RuntimeError, match="injected fixture uninstall failure"),
        _module_auto_compile_patch_scope_factory(),
    ):
        pass

    assert (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV)) == original_env


def test_feature_off_probe_after_package_fixture():
    import torch_neuronx

    integration = torch_neuronx._get_module_auto_compile_integration()
    assert _ENABLE_ENV not in os.environ
    assert _DEBUG_ENV not in os.environ
    assert not torch_neuronx._module_auto_compile_patches_are_active()
    # A fresh default-off process intentionally leaves the integration unloaded.
    if integration is not None:
        assert not integration._module_auto_compile_patches_active
        assert not integration._applied_module_patches
        for owner, attribute, original, _replacement in integration._MODULE_PATCHES:
            assert getattr(owner, attribute) is original
        assert not integration._coerce_tangent_patch_installed
        assert not integration._coerce_tangent_patch_owned
        assert integration._coerce_tangent_patch_value is None


@pytest.mark.parametrize("forked", (False, True), ids=("same-worker", "forked"))
def test_package_fixture_is_process_local(forked):
    if forked:
        pytest.importorskip("pytest_forked")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop(_ENABLE_ENV, None)
    env.pop(_DEBUG_ENV, None)
    original_compile = torch.nn.Module.compile
    original_env = (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV))

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
    ]
    if forked:
        command.append("--forked")
    command.extend(
        (
            "tests/module_auto_compile/test_fixture_scope.py",
            f"{__file__}::test_feature_off_probe_after_package_fixture",
        )
    )
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert torch.nn.Module.compile is original_compile
    assert (os.environ.get(_ENABLE_ENV), os.environ.get(_DEBUG_ENV)) == original_env


@pytest.mark.parametrize("enable_value", (None, "0"), ids=("absent", "explicit-zero"))
def test_default_off_import_does_not_load_or_patch_autocompile(enable_value):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if enable_value is None:
        env.pop(_ENABLE_ENV, None)
    else:
        env[_ENABLE_ENV] = enable_value
    env.pop(_DEBUG_ENV, None)

    script = r"""
import json
import os
import sys
import torch

originals = (
    torch.nn.Module.compile,
    torch.nn.Module.to,
    torch.nn.Module.to_empty,
    torch.nn.Module.__getstate__,
)
import torch_neuronx

fork_exit = None
if hasattr(os, "fork"):
    child = os.fork()
    if child == 0:
        child_ok = (
            "torch_neuronx._module_auto_compile_integration" not in sys.modules
            and "torch_neuronx.module_auto_compile" not in sys.modules
            and originals
            == (
                torch.nn.Module.compile,
                torch.nn.Module.to,
                torch.nn.Module.to_empty,
                torch.nn.Module.__getstate__,
            )
        )
        os._exit(0 if child_ok else 7)
    _, status = os.waitpid(child, 0)
    fork_exit = os.waitstatus_to_exitcode(status)

print(json.dumps({
    "integration_loaded": "torch_neuronx._module_auto_compile_integration" in sys.modules,
    "module_loaded": "torch_neuronx.module_auto_compile" in sys.modules,
    "methods_unchanged": originals == (
        torch.nn.Module.compile,
        torch.nn.Module.to,
        torch.nn.Module.to_empty,
        torch.nn.Module.__getstate__,
    ),
    "fork_exit": fork_exit,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "integration_loaded": False,
        "module_loaded": False,
        "methods_unchanged": True,
        "fork_exit": 0,
    }
