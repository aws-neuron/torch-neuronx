"""Scope module AutoCompile patches to this test package."""

import pytest


@pytest.fixture(scope="package", autouse=True)
def module_auto_compile_patches(_module_auto_compile_patch_scope_factory):
    with _module_auto_compile_patch_scope_factory():
        yield
