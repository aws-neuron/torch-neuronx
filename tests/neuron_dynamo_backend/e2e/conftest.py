"""Shared fixtures for the compile-only e2e tests.

PTN no longer maps a compile target to NRT's funtime instance; the compile-only
driver must set NEURON_RT_FAKE_INSTANCE_TYPE (and the platform/LNC overrides) so
nrt_init runs device-free. This autouse fixture is that driver for the e2e suite.
"""

import os

import pytest

from tests.neuron_dynamo_backend.compile_only_utils import compile_only_platform_env


@pytest.fixture(autouse=True)
def _compile_only_platform_env():
    """Apply the compile-only platform env (target/LNC + funtime fake instance)
    per test, restoring prior values after."""
    prev = {k: os.environ.get(k) for k in compile_only_platform_env()}
    os.environ.update(compile_only_platform_env())
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
