"""Conftest for distributed tests - automatically marks all tests in this folder."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Add 'distributed' marker to all tests in the distributed folder."""
    distributed_marker = pytest.mark.multi_device

    for item in items:
        # Check if this test is in the distributed folder
        if "distributed" in str(item.fspath):
            item.add_marker(distributed_marker)


@pytest.fixture(autouse=True)
def override_main_fixtures():
    # Override any fixtures from main conftest
    os.environ["NEURON_RT_PORT"] = "2025"
