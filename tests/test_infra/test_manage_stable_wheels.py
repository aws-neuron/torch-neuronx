import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "manage_stable_wheels.py"
SPEC = importlib.util.spec_from_file_location("manage_stable_wheels", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_asset(asset_id, name, updated_at):
    return {
        "id": asset_id,
        "name": name,
        "size": 10,
        "digest": f"sha256:{asset_id:064x}",
        "created_at": updated_at,
        "updated_at": updated_at,
        "browser_download_url": f"https://example.test/{name}",
    }


def test_select_current_assets_uses_latest_matching_asset():
    wheel = "torch_neuronx-2.14-cp313-cp313-linux_x86_64.whl"
    matrix = {"include": [{"wheel": wheel}]}
    old = make_asset(1, f"old-{wheel}", "2026-09-20T00:00:00Z")
    current = make_asset(2, wheel, "2026-09-21T00:00:00Z")

    selected, selected_ids = MODULE.select_current_assets(matrix, [old, current])

    assert selected[wheel] == current
    assert selected_ids == {2}


def test_select_current_assets_rejects_missing_asset():
    matrix = {"include": [{"wheel": "missing.whl"}]}

    with pytest.raises(ValueError, match="does not contain missing.whl"):
        MODULE.select_current_assets(matrix, [])


def test_select_stale_assets_requires_initial_manifest():
    matrix = {"include": [{"wheel": "current.whl"}]}
    current = make_asset(1, "current.whl", "2026-09-21T00:00:00Z")

    with pytest.raises(ValueError, match="one-time cleanup"):
        MODULE.select_stale_assets(matrix, [current], {1})


def test_select_stale_assets_rejects_large_backlog():
    matrix = {"include": [{"wheel": "current.whl"}]}
    current = make_asset(1, "current.whl", "2026-09-21T00:00:00Z")
    manifest = make_asset(2, MODULE.MANIFEST_NAME, "2026-09-21T00:00:00Z")
    stale = [make_asset(index, f"old-{index}.whl", "2026-09-20T00:00:00Z") for index in range(3, 6)]

    with pytest.raises(ValueError, match="unexpected wheel backlog"):
        MODULE.select_stale_assets(matrix, [current, manifest, *stale], {1})


def test_select_stale_assets_returns_previous_set():
    matrix = {"include": [{"wheel": "current.whl"}]}
    current = make_asset(1, "current.whl", "2026-09-21T00:00:00Z")
    manifest = make_asset(2, MODULE.MANIFEST_NAME, "2026-09-21T00:00:00Z")
    stale = make_asset(3, "old.whl", "2026-09-20T00:00:00Z")

    assert MODULE.select_stale_assets(matrix, [current, manifest, stale], {1}) == [stale]


def test_build_manifest_includes_current_wheel_metadata():
    wheel_name = "torch_neuronx-2.14-cp313-cp313-linux_x86_64.whl"
    matrix = {
        "include": [
            {
                "id": "py313-ptstable",
                "python_version": "3.13",
                "pytorch_channel": "stable",
                "pytorch_version": "2.14.0",
                "wheel": wheel_name,
            }
        ]
    }
    asset = make_asset(1, wheel_name, "2026-09-21T00:00:00Z")

    manifest = MODULE.build_manifest(
        matrix,
        {wheel_name: asset},
        "owner/repo",
        "abc123",
        "42",
        "https://example.test/run/42",
        "2026-09-21T00:00:00Z",
    )

    assert manifest["source_commit"] == "abc123"
    assert manifest["wheels"][0] == {
        "id": "py313-ptstable",
        "python_version": "3.13",
        "pytorch_channel": "stable",
        "pytorch_version": "2.14.0",
        "filename": wheel_name,
        "sha256": f"{1:064x}",
        "size": 10,
        "url": f"https://example.test/{wheel_name}",
    }
