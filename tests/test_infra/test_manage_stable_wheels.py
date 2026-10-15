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


def test_select_stale_assets_deletes_nothing_without_a_manifest():
    matrix = {"include": [{"wheel": "current.whl"}]}
    current = make_asset(1, "current.whl", "2026-09-21T00:00:00Z")
    previous = make_asset(2, "previous.whl", "2026-09-20T00:00:00Z")

    assert MODULE.select_stale_assets(matrix, [current, previous], {1}) == []


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


def test_update_release_tag_moves_and_verifies_tag(monkeypatch):
    calls = []
    reads = iter([{"object": {"sha": "old123"}}, {"object": {"sha": "abc123"}}])

    def gh_json(*args):
        calls.append(args)
        return next(reads)

    monkeypatch.setattr(MODULE, "gh_json", gh_json)

    MODULE.update_release_tag("owner/repo", "stable-wheels", "abc123")

    assert calls == [
        ("api", "repos/owner/repo/git/ref/tags/stable-wheels"),
        (
            "api",
            "--method",
            "PATCH",
            "repos/owner/repo/git/refs/tags/stable-wheels",
            "-f",
            "sha=abc123",
            "-F",
            "force=true",
        ),
    ]


def test_update_release_tag_keeps_a_tag_that_already_matches(monkeypatch):
    calls = []

    def gh_json(*args):
        calls.append(args)
        return {"object": {"sha": "abc123"}}

    monkeypatch.setattr(MODULE, "gh_json", gh_json)

    MODULE.update_release_tag("owner/repo", "stable-wheels", "abc123")

    assert calls == [("api", "repos/owner/repo/git/ref/tags/stable-wheels")]


def test_update_release_tag_rejects_mismatched_commit(monkeypatch):
    monkeypatch.setattr(MODULE, "gh_json", lambda *args: {"object": {"sha": "old123"}})

    with pytest.raises(RuntimeError, match="points to old123"):
        MODULE.update_release_tag("owner/repo", "stable-wheels", "abc123")


@pytest.mark.parametrize(
    "ref_name, supported",
    [
        ("main", True),
        ("release-2.32", True),
        ("release-3.0", True),
        ("release-2.32.1", True),
        ("release-2.32-cidev-zhanxun-whl", False),
        ("release-foo", False),
        ("release-2", False),
        ("release-", False),
        ("mainline", False),
        ("stable-wheels", False),
        ("", False),
    ],
)
def test_is_supported_ref(ref_name, supported):
    assert MODULE.is_supported_ref(ref_name) is supported


def test_source_commit_uses_the_manual_promote_value(monkeypatch):
    monkeypatch.setenv("SOURCE_COMMIT", "source123")
    monkeypatch.setenv("GITHUB_SHA", "workflow456")

    assert MODULE.source_commit() == "source123"


def test_source_commit_uses_the_workflow_commit_by_default(monkeypatch):
    monkeypatch.delenv("SOURCE_COMMIT", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "workflow456")

    assert MODULE.source_commit() == "workflow456"


def test_source_run_uses_the_manual_promote_values(monkeypatch):
    monkeypatch.setenv("SOURCE_RUN_ID", "42")
    monkeypatch.setenv("SOURCE_RUN_URL", "https://example.test/run/42")
    monkeypatch.setenv("GITHUB_RUN_ID", "84")
    monkeypatch.setenv("GITHUB_RUN_URL", "https://example.test/run/84")

    assert MODULE.source_run() == ("42", "https://example.test/run/42")


def test_source_run_uses_the_workflow_run_by_default(monkeypatch):
    monkeypatch.delenv("SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("SOURCE_RUN_URL", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "84")
    monkeypatch.setenv("GITHUB_RUN_URL", "https://example.test/run/84")

    assert MODULE.source_run() == ("84", "https://example.test/run/84")


def test_update_release_keeps_the_release_title(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(MODULE, "run_gh", lambda *args: commands.append(args))
    manifest = tmp_path / MODULE.MANIFEST_NAME
    manifest.write_text("{}")
    checksums = tmp_path / MODULE.CHECKSUM_NAME
    checksums.write_text("")

    MODULE.update_release("owner/repo", 7, "release-2.32", manifest, checksums, "body")

    patch = next(command for command in commands if "PATCH" in command)
    assert not any(argument.startswith("name=") for argument in patch)
