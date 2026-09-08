"""Test the ModuScope artifact layout and file helpers."""

from __future__ import annotations

import errno

import moduscope_artifact_layout as artifact_layout
import pytest

from conftest import RUN_IDENTITY_FIELDS

INDIVIDUAL_FILE_PATHS = ("individual/G.0/a.safetensors", "SomeGrid.0/individual/x.json")
XLA_DUMP_FILE_PATHS = ("xla_dump/module_0000.hlo.pb", "individual/Cfg.0/xla_dump/m.txt")
MANIFEST_FILE_PATHS = ("manifests/hlo_manifest.json", "manifests/TitanTask/hlo_manifest.json")


def test_run_identity_uses_only_the_fields_added_to_rows(run_identity_env, monkeypatch, capsys):
    assert artifact_layout.read_run_identity() == RUN_IDENTITY_FIELDS
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("MODUSCOPE_GITHUB_RUN_ATTEMPT", "  ")
    assert artifact_layout.read_run_identity() == RUN_IDENTITY_FIELDS
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("MODUSCOPE_GITHUB_REPOSITORY", "  ")
    assert artifact_layout.read_run_identity() is None
    assert capsys.readouterr().out == (
        "RUN_IDENTITY=absent missing=MODUSCOPE_GITHUB_REPOSITORY; "
        + "rows get no workload_id or workload_link\n"
    )


@pytest.mark.parametrize("missing_variable", sorted(artifact_layout.IDENTITY_ENV_VARS))
@pytest.mark.parametrize("variable_value", [None, "  "], ids=["unset", "blank"])
def test_missing_or_blank_run_identity_variables_return_no_identity(
    run_identity_env, monkeypatch, capsys, missing_variable, variable_value
):
    monkeypatch.delenv(missing_variable) if variable_value is None else monkeypatch.setenv(
        missing_variable, variable_value
    )
    assert artifact_layout.read_run_identity() is None
    assert capsys.readouterr().out.startswith(f"RUN_IDENTITY=absent missing={missing_variable};")


def test_run_id_file_helpers_find_and_require_one_file(tmp_path):
    output_file = tmp_path / artifact_layout.run_id_filename("337", "grid_metrics.json")
    output_file.write_text("[]")
    assert artifact_layout.run_id_file_pattern("grid_metrics.json") == "*-grid_metrics.json"
    assert artifact_layout.find_run_id_file(tmp_path, "grid_metrics.json") == output_file
    assert artifact_layout.require_run_id_file(tmp_path, "grid_metrics.json") == output_file
    assert artifact_layout.find_run_id_file(tmp_path, "tokens.txt") is None
    with pytest.raises(FileNotFoundError, match=r"\*-tokens\.txt"):
        artifact_layout.require_run_id_file(tmp_path, "tokens.txt")


def test_run_id_file_lookup_rejects_multiple_matches(tmp_path):
    for run_id in ("337", "338"):
        (tmp_path / artifact_layout.run_id_filename(run_id, "grid_metrics.json")).write_text("[]")
    with pytest.raises(ValueError, match="multiple run-ID files"):
        artifact_layout.find_run_id_file(tmp_path, "grid_metrics.json")


def test_write_grid_uses_two_space_indent_and_removes_staging_file(tmp_path):
    grid_path = tmp_path / "e0" / "grid_metrics.json"
    artifact_layout.write_grid_rows(str(grid_path), [{"status": "pass"}])
    assert grid_path.read_text() == '[\n  {\n    "status": "pass"\n  }\n]'
    assert [child_path.name for child_path in grid_path.parent.iterdir()] == ["grid_metrics.json"]


@pytest.mark.parametrize(
    ("remove_directories", "removed_paths", "kept_paths", "expected_count"),
    [
        (
            lambda root: artifact_layout.remove_named_directories(root, "individual"),
            INDIVIDUAL_FILE_PATHS,
            ("grid_metrics.json", "xla_dump/m.mlir"),
            2,
        ),
        (
            lambda root: artifact_layout.remove_named_directories(
                root, artifact_layout.XLA_DUMP_DIR_NAME
            ),
            XLA_DUMP_FILE_PATHS,
            ("individual/Cfg.0/Cfg.0_metrics.json",),
            2,
        ),
        (artifact_layout.remove_compiler_manifests, MANIFEST_FILE_PATHS,
         ("configs/manifests/nested.json",), 1),
    ],
    ids=["individual-at-any-depth", "xla_dump-at-any-depth", "manifests-at-the-root-only"],
)  # fmt: skip
def test_directory_removers_delete_only_matching_directories(
    tmp_path, remove_directories, removed_paths, kept_paths, expected_count
):
    entry_dir = tmp_path / "entry-1"
    for relative_path in (*removed_paths, *kept_paths):
        (entry_dir / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (entry_dir / relative_path).write_bytes(b"x")
    removal_count = remove_directories(str(entry_dir))
    assert removal_count == expected_count
    remaining_files = sorted(
        path.relative_to(entry_dir).as_posix() for path in entry_dir.rglob("*") if path.is_file()
    )
    assert remaining_files == sorted(kept_paths)
    repeated_removal_count = remove_directories(str(entry_dir))
    absent_entry_removal_count = remove_directories(str(tmp_path / "entry-2"))
    assert repeated_removal_count == 0 and absent_entry_removal_count == 0


def test_directory_removal_warns_when_a_directory_survives(tmp_path, monkeypatch, capsys):
    removed_dir = tmp_path / "removed"
    removed_dir.mkdir()
    directory_removed = artifact_layout.remove_directory_tree(removed_dir)
    assert directory_removed is True
    assert capsys.readouterr().out == ""
    survivor_dir = tmp_path / "survivor"
    survivor_dir.mkdir()
    monkeypatch.setattr(
        artifact_layout.shutil, "rmtree", lambda *_arguments, **_keyword_arguments: None
    )
    survivor_removed = artifact_layout.remove_directory_tree(survivor_dir)
    assert survivor_removed is False
    assert capsys.readouterr().out == f"::warning::survivor survived removal: {survivor_dir}\n"
    survivor_removal_count = artifact_layout.remove_named_directories(str(tmp_path), "survivor")
    assert survivor_removal_count == 0
    assert capsys.readouterr().out == f"::warning::survivor survived removal: {survivor_dir}\n"


def leave_directory_when_errors_are_ignored(
    directory_path, ignore_errors=False, **_ignored_options
):
    if not ignore_errors:
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(directory_path))


@pytest.mark.parametrize(
    ("remove_directories", "directory_name", "survivor_type"),
    [
        (
            lambda root: artifact_layout.remove_named_directories(root, "individual"),
            "individual",
            "directory-recreated",
        ),
        (
            lambda root: artifact_layout.remove_named_directories(
                root, artifact_layout.XLA_DUMP_DIR_NAME
            ),
            "xla_dump",
            "directory-recreated",
        ),
        (artifact_layout.remove_compiler_manifests, "manifests", "directory-recreated"),
        (
            lambda root: artifact_layout.remove_named_directories(
                root, artifact_layout.XLA_DUMP_DIR_NAME
            ),
            "xla_dump",
            "symlink",
        ),
    ],
    ids=[
        "individual-recreated-during-removal",
        "xla-dump-recreated-during-removal",
        "manifests-recreated-during-removal",
        "symlink-survives-removal",
    ],
)
def test_live_writer_and_symlink_survivors_warn_without_raising(
    tmp_path, monkeypatch, capsys, remove_directories, directory_name, survivor_type
):
    entry_dir = tmp_path / "entry-1"
    if survivor_type == "symlink":
        surviving_file = entry_dir / "target" / "keep"
        surviving_file.parent.mkdir(parents=True)
        (entry_dir / directory_name).symlink_to(entry_dir / "target", target_is_directory=True)
    else:
        surviving_file = entry_dir / directory_name / "G.0" / "keep"
        surviving_file.parent.mkdir(parents=True)
        monkeypatch.setattr(
            artifact_layout.shutil,
            "rmtree",
            leave_directory_when_errors_are_ignored,
        )
    surviving_file.write_bytes(b"x")
    survivor_removal_count = remove_directories(str(entry_dir))
    assert survivor_removal_count == 0
    assert surviving_file.is_file() and (entry_dir / directory_name).exists()
    assert (
        capsys.readouterr().out
        == f"::warning::{directory_name} survived removal: {entry_dir / directory_name}\n"
    )


@pytest.mark.parametrize(
    ("file_text", "expected_outcome"),
    [("[{}]", [{}]), ("[]", []), ("{}", ValueError), ("[7]", ValueError),
     (None, FileNotFoundError)],
    ids=["rows", "empty-list", "top-level-dict", "non-dict-row", "missing-file"],
)  # fmt: skip
def test_read_rows_accepts_row_lists_and_rejects_missing_or_invalid_files(
    tmp_path, file_text, expected_outcome
):
    grid_path = tmp_path / "grid_metrics.json"
    if file_text is not None:
        grid_path.write_text(file_text)
    if isinstance(expected_outcome, list):
        assert artifact_layout.read_grid_rows(str(grid_path)) == expected_outcome
    else:
        with pytest.raises(expected_outcome):
            artifact_layout.read_grid_rows(str(grid_path))
