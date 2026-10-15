import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "render_issue.py"
SPEC = importlib.util.spec_from_file_location("render_issue_test", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def issue_spec(**overrides):
    value = {
        "category": "Accuracy",
        "regression": "Unknown",
        "regression_unknown_explanation": "No earlier compiler result is available.",
        "summary": "Output values differ",
        "description": "A standalone HLO reproduces the mismatch.",
        "selected_hlo": "compiler_dump/neuronx-cc/example/module.mlir",
        "exact_error": "Output 0 differs from the reference.",
        "reproduction_command": "python reproduce.py",
        "validation_command": "python validate_hlo.py module.mlir --artdir . --atol 0 --rtol 0",
        "repro_program": "reproduce.py",
        "validator_result": "validator-result.json",
        "repro_exit_code_file": "run.exitcode",
        "versions_file": "versions.txt",
        "environment": "Public package versions and a device type.",
        "impact": "The mismatch blocks correctness validation.",
        "evidence": ["Output 0 differs.", "The stated tolerances detect the difference."],
    }
    value.update(overrides)
    return value


def make_artifacts(tmp_path):
    artdir = tmp_path / "artifacts"
    compiler_dir = artdir / "compiler_dump" / "neuronx-cc" / "example"
    compiler_dir.mkdir(parents=True)
    hlo = compiler_dir / "module.mlir"
    hlo.write_text("module {}", encoding="utf-8")
    (compiler_dir / "compiler.log").write_text("compile completed", encoding="utf-8")
    (compiler_dir / "module.neff").write_bytes(b"\x00NEFF")
    (artdir / "in_input0.pt").write_bytes(b"\x00INPUT")
    (artdir / "out_reference0.pt").write_bytes(b"\x00REFERENCE")
    (artdir / "out_actual0.pt").write_bytes(b"\x00ACTUAL")
    (artdir / "reproduce.py").write_text("print('reproduce')\n", encoding="utf-8")
    (artdir / "run.log").write_text("reproduction output\n", encoding="utf-8")
    (artdir / "run.exitcode").write_text("0\n", encoding="utf-8")
    (artdir / "launch_blocking.txt").write_text("<unset>\n", encoding="utf-8")
    (artdir / "validator-result.exitcode").write_text("0\n", encoding="utf-8")
    (artdir / "versions.txt").write_text(
        'Command: python3 -c "import sys; print(sys.version)"\nPython version output\n',
        encoding="utf-8",
    )
    (artdir / "validator-result.json").write_text(
        json.dumps(
            {
                "state": "REPRODUCED",
                "hlo_path": "compiler_dump/neuronx-cc/example/module.mlir",
                "hlo_sha256": hashlib.sha256(hlo.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (artdir / "issue-spec.json").write_text(json.dumps(issue_spec()), encoding="utf-8")
    return artdir


def make_issue_template(tmp_path):
    template = tmp_path / "bug_report.yml"
    template.write_text(
        """name: Bug
body:
  - type: textarea
    id: details
    attributes:
      label: Bug description
      description: Describe the bug.
    validations:
      required: true
  - type: input
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
""",
        encoding="utf-8",
    )
    return template


def package(tmp_path, artdir, spec=None, **kwargs):
    if "issue_template" not in kwargs:
        kwargs["issue_template"] = make_issue_template(tmp_path)
    return renderer.write_package(
        artdir,
        spec or issue_spec(),
        tmp_path / "issue.md",
        tmp_path / "bundle.tar.gz",
        kwargs.pop("max_archive_bytes", 32_768),
        **kwargs,
    )


def hash_artifact_files(artdir):
    return {
        path.relative_to(artdir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(artdir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_writes_local_draft_and_verified_archive_without_network(tmp_path, monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("A network connection was attempted.")

    monkeypatch.setattr(socket, "socket", fail_network)
    artdir = make_artifacts(tmp_path)
    result = package(tmp_path, artdir)
    assert result["state"] == "DRAFT_READY_FOR_HUMAN_REVIEW"
    assert result["verified"] is True
    assert renderer.verify_archive(tmp_path / "bundle.tar.gz") is True
    draft = (tmp_path / "issue.md").read_text()
    assert draft.startswith("# Issue field plan\n\n## Suggested issue title")
    assert "> [Compiler] Output values differ" in draft
    assert "AI-assistance disclosure:" in draft
    assert "> Status: DRAFT_READY_FOR_HUMAN_REVIEW" in draft
    assert draft.endswith("## Human commentary (human only)\n")


def test_archive_contains_complete_reproduction_evidence(tmp_path):
    artdir = make_artifacts(tmp_path)
    result = package(tmp_path, artdir)
    included = set(result["archive"]["included_paths"])
    assert {
        "issue-spec.json",
        "launch_blocking.txt",
        "reproduce.py",
        "run.log",
        "validator-result.json",
        "validator-result.exitcode",
        "run.exitcode",
        "versions.txt",
    } <= included


def test_binary_artifacts_are_marked_for_human_review(tmp_path):
    artdir = make_artifacts(tmp_path)
    package(tmp_path, artdir)
    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as handle:
        manifest = json.load(handle.extractfile("manifest.json"))
    binary_entries = [item for item in manifest["included"] if item["source"].endswith(".pt")]
    assert binary_entries
    assert all(renderer.BINARY_REVIEW in item["review"] for item in binary_entries)


def test_unlisted_binary_type_is_denied(tmp_path):
    artdir = make_artifacts(tmp_path)
    (artdir / "opaque.bin").write_bytes(b"\x00OPAQUE")
    package(tmp_path, artdir)
    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as handle:
        manifest = json.load(handle.extractfile("manifest.json"))
    assert any(item["source"] == "opaque.bin" for item in manifest["excluded"])


def test_remaining_custom_marker_fails_closed(tmp_path):
    artdir = make_artifacts(tmp_path)
    marker = "SYNTHETIC_BOUNDARY_MARKER"
    (artdir / "run.log").write_text(marker, encoding="utf-8")
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps({"rules": [{"id": "TEST_BOUNDARY", "pattern": marker}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="TEST_BOUNDARY"):
        package(tmp_path, artdir, boundary_rules=[rules])
    assert not (tmp_path / "bundle.tar.gz").exists()


def test_rejects_non_reproduced_validator_result(tmp_path):
    artdir = make_artifacts(tmp_path)
    result_path = artdir / "validator-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["state"] = "NOT_REPRODUCED"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "VALIDATOR_NOT_REPRODUCED"


def test_requires_repro_program_spec_item(tmp_path):
    spec = issue_spec()
    spec.pop("repro_program")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, make_artifacts(tmp_path), spec)
    assert error.value.state == "MISSING_REPRO_PROGRAM"


def test_requires_validator_result_spec_item(tmp_path):
    spec = issue_spec()
    spec.pop("validator_result")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, make_artifacts(tmp_path), spec)
    assert error.value.state == "MISSING_VALIDATOR_RESULT"


def test_requires_repro_exit_code_spec_item(tmp_path):
    spec = issue_spec()
    spec.pop("repro_exit_code_file")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, make_artifacts(tmp_path), spec)
    assert error.value.state == "MISSING_REPRO_EXIT_CODE_FILE"


def test_requires_versions_file_spec_item(tmp_path):
    spec = issue_spec()
    spec.pop("versions_file")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, make_artifacts(tmp_path), spec)
    assert error.value.state == "MISSING_VERSIONS_FILE"


def test_validator_result_must_match_selected_hlo_path_and_sha(tmp_path):
    artdir = make_artifacts(tmp_path)
    result_path = artdir / "validator-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["hlo_path"] = "other/module.mlir"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "VALIDATOR_HLO_MISMATCH"

    result["hlo_path"] = issue_spec()["selected_hlo"]
    result["hlo_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "VALIDATOR_HLO_SHA_MISMATCH"


def test_rejects_invalid_exit_code_and_version_command(tmp_path):
    artdir = make_artifacts(tmp_path)
    (artdir / "run.exitcode").write_text("success\n", encoding="utf-8")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "INVALID_REPRO_EXIT_CODE"

    (artdir / "run.exitcode").write_text("0\n", encoding="utf-8")
    (artdir / "versions.txt").write_text(
        "Command: command-that-does-not-exist --version\noutput\n", encoding="utf-8"
    )
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "VERSION_COMMAND_NOT_FOUND"


def test_parses_fixture_issue_template_in_field_order(tmp_path):
    artdir = make_artifacts(tmp_path)
    template = tmp_path / "bug_report.yml"
    template.write_text(
        """name: Bug
body:
  - type: textarea
    id: details
    attributes:
      label: First field
      description: Describe the bug.
    validations:
      required: true
  - type: input
    id: expected
    attributes:
      label: Second field
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Third field
""",
        encoding="utf-8",
    )
    package(tmp_path, artdir, issue_template=template)
    draft = (tmp_path / "issue.md").read_text(encoding="utf-8")
    first = draft.index("## Field 1: First field (`details`)")
    second = draft.index("## Field 2: Second field (`expected`)")
    third = draft.index("## Field 3: Third field (`environment`)")
    assert first < second < third
    assert "> - [ ] First field (`details`)" in draft
    assert "> - [ ] Second field (`expected`)" in draft


def test_uses_shipped_fallback_when_optional_repository_template_is_absent(tmp_path, monkeypatch):
    scanner = renderer._find_scanner()
    monkeypatch.setattr(renderer, "_find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(renderer, "_find_scanner", lambda: scanner)
    result = package(tmp_path, make_artifacts(tmp_path), issue_template=None)
    assert result["state"] == "DRAFT_READY_FOR_HUMAN_REVIEW"
    draft = (tmp_path / "issue.md").read_text(encoding="ascii")
    assert "Describe the bug" in draft
    assert "Reproduction" in draft
    assert "Expected Behavior" in draft
    assert "Environment" in draft


def test_missing_explicit_issue_template_fails_clearly(tmp_path):
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, make_artifacts(tmp_path), issue_template=tmp_path / "missing.yml")
    assert error.value.state == "ISSUE_TEMPLATE_MISSING"
    assert "explicit issue template" in str(error.value).lower()


@pytest.mark.parametrize(
    "relative",
    ["issue-spec.json", "launch_blocking.txt", "run.log", "validator-result.exitcode"],
)
def test_requires_each_conventional_evidence_file(tmp_path, relative):
    artdir = make_artifacts(tmp_path)
    (artdir / relative).unlink()
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == "MISSING_REQUIRED_EVIDENCE"
    assert relative in str(error.value)


@pytest.mark.parametrize(
    ("value", "state"),
    [("1\n", "VALIDATOR_NOT_REPRODUCED"), ("invalid\n", "INVALID_VALIDATOR_EXIT_CODE")],
)
def test_requires_successful_validator_process_status(tmp_path, value, state):
    artdir = make_artifacts(tmp_path)
    (artdir / "validator-result.exitcode").write_text(value, encoding="ascii")
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir)
    assert error.value.state == state


def test_cli_runs_from_outside_any_git_repo(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    assert not any((path / ".git").exists() for path in (outside, *outside.parents))
    artdir = make_artifacts(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(issue_spec()), encoding="ascii")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artdir",
            str(artdir),
            "--spec",
            str(spec_path),
            "--draft",
            str(tmp_path / "cli-issue.md"),
            "--archive",
            str(tmp_path / "cli-bundle.tar.gz"),
            "--issue-template",
            str(make_issue_template(tmp_path)),
            "--max-archive-bytes",
            "32768",
        ],
        cwd=outside,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout or result.stderr
    assert json.loads(result.stdout)["state"] == "DRAFT_READY_FOR_HUMAN_REVIEW"


def test_requires_compiler_logs(tmp_path):
    artdir = make_artifacts(tmp_path)
    next(artdir.rglob("compiler.log")).unlink()
    with pytest.raises(ValueError, match="Compiler logs are required"):
        package(tmp_path, artdir)


def test_requires_neff_after_numerical_compile(tmp_path):
    artdir = make_artifacts(tmp_path)
    next(artdir.rglob("module.neff")).unlink()
    with pytest.raises(ValueError, match="A NEFF is required"):
        package(tmp_path, artdir)


def test_requires_accuracy_input(tmp_path):
    artdir = make_artifacts(tmp_path)
    (artdir / "in_input0.pt").unlink()
    with pytest.raises(ValueError, match="input tensor is required"):
        package(tmp_path, artdir)


def test_requires_accuracy_reference(tmp_path):
    artdir = make_artifacts(tmp_path)
    (artdir / "out_reference0.pt").unlink()
    with pytest.raises(ValueError, match="reference tensor is required"):
        package(tmp_path, artdir)


def test_fails_if_size_pruning_would_drop_required_artifact(tmp_path):
    artdir = make_artifacts(tmp_path)
    next(artdir.rglob("module.neff")).write_bytes(b"\x00" + b"N" * 20_000)
    with pytest.raises(ValueError, match="Required artifacts exceed"):
        package(tmp_path, artdir, max_archive_bytes=200)
    assert not (tmp_path / "bundle.tar.gz").exists()


def test_required_artifact_rejected_by_scan_stops_package(tmp_path):
    artdir = make_artifacts(tmp_path)
    marker = "SYNTHETIC_REQUIRED_MARKER"
    next(artdir.rglob("compiler.log")).write_text(marker, encoding="utf-8")
    rules = tmp_path / "required-rules.json"
    rules.write_text(
        json.dumps({"rules": [{"id": "REQUIRED_TEST", "pattern": marker}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REQUIRED_TEST"):
        package(tmp_path, artdir, boundary_rules=[rules])


def test_regression_requires_last_known_good_compiler(tmp_path):
    artdir = make_artifacts(tmp_path)
    with pytest.raises(ValueError, match="last_known_good_compiler"):
        package(
            tmp_path,
            artdir,
            issue_spec(regression="Yes", regression_unknown_explanation=""),
        )


def test_unknown_regression_requires_explanation(tmp_path):
    artdir = make_artifacts(tmp_path)
    spec = issue_spec()
    spec.pop("regression_unknown_explanation")
    with pytest.raises(ValueError, match="regression_unknown_explanation"):
        package(tmp_path, artdir, spec)


def test_accuracy_regression_can_include_good_hlo_evidence(tmp_path):
    artdir = make_artifacts(tmp_path)
    good_hlo = artdir / "good" / "module.mlir"
    good_hlo.parent.mkdir()
    good_hlo.write_text("module { good }", encoding="utf-8")
    result = package(
        tmp_path,
        artdir,
        issue_spec(
            regression="Yes",
            regression_unknown_explanation="",
            last_known_good_compiler="A public compiler version",
            good_hlo_evidence="good/module.mlir",
        ),
    )
    assert "good/module.mlir" in result["archive"]["included_paths"]


def test_rejects_missing_hlo_and_unknown_fields(tmp_path):
    artdir = make_artifacts(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        package(tmp_path, artdir, issue_spec(selected_hlo="missing.mlir"))
    outside = str((artdir / "outside.mlir").resolve())
    with pytest.raises(ValueError, match="safe relative path"):
        package(tmp_path, artdir, issue_spec(selected_hlo=outside))
    with pytest.raises(ValueError, match="unsupported fields"):
        package(tmp_path, artdir, issue_spec(extra_field="value"))


MLIR_BYTECODE = b"ML\xefR\x00\x01\xffreal-bytecode"


def test_mlir_bytecode_is_included_byte_for_byte_as_reviewed_hlo(tmp_path):
    artdir = make_artifacts(tmp_path)
    hlo = next(artdir.rglob("module.mlir"))
    hlo.write_bytes(MLIR_BYTECODE)
    result_path = artdir / "validator-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["hlo_sha256"] = hashlib.sha256(MLIR_BYTECODE).hexdigest()
    result_path.write_text(json.dumps(result), encoding="utf-8")

    package(tmp_path, artdir)
    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        entry = next(
            item for item in manifest["included"] if item["source"].endswith("module.mlir")
        )
        archived = archive.extractfile(f"artifacts/{entry['stored_path']}").read()
    assert archived == MLIR_BYTECODE
    assert "type=MLIR bytecode HLO" in entry["review"]


def test_every_tar_header_has_normalized_metadata_and_mode(tmp_path):
    package(tmp_path, make_artifacts(tmp_path))
    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
    assert members
    for member in members:
        assert (member.uid, member.gid, member.uname, member.gname, member.mtime) == (
            0,
            0,
            "",
            "",
            0,
        )
        assert member.mode == (0o755 if member.isdir() else 0o644)


def _mutated_archive(source, destination, mutation):
    with tarfile.open(source, "r:gz") as original, tarfile.open(destination, "w:gz") as output:
        saved = []
        for member in original.getmembers():
            extracted = original.extractfile(member) if member.isfile() else None
            data = extracted.read() if extracted is not None else None
            saved.append((member, data))
        for member, data in saved:
            if mutation == "wrong-type" and member.name == "artifacts":
                replacement = tarfile.TarInfo("artifacts")
                replacement.type = tarfile.REGTYPE
                replacement.size = 0
                output.addfile(replacement, io.BytesIO())
            else:
                output.addfile(member, io.BytesIO(data) if data is not None else None)
        if mutation == "extra-root":
            extra = tarfile.TarInfo("extra.txt")
            extra.size = 5
            output.addfile(extra, io.BytesIO(b"extra"))
        elif mutation == "duplicate":
            member, data = next(item for item in saved if item[0].name == "manifest.json")
            output.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("mutation", ["extra-root", "duplicate", "wrong-type"])
def test_archive_verification_rejects_extra_duplicate_and_wrong_type_members(tmp_path, mutation):
    package(tmp_path, make_artifacts(tmp_path))
    malformed = tmp_path / f"{mutation}.tar.gz"
    _mutated_archive(tmp_path / "bundle.tar.gz", malformed, mutation)
    assert renderer.verify_archive(malformed) is False


def test_accuracy_requires_and_includes_actual_output_tensor(tmp_path):
    artdir = make_artifacts(tmp_path)
    result = package(tmp_path, artdir)
    assert "out_actual0.pt" in result["archive"]["included_paths"]
    (artdir / "out_actual0.pt").unlink()
    with pytest.raises(ValueError, match="actual output tensor is required"):
        package(tmp_path, artdir)


def test_per_file_and_total_limits_exclude_files_before_preparation(tmp_path, monkeypatch):
    artdir = make_artifacts(tmp_path)
    oversized = artdir / "oversized.log"
    oversized.write_bytes(b"x" * 2048)
    original_prepare = renderer._prepare_file

    def guarded_prepare(path, *args, **kwargs):
        assert path != oversized
        return original_prepare(path, *args, **kwargs)

    monkeypatch.setattr(renderer, "_prepare_file", guarded_prepare)
    result = package(tmp_path, artdir, max_file_bytes=1024)
    excluded = {item["source"]: item["reason"] for item in result["archive"]["excluded"]}
    assert excluded["oversized.log"] == "The file exceeds the per-file size limit."

    bounded = tmp_path / "bounded"
    bounded.mkdir()
    (bounded / "out_actual_a.pt").write_bytes(b"1234")
    (bounded / "out_actual_b.pt").write_bytes(b"5678")
    archive = renderer.create_archive(
        bounded,
        tmp_path / "bounded.tar.gz",
        4096,
        max_file_bytes=4,
        max_total_bytes=4,
    )
    assert archive["included"] == 1
    assert any("total input size limit" in item["reason"] for item in archive["excluded"])


def test_pair_publication_rolls_back_both_outputs_on_second_rename_failure(tmp_path, monkeypatch):
    artdir = make_artifacts(tmp_path)
    draft = tmp_path / "issue.md"
    archive = tmp_path / "bundle.tar.gz"
    draft.write_bytes(b"old draft")
    archive.write_bytes(b"old archive")
    real_replace = renderer.os.replace
    failed = False

    def fail_second_publish(source, destination, *args, **kwargs):
        nonlocal failed
        if not failed and destination == draft.name and str(source).startswith(".compiler-issue-"):
            failed = True
            raise OSError("injected second publish failure")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(renderer.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected"):
        package(tmp_path, artdir)
    assert archive.read_bytes() == b"old archive"
    assert draft.read_bytes() == b"old draft"


def test_non_object_spec_cli_returns_invalid_spec_json(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("[]\n", encoding="ascii")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artdir",
            str(tmp_path),
            "--spec",
            str(spec),
            "--draft",
            str(tmp_path / "issue.md"),
            "--archive",
            str(tmp_path / "bundle.tar.gz"),
            "--max-archive-bytes",
            "1024",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout)["state"] == "INVALID_SPEC"


def test_unexpected_cli_failure_returns_json_without_traceback(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(renderer, "write_package", lambda *_args, **_kwargs: 1 / 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--artdir",
            str(tmp_path),
            "--spec",
            str(spec),
            "--draft",
            str(tmp_path / "issue.md"),
            "--archive",
            str(tmp_path / "bundle.tar.gz"),
            "--max-archive-bytes",
            "1024",
        ],
    )
    assert renderer.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["state"] == "UNEXPECTED_ERROR"
    assert "Traceback" not in captured.out


def test_rejects_non_ascii_user_text_and_normalizes_template_labels(tmp_path):
    artdir = make_artifacts(tmp_path)
    with pytest.raises(renderer.RenderError) as error:
        package(tmp_path, artdir, issue_spec(summary="Unicode snowman \u2603"))
    assert error.value.state == "NON_ASCII_USER_TEXT"

    template = tmp_path / "unicode-template.yml"
    template.write_text(
        """name: Bug
body:
  - type: textarea
    id: first
    attributes:
      label: "\U0001f41b Caf\u00e9"
      description: Describe the bug.
  - type: input
    id: second
    attributes:
      label: "\u2705 Result"
""",
        encoding="utf-8",
    )
    package(tmp_path, artdir, issue_template=template)
    draft = (tmp_path / "issue.md").read_text(encoding="ascii")
    assert draft.index("## Field 1: Cafe") < draft.index("## Field 2: Result")
    assert "\U0001f41b" not in draft and "\u2705" not in draft


def test_archive_verification_rejects_backslash_traversal_member(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    stored_path = r"..\..\escape.txt"
    data = b"escape"
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "format": 2,
        "included": [
            {
                "source": "safe.txt",
                "stored_path": stored_path,
                "stored_size": len(data),
                "sha256": digest,
                "review": "test",
            }
        ],
        "excluded": [],
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in (
            ("manifest.json", (json.dumps(manifest) + "\n").encode()),
            ("SHA256SUMS", f"{digest}  artifacts/{stored_path}\n".encode()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        directory = tarfile.TarInfo("artifacts")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo(f"artifacts/{stored_path}")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    assert renderer.verify_archive(archive_path) is False


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "/absolute",
        "//server/share",
        "C:/escape.txt",
        "part//empty",
        "trailing/",
        "control\nname",
        r"..\..\escape.txt",
    ],
)
def test_portable_path_validator_rejects_unsafe_syntax(value):
    with pytest.raises(ValueError):
        renderer._portable_relative_path(value)


def test_archive_creation_rejects_nonportable_source_path(tmp_path):
    artdir = tmp_path / "artifacts"
    artdir.mkdir()
    (artdir / r"..\..\escape.txt").write_text("escape", encoding="ascii")

    with pytest.raises(ValueError, match="backslash"):
        renderer.create_archive(artdir, tmp_path / "bundle.tar.gz", 4096)


def test_in_artdir_outputs_and_staging_names_are_not_collected(tmp_path):
    artdir = make_artifacts(tmp_path)
    final_draft = artdir / "issue-draft.md"
    final_archive = artdir / "compiler-repro.tar.gz"
    final_draft.write_text("old draft", encoding="ascii")
    final_archive.write_bytes(b"old archive")
    (artdir / ".compiler-issue-leftover.md").write_text("stale", encoding="ascii")

    result = renderer.write_package(
        artdir,
        issue_spec(),
        final_draft,
        final_archive,
        32_768,
        issue_template=make_issue_template(tmp_path),
    )

    collected = set(result["archive"]["included_paths"])
    collected.update(item["source"] for item in result["archive"]["excluded"])
    assert "issue-draft.md" not in collected
    assert "compiler-repro.tar.gz" not in collected
    assert not any(".compiler-issue-" in path for path in collected)
    with tarfile.open(final_archive, "r:gz") as archive:
        assert not any(
            "issue-draft.md" in member.name
            or "compiler-repro.tar.gz" in member.name
            or ".compiler-issue-" in member.name
            for member in archive.getmembers()
        )


def test_rejects_same_or_aliased_output_paths_before_staging(tmp_path, monkeypatch):
    artdir = make_artifacts(tmp_path)
    output = tmp_path / "output"

    def fail_staging(*_args, **_kwargs):
        raise AssertionError("staging must not start")

    monkeypatch.setattr(renderer.tempfile, "TemporaryDirectory", fail_staging)
    with pytest.raises(ValueError, match="must be different"):
        renderer.write_package(
            artdir,
            issue_spec(),
            output,
            output,
            32_768,
            issue_template=make_issue_template(tmp_path),
        )

    alias = tmp_path / "output-alias"
    alias.symlink_to(output)
    with pytest.raises(ValueError, match="must be different"):
        renderer.write_package(
            artdir,
            issue_spec(),
            output,
            alias,
            32_768,
            issue_template=make_issue_template(tmp_path),
        )


def test_rejects_output_symlinked_to_or_below_required_evidence_before_staging(
    tmp_path, monkeypatch
):
    artdir = make_artifacts(tmp_path)
    evidence = artdir / "run.exitcode"
    alias = tmp_path / "evidence-alias"
    alias.symlink_to(evidence)

    def fail_staging(*_args, **_kwargs):
        raise AssertionError("staging must not start")

    monkeypatch.setattr(renderer.tempfile, "TemporaryDirectory", fail_staging)
    with pytest.raises(ValueError, match="required evidence"):
        renderer.write_package(
            artdir,
            issue_spec(),
            alias,
            tmp_path / "bundle.tar.gz",
            32_768,
            issue_template=make_issue_template(tmp_path),
        )
    with pytest.raises(ValueError, match="required evidence"):
        renderer.write_package(
            artdir,
            issue_spec(),
            tmp_path / "issue.md",
            alias / "child.tar.gz",
            32_768,
            issue_template=make_issue_template(tmp_path),
        )


@pytest.mark.parametrize("output_name", ["draft", "archive"])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("existing-directory", "not a regular file"),
        ("artifact-directory", "artifact directory"),
        ("artifact-ancestor", "artifact directory"),
        ("evidence-ancestor", "required evidence"),
    ],
)
def test_unsafe_output_is_rejected_before_staging_without_artifact_changes(
    tmp_path, monkeypatch, output_name, case, message
):
    root = tmp_path / output_name / case
    root.mkdir(parents=True)
    artdir = make_artifacts(root)
    if case == "existing-directory":
        unsafe_output = root / "existing-output"
        unsafe_output.mkdir()
    elif case == "artifact-directory":
        unsafe_output = artdir
    elif case == "artifact-ancestor":
        unsafe_output = root
    else:
        unsafe_output = artdir / "compiler_dump" / "neuronx-cc" / "example"
    draft = unsafe_output if output_name == "draft" else root / "issue.md"
    archive = unsafe_output if output_name == "archive" else root / "bundle.tar.gz"
    before = hash_artifact_files(artdir)

    def fail_staging(*_args, **_kwargs):
        raise AssertionError("staging must not start")

    monkeypatch.setattr(renderer.tempfile, "TemporaryDirectory", fail_staging)
    with pytest.raises(ValueError, match=message):
        renderer.write_package(
            artdir,
            issue_spec(),
            draft,
            archive,
            32_768,
            issue_template=make_issue_template(tmp_path / output_name),
        )

    assert hash_artifact_files(artdir) == before
    safe_output = archive if output_name == "draft" else draft
    assert not safe_output.exists()


@pytest.mark.parametrize("node_type", ["symlink", "fifo"])
def test_render_rejects_other_non_regular_outputs_before_staging(tmp_path, monkeypatch, node_type):
    artdir = make_artifacts(tmp_path)
    draft = tmp_path / "unsafe-output"
    if node_type == "symlink":
        target = tmp_path / "target.md"
        target.write_text("unchanged", encoding="ascii")
        draft.symlink_to(target)
    else:
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.unlink(missing_ok=True)
        os.mkfifo(draft)
    before = hash_artifact_files(artdir)

    def fail_staging(*_args, **_kwargs):
        raise AssertionError("staging must not start")

    monkeypatch.setattr(renderer.tempfile, "TemporaryDirectory", fail_staging)
    with pytest.raises(ValueError, match="not a regular file"):
        renderer.write_package(
            artdir,
            issue_spec(),
            draft,
            tmp_path / "bundle.tar.gz",
            32_768,
            issue_template=make_issue_template(tmp_path),
        )
    assert hash_artifact_files(artdir) == before


def test_parent_swapped_to_symlink_before_publish_aborts_without_external_write(
    tmp_path, monkeypatch
):
    output_parent = tmp_path / "publish"
    output_parent.mkdir()
    artdir = make_artifacts(output_parent)
    draft = output_parent / "issue.md"
    archive = output_parent / "bundle.tar.gz"
    draft.write_bytes(b"old draft")
    archive.write_bytes(b"old archive")
    original_parent = tmp_path / "pinned-parent"
    outside = tmp_path / "outside"
    outside.mkdir()

    def swap_parent():
        output_parent.rename(original_parent)
        output_parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(renderer, "_before_output_install", swap_parent)
    with pytest.raises(OSError, match="output parent changed"):
        renderer.write_package(
            artdir,
            issue_spec(),
            draft,
            archive,
            32_768,
            issue_template=make_issue_template(tmp_path),
        )

    assert (original_parent / draft.name).read_bytes() == b"old draft"
    assert (original_parent / archive.name).read_bytes() == b"old archive"
    assert list(outside.iterdir()) == []
    assert not any(path.name.startswith(".compiler-issue-") for path in original_parent.iterdir())


def _set_hlo_sha(artdir, hlo, data):
    hlo.write_bytes(data)
    result_path = artdir / "validator-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["hlo_sha256"] = hashlib.sha256(data).hexdigest()
    result_path.write_text(json.dumps(result), encoding="utf-8")


def test_required_text_hlo_is_archived_byte_for_byte(tmp_path):
    artdir = make_artifacts(tmp_path)
    hlo = next(artdir.rglob("module.mlir"))
    original = b'module attributes {public_name = "example"} {}\n'
    _set_hlo_sha(artdir, hlo, original)

    package(tmp_path, artdir)

    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        entry = next(
            item for item in manifest["included"] if item["source"].endswith("module.mlir")
        )
        archived = archive.extractfile(f"artifacts/{entry['stored_path']}").read()
    assert archived == original
    assert entry["sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["review"].endswith("HLO bytes preserved")


@pytest.mark.parametrize("hlo_kind", ["selected", "good"])
def test_required_text_hlo_that_needs_redaction_fails_closed(tmp_path, hlo_kind):
    artdir = make_artifacts(tmp_path)
    private_hlo = b'module attributes {source = "/ho' b'me/alice/private/model.py"} {}\n'
    spec = issue_spec()
    if hlo_kind == "selected":
        _set_hlo_sha(artdir, next(artdir.rglob("module.mlir")), private_hlo)
    else:
        good_hlo = artdir / "good" / "module.mlir"
        good_hlo.parent.mkdir()
        good_hlo.write_bytes(private_hlo)
        spec.update(
            regression="Yes",
            regression_unknown_explanation="",
            last_known_good_compiler="A public compiler version",
            good_hlo_evidence="good/module.mlir",
        )

    with pytest.raises(ValueError, match="HLO contains private content"):
        package(tmp_path, artdir, spec)

    assert not (tmp_path / "bundle.tar.gz").exists()


def test_manifest_records_redaction_for_non_hlo_text(tmp_path):
    artdir = make_artifacts(tmp_path)
    private_path = "source=/ho" "me/alice/private/model.py\n"
    (artdir / "run.log").write_text(private_path, encoding="utf-8")

    package(tmp_path, artdir)

    with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        entry = next(item for item in manifest["included"] if item["source"] == "run.log")
        archived = archive.extractfile(f"artifacts/{entry['stored_path']}").read()
    assert entry["review"].endswith("redaction applied")
    assert b"<redacted-home-path>" in archived
