import ast
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
SCRIPT = SCRIPTS / "rca_workspace.py"


def run_rca(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_generated_workspace_is_path_neutral_and_python_parses(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[param]\n"
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[other]\n"
    )
    signatures = tmp_path / "category-tests.txt.signatures.txt"
    signatures.write_text("## jacobian_mismatch\n")
    workspace = tmp_path / "rca"

    result = run_rca(
        tests_file,
        workspace,
        "--category",
        "jacobian mismatch",
        "--signatures",
        signatures,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in workspace.iterdir()} == {
        "README.md",
        "notes.md",
        "pr_body.md",
        "repro.py",
        "signatures.txt",
        "tests.txt",
        "unit_test_template.py",
    }
    unit_test = (workspace / "unit_test_template.py").read_text()
    assert "torch.linalg.svd(x)" in unit_test
    assert unit_test.count("def test_linalg_svd_grad") == 1
    ast.parse(unit_test)
    ast.parse((workspace / "repro.py").read_text())

    pr_body = (workspace / "pr_body.md").read_text()
    canonical_reference = (
        Path(__file__).resolve().parents[1] / "references" / "verification-and-pr.md"
    ).read_text()
    canonical = (
        canonical_reference.split("````markdown\n", maxsplit=1)[1].split("\n````", maxsplit=1)[0]
        + "\n"
    )
    assert pr_body == canonical
    disclosure = (
        "AI-assisted content disclosure: The quoted pull request draft below was generated with "
        "<AI tool name>."
    )
    placeholder = "<!-- REQUIRED: Explain why the quoted change is relevant."
    assert pr_body.startswith(disclosure + "\n")
    assert "generated with AI assistance." not in pr_body
    human_section, agent_section = pr_body.split(
        "## AI-generated pull request body\n\n", maxsplit=1
    )
    assert "## Human commentary\n\n" + placeholder in human_section
    assert "## Human confirmation\n\n" in human_section
    assert "- [ ] I read and understand every changed file." in human_section
    assert "- [ ] I take responsibility for these changes." in human_section
    assert "The agent must not fill this section." in human_section
    assert "\n> ## Problem\n" in pr_body
    assert "\n> ## Root cause\n" in pr_body
    assert "\n> ## Fix\n" in pr_body
    assert "\n> ## Verification\n" in pr_body
    assert "\n## Problem\n" not in pr_body
    assert all(not line or line.startswith(">") for line in agent_section.splitlines())

    generated = "\n".join(path.read_text() for path in workspace.iterdir())
    assert str(tmp_path) not in generated
    assert str(tmp_path) not in result.stdout + result.stderr


def test_missing_or_empty_input_exits_two_without_success(tmp_path):
    missing_result = run_rca(tmp_path / "missing.txt", tmp_path / "workspace", "--category", "x")
    assert missing_result.returncode == 2

    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    empty_result = run_rca(empty, tmp_path / "empty-workspace", "--category", "x")
    assert empty_result.returncode == 2
    assert not (tmp_path / "empty-workspace").exists()


def assert_tree_ascii(root):
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        if path.is_file():
            path.read_text(encoding="ascii")


def test_owned_files_are_ascii():
    assert_tree_ascii(SKILL_ROOT)


def test_ascii_scanner_is_recursive_and_excludes_pycache(tmp_path):
    nested = tmp_path / "references" / "nested"
    nested.mkdir(parents=True)
    (nested / "valid.md").write_text("ASCII\n")
    cache = tmp_path / "tests" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "ignored.txt").write_text("not ASCII: \N{SNOWMAN}\n")
    assert_tree_ascii(tmp_path)

    (nested / "invalid.md").write_text("not ASCII: \N{SNOWMAN}\n")
    with pytest.raises(UnicodeDecodeError):
        assert_tree_ascii(tmp_path)


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_unsafe_nodeid_fields_exit_nonzero_without_writing(tmp_path):
    unsafe_nodeids = {
        "file": "../tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[param]",
        "class": "tests/test_ops.py::Bad-Class::test_fn_grad_linalg_svd_neuron_float32[param]",
        "method": "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32'junk[param]",
        "operation": "tests/test_ops.py::TestGrad::test_fn_grad_linalg-svd_neuron_float32[param]",
        "control": (
            "tests/test_ops.py::TestGrad::" "test_fn_grad_linalg_svd_neuron_float32[bad\tparam]"
        ),
    }
    for label, nodeid in unsafe_nodeids.items():
        tests_file = tmp_path / f"{label}.txt"
        tests_file.write_text(nodeid + "\n")
        workspace = tmp_path / f"workspace-{label}"
        result = run_rca(tests_file, workspace, "--category", "unsafe")
        assert result.returncode == 2, label
        assert "unsafe input" in result.stderr, label
        assert "Traceback" not in result.stderr, label
        assert not workspace.exists(), label


def test_unsafe_category_exits_nonzero_without_writing(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[param]\n"
    )
    workspace = tmp_path / "workspace"
    result = run_rca(tests_file, workspace, "--category", "bad'category")
    assert result.returncode == 2
    assert "unsafe input" in result.stderr
    assert "Traceback" not in result.stderr
    assert not workspace.exists()


def test_existing_outputs_require_force_and_notes_are_never_overwritten(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]\n"
    )
    workspace = tmp_path / "rca"
    first = run_rca(tests_file, workspace, "--category", "shape")
    assert first.returncode == 0, first.stderr
    (workspace / "notes.md").write_text("human evidence\n")
    original_readme = (workspace / "README.md").read_text()

    blocked = run_rca(tests_file, workspace, "--category", "new shape")
    assert blocked.returncode == 2
    assert "use --force" in blocked.stderr
    assert (workspace / "README.md").read_text() == original_readme
    assert (workspace / "notes.md").read_text() == "human evidence\n"

    forced = run_rca(tests_file, workspace, "--category", "new shape", "--force")
    assert forced.returncode == 0, forced.stderr
    assert "new_shape" in (workspace / "README.md").read_text()
    assert (workspace / "notes.md").read_text() == "human evidence\n"


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_rejects_nonregular_destination_without_touching_outputs(tmp_path, kind):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]\n"
    )
    workspace = tmp_path / "rca"
    workspace.mkdir()
    destination = workspace / "README.md"
    target = tmp_path / "outside.txt"
    target.write_text("keep\n")
    if kind == "directory":
        destination.mkdir()
    elif kind == "symlink":
        destination.symlink_to(target)
    else:
        os.mkfifo(destination)

    result = run_rca(tests_file, workspace, "--category", "shape", "--force")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert target.read_text() == "keep\n"
    if kind == "directory":
        assert destination.is_dir()
    elif kind == "symlink":
        assert destination.is_symlink()
    else:
        assert stat.S_ISFIFO(destination.lstat().st_mode)
    assert {path.name for path in workspace.iterdir()} == {"README.md"}


def test_rejects_symlink_workspace_without_writing_through_it(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "rca"
    workspace.symlink_to(outside, target_is_directory=True)

    result = run_rca(tests_file, workspace, "--category", "shape", "--force")
    assert result.returncode == 2
    assert "RCA workspace must not be a symlink" in result.stderr
    assert list(outside.iterdir()) == []


def test_tests_file_already_inside_rca_root_is_left_in_place(tmp_path):
    workspace = tmp_path / "rca"
    workspace.mkdir()
    tests_file = workspace / "tests.txt"
    original = "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]\n"
    tests_file.write_text(original)

    result = run_rca(tests_file, workspace, "--category", "documented shape")
    assert result.returncode == 0, result.stderr
    assert tests_file.read_text() == original
    assert (workspace / "notes.md").exists()


def test_balanced_parameter_suffix_keeps_selector_components_intact(tmp_path):
    variants = ("a::b", "a-b", "outer[inner]", "value with spaces")
    for index, parameter in enumerate(variants):
        tests_file = tmp_path / f"tests-{index}.txt"
        tests_file.write_text(
            "tests/test_ops.py::TestGrad::" f"test_fn_grad_linalg_svd_neuron_float32[{parameter}]\n"
        )
        workspace = tmp_path / f"rca-{index}"
        result = run_rca(tests_file, workspace, "--category", "parameters")
        assert result.returncode == 0, (parameter, result.stderr)
        assert "class TestGrad" not in (workspace / "unit_test_template.py").read_text()
        assert "TEST_CLASS = 'TestGrad'" in (workspace / "unit_test_template.py").read_text()


def test_nodeid_and_identifier_length_limits_exit_two_without_traceback(tmp_path):
    overlong_method = "test_" + "x" * 201
    tests_file = tmp_path / "long.txt"
    tests_file.write_text(f"tests/test_ops.py::TestGrad::{overlong_method}[p]\n")
    result = run_rca(tests_file, tmp_path / "rca", "--category", "length")
    assert result.returncode == 2
    assert "exceeds 200" in result.stderr
    assert "Traceback" not in result.stderr


def test_force_mode_late_failure_restores_complete_workspace(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    tests_file.write_text(
        "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]\n"
    )
    workspace = tmp_path / "rca"
    first = run_rca(tests_file, workspace, "--category", "before")
    assert first.returncode == 0, first.stderr
    (workspace / "notes.md").write_text("human evidence\n")
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}

    env = os.environ.copy()
    env["SAFE_OUTPUT_TEST_FAIL_INSTALL"] = "4"
    failed = run_rca(
        tests_file,
        workspace,
        "--category",
        "after",
        "--force",
        env=env,
    )
    assert failed.returncode == 2
    assert "injected late install failure" in failed.stderr
    after = {path.name: path.read_bytes() for path in workspace.iterdir()}
    assert after == before


def test_rca_rederives_and_redacts_structured_signatures(tmp_path):
    tests_file = tmp_path / "category-tests.txt"
    nodeid = "tests/test_ops.py::TestGrad::test_fn_grad_linalg_svd_neuron_float32[p]"
    tests_file.write_text(nodeid + "\n")
    home = "/local/" + "home/person/work.py"
    account = "123456" * 2
    key_id = "AKIA" + "C" * 16
    import socket

    hostname = socket.gethostname()
    signatures = tmp_path / "category-tests.txt.signatures.txt"
    signatures.write_text(
        "# Selected 1 tests from 1 failures\n"
        "# Pinned: nodeids=1 error_regexes=0\n"
        "# Signatures matched: 1\n\n"
        "## RuntimeError  (1 tests)\n"
        f"    example: {nodeid}\n"
        f"    message: at {home} {account} {key_id} token=hidden {hostname}\n"
    )
    workspace = tmp_path / "rca"
    result = run_rca(
        tests_file,
        workspace,
        "--category",
        "redaction",
        "--signatures",
        signatures,
    )
    assert result.returncode == 0, result.stderr
    output = (workspace / "signatures.txt").read_text()
    for sensitive in (home, account, key_id, hostname, "hidden"):
        assert sensitive not in output
    assert "<redacted-home>" in output
    assert "<redacted-account>" in output
    assert "<redacted-key-id>" in output
    assert "<redacted-secret>" in output
    assert "<redacted-hostname>" in output
