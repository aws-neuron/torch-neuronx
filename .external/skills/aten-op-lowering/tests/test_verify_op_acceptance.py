import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_op_acceptance.py"
SPEC = importlib.util.spec_from_file_location("verify_op_acceptance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

NODE_A = "tests/pytorch_tests_native/test/test_ops.py::Test::test_add_neuron_float32"
NODE_B = "tests/pytorch_tests_native/test/test_ops.py::Test::test_add_neuron_complex64"
SEED_TEST = "test_seed.py::TestSeed::test_seed"


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def record(nodeid, outcome, message=""):
    phase_outcome = "failed" if outcome in {"failed", "error"} else "passed"
    if outcome in {"skipped", "xfailed"}:
        phase_outcome = "skipped"
    call = {"outcome": phase_outcome}
    if message:
        call["longrepr"] = message
        call["crash"] = {"message": message, "path": "test_ops.py", "lineno": 1}
    return {
        "nodeid": nodeid,
        "outcome": outcome,
        "keywords": ["xfail"] if outcome in {"xfailed", "xpassed"} else [],
        "setup": {"outcome": "passed"},
        "call": call,
        "teardown": {"outcome": "passed"},
    }


def write_report(path, outcomes):
    pairs = outcomes.items() if isinstance(outcomes, dict) else outcomes
    return write_json(
        path,
        {
            "tests": [
                record(nodeid, *value) if isinstance(value, tuple) else record(nodeid, value)
                for nodeid, value in pairs
            ]
        },
    )


def runner_rc(outcomes):
    values = outcomes.values() if isinstance(outcomes, dict) else (value for _, value in outcomes)
    names = [value[0] if isinstance(value, tuple) else value for value in values]
    return 1 if any(name in {"failed", "error"} for name in names) else 0


def write_rc(path, value):
    path.write_text(f"{value}\n", encoding="utf-8")
    return str(path)


def attach_runxfail(args, path, outcomes):
    args.upstream_report_runxfail = write_report(path, outcomes)
    args.runxfail_runner_rc = write_rc(path.with_suffix(".rc"), runner_rc(outcomes))


def write_csv(path, rows):
    fields = ["op_name", "status", "correctness_test"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_limitations(path):
    path.write_text(
        "## Documented limitations\n"
        "- `complex-dtype` | reason: `complex dtype.*not supported` | "
        "failure signature: `complex dtype.*not supported`\n"
        "- `sparse-data-model` | reason: `sparse.*not supported` | "
        "failure signature: `sparse.*not supported`\n"
        "## Denied reason patterns\n- `wrong value|cpu fallback|layout`\n",
        encoding="utf-8",
    )
    return str(path)


def registry_group(
    *,
    reason="complex dtype not supported",
    tests=None,
    patterns=None,
    folders=None,
    kind="xfail",
    match_target="test_method",
):
    return {
        "kind": kind,
        "reason": reason,
        "tests": tests or [],
        "patterns": patterns or [],
        "folders": folders or [],
        "match_target": match_target,
    }


def seed_registry():
    return [registry_group(reason="sparse data model not supported", tests=[SEED_TEST])]


def artifacts(tmp_path, *, op="aten::add.Tensor", candidate=None, baseline=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = candidate or {NODE_A: "passed"}
    baseline = baseline or dict.fromkeys(candidate, "passed")
    nodeids = tmp_path / "nodeids.txt"
    nodeids.write_text("\n".join(candidate) + "\n", encoding="utf-8")
    upstream = write_report(tmp_path / "upstream.json", candidate)
    baseline_upstream = write_report(tmp_path / "baseline-upstream.json", baseline)
    candidate_runner_rc = write_rc(tmp_path / "candidate-runner.rc", runner_rc(candidate))
    baseline_runner_rc = write_rc(tmp_path / "baseline-runner.rc", runner_rc(baseline))
    base_registry = seed_registry()
    baseline_registry = write_json(tmp_path / "baseline-registry.json", base_registry)
    candidate_registry = write_json(tmp_path / "candidate-registry.json", base_registry)
    limitations = write_limitations(tmp_path / "limitations.md")
    eager = write_csv(
        tmp_path / "eager.csv",
        [{"op_name": op, "status": "runs_on_neuron", "correctness_test": "pass"}],
    )
    compile_report = write_csv(
        tmp_path / "compile.csv",
        [{"op_name": op, "status": "supported", "correctness_test": "pass"}],
    )
    probe = write_json(
        tmp_path / "probe.json",
        {
            "op": op,
            "differentiable": True,
            "backward_kind": "derived",
            "backward_ops": [],
            "backward_dispatch_all": ["aten::mul"],
        },
    )
    target_ops = tmp_path / "target_ops.py"
    target_ops.write_text(
        f'TARGET_ATEN_OPS = frozenset(["{op}"])\n'
        f'TARGET_ATEN_OP_GROUP = {{"{op}": "elementwise"}}\n',
        encoding="utf-8",
    )
    return SimpleNamespace(
        op=op,
        claimed_overload=[],
        upstream_report=upstream,
        baseline_upstream_report=baseline_upstream,
        upstream_report_runxfail=None,
        candidate_runner_rc=candidate_runner_rc,
        baseline_runner_rc=baseline_runner_rc,
        runxfail_runner_rc=None,
        op_nodeids=str(nodeids),
        baseline_registry=baseline_registry,
        candidate_registry=candidate_registry,
        limitations=limitations,
        accept_limitation=[],
        coverage_csv=eager,
        compile_coverage_csv=compile_report,
        require_compile=True,
        autograd_probe=probe,
        unit_rc=0,
        target_ops_file=str(target_ops),
        out=str(tmp_path / "verdict.json"),
    )


def payload(args):
    return json.loads(Path(args.out).read_text(encoding="utf-8"))


def check(value, name):
    return next(item for item in value["checks"] if item["name"] == name)


def add_new_mark(args, group):
    write_json(args.candidate_registry, [*seed_registry(), group])


def test_success_names_checks_and_evidence(tmp_path):
    args = artifacts(tmp_path)
    assert MODULE.command_verify(args) == 0
    result = payload(args)
    assert result["verdict"] == MODULE.PASS
    upstream = check(result, "upstream")
    assert upstream["status"] == MODULE.OK
    assert upstream["evidence"] == [NODE_A]
    assert "Accounted for 1" in upstream["detail"]
    assert check(result, "unit_tests")["evidence"] == ["unit_rc=0"]
    assert check(result, "runxfail_evidence_pair")["status"] == MODULE.OK
    assert "both absent" in check(result, "runxfail_evidence_pair")["detail"]


def test_nonexistent_operator_fails_repository_target_check(tmp_path):
    args = artifacts(tmp_path, op="aten::definitely_not_real")
    args.require_compile = False
    args.compile_coverage_csv = None
    assert MODULE.command_verify(args) == 1
    target = check(payload(args), "repository_target_ops")
    assert target["status"] == MODULE.FAILED
    assert target["evidence"] == ["aten::definitely_not_real"]


def test_unrelated_coverage_row_may_have_empty_correctness(tmp_path):
    args = artifacts(tmp_path)
    write_csv(
        Path(args.coverage_csv),
        [
            {"op_name": "aten::add.Tensor", "status": "runs_on_neuron", "correctness_test": "pass"},
            {"op_name": "aten::other", "status": "cpu_fallback", "correctness_test": ""},
        ],
    )
    assert MODULE.command_verify(args) == 0


def test_target_op_in_comment_only_fails(tmp_path):
    args = artifacts(tmp_path)
    Path(args.target_ops_file).write_text(
        '# "aten::add.Tensor"\nTARGET_ATEN_OPS = []\nTARGET_ATEN_OP_GROUP = {}\n',
        encoding="utf-8",
    )
    assert MODULE.command_verify(args) == 1
    assert check(payload(args), "target_ops")["evidence"] == ["aten::add.Tensor"]


def test_target_op_in_unrelated_string_only_fails(tmp_path):
    args = artifacts(tmp_path)
    Path(args.target_ops_file).write_text(
        'UNRELATED = "aten::add.Tensor"\n'
        "TARGET_ATEN_OPS = ()\n"
        'TARGET_ATEN_OP_GROUP = {"aten::other": "aten::add.Tensor"}\n',
        encoding="utf-8",
    )
    assert MODULE.command_verify(args) == 1
    assert check(payload(args), "target_ops")["evidence"] == ["aten::add.Tensor"]


def test_target_op_in_real_collection_passes(tmp_path):
    args = artifacts(tmp_path)
    Path(args.target_ops_file).write_text(
        'TARGET_ATEN_OPS = ()\nTARGET_ATEN_OP_GROUP = {"aten::add.Tensor": "elementwise"}\n',
        encoding="utf-8",
    )
    assert MODULE.command_verify(args) == 0
    assert check(payload(args), "target_ops")["status"] == MODULE.OK


def test_real_target_ops_file_parses_existing_ops():
    target_ops = Path(__file__).resolve().parents[4] / "tests/vertical/internal/target_ops.py"
    parsed = MODULE.parse_target_ops(target_ops.read_text(encoding="utf-8"), str(target_ops))
    assert {"aten::add.Tensor", "aten::topk"}.issubset(parsed)


def test_candidate_report_rejects_extra_failing_node(tmp_path):
    args = artifacts(tmp_path)
    write_report(args.upstream_report, {NODE_A: "passed", NODE_B: "failed"})
    write_rc(Path(args.candidate_runner_rc), 1)
    assert MODULE.command_verify(args) == 1
    node_ids = check(payload(args), "candidate_report_node_ids")
    assert node_ids["status"] == MODULE.FAILED
    assert node_ids["evidence"] == [f"extra:{NODE_B}"]


def test_baseline_report_rejects_extra_infrastructure_node_as_fail(tmp_path):
    args = artifacts(tmp_path)
    write_report(
        args.baseline_upstream_report,
        {NODE_A: "passed", NODE_B: ("error", "ModuleNotFoundError: dependency")},
    )
    write_rc(Path(args.baseline_runner_rc), 1)
    assert MODULE.command_verify(args) == 1
    result = payload(args)
    assert result["verdict"] == MODULE.FAIL
    node_ids = check(result, "baseline_report_node_ids")
    assert node_ids["status"] == MODULE.FAILED
    assert node_ids["evidence"] == [f"extra:{NODE_B}"]


def test_new_xfail_is_stale_when_runxfail_leg_passes(tmp_path):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: "passed"})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    result = payload(args)
    assert check(result, "runxfail")["status"] == MODULE.FAILED
    assert any("runxfail leg passed" in item for item in check(result, "runxfail")["evidence"])


def test_xfail_failure_requires_exact_accepted_signature(tmp_path):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    attach_runxfail(
        args,
        tmp_path / "runxfail.json",
        {NODE_A: ("failed", "complex dtype is not supported")},
    )
    assert MODULE.command_verify(args) == 1
    assert "no --accept-limitation" in check(payload(args), "runxfail")["evidence"][0]
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 0
    accepted = check(payload(args), "accepted_limitation")
    assert accepted["status"] == MODULE.OK
    assert "complex-dtype" in accepted["evidence"][0]


def test_runxfail_report_rejects_extra_node(tmp_path):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    attach_runxfail(
        args,
        tmp_path / "runxfail.json",
        {
            NODE_A: ("failed", "complex dtype is not supported"),
            NODE_B: ("failed", "complex dtype is not supported"),
        },
    )
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    node_ids = check(payload(args), "runxfail_report_node_ids")
    assert node_ids["status"] == MODULE.FAILED
    assert node_ids["evidence"] == [f"extra:{NODE_B}"]


@pytest.mark.parametrize("outcome", ["skipped", "xfailed", "xpassed"])
def test_xfail_rejects_nonexecuted_or_xfail_runxfail_outcomes(tmp_path, outcome):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: outcome})
    assert MODULE.command_verify(args) == 1
    runxfail = check(payload(args), "runxfail")
    assert runxfail["status"] == MODULE.FAILED
    assert f"rejected runxfail outcome {outcome}" in runxfail["evidence"][0]


def test_xfail_rejects_missing_runxfail_node(tmp_path):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_B: "passed"})
    assert MODULE.command_verify(args) == 1
    assert "missing runxfail outcome" in check(payload(args), "runxfail")["evidence"][0]


def test_new_skip_requires_reason_matched_acceptance(tmp_path):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: "skipped"},
        baseline={NODE_A: ("failed", "complex dtype is not supported")},
    )
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: "skipped"})
    assert MODULE.command_verify(args) == 1
    assert "no --accept-limitation" in check(payload(args), "runxfail")["evidence"][0]
    args.accept_limitation = [(NODE_A, "sparse-data-model")]
    assert MODULE.command_verify(args) == 1
    assert any("reason pattern" in item for item in check(payload(args), "runxfail")["evidence"])
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 0
    assert "skip:complex-dtype" in check(payload(args), "accepted_limitation")["evidence"][0]


@pytest.mark.parametrize("outcome", ["passed", "failed", "error", "xfailed", "xpassed"])
def test_new_skip_rejects_every_non_skipped_runxfail_outcome(tmp_path, outcome):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: "skipped"},
        baseline={NODE_A: ("failed", "complex dtype is not supported")},
    )
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: outcome})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    assert any(
        f"rejected runxfail outcome {outcome} for new skip" in item
        for item in check(payload(args), "runxfail")["evidence"]
    )


@pytest.mark.parametrize("outcome", ["passed", "skipped", "xfailed", "xpassed"])
def test_new_skip_rejects_every_nonfailure_baseline_outcome(tmp_path, outcome):
    args = artifacts(tmp_path, candidate={NODE_A: "skipped"}, baseline={NODE_A: outcome})
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: "skipped"})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    evidence = check(payload(args), "runxfail")["evidence"]
    if outcome == "passed":
        assert any("passed in the baseline" in item for item in evidence)
    else:
        assert any(f"got {outcome}" in item for item in evidence)


@pytest.mark.parametrize("outcome", ["failed", "error"])
def test_new_skip_accepts_failed_or_error_baseline_with_matching_signature(tmp_path, outcome):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: "skipped"},
        baseline={NODE_A: (outcome, "complex dtype is not supported")},
    )
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: "skipped"})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 0


def test_new_skip_rejects_baseline_failure_signature_mismatch(tmp_path):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: "skipped"},
        baseline={NODE_A: ("failed", "different failure")},
    )
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_A: "skipped"})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    assert any(
        "baseline failure signature did not match" in item
        for item in check(payload(args), "runxfail")["evidence"]
    )


def test_new_skip_requires_node_in_runxfail_report(tmp_path):
    args = artifacts(tmp_path, candidate={NODE_A: "skipped"}, baseline={NODE_A: "failed"})
    add_new_mark(
        args,
        registry_group(kind="skip", tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(args, tmp_path / "runxfail.json", {NODE_B: "passed"})
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    assert any(
        "missing runxfail outcome for new skip" in item
        for item in check(payload(args), "runxfail")["evidence"]
    )


@pytest.mark.parametrize(("kind", "candidate_outcome"), [("xfail", "xfailed"), ("skip", "skipped")])
def test_new_mark_rejects_node_that_passed_in_baseline(tmp_path, kind, candidate_outcome):
    args = artifacts(tmp_path, candidate={NODE_A: candidate_outcome}, baseline={NODE_A: "passed"})
    add_new_mark(
        args,
        registry_group(kind=kind, tests=["test_ops.py::Test::test_add_neuron_float32"]),
    )
    attach_runxfail(
        args,
        tmp_path / "runxfail.json",
        {NODE_A: ("failed", "complex dtype is not supported")},
    )
    args.accept_limitation = [(NODE_A, "complex-dtype")]
    assert MODULE.command_verify(args) == 1
    assert "passed in the baseline" in check(payload(args), "runxfail")["evidence"][0]


def test_upstream_rejects_missing_and_unaccounted_skip(tmp_path):
    args = artifacts(tmp_path)
    Path(args.op_nodeids).write_text(f"{NODE_A}\n{NODE_B}\n", encoding="utf-8")
    assert MODULE.command_verify(args) == 1
    assert "candidate:" + NODE_B in check(payload(args), "upstream")["evidence"]
    args = artifacts(
        tmp_path / "second", candidate={NODE_A: "skipped"}, baseline={NODE_A: "failed"}
    )
    assert MODULE.command_verify(args) == 1
    upstream = check(payload(args), "upstream")
    assert upstream["status"] == MODULE.FAILED
    assert "unaccounted outcome skipped" in upstream["evidence"][0]


def test_unknown_report_outcome_is_schema_error(tmp_path, capsys):
    args = artifacts(tmp_path)
    report = json.loads(Path(args.upstream_report).read_text(encoding="utf-8"))
    report["tests"][0]["outcome"] = "unknown"
    write_json(args.upstream_report, report)
    assert MODULE.command_verify(args) == 2
    assert "outcome is missing or invalid" in capsys.readouterr().out


def test_registry_reason_and_target_only_edits_are_full_changes():
    baseline = [registry_group(tests=["test_x"])]
    reason_edit = [registry_group(reason="complex dtype remains not supported", tests=["test_x"])]
    target_edit = [registry_group(tests=["test_x"], match_target="test_class")]
    reason_groups = MODULE.new_registry_groups(baseline, reason_edit)
    target_groups = MODULE.new_registry_groups(baseline, target_edit)
    assert reason_groups[0]["reason"] == "complex dtype remains not supported"
    assert target_groups[0]["match_target"] == "test_class"


def test_registry_delta_lists_added_removed_and_modified():
    baseline = [
        registry_group(tests=["test_removed"]),
        registry_group(reason="complex dtype not supported", tests=["test_modified"]),
    ]
    candidate = [
        registry_group(tests=["test_added"]),
        registry_group(reason="complex dtype remains not supported", tests=["test_modified"]),
    ]
    delta = MODULE.registry_delta(baseline, candidate)
    assert [entry[2] for entry in delta.added] == ["test_added"]
    assert [entry[2] for entry in delta.removed] == ["test_removed"]
    assert [(old[2], new[2]) for old, new in delta.modified] == [("test_modified", "test_modified")]


@pytest.mark.parametrize("kind", ["xfail", "skip"])
def test_removed_mark_passes_only_when_collected_node_passes(tmp_path, kind):
    removed = registry_group(kind=kind, tests=["test_ops.py::Test::test_add_neuron_float32"])
    args = artifacts(
        tmp_path / "passing", candidate={NODE_A: "passed"}, baseline={NODE_A: "failed"}
    )
    write_json(args.baseline_registry, [*seed_registry(), removed])
    assert MODULE.command_verify(args) == 0
    result = payload(args)
    assert check(result, "registry")["status"] == MODULE.OK
    assert check(result, "registry")["detail"].endswith("1 removed, 0 modified.")
    assert result["advisory"]["registry_delta"]["removed"]

    args = artifacts(
        tmp_path / "failing", candidate={NODE_A: "failed"}, baseline={NODE_A: "failed"}
    )
    write_json(args.baseline_registry, [*seed_registry(), removed])
    assert MODULE.command_verify(args) == 1
    assert any(
        "still has outcome failed" in item for item in check(payload(args), "registry")["evidence"]
    )


def test_registry_check_lists_empty_delta_categories(tmp_path):
    args = artifacts(tmp_path)
    assert MODULE.command_verify(args) == 0
    result = payload(args)
    assert result["advisory"]["registry_delta"] == {
        "added": [],
        "removed": [],
        "modified": [],
    }
    assert {"added: none", "removed: none", "modified: none"}.issubset(
        check(result, "registry")["evidence"]
    )


def test_environment_block_uses_verdict_and_reason_fields(tmp_path):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: ("failed", "ModuleNotFoundError: no module named dependency")},
        baseline={NODE_A: "passed"},
    )
    assert MODULE.command_verify(args) == 2
    result = payload(args)
    assert result["verdict"] == "BLOCKED"
    assert result["reason"] == "ENVIRONMENT"
    legacy_verdict = "BLOCKED_" + "ENVIRONMENT"
    assert legacy_verdict not in json.dumps(result)


def test_runner_rc_seven_with_passing_json_is_environment_block(tmp_path):
    args = artifacts(tmp_path)
    write_rc(Path(args.candidate_runner_rc), 7)
    assert MODULE.command_verify(args) == 2
    result = payload(args)
    assert result["verdict"] == MODULE.BLOCKED
    assert result["reason"] == MODULE.ENVIRONMENT
    assert check(result, "candidate_runner")["status"] == MODULE.INFRA


def test_runner_rc_zero_with_failing_json_is_inconsistent_fail(tmp_path):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: ("failed", "operator returned the wrong value")},
        baseline={NODE_A: "failed"},
    )
    write_rc(Path(args.candidate_runner_rc), 0)
    assert MODULE.command_verify(args) == 1
    result = payload(args)
    runner = check(result, "candidate_runner")
    assert result["verdict"] == MODULE.FAIL
    assert runner["status"] == MODULE.FAILED
    assert "exit code is zero" in runner["detail"]


def test_runner_rc_file_requires_one_integer(tmp_path, capsys):
    args = artifacts(tmp_path)
    Path(args.candidate_runner_rc).write_text("not-an-integer\n", encoding="utf-8")
    assert MODULE.command_verify(args) == 2
    assert "must contain one integer" in capsys.readouterr().out


@pytest.mark.parametrize(
    "message",
    [
        "ModuleNotFoundError: dependency",
        "ImportError: dependency",
        "No module named dependency",
        "device not available",
        "collection error while importing test",
        "INTERNALERROR in pytest",
    ],
)
def test_baseline_infrastructure_failure_blocks_environment(tmp_path, message):
    args = artifacts(
        tmp_path,
        candidate={NODE_A: "passed"},
        baseline={NODE_A: ("error", message)},
    )
    assert MODULE.command_verify(args) == 2
    result = payload(args)
    assert result["verdict"] == MODULE.BLOCKED
    assert result["reason"] == MODULE.ENVIRONMENT
    assert "baseline " + NODE_A in check(result, "upstream")["evidence"][0]


@pytest.mark.parametrize("unsafe_kind", ["directory", "symlink", "ancestor"])
def test_verify_rejects_unsafe_output_paths(tmp_path, unsafe_kind, capsys):
    args = artifacts(tmp_path)
    if unsafe_kind == "directory":
        output = tmp_path / "unsafe-output"
        output.mkdir()
    elif unsafe_kind == "symlink":
        target = tmp_path / "regular-target"
        target.write_text("keep", encoding="utf-8")
        output = tmp_path / "unsafe-output"
        output.symlink_to(target)
    else:
        output = tmp_path
    args.out = str(output)
    assert MODULE.command_verify(args) == 2
    assert "Verification input error" in capsys.readouterr().out


def test_verify_rejects_parent_swapped_for_symlink_after_preflight(monkeypatch, tmp_path, capsys):
    args = artifacts(tmp_path / "inputs")
    output_parent = tmp_path / "output"
    moved_parent = tmp_path / "moved-output"
    outside = tmp_path / "outside"
    output_parent.mkdir()
    outside.mkdir()
    args.out = str(output_parent / "verdict.json")

    def swap_parent(_output):
        output_parent.rename(moved_parent)
        output_parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(MODULE, "_after_output_preflight", swap_parent)
    assert MODULE.command_verify(args) == 2
    assert "Verification input error" in capsys.readouterr().out
    assert not (outside / "verdict.json").exists()
    assert not (moved_parent / "verdict.json").exists()


def test_snapshot_merge_and_report_paths_reject_unsafe_outputs(tmp_path):
    registry = tmp_path / "repo/tests/pytorch_tests_native/skip_registry"
    registry.mkdir(parents=True)
    (registry / "sample_registry.py").write_text("SKIP_GROUPS = []\n", encoding="utf-8")
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    snapshot_args = SimpleNamespace(repo_root=str(tmp_path / "repo"), out=str(unsafe_dir))
    assert MODULE.command_snapshot(snapshot_args) == 2

    items = tmp_path / "items.txt"
    items.write_text("test_ops.py\n", encoding="utf-8")
    paths_args = SimpleNamespace(
        output_dir=str(tmp_path), items_file=str(items), shard_id=None, out=str(items)
    )
    assert MODULE.command_report_paths(paths_args) == 2

    main = tmp_path / "main.json"
    write_report(main, {NODE_A: "passed"})
    manifest = write_json(
        tmp_path / "manifest.json", {"main": str(main), "retry": str(tmp_path / "retry.json")}
    )
    merge_args = SimpleNamespace(paths=manifest, out=str(main))
    assert MODULE.command_merge_reports(merge_args) == 2


def test_registry_rejects_empty_invalid_and_duplicate_snapshots(tmp_path, capsys):
    args = artifacts(tmp_path)
    write_json(args.candidate_registry, [])
    assert MODULE.command_verify(args) == 2
    assert "nonempty list" in capsys.readouterr().out
    write_json(args.candidate_registry, [{"kind": "xfail"}])
    assert MODULE.command_verify(args) == 2
    assert "contain exactly" in capsys.readouterr().out
    duplicate = registry_group(tests=["test_x", "test_x"])
    write_json(args.candidate_registry, [duplicate])
    assert MODULE.command_verify(args) == 2
    assert "contains duplicates" in capsys.readouterr().out


def test_registry_rejects_duplicate_full_entries(tmp_path, capsys):
    args = artifacts(tmp_path)
    write_json(
        args.candidate_registry,
        [registry_group(tests=["test_x"]), registry_group(tests=["test_x"])],
    )
    assert MODULE.command_verify(args) == 2
    assert "duplicate entry" in capsys.readouterr().out


def test_folder_and_class_target_matching():
    folder = registry_group(folders=["distributed/"])
    class_group = registry_group(patterns=["TestSpecial*"], match_target="test_class")
    node = "tests/pytorch_tests_native/test/distributed/test_x.py::TestSpecialOps::test_add"
    assert MODULE.matches_group(folder, node)
    assert MODULE.matches_group(class_group, node)
    assert not MODULE.matches_group(class_group, node.replace("TestSpecialOps", "TestOther"))


@pytest.mark.parametrize("report_name", ["upstream_report", "baseline_upstream_report"])
def test_duplicate_candidate_or_baseline_report_is_input_error(tmp_path, report_name, capsys):
    args = artifacts(tmp_path)
    path = Path(getattr(args, report_name))
    write_json(path, {"tests": [record(NODE_A, "passed"), record(NODE_A, "passed")]})
    assert MODULE.command_verify(args) == 2
    assert "duplicate node ID" in capsys.readouterr().out


def test_duplicate_runxfail_report_is_input_error(tmp_path, capsys):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    args.upstream_report_runxfail = write_json(
        tmp_path / "runxfail.json",
        {"tests": [record(NODE_A, "passed"), record(NODE_A, "passed")]},
    )
    args.runxfail_runner_rc = write_rc(tmp_path / "runxfail.rc", 0)
    assert MODULE.command_verify(args) == 2
    assert "duplicate node ID" in capsys.readouterr().out


@pytest.mark.parametrize("duplicate_in", ["main", "retry"])
def test_merge_rejects_duplicate_main_or_retry_reports(tmp_path, duplicate_in):
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    write_report(main, {NODE_A: "passed"})
    write_report(retry, {NODE_A: "passed"})
    target = main if duplicate_in == "main" else retry
    write_json(target, {"tests": [record(NODE_A, "passed"), record(NODE_A, "passed")]})
    manifest = write_json(tmp_path / "manifest.json", {"main": str(main), "retry": str(retry)})
    with pytest.raises(ValueError, match="duplicate node ID"):
        MODULE.merge_report_files(manifest, str(tmp_path / "out.json"))


def backward_args(tmp_path):
    args = artifacts(tmp_path)
    write_json(
        args.autograd_probe,
        {
            "op": "aten::add.Tensor",
            "differentiable": True,
            "backward_kind": "dedicated_op",
            "backward_ops": ["aten::add_backward"],
        },
    )
    return args


def add_backward_rows(args, *, eager=True, compile=True, compile_status="supported"):
    eager_rows = [
        {
            "op_name": "aten::add.Tensor",
            "status": "runs_on_neuron",
            "correctness_test": "pass",
        }
    ]
    compile_rows = [
        {
            "op_name": "aten::add.Tensor",
            "status": "supported",
            "correctness_test": "pass",
        }
    ]
    if eager:
        eager_rows.append(
            {
                "op_name": "aten::add_backward",
                "status": "runs_on_neuron",
                "correctness_test": "pass",
            }
        )
    if compile:
        compile_rows.append(
            {"op_name": "aten::add_backward", "status": compile_status, "correctness_test": "pass"}
        )
    write_csv(Path(args.coverage_csv), eager_rows)
    write_csv(Path(args.compile_coverage_csv), compile_rows)


def test_full_csv_backward_rows_succeed(tmp_path):
    args = backward_args(tmp_path)
    add_backward_rows(args)
    assert MODULE.command_verify(args) == 0
    backward = check(payload(args), "backward")
    assert backward["status"] == MODULE.OK
    assert backward["evidence"] == ["aten::add_backward"]


@pytest.mark.parametrize(
    ("eager", "compile", "compile_status", "message"),
    [
        (False, True, "supported", "no eager row"),
        (True, False, "supported", "no compile row"),
        (True, True, "unsupported", "failing compile coverage"),
    ],
)
def test_backward_missing_or_bad_rows_fail(tmp_path, eager, compile, compile_status, message):
    args = backward_args(tmp_path)
    add_backward_rows(args, eager=eager, compile=compile, compile_status=compile_status)
    assert MODULE.command_verify(args) == 1
    backward = check(payload(args), "backward")
    assert backward["status"] == MODULE.FAILED
    assert message in backward["evidence"][0]


def test_strict_report_root_record_coverage_manifest_and_probe_schemas(tmp_path, capsys):
    args = artifacts(tmp_path)
    write_json(args.upstream_report, [])
    assert MODULE.command_verify(args) == 2
    assert "root must be an object" in capsys.readouterr().out

    args = artifacts(tmp_path / "record")
    write_json(args.upstream_report, {"tests": [None]})
    assert MODULE.command_verify(args) == 2
    assert "must be an object" in capsys.readouterr().out

    args = artifacts(tmp_path / "coverage")
    Path(args.coverage_csv).write_text(
        "op_name,status\naten::add,runs_on_neuron\n", encoding="utf-8"
    )
    assert MODULE.command_verify(args) == 2
    assert "requires headers" in capsys.readouterr().out

    manifest = write_json(tmp_path / "bad-manifest.json", {"main": "x"})
    with pytest.raises(ValueError, match="exactly main and retry"):
        MODULE.merge_report_files(manifest, str(tmp_path / "out.json"))

    args = artifacts(tmp_path / "probe")
    write_json(args.autograd_probe, {"op": "aten::add.Tensor"})
    assert MODULE.command_verify(args) == 2
    assert "requires fields" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("differentiable", "backward_kind", "backward_ops"),
    [
        (True, "none", []),
        (False, "derived", []),
        (False, "dedicated_op", ["aten::add_backward"]),
        (None, "derived", []),
    ],
)
def test_probe_rejects_conflicting_differentiability(
    tmp_path, capsys, differentiable, backward_kind, backward_ops
):
    args = artifacts(tmp_path)
    write_json(
        args.autograd_probe,
        {
            "op": "aten::add.Tensor",
            "differentiable": differentiable,
            "backward_kind": backward_kind,
            "backward_ops": backward_ops,
        },
    )
    assert MODULE.command_verify(args) == 2
    assert "conflicts with backward_kind" in capsys.readouterr().out


def test_command_verify_unwritable_output_returns_two(tmp_path, capsys):
    args = artifacts(tmp_path)
    args.out = str(tmp_path / "missing" / "verdict.json")
    assert MODULE.command_verify(args) == 2
    assert "Verification input error" in capsys.readouterr().out


def test_direct_noninteger_unit_rc_is_rejected(tmp_path, capsys):
    args = artifacts(tmp_path)
    args.unit_rc = "0"
    assert MODULE.command_verify(args) == 2
    assert "--unit-rc must be an integer" in capsys.readouterr().out


def load_runner_module(tmp_path):
    source_runner = (
        Path(__file__).resolve().parents[4]
        / "tests/pytorch_tests_native/runner/run_upstream_tests.py"
    )
    copied_runner_dir = tmp_path / "repo/tests/pytorch_tests_native/runner"
    copied_runner_dir.mkdir(parents=True)
    assert not (tmp_path / "repo/.github").exists()
    (tmp_path / "repo/pyproject.toml").touch()
    (tmp_path / "repo/torch_neuronx").mkdir()
    (tmp_path / "repo/torch_neuronx/__init__.py").touch()
    copied_runner = copied_runner_dir / source_runner.name
    shutil.copy2(source_runner, copied_runner)

    spec = importlib.util.spec_from_file_location("real_run_upstream_tests", copied_runner)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("shard_id", [None, 3])
def test_report_safe_name_matches_real_runner_for_item_counts(tmp_path, count, shard_id):
    runner = load_runner_module(tmp_path)
    items = [f"folder/test_{index}.py" for index in range(count)]
    assert MODULE.derive_safe_name(items, shard_id) == runner.derive_safe_name(items, shard_id)


def test_report_path_manifest_has_canonical_retry_and_shard(tmp_path):
    items = tmp_path / "items.txt"
    items.write_text("test_a.py\ntest_b.py\n", encoding="utf-8")
    manifest = MODULE.report_path_manifest(str(tmp_path), str(items), 2)
    assert manifest["main"].endswith("test_results_test_a_and_2_more_shard2.json")
    assert manifest["retry"].endswith("test_results_test_a_and_2_more_shard2_retry.json")


def test_snapshot_does_not_execute_candidate_code(tmp_path):
    registry = tmp_path / "tests/pytorch_tests_native/skip_registry"
    registry.mkdir(parents=True)
    side_effect = tmp_path / "executed"
    (registry / "evil_registry.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(side_effect)!r}).write_text('executed')\n"
        "SKIP_GROUPS = [SkipGroup(reason='complex dtype not supported', "
        "tests=['test_ops.py::Test::test_add'])]\n",
        encoding="utf-8",
    )
    snapshot = MODULE.snapshot_registry(str(tmp_path))
    assert snapshot[0]["kind"] == "skip"
    assert not side_effect.exists()


def test_cli_requires_integer_unit_rc(capsys):
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(["verify"])
    assert missing.value.code == 2
    missing_error = capsys.readouterr().err
    assert "--unit-rc" in missing_error
    assert "--candidate-runner-rc" in missing_error
    assert "--baseline-runner-rc" in missing_error
    assert "required" in missing_error
    with pytest.raises(SystemExit) as invalid:
        parser.parse_args(["verify", "--unit-rc", "not-an-int"])
    assert invalid.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


@pytest.mark.parametrize(
    "subcommand", ["snapshot-registry", "report-paths", "merge-reports", "verify"]
)
def test_each_subcommand_help(subcommand, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_op_acceptance.py"
    assert script == SCRIPT
    with pytest.raises(SystemExit) as result:
        MODULE.main([subcommand, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_verify_op_acceptance_script_help_subprocess():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_runxfail_report_requires_runner_rc_file(tmp_path, capsys):
    args = artifacts(tmp_path, candidate={NODE_A: "xfailed"}, baseline={NODE_A: "failed"})
    add_new_mark(args, registry_group(tests=["test_ops.py::Test::test_add_neuron_float32"]))
    args.upstream_report_runxfail = write_report(
        tmp_path / "runxfail.json",
        {NODE_A: ("failed", "complex dtype is not supported")},
    )
    assert MODULE.command_verify(args) == 2
    assert "must be both present or both absent" in capsys.readouterr().out


def test_runxfail_runner_rc_without_report_is_usage_error(tmp_path, capsys):
    args = artifacts(tmp_path)
    args.runxfail_runner_rc = write_rc(tmp_path / "orphan-runxfail.rc", 7)
    assert MODULE.command_verify(args) == 2
    assert "must be both present or both absent" in capsys.readouterr().out
