"""Check the shared workflow routes and persistent golden write controls."""

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"


@pytest.fixture
def workflows():
    return {
        path.name: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for path in WORKFLOW_DIR.glob("*.yml")
    }


@pytest.mark.parametrize(
    "name, parameter",
    [
        ("_reusable-wheel-pipeline.yml", "refresh_persistent_goldens"),
        ("_reusable-wheel-tests.yml", "refresh_persistent_goldens"),
        ("_reusable-full-wheel-tests.yml", "refresh_persistent_goldens"),
        ("_reusable-block-perf.yml", "refresh-persistent-goldens"),
    ],
)
def test_golden_refresh_requires_explicit_workflow_input(workflows, name, parameter):
    config = workflows[name]["on"]["workflow_call"]["inputs"][parameter]

    assert config["type"] == "boolean"
    assert config["required"] == "false"
    assert config["default"] == "false"


def test_nightly_forwards_refresh_to_the_full_test_wheel(workflows):
    nightly = workflows["nightly.yml"]["jobs"]["wheels"]
    assert nightly["uses"] == "./.github/workflows/_reusable-wheel-pipeline.yml"
    assert nightly["with"]["refresh_persistent_goldens"] == "${{ github.ref == 'refs/heads/main' }}"

    validate = workflows["_reusable-wheel-pipeline.yml"]["jobs"]["validate-wheel"]
    assert validate["uses"] == "./.github/workflows/_reusable-wheel-tests.yml"
    assert validate["with"]["run_full_suite"] == "${{ matrix.run_full_suite }}"
    assert (
        validate["with"]["refresh_persistent_goldens"] == "${{ inputs.refresh_persistent_goldens }}"
    )

    full = workflows["_reusable-wheel-tests.yml"]["jobs"]["full-test-suite"]
    assert full["if"] == "inputs.run_full_suite"
    assert full["uses"] == "./.github/workflows/_reusable-full-wheel-tests.yml"
    assert full["with"]["refresh_persistent_goldens"] == "${{ inputs.refresh_persistent_goldens }}"
    assert full["with"]["enforce_mfu_threshold"] == "true"
    assert full["with"]["mfu_threshold_multiplier"] == "1.0"

    block = workflows["_reusable-full-wheel-tests.yml"]["jobs"]["test-block-perf"]
    assert block["uses"] == "./.github/workflows/_reusable-block-perf.yml"
    assert block["with"]["refresh-persistent-goldens"] == "${{ inputs.refresh_persistent_goldens }}"
    assert block["with"]["include-trn3pds-b1"] == "${{ vars.TRN3PDS_B1_ENABLED != 'false' }}"
    assert block["with"]["enforce-mfu-threshold"] == "${{ inputs.enforce_mfu_threshold }}"
    assert block["with"]["mfu-threshold-multiplier"] == "${{ inputs.mfu_threshold_multiplier }}"


def test_dependency_guard_enforces_gates_without_refresh(workflows):
    suite = workflows["dependency-guard.yml"]["jobs"]["full-test-suite"]
    assert suite["uses"] == "./.github/workflows/_reusable-full-wheel-tests.yml"
    assert suite["with"]["refresh_persistent_goldens"] == "false"
    assert suite["with"]["enforce_mfu_threshold"] == "true"

    inputs = workflows["_reusable-full-wheel-tests.yml"]["on"]["workflow_call"]["inputs"]
    assert (
        suite["with"].get("mfu_threshold_multiplier", inputs["mfu_threshold_multiplier"]["default"])
        == "1.0"
    )


def test_all_block_callers_run_default_then_advisory_hostcc(workflows):
    for name in ("pull.yml", "_reusable-full-wheel-tests.yml"):
        callers = [
            job
            for job in workflows[name]["jobs"].values()
            if job.get("uses") == "./.github/workflows/_reusable-block-perf.yml"
        ]
        assert callers
        assert all("run-hostcc" not in job.get("with", {}) for job in callers)
    release = workflows["neuron_sdk_release.yml"]["jobs"]["wheels"]
    assert release["uses"] == "./.github/workflows/_reusable-wheel-pipeline.yml"
    assert release["with"]["refresh_persistent_goldens"] == "false"
    workflow = workflows["_reusable-block-perf.yml"]
    assert "run-hostcc" not in workflow["on"]["workflow_call"]["inputs"]
    job = workflow["jobs"]["block-perf"]
    assert "continue-on-error" not in job
    command = next(
        step["with"]["test-command"]
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/run-test-suite"
    )
    assert "TORCH_NEURONX_ENABLE_STREAMS=1" in command
    assert "TORCH_NEURONX_ENABLE_HOST_CC=0" in command
    assert (
        "${{ matrix.group.id != 'infra-group' && '--both-modes --hostcc-advisory' || '' }}"
        in command
    )
    assert "--both-modes --hostcc-advisory" in command


def test_pr_block_checks_have_read_only_storage(workflows):
    block = workflows["pull.yml"]["jobs"]["block-perf"]
    assert block["uses"] == "./.github/workflows/_reusable-block-perf.yml"
    assert block["with"].get("refresh-persistent-goldens", "false") == "false"
    assert "neuron.internal_compute.persistent-storage:read" in block["with"]["runner-labels"]
    assert "neuron.internal_compute.persistent-storage:write" not in block["with"]["runner-labels"]


def test_pr_checks_skip_the_suite_for_release_branches(workflows):
    steps = workflows["pull.yml"]["jobs"]["check-approval"]["steps"]
    script = next(step["with"]["script"] for step in steps if step.get("id") == "check")

    assert "context.payload.pull_request.base.ref.startsWith('release-')" in script
    assert script.count("approved && !draft && !release && !skip") == 2


def test_block_workflow_guards_all_write_paths(workflows):
    block = workflows["_reusable-block-perf.yml"]["jobs"]["block-perf"]
    guard = (
        "inputs.refresh-persistent-goldens && github.ref == 'refs/heads/main' && "
        "endsWith(github.workflow_ref, '/.github/workflows/nightly.yml@refs/heads/main')"
    )
    assert f"{guard} && 'write' || 'read'" in block["runs-on"]
    assert "matrix.group.id != 'infra-group' && format(" in block["runs-on"]

    writer = f"{guard} && matrix.group.id != 'infra-group'"
    assert (
        f"{writer} && format('block-perf-goldens-{{0}}-{{1}}', matrix.platform, matrix.group.id)"
        in block["concurrency"]["group"]
    )
    assert block["concurrency"]["cancel-in-progress"] == "false"

    step = next(
        step for step in block["steps"] if step.get("uses") == "./.github/actions/run-test-suite"
    )
    assert step["if"] == "${{ !inputs.noop }}"
    assert f"{writer} && '--refresh-persistent-goldens' || ''" in step["with"]["test-command"]
    assert "GITHUB_REF_NAME=${{ github.ref_name }}" in step["with"]["test-command"]


def test_only_nightly_enables_refresh_in_callers(workflows):
    enabled = []
    for name, workflow in workflows.items():
        for job in workflow.get("jobs", {}).values():
            value = job.get("with", {}).get("refresh_persistent_goldens")
            if value and value not in ("false", "${{ inputs.refresh_persistent_goldens }}"):
                enabled.append(name)

    assert enabled == ["nightly.yml"]


def test_shared_workflow_references_resolve(workflows):
    assert "nightly-cp313.yml" not in workflows
    assert "_block-perf.yml" not in workflows
    for workflow in workflows.values():
        for job in workflow.get("jobs", {}).values():
            target = job.get("uses", "")
            if target.startswith("./.github/workflows/"):
                assert Path(target).name in workflows, target
