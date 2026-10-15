"""Check that every Neuron compute job selects a Neuron driver."""

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
COMPUTE_LABEL = "neuron.internal_compute"
DKMS_PREFIX = "neuron.dkms:"
MAIN_TAG = "private-torch-neuronx-main-stable"

# These reusable workflows start Neuron compute jobs. Each one takes a dkms input.
REUSABLES = {
    "_reusable-block-perf.yml",
    "_reusable-internal-tests.yml",
    "_reusable-upstream-tests.yml",
    "_reusable-vertical-tests.yml",
    "_reusable-moduscope-tests.yml",
    "_reusable-wheel-tests.yml",
    "_reusable-full-wheel-tests.yml",
    "_reusable-wheel-pipeline.yml",
}


@pytest.fixture
def workflows():
    return {
        path.name: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for path in WORKFLOW_DIR.glob("*.yml")
    }


def runs_on_text(job):
    return yaml.dump(job.get("runs-on", ""))


def test_every_compute_job_selects_a_driver(workflows):
    missing = []
    for name, workflow in workflows.items():
        for job_id, job in workflow.get("jobs", {}).items():
            labels = runs_on_text(job)
            if COMPUTE_LABEL in labels and DKMS_PREFIX not in labels:
                missing.append(f"{name}:{job_id}")

    assert missing == []


def test_every_caller_of_a_compute_reusable_passes_a_driver(workflows):
    missing = []
    for name, workflow in workflows.items():
        for job_id, job in workflow.get("jobs", {}).items():
            target = Path(job.get("uses", "")).name
            if target not in REUSABLES:
                continue
            # pull.yml block-perf sets the label in its full runner-labels override.
            given = job.get("with", {})
            if "dkms" in given or DKMS_PREFIX in given.get("runner-labels", ""):
                continue
            missing.append(f"{name}:{job_id}")

    assert missing == []


@pytest.mark.parametrize("name", sorted(REUSABLES))
def test_a_compute_reusable_takes_a_driver_input(workflows, name):
    assert "dkms" in workflows[name]["on"]["workflow_call"]["inputs"]


def test_only_a_tested_driver_reaches_the_stable_tag(workflows):
    promoters = []
    for name, workflow in workflows.items():
        for job_id, job in workflow.get("jobs", {}).items():
            labels = runs_on_text(job)
            if "neuron.dkms-tag-suffix:" not in labels:
                continue
            promoters.append(f"{name}:{job_id}")
            # A channel name here would promote the driver of the day, not the
            # driver the tests used.
            assert MAIN_TAG not in labels
            assert "neuron.dkms:alpha" not in labels
            assert "neuron.dkms:release-" not in labels
            assert "neuron.dkms:public" not in labels

    assert sorted(promoters) == [
        "dependency-guard.yml:promote-bases",
        "manual-promote-dep-guard.yml:promote",
    ]


def test_the_release_driver_channel_follows_the_train(workflows):
    guard = workflows["neuron_sdk_release.yml"]["jobs"]["preflight"]["steps"][1]

    assert 'echo "dkms_channel=release-$TRAIN"' in guard["run"]


def test_the_release_driver_lookup_does_not_block_a_new_train(workflows):
    step = workflows["neuron_sdk_release.yml"]["jobs"]["build-base"]["steps"][0]

    assert step["id"] == "dkms"
    assert "::warning::" in step["run"]
    assert "exit 0" in step["run"]


def test_the_dependency_guard_driver_lookup_is_strict(workflows):
    build = workflows["dependency-guard.yml"]["jobs"]["build-complete"]
    step = next(step for step in build["steps"] if step.get("id") == "dkms")

    assert "::warning::" not in step["run"]
    assert "exit 0" not in step["run"]
    assert build["outputs"]["dkms_version"] == "${{ steps.dkms.outputs.version }}"
