"""Check the release wheel publish controls."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = "./.github/workflows/_reusable-wheel-publish.yml"
MANUAL_PROMOTE_WORKFLOW = "./.github/workflows/_reusable-manual-wheel-promote.yml"


def load_workflow(name):
    return yaml.load((WORKFLOW_DIR / name).read_text(), Loader=yaml.BaseLoader)


def test_the_sdk_publish_has_no_second_approval_and_runs_for_a_successful_push():
    release = load_workflow("neuron_sdk_release.yml")
    gate = release["jobs"]["publish-gate"]

    assert "environment" not in gate
    assert "success()" in gate["if"]
    assert "github.event_name == 'push'" in gate["if"]
    assert "inputs.publish_wheels" in gate["if"]


def test_a_release_push_cancels_only_an_older_push():
    release = load_workflow("neuron_sdk_release.yml")

    assert release["concurrency"]["group"] == (
        "neuron-sdk-release-${{ github.ref_name }}-${{ github.event_name }}"
    )
    assert release["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'push' }}"


def test_the_sdk_publish_uses_the_shared_workflow():
    release = load_workflow("neuron_sdk_release.yml")
    publish = release["jobs"]["publish"]

    assert publish["uses"] == PUBLISH_WORKFLOW
    assert publish["needs"] == ["wheels", "publish-gate"]
    assert publish["if"] == "success()"
    assert publish["with"]["matrix"] == "${{ needs.wheels.outputs.matrix }}"
    assert publish["with"]["release_tag"] == "${{ github.ref_name }}"
    assert publish["with"]["source_commit"] == "${{ github.sha }}"


def test_the_nightly_publish_uses_the_shared_workflow():
    nightly = load_workflow("nightly.yml")
    publish = nightly["jobs"]["publish-wheels"]

    assert publish["uses"] == PUBLISH_WORKFLOW
    assert publish["needs"] == ["wheels"]
    assert publish["if"] == "success() && github.ref_name == 'main'"
    assert publish["with"]["matrix"] == "${{ needs.wheels.outputs.publish_matrix }}"
    assert publish["with"]["release_tag"] == "stable-wheels"
    assert publish["with"]["source_commit"] == "${{ github.sha }}"
    assert nightly["jobs"]["publish-primary-image"]["needs"] == ["wheels", "publish-wheels"]


def test_the_manual_promote_wrappers_select_the_source_type():
    cases = [
        ("manual-promote-nightly.yml", "nightly"),
        ("manual-promote-release.yml", "release"),
    ]

    for workflow_name, promotion_kind in cases:
        promote = load_workflow(workflow_name)
        job = promote["jobs"]["promote"]

        assert promote["permissions"]["actions"] == "read"
        assert promote["permissions"]["contents"] == "write"
        assert promote["permissions"]["id-token"] == "write"
        assert job["uses"] == MANUAL_PROMOTE_WORKFLOW
        assert job["with"]["promotion_kind"] == promotion_kind
        assert job["with"]["source_run_id"] == "${{ inputs.source_run_id }}"
        assert job["with"]["override_reason"] == "${{ inputs.override_reason }}"


def test_the_manual_promote_keeps_two_access_controls():
    promote = load_workflow("_reusable-manual-wheel-promote.yml")
    jobs = promote["jobs"]

    assert "MANUAL_PROMOTE_APPROVERS" in jobs["authorize"]["steps"][0]["env"]["APPROVERS"]
    assert jobs["review-request"]["needs"] == ["collect"]
    assert jobs["approve"]["needs"] == ["collect", "review-request"]
    assert jobs["approve"]["environment"]["name"] == "stable-promote"
    assert "actions/runs" in jobs["approve"]["environment"]["url"]
    assert jobs["publish"]["needs"] == ["collect", "approve"]
    assert jobs["publish"]["uses"] == PUBLISH_WORKFLOW
    assert jobs["publish"]["with"]["source_run_id"] == "${{ inputs.source_run_id }}"

    review = jobs["review-request"]["steps"][0]
    assert review["env"]["OVERRIDE_REASON"] == "${{ inputs.override_reason }}"
    assert review["env"]["SOURCE_CONCLUSION"] == "${{ needs.collect.outputs.source_conclusion }}"
    assert review["env"]["WHEEL_MATRIX"] == "${{ needs.collect.outputs.matrix }}"
    assert "Manual wheel promotion request" in review["with"]["script"]
    assert "${{ inputs.override_reason }}" not in review["with"]["script"]


def test_the_manual_promote_checks_the_source_run():
    promote = load_workflow("_reusable-manual-wheel-promote.yml")
    script = promote["jobs"]["collect"]["steps"][0]["with"]["script"]

    assert "process.env.PROMOTION_KIND === 'nightly'" in script
    assert "run.path !== nightlyWorkflow" in script
    assert "run.path !== releaseWorkflow" in script
    assert "dispatchBranch !== 'main'" in script
    assert "releaseBranch.test(run.head_branch)" in script
    assert "dispatchBranch !== run.head_branch" in script
    assert "run.status !== 'completed'" in script
    assert "core.setOutput('sha', run.head_sha)" in script
    assert "core.setOutput('conclusion', run.conclusion)" in script
    assert "core.setOutput('url', run.html_url)" in script


def test_the_manual_promote_needs_a_complete_wheel_set():
    promote = load_workflow("_reusable-manual-wheel-promote.yml")
    script = promote["jobs"]["collect"]["steps"][3]["with"]["script"]

    assert "expectedIds" in script
    assert "missing.length || unexpected.length || duplicateCount" in script
    assert "!wheel.request_id" in script


def test_the_manual_promote_publishes_only_the_nightly_image():
    promote = load_workflow("_reusable-manual-wheel-promote.yml")
    jobs = promote["jobs"]

    assert jobs["publish-primary-image"]["needs"] == ["collect", "publish"]
    assert jobs["publish-primary-image"]["if"] == (
        "success() && needs.collect.outputs.publish_image == 'true'"
    )
    assert jobs["publish"]["with"]["release_tag"] == "${{ needs.collect.outputs.release_tag }}"


def test_the_shared_publisher_checks_the_branch_target():
    publish = load_workflow("_reusable-wheel-publish.yml")
    jobs = publish["jobs"]
    script = jobs["validate-target"]["steps"][0]["run"]

    assert jobs["publish-wheel"]["needs"] == ["validate-target"]
    assert 'WORKFLOW_BRANCH}" == "main"' in script
    assert 'RELEASE_TAG}" == "stable-wheels"' in script
    assert "release-[0-9]+" in script
    assert 'RELEASE_TAG}" == "${WORKFLOW_BRANCH}"' in script


def test_the_wheel_metadata_outlives_one_day():
    pipeline = load_workflow("_reusable-wheel-pipeline.yml")
    upload = pipeline["jobs"]["build-wheel"]["steps"][-1]

    assert upload["with"]["name"] == "wheel-metadata-${{ matrix.id }}"
    assert int(upload["with"]["retention-days"]) >= 7
