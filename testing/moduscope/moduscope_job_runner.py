from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

#: TTFI clears the compile caches before every model, so a run pays a recompile per model for
#: numbers nothing reads. Set explicitly rather than left to the suite config's default.
TTFI_DISABLE_OVERRIDE = "++task_config.enable_ttfi_metrics=false"

#: The id lands in a command the pod re-runs through a shell, so anything outside this character
#: set could execute there.
MODEL_ID_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

MODEL_IDS_ENV_VAR = "MODEL_IDS"
SUITE_TAG_ENV_VAR = "SUITE_TAG"
JOB_TIMEOUT_MINUTES_ENV_VAR = "MODUSCOPE_JOB_TIMEOUT_MINUTES"

#: Minutes of the job bound that are not ModuScope's to spend: the image scan gate and pod start
#: both land before it. Subtracted here rather than in the workflow so the arithmetic is testable.
#: Too small a reserve and the job bound fires first, killing the step before run_test.sh copies
#: outputs/ -- losing the grid that shows the hang.
COPY_BUFFER_MINUTES = 40

MODUSCOPE_ENTRY_POINT = "cli_scripts.task_runner"

#: The exit code the `timeout` command uses for something it killed.
TIMEOUT_EXIT_CODE = 124

OUTPUT_SUBDIR = os.path.join("outputs", "moduscope")


def normalize_model_id(model_id: str) -> str:
    """Return the form of ``model_id`` a grid row carries: ``org/name`` -> ``org_name``.

    Lives here rather than in ``moduscope_grid`` so dispatch refuses a colliding pair at the
    spelling grading later reads rows with: the grid module imports this, and a test forbids the
    reverse edge.
    """
    return model_id.replace("/", "_")


def resolve_model_ids(raw: str) -> tuple[str, ...]:
    """Parse the comma-separated model-id list into the ids to run, in the order given."""
    asked = [part.strip() for part in raw.split(",") if part.strip()]
    if not asked:
        raise ValueError(
            f"{MODEL_IDS_ENV_VAR}={raw!r} names no model. The list comes from the workflow "
            f"input, so there is no built-in set to fall back to."
        )
    malformed = [model_id for model_id in asked if not MODEL_ID_RE.fullmatch(model_id)]
    if malformed:
        raise ValueError(
            f"{MODEL_IDS_ENV_VAR}={raw!r} contains ids that are not a bare owner/name over "
            f"[A-Za-z0-9._-]: {malformed}. The value is interpolated into a command the pod "
            f"re-executes with bash -c, so anything outside that charset can run there."
        )
    # Exact repeats collapse first. What survives is grading-fatal: foo/bar_baz and foo_bar/baz
    # both become foo_bar_baz, so one passing row would prove both.
    model_ids = tuple(dict.fromkeys(asked))
    groups: dict[str, list[str]] = {}
    for model_id in model_ids:
        groups.setdefault(normalize_model_id(model_id), []).append(model_id)
    colliding = {norm: ids for norm, ids in groups.items() if len(ids) > 1}
    if colliding:
        raise ValueError(
            f"{MODEL_IDS_ENV_VAR}={raw!r} names ids that collide once normalized: "
            f"{colliding}. A grid row carries only the normalized form, so one row "
            f"would satisfy every id in a colliding group."
        )
    return model_ids


def read_env() -> tuple[str, str, int]:
    """Return ``(model_ids_raw, suite_tag, timeout_seconds)`` from the environment.

    The workflow passes its whole job bound; the reserve is subtracted here.
    """
    names = (MODEL_IDS_ENV_VAR, SUITE_TAG_ENV_VAR, JOB_TIMEOUT_MINUTES_ENV_VAR)
    values = {name: os.environ.get(name, "") for name in names}
    missing = [name for name in names if not values[name].strip()]
    if missing:
        raise ValueError(
            f"required environment variable(s) unset or blank: {missing}. The workflow "
            f"exports all three before invoking this module."
        )
    raw_minutes = values[JOB_TIMEOUT_MINUTES_ENV_VAR].strip()
    try:
        job_minutes = int(raw_minutes)
    except ValueError:
        raise ValueError(
            f"{JOB_TIMEOUT_MINUTES_ENV_VAR}={raw_minutes!r} is not an integer number of minutes."
        ) from None
    timeout_seconds = (job_minutes - COPY_BUFFER_MINUTES) * 60
    if timeout_seconds <= 0:
        raise ValueError(
            f"{JOB_TIMEOUT_MINUTES_ENV_VAR}={raw_minutes!r} leaves nothing after the "
            f"{COPY_BUFFER_MINUTES}-minute reserve, so ModuScope would be killed before it "
            f"starts. Raise the job bound above {COPY_BUFFER_MINUTES}."
        )
    return (
        values[MODEL_IDS_ENV_VAR],
        values[SUITE_TAG_ENV_VAR].strip(),
        timeout_seconds,
    )


def output_dir(cwd: str | None = None) -> str:
    """Return the directory ModuScope must write into, derived from the working directory.

    Derived rather than configured: run_test.sh copies ``outputs/`` off the pod after the test
    command returns, so anything written elsewhere never leaves the pod.
    """
    return os.path.join(cwd if cwd is not None else os.getcwd(), OUTPUT_SUBDIR)


def prepare_output_dir(cwd: str | None = None) -> str:
    """Create an empty ``output_dir()``, removing whatever was there; return the path.

    Emptied because the grader globs the artifact for every ``grid_metrics.json`` and requires all
    of them to pass: one grid left by an earlier run would fail a good run.
    """
    path = output_dir(cwd)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


#: Only these suites read ``hf_cfg_wishlist``, so only here do the model ids decide what runs.
#: Any other suite takes its model set from ``model_configurations`` and the ids silently do
#: nothing. Before adding one, check that its task config reads the wishlist.
WISHLIST_SUITE_TAGS = ("full_dense_decoder", "full_dense_decoder_compile")


def assert_suite_reads_wishlist(suite_tag: str) -> str:
    """Return ``suite_tag`` if it is a suite the model ids can cap, else raise.

    The dispatch screen offers only these as a ``choice``, but a ``workflow_call`` input cannot
    be a choice, so this is the only thing standing between a nightly or dep-guard caller and a
    run that quietly measures a different model set. Membership in a closed set of literals is
    also what makes a separate shell-safety pattern unnecessary.
    """
    if suite_tag not in WISHLIST_SUITE_TAGS:
        raise ValueError(
            f"suite_tag={suite_tag!r} is not a suite that reads hf_cfg_wishlist "
            f"(allowed: {list(WISHLIST_SUITE_TAGS)}). It would take its model set from "
            f"model_configurations instead, so the requested ids would silently do nothing "
            f"and the run would measure a set nobody asked for."
        )
    return suite_tag


def build_moduscope_argv(model_ids: tuple[str, ...], out_dir: str, suite_tag: str) -> list[str]:
    """Build the ModuScope command line for ``model_ids``."""
    if not model_ids:
        raise ValueError(
            "model_ids is empty, which would render the wishlist override falsy and "
            "reinstate ModuScope's own model discovery -- running the suite's own model set "
            "instead of these ids."
        )
    assert_suite_reads_wishlist(suite_tag)
    wishlist = ",".join(f"'{model_id}'" for model_id in model_ids)
    return [
        sys.executable,
        "-m",
        MODUSCOPE_ENTRY_POINT,
        f"--config-name={suite_tag}",
        f"++task_config.hf_cfg_wishlist=[{wishlist}]",
        f"++task_config.artifact_directory={out_dir}",
        TTFI_DISABLE_OVERRIDE,
    ]


def main() -> int:
    """Resolve the inputs, run ModuScope, return its exit code.

    The inner timeout is what keeps the grid alive through a hang: letting the job's own timeout
    fire kills the step before run_test.sh reaches the line that copies ``outputs/`` off the pod,
    taking the grid with it.
    """
    # Set here rather than in the workflow so tests cover it. Without it, writes hit the Brazil
    # production bucket and uploads try roles this side does not have.
    os.environ["MODUSCOPE_CI"] = "github"
    model_ids_raw, suite_tag, timeout_seconds = read_env()
    model_ids = resolve_model_ids(model_ids_raw)
    out_dir = prepare_output_dir()
    print(f"Resolved {len(model_ids)} model id(s): {', '.join(model_ids)}")
    print(f"Output directory (emptied): {out_dir}")

    argv_out = build_moduscope_argv(model_ids, out_dir, suite_tag)
    print("Running: " + " ".join(argv_out))
    try:
        returncode = subprocess.run(argv_out, check=False, timeout=timeout_seconds).returncode
    except subprocess.TimeoutExpired:
        print(f"ModuScope exceeded {timeout_seconds}s and was killed.")
        return TIMEOUT_EXIT_CODE
    print(f"ModuScope exited {returncode}.")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
