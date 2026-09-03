"""Tests for the ModuScope suite wrapper: ``moduscope_job_runner`` + ``moduscope_grid``.

Grid fixtures are copied verbatim from a real executed run, trailing ``" X"`` included.

Three conventions:

1. Override strings, the grid file name, the ``_config`` anchor, the ``outputs/`` path
   parts and the id grammar are literals rather than read back from the modules: a
   fixture derived from the constant it guards cannot notice that constant weakening.
2. The workflows' ``model-ids`` defaults are split here by the tests' own parse. The
   runner is what resolves them, so it cannot also be the witness for what they say.
3. The runner must not import ModuScope: these tests run where it is absent, and a
   top-level import would make the whole file uncollectable rather than failing one
   test. ModuScope is reached only as a subprocess.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# The modules under test are in testing/moduscope/ -- test infrastructure, which is not a test
# and so does not belong under tests/. That directory is not a package either, so pytest's
# rootdir-based collection cannot import from it and the path goes on sys.path by hand. This
# stays above the imports and stays a sys.path call: an assignment here would make every import
# below it an E402.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing" / "moduscope"))

import moduscope_grid as grid
import moduscope_job_runner as mjr

#: Where the grid module lives, for the subprocess probe further down.
GRID_DIR = str(Path(grid.__file__).resolve().parent)

#: parents[3] from tests/testing/moduscope/<file> is the repo root. Derived from this file
#: rather than from an imported module, so the tests that assert where those modules live are
#: not comparing a path against itself.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The reusable workflow owns the suite; the wrapper owns the dispatch screen.
REUSABLE_WORKFLOW = REPO_ROOT / ".github/workflows/_moduscope-tests.yml"
WRAPPER_WORKFLOW = REPO_ROOT / ".github/workflows/manual-moduscope-tests.yml"

GRID_FILE_NAME = "grid_metrics.json"
ENTRY_POINT = "cli_scripts.task_runner"

#: The two path components run_test.sh's copy of ``outputs/`` fixes. Literals, so a
#: runner that started writing somewhere the copy never reaches fails here.
OUTPUT_PARTS = ("outputs", "moduscope")

#: ``++`` (add-or-override) rather than ``+`` (add, and error if the key exists): no
#: shipped suite config sets this key today, but ``+`` would hard-error at Hydra
#: composition the day one did.
TTFI_DISABLE_OVERRIDE = "++task_config.enable_ttfi_metrics=false"

#: For the call sites whose subject is not the tag. Allowlisted, because building the command
#: line now refuses a suite that ignores the wishlist -- but deliberately NOT the shipped
#: default, so a test claiming the tag reached the command line still cannot be satisfied by a
#: default leaking through.
TAG = "full_dense_decoder_compile"

#: Any positive bound, for the call sites whose subject is not the timeout.
JOB_MINUTES = "90"

TINY = "smeoni/tiny-random-LlamaForCausalLM"
OLMO = "allenai/OLMo-1B-hf"
#: Already carries an underscore and hyphens, so it distinguishes "replace the slash"
#: from "sanitize the id".
STABLELM = "stabilityai/stablelm-2-1_6b"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
LLAMA = "meta-llama/Llama-3.1-8B"

#: Deliberately not among the requested ids, for the "rode along" arms.
DECOY = "nobody/not-requested"

#: The grammar the runner enforces, spelled out here so a weakened pattern in the
#: module cannot move the fixture and the assertion together.
HUB_ID_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

#: Anything either shell could act on. The regex above already excludes these; this
#: states the intent the charset exists to serve.
SHELL_METACHARACTERS = "'\"`$;|&()<>*?!\\{}[]# \t\n"

#: Every spelling of "blank". Shared by the suite-tag guard and read_env so the two
#: cannot drift: both must resist the same one-character weakening (dropping
#: ``.strip()``).
BLANKS = ["", "   ", "\t"]
BLANK_IDS = ["empty", "spaces", "tab"]

#: ModuScope's own separators between fields; ``grid_entry_name`` uses the pipe form.
#: Any token extractor has to stop at all three.
SEPARATORS = [", ", "| ", " "]
SEPARATOR_IDS = ["comma", "pipe", "space"]

ENV_NAMES = ("MODEL_IDS", "SUITE_TAG", "MODUSCOPE_JOB_TIMEOUT_MINUTES")

#: Rows from the real artifact, cut to the two keys under test. No id is transformed by
#: test code, so a broken normalizer cannot drag fixture and assertion together.
REAL_GRID_ROWS = [
    {"module_configs_str": "smeoni_tiny-random-LlamaForCausalLM_config X", "status": "pass"},
    {"module_configs_str": "allenai_OLMo-1B-hf_config X", "status": "pass"},
    {"module_configs_str": "stabilityai_stablelm-2-1_6b_config X", "status": "pass"},
    {"module_configs_str": "microsoft_Phi-3-mini-4k-instruct_config X", "status": "fail"},
    {"module_configs_str": "meta-llama_Llama-3.1-8B_config X", "status": "pass"},
]

#: Headings assert_grid must file each problem class under. Literals on purpose: read
#: back from the module, a label swap would move fixture and assertion together.
MISSING_LABEL = "requested but missing"
UNEXPECTED_LABEL = "in the grid but not requested"
FAILING_LABEL = "status is not"

#: Values that are falsy (or empty) on the far side of Hydra parsing.
EMPTY_VALUES = frozenset({"", "[]", "{}", "''", '""', "null", "none", "[ ]"})

#: Payloads that reach a shell if the grammar does not refuse them. run_test.sh
#: re-executes the command with ``bash -c``, so the single quotes in
#: run-test-suite/action.yml are not containment: a ``$(...)`` survives them literally
#: and is expanded by that second shell. The quote case breaks the first shell outright.
INJECTION_IDS = [
    "owner/name'; touch /tmp/pwned; '",
    "owner/$(id)",
    "owner/`id`",
    "owner/name;id",
    "owner/name|id",
    "owner/name&id",
    "owner/name\nowner2/name2",
    'owner/"name"',
    "owner/name space",
    # No slash at all, and two slashes: neither is a hub id.
    "justaname",
    "a/b/c",
]
INJECTION_ID_LABELS = [
    "quote-escape",
    "command-substitution",
    "backtick",
    "semicolon",
    "pipe",
    "ampersand",
    "newline",
    "double-quote",
    "space",
    "no-slash",
    "two-slashes",
]


def frag(model_id: str) -> str:
    """The slash-free half of a hub id, for asserting a message names an offender."""
    # Slash-free so the assertion holds whether the message prints the hub id or the
    # normalized id. No two fragments used together in one assertion are substrings of
    # each other, so "names A" cannot be satisfied by a message naming only B.
    return model_id.split("/")[1]


def clause_for(message: str, label: str) -> str:
    """The one ``;``-separated clause of an assert_grid message filed under ``label``."""
    # Selecting by heading is what asserts an id was filed under the RIGHT heading
    # rather than merely mentioned somewhere: swapping two labels leaves every id named
    # and every message distinct.
    clauses = [part for part in message.split(";") if label in part]
    assert len(clauses) == 1, (
        f"expected exactly one clause headed {label!r}, found {len(clauses)}: {clauses} "
        f"in message {message!r}"
    )
    return clauses[0]


def row(model_id: str, status: str = "pass") -> dict:
    """One grid row for ``model_id`` in the real artifact's shape."""
    # Spelled out rather than calling grid.normalize_model_id, so a weakened normalizer
    # cannot move the fixture and the assertion together.
    return {"module_configs_str": f"{model_id.replace('/', '_')}_config X", "status": status}


def write_grid(
    tmp_path: Path,
    payload: object,
    dirname: str = "moduscope-out",
    filename: str = GRID_FILE_NAME,
) -> str:
    """Write ``payload`` as the grid ModuScope leaves behind; return output_dir."""
    out = tmp_path / dirname
    out.mkdir(parents=True, exist_ok=True)
    (out / filename).write_text(json.dumps(payload))
    return str(out)


def empty_valued_terms(argv: list[str]) -> list[str]:
    """Every ``key=value`` term in ``argv`` whose value is empty."""
    return [
        term
        for term in argv
        if "=" in term and term.partition("=")[2].strip().lower() in EMPTY_VALUES
    ]


def parse_workflow(workflow: Path) -> dict:
    """``workflow`` as a dict.

    Parsed rather than pattern-matched: assertions over raw text were satisfied three times in
    this PR by a comment quoting the very directive it explained, and a parse sees no comments.
    """
    return yaml.safe_load(workflow.read_text())


def workflow_inputs(workflow: Path) -> dict:
    """The ``inputs:`` of whichever trigger ``workflow`` declares.

    YAML 1.1 reads a bare ``on:`` key as the boolean ``True``, hence the two-key lookup.
    """
    doc = parse_workflow(workflow)
    triggers = doc.get("on", doc.get(True))
    assert triggers, f"{workflow} declares no triggers"
    (trigger,) = triggers.keys()
    return triggers[trigger]["inputs"]


def workflow_input_default(name: str, workflow: Path):
    """The ``default:`` ``workflow`` ships for input ``name``, at its YAML type."""
    inputs = workflow_inputs(workflow)
    assert name in inputs, f"no input named {name} in {workflow}"
    assert "default" in inputs[name], f"input {name} in {workflow} ships no default"
    return inputs[name]["default"]


def job(workflow: Path, name: str) -> dict:
    """Job ``name`` of ``workflow``, as a dict."""
    jobs = parse_workflow(workflow)["jobs"]
    assert name in jobs, f"no job named {name} in {workflow}"
    return jobs[name]


def device_test_command(workflow: Path) -> str:
    """The ``test-command`` the suite job hands the shared composite action."""
    for step in job(workflow, "moduscope-suite")["steps"]:
        if "test-command" in step.get("with", {}):
            return step["with"]["test-command"]
    raise AssertionError(f"no step in {workflow} passes a test-command")


def split_ids(raw: str) -> list[str]:
    """The comma list split by the test's own parse rather than the runner's."""
    return [part.strip() for part in raw.split(",") if part.strip()]


#: Both files ship a ``model-ids`` default, and each is the list its own entry point runs.
WORKFLOW_PARAMS = [
    pytest.param(REUSABLE_WORKFLOW, id="reusable"),
    pytest.param(WRAPPER_WORKFLOW, id="wrapper"),
]

#: ``(workflow name, id)`` for every id either default ships.
SHIPPED_IDS = [
    (workflow.name, model_id)
    for workflow in (REUSABLE_WORKFLOW, WRAPPER_WORKFLOW)
    for model_id in split_ids(workflow_input_default("model-ids", workflow))
]

#: The ids CI dispatches. Resolved through the runner below rather than here, so a
#: default that no longer parses fails one test instead of collection.
CI_MODEL_IDS_RAW = workflow_input_default("model-ids", REUSABLE_WORKFLOW)
CI_MODEL_IDS = tuple(split_ids(CI_MODEL_IDS_RAW))
CI_GRID_ROWS = [row(model_id) for model_id in CI_MODEL_IDS]

WRAPPER_SUITE_TAG_OPTIONS = workflow_inputs(WRAPPER_WORKFLOW)["suite-tag"]["options"]


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (LLAMA, "meta-llama_Llama-3.1-8B"),
        # Over-sanitizing (re.sub(r'[^A-Za-z0-9]', '_', id)) passes the case above and
        # then fails to match stabilityai_stablelm-2-1_6b_config X, a real row.
        (STABLELM, "stabilityai_stablelm-2-1_6b"),
        (PHI3, "microsoft_Phi-3-mini-4k-instruct"),
        # Nothing else changes: no suffix, no lowercasing.
        ("gpt2", "gpt2"),
    ],
)
def test_normalize_replaces_only_the_slash(model_id, expected):
    assert grid.normalize_model_id(model_id) == expected


@pytest.mark.parametrize(
    "raw", ["", "   ", "\t\n ", ",,"], ids=["empty", "spaces", "whitespace", "commas"]
)
def test_resolve_refuses_a_value_naming_no_model(raw):
    """Blank has nothing to mean, so any ``or SOME_DEFAULT`` fallback must be absent."""
    # Returning ``()`` is worse than raising: an empty wishlist reinstates ModuScope's own
    # unbounded discovery. ``",,"`` is not blank, so it must not become an empty tuple either.
    with pytest.raises(ValueError):
        mjr.resolve_model_ids(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"{OLMO},{PHI3}", (OLMO, PHI3)),
        (LLAMA, (LLAMA,)),
        # The value IS the model set, so its order is the operator's: sorting would make
        # the wishlist disagree with the input they can read on the run page.
        (f"{LLAMA},{OLMO}", (LLAMA, OLMO)),
        # The default is a literal in YAML, where a hand-edited list easily gains spaces.
        (f" {OLMO} , {PHI3} ", (OLMO, PHI3)),
        # assert_grid compares SETS, so a duplicate cannot be caught downstream.
        (f"{OLMO},{PHI3},{OLMO}", (OLMO, PHI3)),
    ],
    ids=["pair", "single", "unsorted", "padded", "duplicate"],
)
def test_resolve_returns_exactly_the_ids_given(raw, expected):
    assert mjr.resolve_model_ids(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "foo/bar_baz,foo_bar/baz",
        # Order must not matter: the check groups, it does not compare neighbours.
        "foo_bar/baz,foo/bar_baz",
        f"foo/bar_baz,{OLMO},foo_bar/baz",
        # The slash can sit anywhere, so the pair need not share a prefix.
        "a/b_c_d,a_b_c/d",
    ],
    ids=["pair", "reversed", "with-innocent", "deep"],
)
def test_resolve_refuses_ids_that_collide_once_normalized(raw):
    """A grid row carries only ``org_name``, so a colliding pair cannot be graded apart."""
    # One passing row satisfies both, and a model that never ran is reported present.
    with pytest.raises(ValueError, match="collide"):
        mjr.resolve_model_ids(raw)


def test_resolve_still_collapses_an_exactly_repeated_id():
    """Only two DISTINCT ids sharing a normalized form are an error."""
    # Without collapsing exact repeats before grouping, a doubled id groups with itself.
    assert mjr.resolve_model_ids(f"{OLMO},{OLMO}") == (OLMO,)


@pytest.mark.parametrize("raw", INJECTION_IDS, ids=INJECTION_ID_LABELS)
def test_resolve_refuses_an_id_outside_the_shell_safe_grammar(raw):
    """Every one of these reaches a shell if it is not refused here."""
    with pytest.raises(ValueError):
        mjr.resolve_model_ids(raw)


@pytest.mark.parametrize(("workflow_name", "model_id"), SHIPPED_IDS)
def test_every_shipped_model_id_is_a_bare_hub_id(workflow_name, model_id):
    """The ids reach ModuScope through argv, but a malformed one should fail in review."""
    assert HUB_ID_RE.fullmatch(model_id), f"{workflow_name}: {model_id!r} is not a bare owner/name"
    offenders = sorted(set(model_id) & set(SHELL_METACHARACTERS))
    assert not offenders, f"{workflow_name}: {model_id!r} carries shell metacharacters {offenders}"


@pytest.mark.parametrize("workflow", WORKFLOW_PARAMS)
def test_no_shipped_model_ids_default_has_a_normalized_collision(workflow):
    """Checked with the test's own normalizer, so a weakened guard cannot hide a collision."""
    raw = workflow_input_default("model-ids", workflow)
    normalized = [model_id.replace("/", "_") for model_id in split_ids(raw)]
    assert len(set(normalized)) == len(
        normalized
    ), f"{workflow.name} collides once normalized: {normalized}"


@pytest.mark.parametrize("workflow", WORKFLOW_PARAMS)
def test_the_shipped_model_ids_default_reaches_the_command_line(workflow):
    """The path CI actually runs: each workflow's own ``model-ids`` default."""
    # Every build_argv test hand-builds a tuple, so none notices a YAML default that no
    # longer resolves -- which ships no wishlist term at all, leaving the run unbounded.
    model_ids = mjr.resolve_model_ids(workflow_input_default("model-ids", workflow))
    assert len(model_ids) >= 2, f"{workflow.name}'s default resolves to {model_ids}"
    argv = mjr.build_moduscope_argv(model_ids, "/tmp/out", TAG)
    assert not empty_valued_terms(argv), f"empty-valued term on the default path: {argv}"
    for model_id in model_ids:
        assert model_id in " ".join(argv), f"{model_id} missing from the default-path wishlist"


def test_every_allowlisted_suite_tag_is_shell_safe_by_construction():
    """What replaces a separate shell-safety pattern for the tag.

    The tag rides the same interpolation as the model ids, and there is no charset check on it
    any more -- membership in this closed set is the check. That only holds while every member
    is inert, so a tag added here carrying a shell metacharacter must fail.
    """
    for suite_tag in mjr.WISHLIST_SUITE_TAGS:
        assert re.fullmatch(r"[A-Za-z0-9._-]+", suite_tag), (
            f"{suite_tag!r} is in WISHLIST_SUITE_TAGS but is not inert in a shell; the tag is "
            f"interpolated into a command the pod re-executes with bash -c"
        )


def test_build_argv_refuses_an_empty_model_set():
    """The last line of defence before an uncapped model set."""
    # ModuScope replaces its own discovery only when the wishlist is non-empty, and an
    # empty list is falsy at that decision, so letting one through is a full-suite run.
    with pytest.raises(ValueError):
        mjr.build_moduscope_argv((), "/tmp/out", TAG)


@pytest.mark.parametrize("suite_tag", BLANKS, ids=BLANK_IDS)
def test_build_argv_refuses_a_blank_suite_tag(suite_tag):
    """A blank tag renders ``--config-name=`` and fails Hydra naming neither model nor tag."""
    with pytest.raises(ValueError):
        mjr.build_moduscope_argv((LLAMA,), "/tmp/out", suite_tag)


def test_build_argv_renders_the_whole_invocation():
    """Every term of the ModuScope command line, pinned as literals."""
    argv = mjr.build_moduscope_argv((OLMO, PHI3, LLAMA), "/tmp/moduscope-elsewhere", TAG)
    joined = " ".join(argv)

    # sys.executable rather than a bare "python": the image has several interpreters and
    # only one has torch-neuronx installed. -m rather than a script path: handing Python a
    # file path changes sys.path[0] and breaks Hydra's config discovery.
    assert argv[:3] == [sys.executable, "-m", ENTRY_POINT], f"wrong argv head: {argv[:3]}"

    # A hardcoded tag would let the dispatched tag and the run disagree silently, because
    # the wrong suite still produces a grid.
    assert f"--config-name={TAG}" in argv, f"the tag argument did not reach argv: {argv}"

    # enable_ttfi_metrics defaults to True in BaseTaskConfig and no shipped suite YAML
    # overrides it, so without this term TTFI clears the NEFF/HLO caches before every grid
    # entry -- a recompile per model.
    assert TTFI_DISABLE_OVERRIDE in argv, f"{TTFI_DISABLE_OVERRIDE!r} not in {argv}"

    # Ignoring out_dir sends ModuScope to its own Hydra default, and read_grid then reports
    # "produced nothing" instead of "we looked in the wrong place".
    assert "/tmp/moduscope-elsewhere" in joined, f"out_dir not in {argv}"

    # Catches capping the model set at model_ids[0], so a three-model request runs one and the
    # grid reports two missing ids as if ModuScope had dropped them.
    for model_id in (OLMO, PHI3, LLAMA):
        assert model_id in joined, f"{model_id} missing from the wishlist: {joined}"

    # Name-independent: whatever the wishlist key is called, no override may carry an empty
    # value. An empty wishlist reinstates the suite's own discovered model set.
    assert not empty_valued_terms(argv), f"empty-valued override(s): {empty_valued_terms(argv)}"

    # ModuScope computes effective_wishlist = [resolved] if resolved else hf_cfg_wishlist,
    # where resolved comes from +task_config.custom_parallel_model -- so that override
    # SILENTLY OUTRANKS the wishlist and would defeat the cap with a correct one in argv.
    assert "custom_parallel_model" not in joined, f"custom_parallel_model uncaps: {argv}"


def test_output_dir_is_the_directory_run_test_sh_copies(tmp_path):
    """The grid has to land under ``outputs/`` or run_test.sh never copies it off the pod."""
    path = Path(mjr.output_dir(str(tmp_path)))
    assert path.parts[-2:] == OUTPUT_PARTS, f"{path} is not <cwd>/outputs/moduscope"
    assert path.parent == tmp_path / OUTPUT_PARTS[0], f"{path} is not under {tmp_path}/outputs"


def test_output_dir_defaults_to_the_process_working_directory(tmp_path, monkeypatch):
    """No argument means cwd, which is what main() relies on."""
    monkeypatch.chdir(tmp_path)
    assert mjr.output_dir() == os.path.join(os.getcwd(), *OUTPUT_PARTS)


def test_prepare_output_dir_returns_the_path_output_dir_names(tmp_path):
    """The preparer and the deriver must not be able to disagree."""
    assert mjr.prepare_output_dir(str(tmp_path)) == mjr.output_dir(str(tmp_path))


def test_prepare_output_dir_creates_it_empty(tmp_path):
    """A run whose directory does not exist yet is the normal case on a fresh pod."""
    path = Path(mjr.prepare_output_dir(str(tmp_path)))
    assert path.is_dir(), f"{path} was not created"
    assert list(path.iterdir()) == [], f"{path} is not empty: {list(path.iterdir())}"


def test_prepare_output_dir_removes_a_stale_grid(tmp_path):
    """The point of clearing: the grader requires EVERY grid it globs to pass."""
    # A grid left by an earlier run in the same working directory would fail a good run,
    # so merging into the directory is not an option.
    stale = tmp_path / Path(*OUTPUT_PARTS) / GRID_FILE_NAME
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(REAL_GRID_ROWS))
    nested = stale.parent / "sub" / "nested_grid.json"
    nested.parent.mkdir()
    nested.write_text("{}")
    sibling = tmp_path / OUTPUT_PARTS[0] / "keep-me.txt"
    sibling.write_text("other suites write here")

    path = Path(mjr.prepare_output_dir(str(tmp_path)))

    assert not stale.exists(), "a stale grid survived; the grader would fail a good run"
    assert not nested.exists(), "a nested stale grid survived"
    assert list(path.iterdir()) == [], f"{path} is not empty: {list(path.iterdir())}"
    # Only the suite's own directory is cleared, not everything the pod copies back.
    assert sibling.exists(), "prepare_output_dir removed outputs/ wholesale"


def test_the_output_directory_is_not_an_environment_variable(env_set, tmp_path, monkeypatch):
    """The path is plumbing now, so an ambient OUTPUT_DIR must not steer the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "decoy"))
    env_set()
    assert len(mjr.read_env()) == 3, "read_env still returns an output directory"
    assert "decoy" not in mjr.output_dir(), "an ambient OUTPUT_DIR steered the derived path"


def test_the_reusable_workflow_requires_an_image():
    """No default image, so a caller cannot silently run a stale one."""
    image = workflow_inputs(REUSABLE_WORKFLOW)["image"]
    assert image["required"] is True, f"image is not required: {image}"
    assert "default" not in image, f"image carries a default: {image}"


def test_the_test_command_block_has_no_apostrophe():
    """The composite wraps the command in ``TEST_COMMAND='...'``, so one apostrophe truncates it."""
    command = device_test_command(REUSABLE_WORKFLOW)
    assert "'" not in command, f"apostrophe in the test-command block:\n{command}"


def test_the_workflow_launches_the_runner_from_the_path_it_lives_at():
    """The workflow names the runner as a path, so a move that misses it fails on the pod."""
    relative = Path(mjr.__file__).resolve().relative_to(REPO_ROOT)
    assert (
        f"python {relative}" in REUSABLE_WORKFLOW.read_text()
    ), f"the workflow does not launch {relative}; the runner moved and the workflow did not"


def test_the_workflow_grades_the_grid_after_the_device_job():
    """ModuScope's exit code cannot say WHICH models ran, so the grader is the verdict."""
    # Three properties, because dropping any one silently restores that hole: the grader
    # is invoked, it runs under always() so a failed device job is still graded, and it
    # is not excused by continue-on-error.
    grader = Path(grid.__file__).resolve().relative_to(REPO_ROOT)
    assert (
        f"python3 {grader}" in REUSABLE_WORKFLOW.read_text()
    ), f"nothing invokes {grader}; the grid is not graded"
    grade = job(REUSABLE_WORKFLOW, "grade-grid")
    # On the job, not a step: a step-level always() cannot start a job the failure skipped.
    assert (
        grade.get("if") == "always()"
    ), "job-level if: always() is gone, so the grader never starts after a failed device job"
    assert "continue-on-error" not in grade, "the grader cannot fail the workflow"
    # always() without needs starts the grader at t=0, where it races the device job and
    # downloads an artifact that does not exist yet.
    assert grade.get("needs") == [
        "moduscope-suite"
    ], f"the grader does not wait for the job it grades: needs={grade.get('needs')!r}"


def test_the_reusable_workflow_is_callable_and_the_wrapper_is_dispatchable():
    """The split only works if each file carries the trigger its role needs."""
    # A mutant that swapped workflow_call for workflow_dispatch left every other test
    # green while breaking the wrapper's `uses:` at runtime: a workflow can only be
    # called if it declares workflow_call, and that failure appears nowhere else.
    reusable = REUSABLE_WORKFLOW.read_text()
    wrapper = WRAPPER_WORKFLOW.read_text()
    assert re.search(
        r"^on:\n  workflow_call:", reusable, re.MULTILINE
    ), "the reusable workflow does not declare workflow_call, so `uses:` cannot reach it"
    assert (
        "workflow_dispatch:" not in reusable
    ), "the reusable workflow also takes a dispatch, which puts two entry points on one file"
    assert re.search(
        r"^on:\n  workflow_dispatch:", wrapper, re.MULTILINE
    ), "the wrapper is not dispatchable, so there is no manual entry point"
    assert (
        f"uses: ./.github/workflows/{REUSABLE_WORKFLOW.name}" in wrapper
    ), f"the wrapper does not call {REUSABLE_WORKFLOW.name}"


#: The fleet runs an ECR vulnerability scan before internal-compute dispatch and fails the job
#: when it does not finish, printing "Scan timed out after 1800s ... Blocking dispatch". The gate
#: is injected by the runner administrator and configured nowhere in this repo, so this is an
#: assumption about the fleet rather than something the test can enforce.
SCAN_GATE_MAX_MINUTES = 30

#: Dispatch plus pod start to first output, rounded up from just under two minutes observed.
POD_START_MINUTES = 2


def test_the_reserve_covers_every_minute_moduscope_cannot_spend():
    """ModuScope's clock starts when ModuScope does; the job's started long before.

    ModuScope gets ``job bound - reserve`` seconds, counted from after the ECR scan and pod
    start. So its bound expires at ``scan + start + (bound - reserve)``, and requiring that to
    land inside ``bound`` reduces to ``reserve >= scan + start``. Backwards, and the job timeout
    fires first, killing the step before run_test.sh copies ``outputs/`` -- losing the grid in
    precisely the hang the inner bound exists to survive.
    """
    floor = SCAN_GATE_MAX_MINUTES + POD_START_MINUTES
    assert floor <= mjr.COPY_BUFFER_MINUTES, (
        f"COPY_BUFFER_MINUTES is {mjr.COPY_BUFFER_MINUTES} but {floor} minute(s) elapse before "
        f"ModuScope starts ({SCAN_GATE_MAX_MINUTES} of ECR scan, {POD_START_MINUTES} of pod "
        f"start), so its bound outlives the job bound"
    )


def test_the_shipped_job_bound_leaves_moduscope_a_positive_budget():
    """The bound the workflow ships has to survive the runner's own subtraction."""
    job_minutes = workflow_input_default("timeout-minutes", REUSABLE_WORKFLOW)
    assert job_minutes > mjr.COPY_BUFFER_MINUTES, (
        f"timeout-minutes {job_minutes} is not above the {mjr.COPY_BUFFER_MINUTES}-minute "
        f"reserve, so the derived inner bound is not positive"
    )
    # The default that ships is what reserves a trn2.48xlarge, and 240 got here without a
    # measurement. Fails if it drifts back to a multi-hour hold on that hardware.
    assert job_minutes <= 120, (
        f"the shipped timeout-minutes default is {job_minutes}; the suite measured 19m25s, so "
        f"a default this high holds a trn2.48xlarge for hours on a hang instead of failing"
    )


def test_the_device_job_bound_and_the_runner_read_the_same_input():
    """The job's own bound and the number the runner subtracts from must be one value."""
    suite = job(REUSABLE_WORKFLOW, "moduscope-suite")
    assert (
        suite["timeout-minutes"] == "${{ inputs.timeout-minutes }}"
    ), f"the job bound does not come from inputs.timeout-minutes: {suite['timeout-minutes']!r}"
    assert (
        f'{mjr.JOB_TIMEOUT_MINUTES_ENV_VAR}="${{{{ inputs.timeout-minutes }}}}"'
        in device_test_command(REUSABLE_WORKFLOW)
    ), "the runner is handed a different number than the job is bounded by"


def test_the_fleet_ceiling_matches_the_shipped_job_bound():
    """neuron.timeout is how long the fleet holds the node; GitHub's own bound does not free it.

    Workflow expressions cannot multiply, so the label is a literal in seconds and this is what
    keeps it equal to timeout-minutes. Drift in either direction is a real failure: a label above
    the bound holds a trn2.48xlarge after GitHub has given up, a label below it kills the pod
    before the inner timeout can save the grid.
    """
    labels = job(REUSABLE_WORKFLOW, "moduscope-suite")["runs-on"]
    ceilings = [
        int(label.split(":", 1)[1]) for label in labels if label.startswith("neuron.timeout:")
    ]
    assert len(ceilings) == 1, f"expected exactly one neuron.timeout label, found {ceilings}"
    job_minutes = workflow_input_default("timeout-minutes", REUSABLE_WORKFLOW)
    assert ceilings[0] == job_minutes * 60, (
        f"neuron.timeout is {ceilings[0]}s but timeout-minutes defaults to {job_minutes} "
        f"({job_minutes * 60}s); the fleet and GitHub would give up at different times"
    )


def test_the_reusable_suite_tag_default_reaches_the_command_line():
    """Catches an emptied or quote-mangled tag, which fails at Hydra composition on the pod."""
    # Asserted through build_moduscope_argv rather than on the string, because the
    # consumer is what decides whether the value is usable.
    tag = workflow_input_default("suite-tag", REUSABLE_WORKFLOW)
    assert tag.strip(), "the workflow ships a blank suite-tag default"
    assert f"--config-name={tag}" in mjr.build_moduscope_argv((LLAMA,), "/tmp/out", tag)


def test_the_wrapper_offers_suite_tags_as_a_closed_choice():
    """A free-form tag accepts a Titan tag, which ignores the wishlist and measures its own set."""
    # The type is pinned, not just the default: the default is only what an unedited
    # dispatch sends, and a string field stays free-form for anyone who edits it.
    spec = workflow_inputs(WRAPPER_WORKFLOW)["suite-tag"]
    assert spec["type"] == "choice", f"suite-tag is not a closed choice: {spec}"
    assert WRAPPER_SUITE_TAG_OPTIONS, f"suite-tag is a choice with no options: {spec}"
    assert (
        spec["default"] in WRAPPER_SUITE_TAG_OPTIONS
    ), f"the shipped default {spec['default']!r} is not among {WRAPPER_SUITE_TAG_OPTIONS}"


def test_the_reusable_suite_tag_default_is_one_the_wrapper_offers():
    """A caller that omits suite-tag must land on a tag the dispatch screen also vouches for."""
    default = workflow_input_default("suite-tag", REUSABLE_WORKFLOW)
    assert (
        default in WRAPPER_SUITE_TAG_OPTIONS
    ), f"the reusable default {default!r} is not among the wrapper's {WRAPPER_SUITE_TAG_OPTIONS}"


def test_the_dispatch_dropdown_is_exactly_the_allowlist():
    """The dropdown and the module allowlist must be the same set, not merely overlap."""
    # A choice input constrains one screen. Every other caller of the reusable workflow passes
    # a plain string, because workflow_call has no choice type, so the module allowlist is the
    # only thing holding the line there. If the two drift, the screen offers a suite the
    # preflight refuses, or stops offering one the preflight still allows.
    assert set(WRAPPER_SUITE_TAG_OPTIONS) == set(mjr.WISHLIST_SUITE_TAGS), (
        f"dropdown {sorted(WRAPPER_SUITE_TAG_OPTIONS)} != "
        f"allowlist {sorted(mjr.WISHLIST_SUITE_TAGS)}"
    )


@pytest.mark.parametrize(
    "suite_tag",
    ["full_titan_multicore_async", "full_diffusers", "full_dense_encoder", "smoke_sentinel_tag"],
    ids=["titan", "diffusers", "encoder", "sentinel"],
)
def test_a_suite_that_ignores_the_wishlist_is_refused(suite_tag):
    """A suite outside the allowlist takes its model set from model_configurations instead, so
    the requested ids do nothing and the run reports success over a set nobody asked for.

    Asserted twice on purpose: once that the guard refuses, once that something calls it. The
    allowlist shipped for a while enforcing nothing, because its only caller was a preflight job
    that got deleted while the function, its constant and its direct tests all stayed green.
    """
    with pytest.raises(ValueError, match="hf_cfg_wishlist"):
        mjr.assert_suite_reads_wishlist(suite_tag)
    with pytest.raises(ValueError, match="hf_cfg_wishlist"):
        mjr.build_moduscope_argv((LLAMA,), "/tmp/out", suite_tag)


@pytest.mark.parametrize("suite_tag", list(mjr.WISHLIST_SUITE_TAGS))
def test_every_allowlisted_suite_is_accepted(suite_tag):
    """The guard has to admit what ships, or it is an outage rather than a guard."""
    assert mjr.assert_suite_reads_wishlist(suite_tag) == suite_tag
    # And through the path that runs, or an over-strict guard breaks the shipped default
    # without any test noticing.
    assert f"--config-name={suite_tag}" in mjr.build_moduscope_argv((LLAMA,), "/tmp/out", suite_tag)


def summary_for(tmp_path, rows, requested, errors=()):
    """Build an artifact and return the markdown summary the grading job would post."""
    out = tmp_path / "outputs" / "moduscope"
    out.mkdir(parents=True, exist_ok=True)
    (out / GRID_FILE_NAME).write_text(json.dumps(rows))
    for name in errors:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text("{}")
    return grid.summarize(str(out), requested)


def test_the_summary_leads_with_the_verdict_when_everything_passed(tmp_path):
    """The first line is what an on-call reads, so it has to be the answer."""
    text = summary_for(tmp_path, [row(TINY), row(OLMO)], (TINY, OLMO))
    assert text.splitlines()[0] == "### ModuScope: 2/2 models passed", text


def test_the_summary_names_the_failing_model_in_its_first_line(tmp_path):
    """Naming the model is the point: a verdict without it sends someone to the 3 MB log."""
    text = summary_for(tmp_path, [row(TINY, "fail"), row(OLMO)], (TINY, OLMO))
    first = text.splitlines()[0]
    assert first.startswith("### ModuScope: FAILED"), first
    assert TINY in first, first
    assert OLMO not in first, f"named a model that passed: {first}"


def test_the_summary_separates_a_failure_from_a_model_that_never_ran(tmp_path):
    """The two need different next steps, so the headline must not merge them."""
    # A failure means read that model error file. A no-show means the run stopped early and
    # the device log is the place to look. Merging them costs an on-call the distinction.
    text = summary_for(tmp_path, [row(TINY, "fail")], (TINY, OLMO))
    first = text.splitlines()[0]
    assert "1 failed" in first and TINY in first, first
    assert "1 never ran" in first and OLMO in first, first
    assert "DID NOT RUN" in text, text


def test_the_summary_flags_a_model_that_rode_along(tmp_path):
    """A model nobody asked for is a finding, not noise to drop from the table."""
    text = summary_for(tmp_path, [row(TINY), row(OLMO), row(DECOY)], (TINY, OLMO))
    assert "not requested" in text.splitlines()[0], text
    assert "ran but was not requested" in text


def test_the_summary_names_the_error_files_to_read_next(tmp_path):
    """On-call reads these next, so the summary lists them rather than making someone glob."""
    text = summary_for(
        tmp_path, [row(TINY, "fail")], (TINY,), errors=("errors/DenseDecoder.0_error.json",)
    )
    assert "errors/DenseDecoder.0_error.json" in text, text


def test_the_summary_says_so_when_there_are_no_error_files(tmp_path):
    """Silence would read as "I did not look", which is the same shape as a broken glob."""
    text = summary_for(tmp_path, [row(TINY)], (TINY,))
    assert "No error files" in text, text


def test_the_summary_is_loud_rather_than_empty_when_no_grid_exists(tmp_path):
    """The case most likely to be misread as nothing-to-see is the one that proved nothing."""
    text = grid.summarize(str(tmp_path / "absent"), (TINY,))
    assert "NO RESULT GRID" in text, text
    assert "device job log" in text, text


def test_write_step_summary_appends_and_reports_whether_github_is_listening(tmp_path, monkeypatch):
    """Two runs must both land: the grader writes once per grid found."""
    path = tmp_path / "summary.md"
    monkeypatch.setenv(grid.STEP_SUMMARY_ENV_VAR, str(path))
    assert grid.write_step_summary("first\n") is True
    assert grid.write_step_summary("second\n") is True
    assert path.read_text() == "first\nsecond\n"
    # Control: unset means no file and no crash, which is how the unit tests run it.
    monkeypatch.delenv(grid.STEP_SUMMARY_ENV_VAR, raising=False)
    assert grid.write_step_summary("third\n") is False


def test_the_grader_writes_a_summary_on_the_failing_path(tmp_path, monkeypatch):
    """A red run is exactly when the summary matters, so it cannot be the green-only path."""
    path = tmp_path / "summary.md"
    monkeypatch.setenv(grid.STEP_SUMMARY_ENV_VAR, str(path))
    out = tmp_path / "artifact" / "outputs" / "moduscope"
    out.mkdir(parents=True)
    (out / GRID_FILE_NAME).write_text(json.dumps([row(TINY, "fail")]))
    assert grid.main([str(tmp_path / "artifact"), "--model-ids", TINY]) == 1
    assert "FAILED" in path.read_text()


def test_the_grader_annotates_a_failure_so_it_shows_without_opening_the_job(tmp_path, capsys):
    """An annotation is what appears on the run page; stderr alone needs a click to find."""
    out = tmp_path / "artifact" / "outputs" / "moduscope"
    out.mkdir(parents=True)
    (out / GRID_FILE_NAME).write_text(json.dumps([row(TINY, "fail")]))
    assert grid.main([str(tmp_path / "artifact"), "--model-ids", TINY]) == 1
    captured = capsys.readouterr()
    assert "::error title=ModuScope grid::" in captured.out, captured.out
    # Control: a passing run must NOT annotate, or every green run cries wolf.
    ok = tmp_path / "green" / "outputs" / "moduscope"
    ok.mkdir(parents=True)
    (ok / GRID_FILE_NAME).write_text(json.dumps([row(TINY)]))
    assert grid.main([str(tmp_path / "green"), "--model-ids", TINY]) == 0
    assert "::error" not in capsys.readouterr().out


@pytest.mark.parametrize("make_dir", [True, False], ids=["empty-dir", "never-downloaded"])
def test_the_grader_writes_a_summary_when_the_artifact_has_no_grid(tmp_path, monkeypatch, make_dir):
    """No grid is a lost run; the job page must say that rather than stay blank.

    Both shapes: an artifact that arrived without a grid, and a download that never produced a
    directory at all. The second is the failed-download path, which takes a different branch.
    """
    path = tmp_path / "summary.md"
    monkeypatch.setenv(grid.STEP_SUMMARY_ENV_VAR, str(path))
    artifact = tmp_path / "artifact"
    if make_dir:
        artifact.mkdir()
    assert grid.main([str(artifact), "--model-ids", TINY]) == 1
    assert "NO RESULT GRID" in path.read_text()


def test_the_grade_grid_job_and_its_grader_step_both_run_on_always():
    """Two separate always() are needed, and a comment mentioning one must not satisfy either."""
    # Comments stripped first: the explanation next to the step-level if: always() contains that
    # exact string, so a substring check over raw text passed even with the directive deleted.
    # Third time this shape bit in this PR, after continue-on-error and the allowlist call.
    #
    # Job-level alone is not enough: steps default to success(), so a failed download-artifact
    # skips the grader in the one case its no-grid diagnostic exists for. Step-level alone is
    # not enough either: the job would not start after a failed device job.
    grade = job(REUSABLE_WORKFLOW, "grade-grid")
    assert grade.get("if") == "always()", "job-level if: always() is gone"
    grader = [step for step in grade["steps"] if "moduscope_grid.py" in step.get("run", "")]
    assert len(grader) == 1, f"expected one grading step, found {len(grader)}"
    assert (
        grader[0].get("if") == "always()"
    ), "step-level if: always() is gone, so a failed download skips the no-grid diagnostic"


def test_the_wrapper_delegates_to_the_reusable_workflow():
    """The wrapper builds the images; the reusable workflow owns the suite."""
    block = job(WRAPPER_WORKFLOW, "moduscope")
    assert (
        block.get("uses") == f"./.github/workflows/{REUSABLE_WORKFLOW.name}"
    ), f"the wrapper does not call {REUSABLE_WORKFLOW.name}: uses={block.get('uses')!r}"
    # image has no default, so a wrapper that forgets it fails at dispatch rather than
    # running the suite in whatever image the reusable workflow last named.
    image = block.get("with", {}).get("image")
    assert image, f"the wrapper passes no image: with={block.get('with')!r}"
    assert "${{" in image, f"the image is not the one the wrapper built: {image!r}"


def test_the_grid_module_normalizes_through_the_runner():
    """One normalizer, or the collision check and the grid read can disagree."""
    assert grid.normalize_model_id is mjr.normalize_model_id
    tree = ast.parse(Path(grid.__file__).read_text())
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_model_id"
    ], "moduscope_grid defines its own normalizer again"


def test_read_grid_returns_the_rows(tmp_path):
    """Baseline: the real artifact loads and keeps its rows and their order."""
    rows = grid.read_grid(write_grid(tmp_path, REAL_GRID_ROWS))
    assert [r["status"] for r in rows] == ["pass", "pass", "pass", "fail", "pass"]


@pytest.mark.parametrize("make_dir", [True, False], ids=["output-dir-exists", "no-output-dir"])
def test_read_grid_missing_grid_raises_file_not_found(tmp_path, make_dir):
    """Catches ``if not path.exists(): return []``, which blames the models for a dead run."""
    # Both real shapes: ModuScope made its output dir and then died, and the job never
    # got far enough to make the directory at all.
    out = tmp_path / "moduscope-out"
    if make_dir:
        out.mkdir()
    with pytest.raises(FileNotFoundError):
        grid.read_grid(str(out))


def test_read_grid_empty_list_is_returned_not_raised(tmp_path):
    """An empty grid is a legitimate result and must not collapse into "no grid"."""
    assert grid.read_grid(write_grid(tmp_path, [])) == []


@pytest.mark.parametrize(
    "payload",
    [{"rows": []}, "not a grid", 42, [1, 2]],
    ids=["dict", "str", "int", "list-of-ints"],
)
def test_read_grid_non_list_raises_value_error(tmp_path, payload):
    """Catches ``json.load`` straight into the caller, which fails later naming a Python frame."""
    with pytest.raises(ValueError):
        grid.read_grid(write_grid(tmp_path, payload))


def test_read_grid_ignores_an_ambient_grid_filename_override(tmp_path, monkeypatch):
    """ModuScope's own GRID_SKELETON_FILENAME belongs to the writer, on the pod.

    This module reads the artifact on a hosted runner, where that variable is a different
    process's setting. Honouring it here would make the grader miss a grid that exists.
    """
    monkeypatch.setenv("GRID_SKELETON_FILENAME", "custom_grid.json")
    default = write_grid(tmp_path, REAL_GRID_ROWS, dirname="env-set")
    assert len(grid.read_grid(default)) == 5, "an ambient override hid the grid that is there"


def test_grid_model_ids_returns_every_id_across_every_row():
    """Set equality, not a superset check: this value is BOTH halves of assert_grid's sums."""
    # An extractor that drops or invents a token corrupts missing and unexpected at the
    # same time. Five rows, so reading only the first fails here.
    assert grid.grid_model_ids(REAL_GRID_ROWS) == {
        "smeoni_tiny-random-LlamaForCausalLM",
        "allenai_OLMo-1B-hf",
        "stabilityai_stablelm-2-1_6b",
        "microsoft_Phi-3-mini-4k-instruct",
        "meta-llama_Llama-3.1-8B",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Normalized LLAMA is a strict prefix of the Instruct variant's normalized form,
        # and the grid holds ONLY the Instruct model. An extractor that yields prefixes
        # reports the requested base model as present when the wrong variant ran.
        pytest.param(
            "meta-llama_Llama-3.1-8B-Instruct_config X",
            {"meta-llama_Llama-3.1-8B-Instruct"},
            id="prefix-collision",
        ),
        # The mirror: under a right-anchored substring test, zzz_allenai_OLMo-1B-hf_config
        # counts OLMO as PRESENT.
        pytest.param(
            "zzz_allenai_OLMo-1B-hf_config X",
            {"zzz_allenai_OLMo-1B-hf"},
            id="left-superstring",
        ),
        # There is deliberately NO trailing anchor. llama3_1_8B_config_cpu.json is a real
        # file in ModuScope's config tree, so a trailing (?![\w.-]) makes such a row match
        # nothing and a model that really ran is reported MISSING.
        pytest.param(
            "stabilityai_stablelm-2-1_6b_config_cpu X",
            {"stabilityai_stablelm-2-1_6b"},
            id="config-variant-suffix",
        ),
        # A greedy class such as ([\w./-]+)_config swallows / and ., so a value carrying a
        # filesystem path yields the path and a good run reads as "not requested".
        pytest.param(
            "/opt/moduscope/configs/allenai_OLMo-1B-hf_config X",
            {"allenai_OLMo-1B-hf"},
            id="path-like-prefix",
        ),
        # _FULL_MODEL_OVERLAY puts 'quantization_config' in this override dict on every
        # full-model run, and an extractor without the dict-key lookahead reads that KEY
        # as a second model.
        pytest.param(
            "('meta-llama_Llama-3.1-8B_config', {'torch_dtype': 'bfloat16', "
            "'quantization_config': {}, 'auto_map': {}}) X",
            {"meta-llama_Llama-3.1-8B"},
            id="tuple-shape-with-overrides",
        ),
        # ModuScope's other tuple shape (diffusers / no-overwrite path): a flat tuple whose
        # second element is None, so only a quote and a comma follow the id.
        pytest.param(
            "('Wan-AI_Wan2.1-T2V-1.3B-Diffusers_config', None) X",
            {"Wan-AI_Wan2.1-T2V-1.3B-Diffusers"},
            id="tuple-shape-without-overrides",
        ),
    ],
)
def test_grid_model_ids_yields_whole_tokens(value, expected):
    """Row strings are literals here, so no collision can be normalized away."""
    assert grid.grid_model_ids([{"module_configs_str": value}]) == expected


def test_assert_grid_accepts_a_complete_passing_grid(tmp_path):
    """Baseline: the happy path must not raise."""
    grid.assert_grid(write_grid(tmp_path, [row(OLMO), row(LLAMA)]), (OLMO, LLAMA))


def test_assert_grid_accepts_a_requested_model_with_a_config_variant_suffix(tmp_path):
    """A requested model whose row carries a ``_config_cpu`` variant must not read as missing."""
    out = write_grid(
        tmp_path,
        [{"module_configs_str": "stabilityai_stablelm-2-1_6b_config_cpu X", "status": "pass"}],
    )
    grid.assert_grid(out, (STABLELM,))


def test_assert_grid_fails_when_every_row_passes_but_the_ids_are_wrong(tmp_path):
    """All rows say pass; none is a requested model. ModuScope exits 0 either way."""
    # Catches a status-only check, under which a substituted model set, a stale grid or
    # discovery running its own models all produce a green proof.
    out = write_grid(tmp_path, [row(STABLELM), row(TINY)])
    with pytest.raises(AssertionError) as excinfo:
        grid.assert_grid(out, (OLMO, PHI3))
    assert frag(OLMO) in clause_for(str(excinfo.value), MISSING_LABEL)


def test_assert_grid_files_missing_unexpected_and_failing_separately(tmp_path):
    """One grid, all three faults at once, each filed under its own heading."""
    # Catches reporting the first category found and returning. The negative assertions
    # are what make a heading swap detectable; see clause_for.
    out = write_grid(tmp_path, [row(PHI3, status="fail"), row(STABLELM)])
    with pytest.raises(AssertionError) as excinfo:
        grid.assert_grid(out, (OLMO, PHI3))
    message = str(excinfo.value)
    missing = clause_for(message, MISSING_LABEL)
    unexpected = clause_for(message, UNEXPECTED_LABEL)
    failing = clause_for(message, FAILING_LABEL)

    assert frag(OLMO) in missing, f"missing id not filed as missing: {missing}"
    assert frag(STABLELM) in unexpected, f"unexpected id not filed as unexpected: {unexpected}"
    assert frag(PHI3) in failing, f"failing row not filed as failing: {failing}"
    assert frag(STABLELM) not in missing, f"unexpected id reported as missing: {missing}"
    assert frag(OLMO) not in unexpected, f"missing id reported as unexpected: {unexpected}"


@pytest.mark.parametrize("separator", SEPARATORS, ids=SEPARATOR_IDS)
def test_assert_grid_catches_an_extra_model_sharing_a_row_with_a_requested_one(tmp_path, separator):
    """``module_configs_str`` is PLURAL, so one entry can name several module configs."""
    # That shape defeats a set check written as two independent questions: ask "which
    # requested ids are missing?" and separately "which rows matched nothing I wanted?"
    # and a single combined row answers both innocently.
    #
    # LOAD-BEARING -- do not consolidate the separator params or the two calls away. The
    # params are the sole killers of the "revert to the two-question form" mutant, which
    # rewrites assert_grid's arithmetic and never touches the extractor. Without the
    # second call, an implementation that rejects any multi-config row would pass.
    combined = f"allenai_OLMo-1B-hf_config X{separator}stabilityai_stablelm-2-1_6b_config X"
    out = write_grid(
        tmp_path,
        [{"module_configs_str": combined, "status": "pass"}],
        dirname=f"stowaway-{separator.strip() or 'space'}",
    )

    with pytest.raises(AssertionError) as excinfo:
        grid.assert_grid(out, (OLMO,))
    unexpected = clause_for(str(excinfo.value), UNEXPECTED_LABEL)
    assert frag(STABLELM) in unexpected, f"stowaway not filed as unexpected: {excinfo.value}"

    grid.assert_grid(out, (OLMO, STABLELM))


@pytest.mark.parametrize("separator", SEPARATORS, ids=SEPARATOR_IDS)
def test_assert_grid_does_not_fail_a_row_carrying_a_path_like_prefix(tmp_path, separator):
    """The delimiter property at the level that matters: a FALSE FAILURE on a good run."""
    # A greedy token class absorbs / and ., so a value carrying a filesystem path yields
    # a token unequal to any requested id and set subtraction calls it unexpected.
    combined = (
        f"/opt/moduscope/configs/allenai_OLMo-1B-hf_config X"
        f"{separator}stabilityai_stablelm-2-1_6b_config X"
    )
    out = write_grid(
        tmp_path,
        [{"module_configs_str": combined, "status": "pass"}],
        dirname=f"pathprefix-{separator.strip() or 'space'}",
    )
    grid.assert_grid(out, (OLMO, STABLELM))


#: Runs assert_grid in a child interpreter and reports what happened as JSON.
#: ``__debug__`` is an instrument control: under -OO it must be False, so a dropped flag
#: fails the assertion instead of faking a pass.
OPTIMIZE_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
import moduscope_grid as grid

report = {
    "debug": __debug__,
    "docstring_stripped": grid.assert_grid.__doc__ is None,
}
try:
    grid.assert_grid(sys.argv[2], tuple(json.loads(sys.argv[3])))
    report["raised"] = None
except Exception as exc:
    report["raised"] = type(exc).__name__
print(json.dumps(report))
"""


def run_optimize_probe(output_dir: str, requested: tuple[str, ...]) -> dict:
    """Call assert_grid under ``-OO`` in a child interpreter; return its report."""
    argv = [sys.executable, "-OO", "-c", OPTIMIZE_PROBE]
    argv += [GRID_DIR, output_dir, json.dumps(requested)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, f"probe crashed: rc={proc.returncode}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_assert_grid_still_fails_under_an_optimize_flag(tmp_path):
    """``python -O`` deletes every ``assert``, so a bare-assert grid check becomes a no-op."""
    # It passes every normal test run and silently stops checking the moment
    # PYTHONOPTIMIZE is set in an image. Both arms are in one test on purpose: the
    # failing arm alone is satisfied by an implementation that raises unconditionally.
    failing = write_grid(tmp_path, [row(STABLELM)], dirname="oo-failing")
    passing = write_grid(tmp_path, [row(OLMO)], dirname="oo-passing")

    failing_report = run_optimize_probe(failing, (OLMO,))
    passing_report = run_optimize_probe(passing, (OLMO,))

    # If -OO were ignored, everything below would only describe a normal interpreter.
    assert failing_report["debug"] is False, "-OO was not in effect; __debug__ is still True"
    assert failing_report["docstring_stripped"] is True, "-OO did not strip docstrings"

    assert failing_report["raised"] == "AssertionError", (
        f"a wrong-model grid did NOT fail under -OO (got {failing_report['raised']!r}) -- "
        "the check is compiled away, so the proof is unfalsifiable in an optimized image"
    )
    assert passing_report["raised"] is None, (
        f"a correct grid failed under -OO (got {passing_report['raised']!r}) -- "
        "the check raises unconditionally, so the arm above proves nothing"
    )


@pytest.fixture
def env_set(monkeypatch):
    """Set all three runner inputs, then let a test unset (``None``) or blank one."""
    # Set here rather than left ambient: read_env reads the environment, so a
    # developer's shell could otherwise supply them.

    def _set(**overrides):
        values = {
            "MODEL_IDS": CI_MODEL_IDS_RAW,
            "SUITE_TAG": TAG,
            "MODUSCOPE_JOB_TIMEOUT_MINUTES": JOB_MINUTES,
        }
        values.update(overrides)
        for name, value in values.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    return _set


def test_read_env_returns_all_three_in_order_with_the_reserve_subtracted(env_set):
    """Out of order would send the tag where the ids go; unsubtracted would overrun the job."""
    env_set()
    expected_seconds = (int(JOB_MINUTES) - mjr.COPY_BUFFER_MINUTES) * 60
    assert mjr.read_env() == (CI_MODEL_IDS_RAW, TAG, expected_seconds)
    assert expected_seconds < int(JOB_MINUTES) * 60, "the reserve was not taken off the job bound"


@pytest.mark.parametrize(
    "raw",
    ["nine hundred", "90.5", "9e1", "", "  ", "0x5a"],
    ids=["words", "float", "exponent", "empty", "spaces", "hex"],
)
def test_read_env_refuses_a_job_bound_that_is_not_an_integer(env_set, raw):
    """A bound that does not parse must be refused, not coerced."""
    env_set(MODUSCOPE_JOB_TIMEOUT_MINUTES=raw)
    with pytest.raises(ValueError, match="MODUSCOPE_JOB_TIMEOUT_MINUTES"):
        mjr.read_env()


@pytest.mark.parametrize("raw", ["0", "-1", "40", "39"], ids=["zero", "negative", "equal", "under"])
def test_read_env_refuses_a_job_bound_the_reserve_swallows(env_set, raw):
    """These parse, but leave ModuScope zero or negative seconds, which reads as a hang.

    ``equal`` and ``under`` are new cases created by moving the reserve into this module: a bound
    at or below COPY_BUFFER_MINUTES is a positive number that still yields no budget.
    """
    env_set(MODUSCOPE_JOB_TIMEOUT_MINUTES=raw)
    with pytest.raises(ValueError, match="MODUSCOPE_JOB_TIMEOUT_MINUTES"):
        mjr.read_env()


@pytest.mark.parametrize("value", [None, *BLANKS], ids=["unset", *BLANK_IDS])
@pytest.mark.parametrize("name", ENV_NAMES)
def test_read_env_refuses_an_unset_or_blank_variable(env_set, name, value):
    """Every variable gets its own case; a guard checking two of three reads like one."""
    # Blank is the interesting value rather than unset: the workflow always exports
    # these, so a cleared dispatch field arrives as whitespace, which a bare truthiness
    # check passes.
    env_set(**{name: value})
    with pytest.raises(ValueError, match=name):
        mjr.read_env()


def test_read_env_names_every_missing_variable_not_just_the_first(env_set):
    """Two misconfigured variables should cost one round trip, not two."""
    env_set(MODEL_IDS=None, MODUSCOPE_JOB_TIMEOUT_MINUTES="")
    with pytest.raises(ValueError) as excinfo:
        mjr.read_env()
    message = str(excinfo.value)
    assert "MODEL_IDS" in message and "MODUSCOPE_JOB_TIMEOUT_MINUTES" in message, message
    assert "SUITE_TAG" not in message, f"blamed a variable that was set: {message}"


def test_read_env_survives_padding_on_the_ids_and_strips_the_tag(env_set):
    """A YAML-sourced value easily gains a trailing space, which both grammars then refuse."""
    # The tag is stripped here; padding on the ids is the resolver's to strip, per part.
    env_set(MODEL_IDS=f"  {CI_MODEL_IDS_RAW}  ", SUITE_TAG=f"  {TAG}  ")
    raw, tag, _ = mjr.read_env()
    assert tag == TAG
    assert mjr.resolve_model_ids(raw) == CI_MODEL_IDS


@pytest.mark.parametrize("returncode", [0, 1, 2, 137])
def test_main_runs_the_built_command_and_returns_its_exit_code(
    env_set, tmp_path, monkeypatch, returncode
):
    """main() is a launcher: the command is build_moduscope_argv's output, the code ModuScope's."""
    # Catches remapping -- collapsing every non-zero to 1, or swallowing one to return 0.
    # 137 is in the set because Kueue evicting the pod surfaces as that code, so
    # rewriting it to 1 would make an eviction look like a test failure.
    monkeypatch.chdir(tmp_path)
    env_set()
    expected_out = os.path.join(os.getcwd(), *OUTPUT_PARTS)
    ran = []

    def fake_run(cmd, *args, **kwargs):
        ran.append(cmd)
        return SimpleNamespace(returncode=returncode)

    # Every subprocess entry point is patched, so the claim covers all of them rather
    # than the one this test guessed.
    for name in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, name, fake_run)

    assert mjr.main() == returncode, "exit code was not passed through"
    assert ran == [mjr.build_moduscope_argv(CI_MODEL_IDS, expected_out, TAG)], ran


def test_main_marks_the_run_as_github_ci_before_launching_moduscope(env_set, tmp_path, monkeypatch):
    """Without this ModuScope writes to the Brazil prod bucket and tries roles this side lacks.

    Read inside the fake subprocess rather than after main() returns: the value only helps if it
    is set by the time ModuScope starts. It moved here from the workflow, where nothing tested it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODUSCOPE_CI", raising=False)
    env_set()
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["MODUSCOPE_CI"] = os.environ.get("MODUSCOPE_CI")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert mjr.main() == 0
    assert seen["MODUSCOPE_CI"] == "github", f"ModuScope started without the CI switch: {seen}"


def test_main_empties_the_output_dir_before_running(env_set, tmp_path, monkeypatch):
    """A stale grid from an earlier run in the same directory would fail a good run."""
    monkeypatch.chdir(tmp_path)
    env_set()
    stale = tmp_path / Path(*OUTPUT_PARTS) / GRID_FILE_NAME
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(REAL_GRID_ROWS))
    ran = []

    def fake_run(cmd, *args, **kwargs):
        ran.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert mjr.main() == 0
    assert not stale.exists(), "main() did not empty the output directory"
    assert str(stale.parent) in " ".join(ran[0]), f"ModuScope was not aimed there: {ran}"


def test_main_bounds_moduscope_with_the_configured_timeout(env_set, tmp_path, monkeypatch):
    """The bound has to reach subprocess.run as ``timeout=``, carrying the value from the env."""
    # Asserted on the keyword rather than on elapsed time: a positional or renamed
    # argument would be silently ignored by subprocess.run, leaving the run unbounded.
    monkeypatch.chdir(tmp_path)
    job_minutes = mjr.COPY_BUFFER_MINUTES + 100
    env_set(MODUSCOPE_JOB_TIMEOUT_MINUTES=str(job_minutes))
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert mjr.main() == 0
    assert seen.get("timeout") == 100 * 60, f"ModuScope was not bounded: {seen}"


def test_main_returns_124_when_moduscope_outlasts_the_timeout(env_set, tmp_path, monkeypatch):
    """A killed run must report a distinct code, not ModuScope's or a bare 1."""
    # The point of the bound is that run_test.sh still gets to copy outputs/ off the pod,
    # so the exit code is the only thing that says the grid describes a killed run.
    monkeypatch.chdir(tmp_path)
    env_set(MODUSCOPE_JOB_TIMEOUT_MINUTES=str(mjr.COPY_BUFFER_MINUTES + 7))

    def fake_run(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert mjr.main() == mjr.TIMEOUT_EXIT_CODE
    assert mjr.TIMEOUT_EXIT_CODE == 124, "124 is timeout(1)'s code; a different one misleads"


def test_the_runner_does_not_import_the_grid_module():
    """The runner launches; grading lives in moduscope_grid, which is why it is unit testable."""
    # Parsed rather than grepped: a substring scan cannot tell a call from a docstring
    # pointer, and main's docstring names the grid module on purpose.
    tree = ast.parse(Path(mjr.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "moduscope_grid" not in imported, f"the runner imports the grid module: {imported}"


def artifact_tree(tmp_path: Path, *relative_dirs: str, rows: object = None) -> Path:
    """Write ``rows`` as a grid under each of ``relative_dirs``; return the artifact root."""
    # Nested on purpose: internal compute delivers the grid several levels down, and
    # under a test-results/ mirror as well, so a grader joining a fixed path finds nothing.
    root = tmp_path / "artifact"
    payload = CI_GRID_ROWS if rows is None else rows
    for relative in relative_dirs:
        out = root / relative
        out.mkdir(parents=True, exist_ok=True)
        (out / GRID_FILE_NAME).write_text(json.dumps(payload))
    return root


def test_find_grid_dirs_reaches_a_nested_grid(tmp_path):
    """The grader is handed an artifact root, not the grid's directory."""
    root = artifact_tree(tmp_path, "outputs/moduscope")
    assert grid.find_grid_dirs(str(root)) == [str(root / "outputs/moduscope")]


def test_find_grid_dirs_returns_both_copies_of_a_mirrored_artifact(tmp_path):
    """Returning one would make which copy gets graded depend on glob order."""
    root = artifact_tree(tmp_path, "outputs/moduscope", "test-results/outputs/moduscope")
    assert len(grid.find_grid_dirs(str(root))) == 2


def test_find_grid_dirs_is_empty_when_there_is_no_grid(tmp_path):
    """Control for the two cases above: the glob must be capable of finding nothing."""
    (tmp_path / "artifact").mkdir()
    assert grid.find_grid_dirs(str(tmp_path / "artifact")) == []


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # The happy path, and the one check that the shipped default grades what it dispatches.
        (CI_GRID_ROWS, 0),
        # The whole reason this job exists: ModuScope exits 0 having run two of three.
        (CI_GRID_ROWS[:-1], 1),
        # Naming exactly the requested ids is not enough; the rows have to say pass.
        ([*CI_GRID_ROWS[:-1], dict(CI_GRID_ROWS[-1], status="fail")], 1),
    ],
    ids=["complete", "model-absent", "row-failed"],
)
def test_grader_cli_verdict_follows_the_grid(tmp_path, rows, expected):
    """Exit code 0 only when the grid proves every requested model ran and passed."""
    root = artifact_tree(tmp_path, "outputs/moduscope", rows=rows)
    assert grid.main([str(root), "--model-ids", CI_MODEL_IDS_RAW]) == expected


def test_grader_cli_fails_when_one_mirrored_copy_disagrees(tmp_path):
    """Every grid found must pass, so an extra copy cannot be the one that happens to grade."""
    # Catches a grader that stops at its first success: the good copy is found first by
    # sort order, and returning 0 on it would hide the mirror naming the wrong model.
    root = artifact_tree(tmp_path, "outputs/moduscope")
    mirror = root / "test-results/outputs/moduscope"
    mirror.mkdir(parents=True)
    (mirror / GRID_FILE_NAME).write_text(json.dumps([row(DECOY)]))
    assert grid.main([str(root), "--model-ids", CI_MODEL_IDS_RAW]) == 1


def test_grader_cli_refuses_ids_it_cannot_grade(tmp_path):
    """The CLI resolves through the runner, so a colliding pair is refused here too."""
    root = artifact_tree(tmp_path, "outputs/moduscope")
    with pytest.raises(ValueError, match="collide"):
        grid.main([str(root), "--model-ids", "foo/bar_baz,foo_bar/baz"])


def test_grader_cli_refuses_a_model_id_outside_the_shell_safe_grammar(tmp_path):
    """The grader shares the runner's grammar, so the two cannot accept different ids."""
    root = artifact_tree(tmp_path, "outputs/moduscope")
    with pytest.raises(ValueError):
        grid.main([str(root), "--model-ids", "owner/$(id)"])


def test_grader_cli_requires_model_ids(tmp_path):
    """Without them the grader has nothing to compare the grid to, so it must not default."""
    root = artifact_tree(tmp_path, "outputs/moduscope")
    with pytest.raises(SystemExit) as excinfo:
        grid.main([str(root)])
    assert excinfo.value.code == 2


def test_grader_cli_ignores_an_ambient_grid_filename_override(tmp_path, monkeypatch):
    """End to end through main(), not just read_grid: the whole CLI uses the fixed name."""
    monkeypatch.setenv("GRID_SKELETON_FILENAME", "renamed_grid.json")
    root = artifact_tree(tmp_path, "outputs/moduscope", rows=CI_GRID_ROWS)
    assert grid.main([str(root), "--model-ids", CI_MODEL_IDS_RAW]) == 0
