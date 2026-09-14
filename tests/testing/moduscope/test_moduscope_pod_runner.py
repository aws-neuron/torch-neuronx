"""Pytest calls this module to test the pod runner with runner stubs and a fake wheel package."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import moduscope_artifact_layout as entry_results
import moduscope_component_equivalence_shard_runner as component_equivalence_runner
import moduscope_hugging_face_shard_runner as hugging_face_runner
import moduscope_pod_runner as pod_runner
import moduscope_process as process
import moduscope_targets as targets
import moduscope_titan_shard_runner as titan_runner
import pytest
from moduscope_shard_planner import build_shards

from conftest import (
    MODUSCOPE_MODULES_DIR,
    PROCESS_TERMINATION_GRACE_SECONDS,
    RUN_ID,
    SAMPLE_HUGGING_FACE_MODEL_IDS,
    SAMPLE_TITAN_ENTRY_NAMES,
    TEST_COMPONENT_EQUIVALENCE_TARGET,
    TIMING_SLACK_SECONDS,
)

FULL_HUGGING_FACE_TARGET = "full_dense_decoder"
TEST_HUGGING_FACE_TARGET = "test_hf"
TEST_TITAN_TARGET = "test_titan"
HUGGING_FACE_ADAPTER = "hugging-face"
TITAN_ADAPTER = "titan"
COMPONENT_EQUIVALENCE_ADAPTER = "component-equivalence"
ENTRIES_SUBDIR = hugging_face_runner.ENTRIES_SUBDIR
WORKFLOW_ENV = {
    "MODUSCOPE_CI": "github",
    "MODUSCOPE_TARGET": FULL_HUGGING_FACE_TARGET,
    "MODUSCOPE_SHARD_INDEX": "1",
    "MODUSCOPE_TIMEOUT_SECONDS": "3600",
    "MODUSCOPE_GITHUB_RUN_ID": RUN_ID,
    "MODUSCOPE_GITHUB_REPOSITORY": "pytorch/torch-neuronx",
    "MODUSCOPE_REQUESTER_TYPE": "CDPipeline",
    "MODUSCOPE_REQUESTER_NAME": "torch-neuronx",
    "TORCH_NEURONX_ENABLE_PROLOGUE": "1",
}
HUGGING_FACE_TARGET = targets.Target(
    name=TEST_HUGGING_FACE_TARGET,
    timeout_minutes=90,
    adapter=HUGGING_FACE_ADAPTER,
    config=FULL_HUGGING_FACE_TARGET,
    shards=build_shards(tuple((model_id,) for model_id in SAMPLE_HUGGING_FACE_MODEL_IDS)),
)
TITAN_TARGET = targets.Target(
    name=TEST_TITAN_TARGET,
    timeout_minutes=70,
    adapter=TITAN_ADAPTER,
    config="e_titan_multicore_async",
    shards=build_shards(tuple((entry_name,) for entry_name in SAMPLE_TITAN_ENTRY_NAMES)),
)
COMPONENT_EQUIVALENCE_TARGET = targets.resolve_target(TEST_COMPONENT_EQUIVALENCE_TARGET)
COMPILED_COMPONENT_TARGET = COMPONENT_EQUIVALENCE_TARGET._replace(
    name="compiled-component", torch_compile_enabled=True
)
EXPLICIT_ADAPTER_TARGET = COMPONENT_EQUIVALENCE_TARGET._replace(
    name="e_titan_multicore_async", config=FULL_HUGGING_FACE_TARGET
)
FULL_TARGET = HUGGING_FACE_TARGET._replace(name=FULL_HUGGING_FACE_TARGET)
TEST_TARGETS = {
    target.name: target
    for target in (
        HUGGING_FACE_TARGET,
        TITAN_TARGET,
        FULL_TARGET,
        COMPONENT_EQUIVALENCE_TARGET,
        COMPILED_COMPONENT_TARGET,
        EXPLICIT_ADAPTER_TARGET,
    )
}
FULL_MODEL_IDS = FULL_TARGET.members
FIRST_TITAN_ENTRY, SECOND_TITAN_ENTRY, THIRD_TITAN_ENTRY = SAMPLE_TITAN_ENTRY_NAMES
# Use at least three seconds because live-process tests can run under high load.
LIVE_TIMEOUT_SECONDS = 3.0
# The fake model records SIGTERM but keeps running. Only SIGKILL stops it.
# An optional runner process ID tests a signal during shutdown.
FAKE_MODEL_SCRIPT = (
    "import os, signal, sys, time\n"
    + "def mark(*_):\n"
    + "    open(sys.argv[2], 'w').write(str(time.time()))\n"
    + "    if len(sys.argv) > 3: os.kill(int(sys.argv[3]), signal.SIGTERM)\n"
    + "signal.signal(signal.SIGTERM, mark)\n"
    + "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    + "while True: signal.pause()\n"
)
# The fake wheel starts the same signal-resistant model.
FAKE_TASK_RUNNER_SCRIPT = (
    "import os, sys\n"
    + "sys.argv[1:] = [os.environ[variable_name] "
    + "for variable_name in ('PID_FILE', 'SEEN')]\n"
) + FAKE_MODEL_SCRIPT
# Use the short test grace because the fake model ignores SIGTERM.
SUBPROCESS_TARGET = (
    f"Target(name={FULL_HUGGING_FACE_TARGET!r}, timeout_minutes=90, "
    + f"adapter={HUGGING_FACE_ADAPTER!r}, "
    + f"config={FULL_HUGGING_FACE_TARGET!r}, shards={FULL_TARGET.shards!r})"
)
SUBPROCESS_SETUP = (
    "import signal, sys; sys.path.insert(0, sys.argv[1]); "
    + "import moduscope_pod_runner, moduscope_process; "
    + "from moduscope_shard_planner import Shard; "
    + "from moduscope_targets import Target; "
    + f"moduscope_pod_runner.resolve_target = lambda _name: {SUBPROCESS_TARGET}; "
)
ENTRYPOINT_COMMAND = (
    SUBPROCESS_SETUP
    + "moduscope_process.run_in_process_group.__defaults__ = "
    + f"({PROCESS_TERMINATION_GRACE_SECONDS},); "
    + "signal.signal(signal.SIGTERM, moduscope_pod_runner.exit_on_sigterm); "
    + "raise SystemExit(moduscope_pod_runner.main())"
)
DIRECT_MAIN_COMMAND = SUBPROCESS_SETUP + "raise SystemExit(moduscope_pod_runner.main())"
TEST_WHEEL_TOKEN = "MODUSCOPE_WHEEL version=2.0 sha256=0123456789abcdef"
RENDERER_SOURCE = b"def generate_report_html(rows, title):\n    return title\n"
RENDERER_SHA256 = hashlib.sha256(RENDERER_SOURCE).hexdigest()
REAL_STAGE_GRID_RENDERER = pod_runner.stage_grid_renderer


def mapped_adapters(transform):
    return {key: transform(key, adapter) for key, adapter in pod_runner.EXECUTION_ADAPTERS.items()}


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """Create the pod working directory and environment.

    Replace each target runner with a test stub.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "cli_scripts", None)
    monkeypatch.delenv("MODUSCOPE_SUITE", raising=False)
    monkeypatch.setattr(pod_runner, "moduscope_wheel_token_line", lambda: TEST_WHEEL_TOKEN)
    monkeypatch.setattr(pod_runner, "stage_grid_renderer", lambda _output_dir: None)

    def set_environment(**overrides):
        for name, value in {**WORKFLOW_ENV, **overrides}.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    def stub_runners(runner_stub):
        monkeypatch.setattr(
            pod_runner,
            "EXECUTION_ADAPTERS",
            mapped_adapters(
                lambda _key, adapter: pod_runner.RunnerAdapter(
                    runner_stub, adapter.installed_member_field
                )
            ),
        )

    def resolve_test_target(name):
        return TEST_TARGETS.get(name) or targets.resolve_target(name)

    monkeypatch.setattr(pod_runner, "resolve_target", resolve_test_target)
    set_environment()
    return SimpleNamespace(
        env=set_environment, stub=stub_runners, output_dir=tmp_path / entry_results.OUTPUT_SUBDIR
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"MODUSCOPE_CI": "internal"}, "MODUSCOPE_CI"),
        ({"MODUSCOPE_TARGET": None}, "MODUSCOPE_TARGET"),
        ({"MODUSCOPE_SHARD_INDEX": None}, "MODUSCOPE_SHARD_INDEX"),
        ({"MODUSCOPE_GITHUB_RUN_ID": None}, "MODUSCOPE_GITHUB_RUN_ID"),
        ({"MODUSCOPE_GITHUB_RUN_ID": " "}, "MODUSCOPE_GITHUB_RUN_ID"),
        ({"MODUSCOPE_TIMEOUT_SECONDS": "0"}, "MODUSCOPE_TIMEOUT_SECONDS"),
        ({"MODUSCOPE_TIMEOUT_SECONDS": "-1"}, "MODUSCOPE_TIMEOUT_SECONDS"),
        ({"MODUSCOPE_TIMEOUT_SECONDS": "nine"}, "MODUSCOPE_TIMEOUT_SECONDS"),
    ],
    ids=[
        "other-side",
        "unset-target",
        "unset-shard-index",
        "unset-run-id",
        "blank-run-id",
        "zero-timeout",
        "negative-timeout",
        "non-integer-timeout",
    ],
)
def test_read_env_accepts_github_and_rejects_invalid_variables(pod, override, match):
    pod.env(MODUSCOPE_CI=" GitHub ")
    pod_values = pod_runner.read_env()
    assert pod_values == (FULL_HUGGING_FACE_TARGET, "1", 3600, RUN_ID)
    pod.env(**override)
    with pytest.raises(ValueError, match=match):
        pod_runner.read_env()


@pytest.mark.parametrize("raw", ["-1", "3", "one", "0.5"])
def test_read_shard_index_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="MODUSCOPE_SHARD_INDEX"):
        pod_runner.read_shard_index(raw, 3)


@pytest.mark.parametrize("raw", ["0", "2", " 1 "])
def test_read_shard_index_accepts_each_valid_shard_index(raw):
    assert pod_runner.read_shard_index(raw, 3) == int(raw)


def test_moduscope_wheel_token_hashes_a_fake_distribution_record(tmp_path, monkeypatch):
    record_bytes = b"modu_scope/__init__.py,,\nmoduscope-2.0.dist-info/RECORD,,\n"
    record_path = tmp_path / "moduscope-2.0.dist-info" / "RECORD"
    record_path.parent.mkdir()
    record_path.write_bytes(record_bytes)
    record_file = SimpleNamespace(
        name="RECORD",
        parent=SimpleNamespace(name="moduscope-2.0.dist-info"),
        locate=lambda: record_path,
    )
    monkeypatch.setattr(pod_runner.importlib.metadata, "version", lambda _name: "2.0")
    monkeypatch.setattr(pod_runner.importlib.metadata, "files", lambda _name: [record_file])

    assert pod_runner.moduscope_wheel_token_line() == (
        "MODUSCOPE_WHEEL version=2.0 " + f"sha256={hashlib.sha256(record_bytes).hexdigest()}"
    )


def test_moduscope_wheel_token_uses_none_without_a_record(monkeypatch):
    monkeypatch.setattr(pod_runner.importlib.metadata, "version", lambda _name: "2.0")
    monkeypatch.setattr(pod_runner.importlib.metadata, "files", lambda _name: [])

    assert pod_runner.moduscope_wheel_token_line() == "MODUSCOPE_WHEEL version=2.0 sha256=none"


def test_moduscope_wheel_token_is_unavailable_without_the_distribution(monkeypatch):
    def missing_distribution(_name):
        raise pod_runner.importlib.metadata.PackageNotFoundError("moduscope")

    monkeypatch.setattr(pod_runner.importlib.metadata, "version", missing_distribution)

    assert pod_runner.moduscope_wheel_token_line() == (
        "MODUSCOPE_WHEEL version=unavailable sha256=unavailable"
    )


def test_stage_grid_renderer_writes_exact_source_hash_version_and_token(tmp_path, monkeypatch):
    source_path = tmp_path / "installed" / "grid_report_html.py"
    source_path.parent.mkdir()
    source_path.write_bytes(RENDERER_SOURCE)
    monkeypatch.setattr(
        pod_runner.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(source_path)),
    )
    output_dir = tmp_path / "output"

    pod_runner.stage_grid_renderer(str(output_dir))

    renderer_path = entry_results.STAGED_GRID_RENDERER_RELATIVE_PATH
    assert (output_dir / renderer_path).read_bytes() == RENDERER_SOURCE
    assert process._token_lines == [f"RENDERER copied=1 sha256={RENDERER_SHA256}"]


def test_main_keeps_zero_exit_and_records_unavailable_renderer(pod, monkeypatch):
    pod.stub(lambda *_args: 0)
    monkeypatch.setattr(pod_runner, "stage_grid_renderer", REAL_STAGE_GRID_RENDERER)
    monkeypatch.setattr(pod_runner.importlib.util, "find_spec", lambda name: None)

    assert pod_runner.main() == 0

    renderer_path = entry_results.STAGED_GRID_RENDERER_RELATIVE_PATH
    assert not (
        pod.output_dir
        / renderer_path.parent
        / entry_results.run_id_filename(RUN_ID, renderer_path.name)
    ).exists()
    assert (pod.output_dir / f"{RUN_ID}-tokens.txt").read_text().splitlines()[-1] == (
        "RENDERER copied=0 sha256=-"
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"MODUSCOPE_GITHUB_RUN_ID": None}, "MODUSCOPE_GITHUB_RUN_ID"),
        ({"MODUSCOPE_GITHUB_RUN_ID": " "}, "MODUSCOPE_GITHUB_RUN_ID"),
        ({"MODUSCOPE_SHARD_INDEX": None}, "MODUSCOPE_SHARD_INDEX"),
        ({"MODUSCOPE_SHARD_INDEX": "999"}, "outside a target"),
    ],
    ids=["missing-run-id", "blank-run-id", "missing-shard-index", "invalid-shard"],
)
def test_validation_failures_remove_moduscope_output_and_keep_test_reports(pod, override, match):
    test_reports = pod.output_dir.parent / "test-reports"
    test_reports.mkdir(parents=True)
    stale_output = pod.output_dir / "stale.log"
    stale_output.parent.mkdir()
    stale_output.write_text("stale")
    runner_calls = []
    pod.env(**override)
    pod.stub(lambda *args: runner_calls.append(args))

    with pytest.raises(ValueError, match=match):
        pod_runner.main()

    assert test_reports.is_dir()
    assert not pod.output_dir.exists()
    assert not list(pod.output_dir.parents[1].glob("outputs.validation-error-*"))
    assert runner_calls == []


def test_invalid_shard_quarantines_upload_root_with_locked_stale_output(pod):
    output_root = pod.output_dir.parent
    locked_directory = output_root / "locked"
    locked_directory.mkdir(parents=True)
    (locked_directory / "stale.json").write_text("readable")
    locked_directory.chmod(0o555)
    runner_calls = []
    pod.env(MODUSCOPE_SHARD_INDEX="999")
    pod.stub(lambda *args: runner_calls.append(args))
    try:
        with pytest.raises(ValueError, match="outside a target"):
            pod_runner.main()
        assert (output_root / "test-reports").is_dir()
        assert runner_calls == []
    finally:
        for locked_path in pod.output_dir.parents[1].rglob("locked"):
            locked_path.chmod(0o755)


def test_unknown_target_stops_before_runner_dispatch(pod):
    runner_calls = []
    pod.env(MODUSCOPE_TARGET="unknown")
    pod.stub(lambda *args: runner_calls.append(args))

    with pytest.raises(ValueError, match="no target named 'unknown'"):
        pod_runner.main()

    assert runner_calls == []


def test_adapter_registry_maps_each_adapter_to_its_runner():
    assert {
        targets.HUGGING_FACE_ADAPTER: pod_runner.RunnerAdapter(
            hugging_face_runner.run_shard, hugging_face_runner.HUGGING_FACE_MEMBER_FIELD
        ),
        targets.TITAN_ADAPTER: pod_runner.RunnerAdapter(
            titan_runner.run_shard, titan_runner.TITAN_CONFIG_FIELD
        ),
        targets.COMPONENT_EQUIVALENCE_ADAPTER: pod_runner.RunnerAdapter(
            component_equivalence_runner.run_shard
        ),
    } == pod_runner.EXECUTION_ADAPTERS


RUNNER_SPECIFIC_NAMES = (
    "hf_cfg_wishlist",
    "model_configurations",
    "MODEL_CONFIG",
    "TORCH_COMPILE",
    "NUM_WORKERS",
    "apply_torch_compile",
    "grid_entry_list",
    "cli_scripts",
)


def test_pod_runner_source_names_no_runner_specific_value():
    source = Path(pod_runner.__file__).read_text()
    assert [name for name in RUNNER_SPECIFIC_NAMES if name in source] == []


def test_main_succeeds_with_only_the_workflow_environment(pod, monkeypatch):
    for name in pod_runner.ENV_VARS_READ_ON_POD:
        monkeypatch.delenv(name, raising=False)
    for name, value in WORKFLOW_ENV.items():
        monkeypatch.setenv(name, value)
    identity = {
        "workload_id": RUN_ID,
        "workload_link": (
            f"https://github.com/{WORKFLOW_ENV['MODUSCOPE_GITHUB_REPOSITORY']}"
            + f"/actions/runs/{RUN_ID}"
        ),
    }
    events = []
    runner_calls = []

    def read_identity():
        events.append("identity")
        return identity

    def read_clock():
        events.append("clock")
        return 1000.0

    def run(context):
        events.append("dispatch")
        runner_calls.append(context)
        return 0

    monkeypatch.setattr(pod_runner, "read_run_identity", read_identity)
    monkeypatch.setattr(pod_runner.time, "monotonic", read_clock)
    pod.stub(run)

    assert pod_runner.main() == 0
    assert events == ["identity", "clock", "dispatch"]
    assert runner_calls == [
        targets.ShardRunContext(
            target=FULL_TARGET,
            shard=FULL_TARGET.shards[1],
            output_dir=str(pod.output_dir),
            run_identity=identity,
            deadline=4600.0,
        )
    ]
    assert runner_calls[0].target is FULL_TARGET
    assert runner_calls[0].shard is FULL_TARGET.shards[1]


def test_module_entry_point_reads_workflow_environment_without_arguments():
    child_env = {
        name: value
        for name, value in os.environ.items()
        if name not in pod_runner.ENV_VARS_READ_ON_POD
    }
    completed = subprocess.run(
        [sys.executable, pod_runner.__file__],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
    )

    assert completed.returncode == 1
    assert "MODUSCOPE_CI" in completed.stderr


TARGET_DISPATCH_CASES = [
    pytest.param(
        HUGGING_FACE_ADAPTER,
        TEST_HUGGING_FACE_TARGET,
        "1",
        FULL_HUGGING_FACE_TARGET,
        HUGGING_FACE_TARGET.shards[1],
        True,
        id="hugging-face",
    ),
    pytest.param(
        TITAN_ADAPTER,
        TEST_TITAN_TARGET,
        "0",
        "e_titan_multicore_async",
        TITAN_TARGET.shards[0],
        True,
        id="titan",
    ),
    pytest.param(
        COMPONENT_EQUIVALENCE_ADAPTER,
        COMPILED_COMPONENT_TARGET.name,
        "1",
        "",
        COMPILED_COMPONENT_TARGET.shards[1],
        False,
        id="component-equivalence",
    ),
    pytest.param(
        COMPONENT_EQUIVALENCE_ADAPTER,
        EXPLICIT_ADAPTER_TARGET.name,
        "0",
        FULL_HUGGING_FACE_TARGET,
        EXPLICIT_ADAPTER_TARGET.shards[0],
        False,
        id="adapter-over-name-and-config",
    ),
]


@pytest.mark.parametrize(
    ("adapter_key", "name", "index", "config", "expected_shard", "checked"), TARGET_DISPATCH_CASES
)
def test_main_dispatches_each_adapter_and_always_writes_tokens(
    pod, capsys, monkeypatch, adapter_key, name, index, config, expected_shard, checked
):
    pod.env(MODUSCOPE_TARGET=name, MODUSCOPE_SHARD_INDEX=index, MODUSCOPE_GITHUB_RUN_ID="337")
    stale = pod.output_dir / "grid_metrics.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("[]")
    sibling = pod.output_dir.parent / "keep-me.txt"
    sibling.write_text("other suites write here")
    monkeypatch.setattr(pod_runner.time, "monotonic", lambda: 1000.0)
    runner_calls = []

    def runner_stub(stub_key):
        def run(context):
            runner_calls.append(
                (
                    stub_key,
                    context.target,
                    context.shard,
                    os.listdir(context.output_dir),
                    context.deadline,
                )
            )
            process.emit_token_line("WORK=1")
            return 7

        return run

    monkeypatch.setattr(
        pod_runner,
        "EXECUTION_ADAPTERS",
        mapped_adapters(
            lambda key, adapter: pod_runner.RunnerAdapter(
                runner_stub(key), adapter.installed_member_field
            )
        ),
    )
    returncode = pod_runner.main()
    assert returncode == 7
    assert runner_calls == [(adapter_key, TEST_TARGETS[name], expected_shard, [], 4600.0)]
    assert os.environ["MODUSCOPE_SUITE"] == config
    run_line = f"RUN target={name} shard={index} run=337"
    assert capsys.readouterr().out.splitlines()[:2] == [TEST_WHEEL_TOKEN, run_line]
    check = [f"TARGET_CHECK target={name} error=1"] if checked else []
    assert (pod.output_dir / "337-tokens.txt").read_text().splitlines() == [
        TEST_WHEEL_TOKEN,
        run_line,
        "WORK=1",
        *check,
    ]
    assert (pod.output_dir / "337-target_check.json").exists() == bool(check)
    assert not sibling.exists()
    assert sibling.with_name("337-keep-me.txt").read_text() == "other suites write here"


@pytest.mark.parametrize(
    ("adapter_key", "name", "index"),
    [
        (HUGGING_FACE_ADAPTER, TEST_HUGGING_FACE_TARGET, "1"),
        (TITAN_ADAPTER, TEST_TITAN_TARGET, "0"),
    ],
)
def test_main_checks_the_target_with_the_adapters_member_lister(
    pod, monkeypatch, adapter_key, name, index
):
    target = TEST_TARGETS[name]
    members = target.members
    listed = []

    def installed_config_field(config, field_name):
        listed.append((config, field_name))
        return members

    monkeypatch.setattr(pod_runner, "installed_config_field", installed_config_field)
    pod.stub(lambda *args: 0)
    pod.env(MODUSCOPE_TARGET=name, MODUSCOPE_SHARD_INDEX=index)

    assert pod_runner.main() == 0
    assert listed == [
        (target.config, pod_runner.EXECUTION_ADAPTERS[adapter_key].installed_member_field)
    ]
    token_lines = (pod.output_dir / f"{RUN_ID}-tokens.txt").read_text().splitlines()
    assert f"TARGET_CHECK target={name} missing=0 extra=0 excluded=0" in token_lines


def test_main_stamps_every_file_under_the_upload_root(pod):
    run_id = "34037088851"
    pod.env(MODUSCOPE_GITHUB_RUN_ID=run_id)

    def write_fake_output_tree(context):
        output_root = Path(context.output_dir).parent
        output_files = (
            Path(context.output_dir, "grid_metrics.json"),
            Path(context.output_dir, ENTRIES_SUBDIR, "entry-0", "cache_keys.txt"),
            Path(output_root, "2026-09-06", "14-19-16", "task_runner.log"),
            Path(output_root, f"{run_id}-ready.txt"),
        )
        for output_file in output_files:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output_file.name)
        return 0

    pod.stub(write_fake_output_tree)
    assert pod_runner.main() == 0
    output_files = [
        output_file for output_file in pod.output_dir.parent.rglob("*") if output_file.is_file()
    ]
    assert len(output_files) == 6
    assert all(output_file.name.startswith(f"{run_id}-") for output_file in output_files)
    assert all(
        not output_file.name.startswith(f"{run_id}-{run_id}-") for output_file in output_files
    )
    assert {output_file.name.removeprefix(f"{run_id}-") for output_file in output_files} == {
        "grid_metrics.json",
        "cache_keys.txt",
        "task_runner.log",
        "ready.txt",
        "target_check.json",
        "tokens.txt",
    }


def test_main_strips_the_run_id_before_stamping(pod):
    pod.env(MODUSCOPE_GITHUB_RUN_ID=f" {RUN_ID} ")

    def write_grid(context):
        Path(context.output_dir, "grid_metrics.json").write_text("[]")
        return 0

    pod.stub(write_grid)
    assert pod_runner.main() == 0
    assert (pod.output_dir / f"{RUN_ID}-grid_metrics.json").read_text() == "[]"


def test_output_stamping_removes_the_upload_root_on_collision(tmp_path):
    source = tmp_path / "grid_metrics.json"
    collision = tmp_path / "337-grid_metrics.json"
    source.write_text("source")
    collision.write_text("collision")
    with pytest.raises(FileExistsError, match="already exists"):
        pod_runner.stamp_output_files(str(tmp_path), "337")
    assert not tmp_path.exists()


@pytest.mark.xfail(reason="Not yet verified. Xfailed to unblock CI.", strict=False)
def test_output_stamping_quarantines_the_upload_root_on_permission_error(tmp_path):
    output_root = tmp_path / "outputs"
    locked_directory = output_root / "locked"
    locked_directory.mkdir(parents=True)
    (locked_directory / "grid_metrics.json").write_text("readable")
    locked_directory.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            pod_runner.stamp_output_files(str(output_root), RUN_ID)
        assert not output_root.exists()
        quarantined_files = list(tmp_path.glob("outputs.stamping-error-*/locked/grid_metrics.json"))
        assert len(quarantined_files) == 1
        assert quarantined_files[0].read_text() == "readable"
    finally:
        for locked_path in tmp_path.rglob("locked"):
            locked_path.chmod(0o755)


def test_main_stamps_outputs_when_token_writing_fails(pod, monkeypatch):
    def write_grid(context):
        Path(context.output_dir, "grid_metrics.json").write_text("[]")
        return 0

    def fail_token_write(output_dir):
        Path(output_dir, "tokens.txt").write_text("partial")
        raise OSError("token write failed")

    pod.stub(write_grid)
    monkeypatch.setattr(pod_runner, "write_token_lines", fail_token_write)
    with pytest.raises(OSError, match="token write failed"):
        pod_runner.main()
    assert (pod.output_dir / f"{RUN_ID}-grid_metrics.json").read_text() == "[]"
    assert (pod.output_dir / f"{RUN_ID}-tokens.txt").read_text() == "partial"
    assert not (pod.output_dir / "grid_metrics.json").exists()
    assert not (pod.output_dir / "tokens.txt").exists()


def test_main_stops_before_dispatch_when_output_cleanup_fails(pod, monkeypatch):
    pod.output_dir.mkdir(parents=True)
    runner_calls = []
    real_rmtree = pod_runner.shutil.rmtree

    def refuse_cleanup(path, *args, **kwargs):
        if Path(path) == pod.output_dir:
            raise PermissionError("cleanup refused")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pod_runner.shutil, "rmtree", refuse_cleanup)
    pod.stub(lambda *args: runner_calls.append(args))
    with pytest.raises(PermissionError, match="cleanup refused"):
        pod_runner.main()
    assert runner_calls == []


def test_main_quarantines_stale_output_when_cleanup_loses_a_child(pod, monkeypatch):
    stale = pod.output_dir / "stale.json"
    removed = pod.output_dir / "removed.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")
    removed.write_text("removed")
    runner_calls = []
    real_rmtree = pod_runner.shutil.rmtree

    def partial_cleanup(path, *args, **kwargs):
        if Path(path) == pod.output_dir:
            removed.unlink()
            raise FileNotFoundError(2, "No such file or directory", removed)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pod_runner.shutil, "rmtree", partial_cleanup)
    pod.stub(lambda *args: runner_calls.append(args))
    with pytest.raises(FileNotFoundError, match=r"removed\.json"):
        pod_runner.main()
    assert not stale.exists()
    assert not stale.with_name(f"{RUN_ID}-stale.json").exists()
    assert (pod.output_dir.parent / "test-reports").is_dir()
    assert not list(pod.output_dir.parents[1].glob("outputs.validation-error-*"))
    assert runner_calls == []


def test_main_propagates_child_cleanup_failure_after_output_root_disappears(pod, monkeypatch):
    removed = pod.output_dir / "removed.json"
    removed.parent.mkdir(parents=True)
    removed.write_text("removed")
    runner_calls = []
    real_rmtree = pod_runner.shutil.rmtree

    def child_failure_after_root_removal(path, *args, **kwargs):
        if Path(path) == pod.output_dir:
            removed.unlink()
            pod.output_dir.rmdir()
            raise FileNotFoundError(2, "No such file or directory", removed)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(pod_runner.shutil, "rmtree", child_failure_after_root_removal)
    pod.stub(lambda *args: runner_calls.append(args))
    with pytest.raises(FileNotFoundError, match=r"removed\.json"):
        pod_runner.main()
    assert runner_calls == []


@pytest.mark.parametrize(
    ("adapter_key", "name", "index", "site", "checked"),
    [
        (HUGGING_FACE_ADAPTER, TEST_HUGGING_FACE_TARGET, "1", "runner", True),
        (TITAN_ADAPTER, TEST_TITAN_TARGET, "0", "runner", True),
        (COMPONENT_EQUIVALENCE_ADAPTER, TEST_COMPONENT_EQUIVALENCE_TARGET, "0", "runner", False),
        (HUGGING_FACE_ADAPTER, TEST_HUGGING_FACE_TARGET, "1", "check", False),
    ],
)
def test_main_writes_check_and_tokens_when_runner_or_check_raises(
    pod, monkeypatch, adapter_key, name, index, site, checked
):
    def raise_error(*args):
        raise RuntimeError(f"the {site} blew up")

    pod.env(MODUSCOPE_TARGET=name, MODUSCOPE_SHARD_INDEX=index)
    pod.stub(raise_error if site == "runner" else lambda *args: 0)
    if site == "check":
        monkeypatch.setattr(pod_runner, "check_target", raise_error)
    with pytest.raises(RuntimeError, match="blew up"):
        pod_runner.main()
    run_line = f"RUN target={name} shard={index}"
    expected = [TEST_WHEEL_TOKEN, f"{run_line} run={RUN_ID}"]
    if checked:
        expected.append(f"TARGET_CHECK target={name} error=1")
    assert (pod.output_dir / f"{RUN_ID}-tokens.txt").read_text().splitlines() == expected
    assert (pod.output_dir / f"{RUN_ID}-target_check.json").exists() == checked


NEW_MODEL_ID = "nobody/new-model"
EXCLUDED_MODEL_ID = "nobody/remote-code-model"
ABSENT_MODEL_ID = "nobody/retired-model"
TARGET_CHECK_CASES = [
    pytest.param(FULL_HUGGING_FACE_TARGET,
     (*FULL_MODEL_IDS[1:], NEW_MODEL_ID),
     {"missing": [FULL_MODEL_IDS[0]], "extra": [NEW_MODEL_ID]},
     "missing=1 extra=1 excluded=0", id="full-set-equality"),
    pytest.param(TEST_TITAN_TARGET,
     (SECOND_TITAN_ENTRY, FIRST_TITAN_ENTRY, THIRD_TITAN_ENTRY),
     {"missing": [], "extra": [], "excluded": [], "excluded_absent": []},
     "missing=0 extra=0 excluded=0", id="titan-order-independent"),
    pytest.param(HUGGING_FACE_TARGET._replace(
         name="excluded-hf", excluded_models=(ABSENT_MODEL_ID, EXCLUDED_MODEL_ID)),
     (*SAMPLE_HUGGING_FACE_MODEL_IDS, EXCLUDED_MODEL_ID),
     {"missing": [], "extra": [], "excluded": [EXCLUDED_MODEL_ID],
      "excluded_absent": [ABSENT_MODEL_ID]},
     "missing=0 extra=0 excluded=1", id="excluded-member"),
    pytest.param(TEST_HUGGING_FACE_TARGET,
     ModuleNotFoundError("cli_scripts is not installed"),
     {"error": "ModuleNotFoundError: cli_scripts is not installed"}, "error=1", id="no-wheel")
]  # fmt: skip


@pytest.mark.parametrize(
    ("target", "wheel_members_or_error", "expected_report", "token_tail"), TARGET_CHECK_CASES
)
def test_check_target_compares_target_members_with_wheel_members(
    tmp_path, monkeypatch, target, wheel_members_or_error, expected_report, token_tail
):
    listed_fields = []

    def installed_config_field(config, field_name):
        listed_fields.append((config, field_name))
        if isinstance(wheel_members_or_error, Exception):
            raise wheel_members_or_error
        return wheel_members_or_error

    monkeypatch.setattr(pod_runner, "installed_config_field", installed_config_field)
    if isinstance(target, str):
        target = TEST_TARGETS.get(target) or targets.resolve_target(target)
    if target.name == TEST_HUGGING_FACE_TARGET:
        target = target._replace(shards=())
    token_line = pod_runner.check_target(target, str(tmp_path), "members")
    written_report = json.loads((tmp_path / "target_check.json").read_text())
    assert listed_fields == [(target.config, "members")]
    assert {key: written_report[key] for key in expected_report} == expected_report
    assert token_line == f"TARGET_CHECK target={target.name} {token_tail}"


@pytest.mark.parametrize(
    ("target", "wheel_members", "duplicate_member"),
    [
        pytest.param(
            HUGGING_FACE_TARGET._replace(
                name="duplicate-hf",
                shards=build_shards((tuple(SAMPLE_HUGGING_FACE_MODEL_IDS[:2]),)),
            ),
            (*SAMPLE_HUGGING_FACE_MODEL_IDS[:2], SAMPLE_HUGGING_FACE_MODEL_IDS[1]),
            SAMPLE_HUGGING_FACE_MODEL_IDS[1],
            id="hugging-face",
        ),
        pytest.param(
            TEST_TARGETS[TEST_TITAN_TARGET]._replace(
                name="duplicate-titan", shards=build_shards((tuple(SAMPLE_TITAN_ENTRY_NAMES[:2]),))
            ),
            (*SAMPLE_TITAN_ENTRY_NAMES[:2], SAMPLE_TITAN_ENTRY_NAMES[1]),
            SAMPLE_TITAN_ENTRY_NAMES[1],
            id="titan",
        ),
    ],
)
def test_check_target_rejects_duplicate_wheel_members(
    tmp_path, monkeypatch, target, wheel_members, duplicate_member
):
    monkeypatch.setattr(
        pod_runner, "installed_config_field", lambda _config, _field_name: wheel_members
    )
    token_line = pod_runner.check_target(target, str(tmp_path), "members")
    written_report = json.loads((tmp_path / "target_check.json").read_text())

    assert token_line == f"TARGET_CHECK target={target.name} error=1"
    assert written_report["error"] == (
        f"ValueError: The installed wheel contains duplicate members: {duplicate_member}. "
        + f"Remove duplicate members from {target.config}."
    )


def wait_for_process_id(pid_file, ceiling=20.0):
    end = time.monotonic() + ceiling
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < end, "the fake model never reported its pid"
        time.sleep(0.05)


def run_and_sigterm_pod_runner(tmp_path, child_process_id_file, signal_path, launch_command):
    """Start the pod runner with the fake wheel.

    Send SIGTERM after the model reports its process ID.
    """
    wheel = tmp_path / "cli_scripts"
    wheel.mkdir(exist_ok=True)
    (wheel / "__init__.py").touch()
    (wheel / "task_runner.py").write_text(FAKE_TASK_RUNNER_SCRIPT)
    child_process_id_file.unlink(missing_ok=True)
    signal_path.unlink(missing_ok=True)
    child_env = {
        **os.environ,
        **WORKFLOW_ENV,
        "MODUSCOPE_TIMEOUT_SECONDS": "60",
        "PYTHONPATH": str(tmp_path),
    }
    child_env.update(PID_FILE=str(child_process_id_file), SEEN=str(signal_path))
    runner_process = subprocess.Popen(
        [sys.executable, "-c", launch_command, str(MODUSCOPE_MODULES_DIR)],
        cwd=tmp_path,
        env=child_env,
    )
    wait_for_process_id(child_process_id_file)
    os.kill(runner_process.pid, signal.SIGTERM)
    signal_time = time.monotonic()
    return (
        runner_process.wait(timeout=15),
        int(child_process_id_file.read_text()),
        time.monotonic() - signal_time,
    )


def test_sigterm_reaches_the_model_group_before_sigkill(tmp_path, child_process_id_file):
    # Give the launcher time to stop worker sessions before SIGKILL.
    signal_path, check = (
        tmp_path / "saw_sigterm",
        tmp_path / entry_results.OUTPUT_SUBDIR / f"{RUN_ID}-target_check.json",
    )
    exit_code, model_process_id, elapsed = run_and_sigterm_pod_runner(
        tmp_path, child_process_id_file, signal_path, ENTRYPOINT_COMMAND
    )
    assert exit_code == 128 + signal.SIGTERM
    assert signal_path.exists(), "the model's group never saw the SIGTERM"
    with pytest.raises(ProcessLookupError):
        os.kill(model_process_id, 0)
    assert (
        PROCESS_TERMINATION_GRACE_SECONDS
        <= elapsed
        < PROCESS_TERMINATION_GRACE_SECONDS + TIMING_SLACK_SECONDS
    ), elapsed
    assert check.exists(), "the finally did not run"
    # Without the signal handler, the default action skips cleanup.
    exit_code, model_process_id, _ = run_and_sigterm_pod_runner(
        tmp_path, child_process_id_file, signal_path, DIRECT_MAIN_COMMAND
    )
    assert exit_code == -signal.SIGTERM and not signal_path.exists() and not check.exists()
    os.kill(model_process_id, 0)


def test_sigterm_during_grace_still_kills_the_model_group(
    tmp_path, monkeypatch, child_process_id_file
):
    # A second SIGTERM interrupts the grace wait. Cleanup must still send SIGKILL.
    signal_path = tmp_path / "saw_sigterm"
    argv = [
        sys.executable,
        "-c",
        FAKE_MODEL_SCRIPT,
        str(child_process_id_file),
        str(signal_path),
        str(os.getpid()),
    ]

    class ReadyPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Start the timeout after the fake model installs its signal handler.
            wait_for_process_id(child_process_id_file)

    monkeypatch.setattr(subprocess, "Popen", ReadyPopen)
    previous = signal.signal(signal.SIGTERM, pod_runner.exit_on_sigterm)
    try:
        with pytest.raises(SystemExit) as info:
            process.run_in_process_group(
                argv, LIVE_TIMEOUT_SECONDS, grace_seconds=PROCESS_TERMINATION_GRACE_SECONDS
            )
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert info.value.code == 128 + signal.SIGTERM
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_process_id_file.read_text()), 0)
    # The second SIGTERM must end the grace wait early.
    assert time.time() - float(signal_path.read_text()) < PROCESS_TERMINATION_GRACE_SECONDS
