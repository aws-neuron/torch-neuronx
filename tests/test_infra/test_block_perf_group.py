"""Unit tests for platform-specific block performance thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BLOCK_PERF_DIR = Path(__file__).resolve().parent.parent / "blocks" / "block_perf"
sys.path.insert(0, str(BLOCK_PERF_DIR))

import run_group  # noqa: E402
from run_group import build_command  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("TORCH_NEURONX_ENABLE_HOST_CC", raising=False)
    monkeypatch.delenv("PERSISTENT_STORAGE_BUCKET", raising=False)
    monkeypatch.setattr(run_group, "MFU_RETRY_DELAY_SECONDS", 0)


@pytest.fixture
def torchrun_test():
    return {
        "benchmark_runs": 20,
        "block": "qwen3_torchtitan",
        "id": "qwen3-test",
        "mfu": {
            "trn2": {"threshold": 10.0},
            "trn3pds-b1": {"threshold": 20.0},
        },
        "mfu_hostcc": {
            "trn2": {"baseline": 15.0, "baseline_date": "2026-01-01", "threshold": 13.5},
            "trn3pds-b1": {
                "baseline": 25.0,
                "baseline_date": "2026-01-01",
                "threshold": 22.5,
            },
        },
        "host_cc": True,
        "n_layers": 4,
        "nproc": 4,
        "preset": "qwen3-8b-tp",
        "runner": "torchrun",
        "tp_size": 4,
        "warmup_runs": 20,
    }


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("trn2", "9.500000"),
        ("trn3pds-b1", "19.000000"),
    ],
)
def test_build_command_selects_platform_threshold(torchrun_test, platform, expected):
    cmd = build_command(
        torchrun_test,
        platform,
        skip_mfu_threshold=False,
        mfu_threshold_multiplier=0.95,
    )

    threshold_index = cmd.index("--mfu-threshold")
    assert cmd[threshold_index + 1] == expected


def test_build_command_skips_threshold(torchrun_test):
    cmd = build_command(
        torchrun_test,
        "trn3pds-b1",
        skip_mfu_threshold=True,
    )

    assert "--mfu-threshold" not in cmd


@pytest.mark.parametrize("platform, expected", [("trn2", "12.825000"), ("trn3pds-b1", "21.375000")])
def test_build_command_selects_hostcc_threshold(torchrun_test, platform, expected):
    cmd = build_command(
        torchrun_test,
        platform,
        skip_mfu_threshold=False,
        mfu_threshold_multiplier=0.95,
        mfu_key="mfu_hostcc",
    )

    assert cmd[cmd.index("--mfu-threshold") + 1] == expected


def test_build_command_rejects_missing_platform_threshold(torchrun_test):
    del torchrun_test["mfu"]["trn3pds-b1"]

    with pytest.raises(
        ValueError,
        match="Missing MFU threshold for test 'qwen3-test' on platform 'trn3pds-b1'",
    ):
        build_command(
            torchrun_test,
            "trn3pds-b1",
            skip_mfu_threshold=False,
        )


def test_main_rejects_all_missing_thresholds_before_running(tmp_path, monkeypatch, capsys):
    config = tmp_path / "group.json"
    config.write_text(
        json.dumps(
            {
                "tests": [
                    {"id": "missing-explicit", "runner": "torchrun"},
                    {"id": "missing-default"},
                    {"id": "present", "mfu": {"trn3pds-b1": {"threshold": 10.0}}},
                    {"id": "skipped", "runner": "torchrun", "skip": True},
                    {"id": "singlecore", "runner": "singlecore"},
                ]
            }
        )
    )
    monkeypatch.setattr(
        run_group,
        "run_test",
        lambda *args, **kwargs: pytest.fail("validation must happen before the run loop"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
            "--platform",
            "trn3pds-b1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "missing-explicit/trn3pds-b1" in stderr
    assert "missing-default/trn3pds-b1" in stderr
    assert "skipped/trn3pds-b1" not in stderr
    assert "singlecore/trn3pds-b1" not in stderr


@pytest.mark.parametrize("config_name", ["qwen3.json", "gpt_oss.json"])
@pytest.mark.parametrize("mfu_key", ["mfu", "mfu_hostcc"])
def test_active_mfu_tests_define_each_platform(config_name, mfu_key):
    config_path = BLOCK_PERF_DIR / "group_configs" / config_name
    config = json.loads(config_path.read_text())

    for test in config["tests"]:
        if test.get("runner", "torchrun") != "torchrun" or test.get("skip"):
            continue
        if mfu_key == "mfu_hostcc" and not test.get("host_cc"):
            continue

        assert set(test[mfu_key]) == {"trn2", "trn3pds-b1"}, test["id"]
        for platform in ("trn2", "trn3pds-b1"):
            golden = test[mfu_key][platform]
            assert set(golden) == {"baseline", "baseline_date", "threshold"}
            assert 0 < golden["threshold"] <= golden["baseline"]


@pytest.mark.parametrize("hostcc_advisory", [False, True])
@pytest.mark.parametrize(
    "returncodes, statuses, strict_exit, advisory_exit",
    [
        ((0, 0), ("passed", "passed"), 0, 0),
        ((1, 0), ("failed", "passed"), 1, 1),
        ((0, 2), ("passed", "failed"), 2, 0),
        ((2, 1), ("failed", "failed"), 2, 2),
        ((-15, 0), ("interrupted", "passed"), -15, -15),
        ((0, -2), ("passed", "interrupted"), -2, -2),
    ],
)
def test_both_modes_keep_failures_and_separate_passes(
    tmp_path,
    monkeypatch,
    capsys,
    hostcc_advisory,
    returncodes,
    statuses,
    strict_exit,
    advisory_exit,
):
    calls = []

    def run(cmd, *, env):
        calls.append((cmd, env))
        return SimpleNamespace(returncode=returncodes[len(calls) - 1])

    monkeypatch.setattr(run_group.subprocess, "run", run)
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_STREAMS", "1")
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", "1")
    monkeypatch.setenv("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", "1")
    monkeypatch.setenv("NEURON_COMPILE_CACHE_URL", "s3://bucket/cache")
    for variable in run_group.CACHE_ENV_DIRS:
        if variable != "NEURON_COMPILE_CACHE_URL":
            monkeypatch.delenv(variable, raising=False)
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            "group.json",
            "--both-modes",
            "--refresh-persistent-goldens",
            "--output-dir",
            str(output_dir),
        ]
        + (["--hostcc-advisory"] if hostcc_advisory else []),
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == (advisory_exit if hostcc_advisory else strict_exit)
    assert len(calls) == 2
    receipt = json.loads((output_dir / "block-perf-status.json").read_text())
    assert receipt["hostcc_advisory"] is hostcc_advisory
    assert receipt["modes"] == {
        "streams": {"status": statuses[0], "returncode": returncodes[0], "advisory": False},
        "hostcc": {
            "status": statuses[1],
            "returncode": returncodes[1],
            "advisory": hostcc_advisory and statuses[1] == "failed",
        },
    }
    stdout = capsys.readouterr().out
    for mode, status in receipt["modes"].items():
        policy = "advisory" if status["advisory"] else "strict"
        assert (
            f"Block pass result: {mode} {status['status']} "
            f"(exit {status['returncode']}; {policy})"
        ) in stdout
    for (cmd, env), mode, enabled in zip(calls, ("streams", "hostcc"), ("0", "1"), strict=False):
        assert "--both-modes" not in cmd
        assert "--hostcc-advisory" not in cmd
        assert cmd[cmd.index("--config") + 1] == "group.json"
        assert cmd[-2:] == ["--output-dir", str(output_dir / mode)]
        assert "--refresh-persistent-goldens" in cmd
        assert "--skip-mfu-threshold" not in cmd
        assert env["TORCH_NEURONX_ENABLE_STREAMS"] == "1"
        assert env["TORCH_NEURONX_ENABLE_HOST_CC"] == enabled
        assert ("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE" in env) == (mode == "hostcc")
        assert env["NEURON_COMPILE_CACHE_URL"] == f"s3://bucket/cache/{mode}"
        assert env["NKI_COMPILE_CACHE_URL"].endswith(f"/nki-compile/{mode}")
        assert env["TORCHINDUCTOR_CACHE_DIR"].endswith(f"/inductor/{mode}")
        for variable in run_group.CACHE_ENV_DIRS:
            assert env[variable].endswith(f"/{mode}")
            assert calls[0][1][variable] != calls[1][1][variable]


def test_hostcc_advisory_requires_both_modes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_group.py", "--config", str(tmp_path / "unused.json"), "--hostcc-advisory"],
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 2
    assert "--hostcc-advisory requires --both-modes" in capsys.readouterr().err


@pytest.mark.parametrize("failed_mode", ["streams", "hostcc"])
@pytest.mark.parametrize("error, status", [(KeyboardInterrupt, "interrupted"), (OSError, "error")])
def test_both_modes_records_and_propagates_interruption_or_launch_error(
    tmp_path, monkeypatch, failed_mode, error, status
):
    calls = []

    def run(cmd, *, env):
        mode = "hostcc" if env["TORCH_NEURONX_ENABLE_HOST_CC"] == "1" else "streams"
        calls.append(mode)
        if mode == failed_mode:
            raise error
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_group.subprocess, "run", run)

    with pytest.raises(error):
        run_group.run_both_modes(["--both-modes", "--hostcc-advisory"], tmp_path, True)

    receipt = json.loads((tmp_path / "block-perf-status.json").read_text())
    assert receipt["modes"][failed_mode] == {
        "status": status,
        "returncode": None,
        "advisory": False,
    }
    if failed_mode == "streams":
        assert calls == ["streams"]
        assert receipt["modes"]["hostcc"]["status"] == "not-run"
    else:
        assert calls == ["streams", "hostcc"]
        assert receipt["modes"]["streams"]["status"] == "passed"


def test_both_modes_keeps_receipt_write_failure_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_group.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )

    def write_text(*args, **kwargs):
        raise OSError("Cannot write the status receipt.")

    monkeypatch.setattr(Path, "write_text", write_text)

    with pytest.raises(OSError, match="Cannot write the status receipt"):
        run_group.run_both_modes(["--both-modes", "--hostcc-advisory"], tmp_path, True)


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("retry_passes", [False, True])
def test_hostcc_enforces_goldens_and_retries(
    tmp_path, monkeypatch, torchrun_test, refresh, retry_passes
):
    config = tmp_path / "group.json"
    config.write_text(json.dumps({"tests": [torchrun_test, {"id": "default-only"}]}))
    calls = []
    refreshed = []

    def run_test(test, platform, skip_mfu_threshold, timeout, multiplier, output_dir, mfu_key):
        cmd = build_command(test, platform, skip_mfu_threshold, multiplier, output_dir, mfu_key)
        calls.append(cmd)
        assert test["id"] == "qwen3-test"
        assert mfu_key == "mfu_hostcc"
        assert cmd[cmd.index("--mfu-threshold") + 1] == "13.500000"
        passed = len(calls) == 2 and retry_passes
        return run_group.TestRunResult(passed, output=run_group.MFU_GATE_FAILURE_MARKER)

    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", "1")
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(
        run_group.PersistentMfuGoldens,
        "refresh",
        lambda self, output_dir, results: refreshed.append((self.mfu_key, results)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_group.py", "--config", str(config)]
        + (["--refresh-persistent-goldens"] if refresh else []),
    )
    monkeypatch.chdir(tmp_path)

    if retry_passes:
        run_group.main()
    else:
        with pytest.raises(SystemExit) as exc:
            run_group.main()
        assert exc.value.code == 1

    assert len(calls) == 2
    assert refreshed == ([("mfu_hostcc", {"qwen3-test": retry_passes})] if refresh else [])


@pytest.mark.parametrize("missing_map", [False, True])
def test_hostcc_rejects_missing_seed_before_running(
    tmp_path, monkeypatch, capsys, torchrun_test, missing_map
):
    if missing_map:
        del torchrun_test["mfu_hostcc"]
    else:
        del torchrun_test["mfu_hostcc"]["trn3pds-b1"]
    config = tmp_path / "group.json"
    config.write_text(json.dumps({"tests": [torchrun_test]}))
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", "1")
    monkeypatch.setattr(
        run_group, "run_test", lambda *args: pytest.fail("A HostCC seed is required.")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_group.py", "--config", str(config), "--platform", "trn3pds-b1"],
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 2
    assert "qwen3-test/trn3pds-b1" in capsys.readouterr().err


@pytest.mark.parametrize("hostcc", ["0", "1"])
@pytest.mark.parametrize("missing_map", [False, True])
def test_explicit_capture_runs_without_seed_or_golden_storage(
    tmp_path, monkeypatch, torchrun_test, hostcc, missing_map
):
    mfu_key = "mfu_hostcc" if hostcc == "1" else "mfu"
    if missing_map:
        del torchrun_test[mfu_key]
    else:
        del torchrun_test[mfu_key]["trn3pds-b1"]
    config = tmp_path / "group.json"
    config.write_text(json.dumps({"tests": [torchrun_test]}))
    output_dir = tmp_path / "capture"
    calls = []

    def run_test(test, platform, skip_mfu_threshold, timeout, multiplier, outputs, selected_key):
        cmd = build_command(test, platform, skip_mfu_threshold, multiplier, outputs, selected_key)
        calls.append(cmd)
        assert skip_mfu_threshold is True
        assert selected_key == mfu_key
        assert platform == "trn3pds-b1"
        assert "--mfu-threshold" not in cmd
        assert "--save_results" in cmd
        assert cmd[cmd.index("--output_file") + 1] == str(output_dir / "qwen3-test.json")
        assert outputs.is_dir()
        return run_group.TestRunResult(True)

    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", hostcc)
    monkeypatch.setenv("PERSISTENT_STORAGE_BUCKET", "bucket")
    monkeypatch.setattr(
        run_group,
        "PersistentMfuGoldens",
        lambda *args: pytest.fail("Explicit capture must bypass the golden store."),
    )
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
            "--platform",
            "trn3pds-b1",
            "--skip-mfu-threshold",
            "--output-dir",
            str(output_dir),
        ],
    )

    run_group.main()

    assert len(calls) == 1


@pytest.mark.parametrize("hostcc", ["0", "1"])
@pytest.mark.parametrize("both_modes", [False, True])
def test_refresh_requires_threshold_enforcement(tmp_path, monkeypatch, capsys, hostcc, both_modes):
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", hostcc)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(tmp_path / "unused.json"),
            "--refresh-persistent-goldens",
            "--skip-mfu-threshold",
        ]
        + (["--both-modes", "--hostcc-advisory"] if both_modes else []),
    )

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 2
    assert "requires MFU threshold enforcement" in capsys.readouterr().err


def test_group_output_paths_include_pass_and_test(torchrun_test):
    infra = json.loads((BLOCK_PERF_DIR / "group_configs" / "infra.json").read_text())
    for mode in ("streams", "hostcc"):
        output_dir = Path("outputs") / mode
        for test in [torchrun_test, *infra["tests"]]:
            cmd = build_command(test, "trn2", True, output_dir=output_dir)
            if test["runner"] == "singlecore":
                continue
            flag, expected = "--output_file", output_dir / f"{test['id']}.json"
            assert cmd[cmd.index(flag) + 1] == str(expected)
