"""Test target validation and workflow plans."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import moduscope_shard_planner as shard_planner
import moduscope_targets as targets_module
import pytest

from conftest import (
    LLAMA_MODEL_ID,
    MODUSCOPE_MODULES_DIR,
    OLMO_MODEL_ID,
    PHI3_MODEL_ID,
    SAMPLE_HUGGING_FACE_MODEL_IDS,
    SAMPLE_TITAN_ENTRY_NAMES,
    TEST_COMPONENT_EQUIVALENCE_TARGET,
)

SHIPPED_TARGET_DATA = json.loads((MODUSCOPE_MODULES_DIR / "targets.json").read_text())["targets"]
TRAINIUM_INSTANCE = "trn2.48xlarge"
TEST_ADAPTER = "hugging-face"
DENSE_COMPONENT_SHARDS = (
    ("qwen3_8b", "qwen3_14b", "qwen3_32b"),
    ("llama3_8b", "llama3_70b"),
)
DENSE_COMPONENT_ROWS = [
    {"index": 0, "label": "qwen3_8b +2"},
    {"index": 1, "label": "llama3_8b +1"},
]


def target_data(shards=(("member",),), **overrides) -> dict:
    data = {
        "adapter": TEST_ADAPTER,
        "shards": [list(shard) for shard in shards],
        "timeout_minutes": 60,
    }
    return {**data, **overrides}


def read_test_targets(tmp_path, targets: dict) -> dict[str, targets_module.Target]:
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": targets}))
    return targets_module.read_targets(str(path))


def shard_state(target):
    return [(shard.index, shard.members) for shard in target.shards]


def run_plan_command(target_name, timeout_minutes):
    root = MODUSCOPE_MODULES_DIR.parents[1]
    argv = [
        sys.executable,
        "testing/moduscope/moduscope_targets.py",
        "plan",
        "--target",
        target_name,
        "--timeout-minutes",
        str(timeout_minutes),
        "--copy-buffer-minutes",
        "30",
    ]
    return subprocess.run(argv, cwd=root, capture_output=True, text=True)


def test_the_shipped_file_contains_only_the_component_target_from_any_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    assert SHIPPED_TARGET_DATA == {
        TEST_COMPONENT_EQUIVALENCE_TARGET: {
            "adapter": "component-equivalence",
            "shards": [list(shard) for shard in DENSE_COMPONENT_SHARDS],
            "timeout_minutes": 330,
        }
    }
    assert list(targets_module.read_targets()) == [TEST_COMPONENT_EQUIVALENCE_TARGET]


def test_target_loading_rejects_an_unknown_name_or_a_missing_targets_object(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": {"full_a": target_data(((OLMO_MODEL_ID,),))}}))
    with pytest.raises(ValueError, match=r"'full_b'.*full_a"):
        targets_module.resolve_target("full_b", path=str(path))
    path.write_text(json.dumps({"jobs": {}}))
    with pytest.raises(ValueError, match="'targets'"):
        targets_module.read_targets(str(path))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("kind", "hf", id="kind"),
        pytest.param("selector", "models", id="selector"),
        pytest.param("devices", ["neuron"], id="devices"),
        pytest.param("models", [OLMO_MODEL_ID], id="models"),
        pytest.param("entries", ["A"], id="entries"),
        pytest.param("model_config_shards", [["qwen3_8b"]], id="model-config-shards"),
        pytest.param("test_files", ["test_00_rms_norm.py"], id="test-files"),
        pytest.param("excluded_members", {"omitted": "why"}, id="excluded-members"),
        pytest.param("execution_adapter", "hugging-face", id="execution-adapter"),
    ],
)
def test_target_rejects_kind_and_runner_specific_fields(tmp_path, field_name, value):
    data = target_data()
    data[field_name] = value
    with pytest.raises(
        ValueError,
        match=re.escape(f"target 'target', field {field_name!r}"),
    ):
        read_test_targets(tmp_path, {"target": data})


@pytest.mark.parametrize(
    ("data", "field_name", "reason_pattern"),
    [
        pytest.param(
            target_data(instance="trn2;id"),
            "instance",
            "string over",
            id="instance-shell",
        ),
        pytest.param(target_data(instance="TRN2"), "instance", "string over", id="instance-case"),
        pytest.param(target_data(config="full$(id)"), "config", "string over", id="config-shell"),
        pytest.param(target_data(config=3), "config", "must be a string", id="config-type"),
        pytest.param({"shards": [["A"]]}, "timeout_minutes", "integer", id="missing-timeout"),
        pytest.param(
            target_data(timeout_minutes="60"),
            "timeout_minutes",
            "integer",
            id="timeout-str",
        ),
        pytest.param(
            target_data(timeout_minutes=True),
            "timeout_minutes",
            "integer",
            id="timeout-bool",
        ),
        pytest.param(
            target_data(timeout_minutes=0),
            "timeout_minutes",
            "above 0",
            id="timeout-zero",
        ),
        pytest.param(
            target_data(torch_compile_enabled=1),
            "torch_compile_enabled",
            "boolean",
            id="compile-setting-integer",
        ),
        pytest.param(
            target_data(torch_compile_enabled="true"),
            "torch_compile_enabled",
            "boolean",
            id="compile-setting-string",
        ),
        pytest.param(
            target_data(excluded_models=[PHI3_MODEL_ID]),
            "excluded_models",
            "object",
            id="excluded-type",
        ),
        pytest.param(
            target_data(excluded_models={"owner/$(id)": "why"}),
            "excluded_models",
            "not over",
            id="excluded-shell-character",
        ),
        pytest.param(
            target_data(excluded_models={PHI3_MODEL_ID: " "}),
            "excluded_models",
            "needs a reason",
            id="excluded-reason",
        ),
        pytest.param(
            target_data(excluded_models={"member": "why"}),
            "excluded_models",
            "also in shards",
            id="excluded-in-shards",
        ),
        pytest.param(
            target_data(excluded_models={"owner/model": "why"}, adapter="component-equivalence"),
            "excluded_models",
            "not over",
            id="excluded-component-slash",
        ),
        pytest.param(target_data(xfail=["A"]), "xfail", "object", id="xfail-type"),
        pytest.param(target_data(xfail={"missing": "why"}), "xfail", "not in", id="xfail-unknown"),
        pytest.param(
            target_data(xfail={"member": " "}),
            "xfail",
            "needs a reason",
            id="xfail-reason",
        ),
    ],
)
def test_target_schema_errors_name_the_target_field_and_reason(
    tmp_path, data, field_name, reason_pattern
):
    with pytest.raises(ValueError) as raised:
        read_test_targets(tmp_path, {"target": data})
    message = str(raised.value)
    assert "target 'target'" in message
    assert f"field {field_name!r}" in message
    assert re.search(reason_pattern, message)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param(
            {"shards": [["A"]], "timeout_minutes": 60},
            (
                "targets.json: target 'target', field 'adapter': is required. "
                "Set it to one of hugging-face, titan, component-equivalence."
            ),
            id="missing",
        ),
        pytest.param(
            target_data(adapter="unknown"),
            (
                "targets.json: target 'target', field 'adapter': must be one of "
                "hugging-face, titan, component-equivalence, got 'unknown'."
            ),
            id="unknown",
        ),
    ],
)
def test_target_requires_a_known_adapter(tmp_path, data, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        read_test_targets(tmp_path, {"target": data})


@pytest.mark.parametrize(
    ("shards", "reason_pattern"),
    [
        pytest.param([], "nonempty list", id="empty"),
        pytest.param("not-a-list", "nonempty list", id="string"),
        pytest.param(None, "nonempty list", id="null"),
        pytest.param([[]], "shard 0", id="empty-shard"),
        pytest.param(["member"], "shard 0", id="member-not-list"),
        pytest.param([["member", 3]], "strings", id="member-not-string"),
        pytest.param([["owner/$(id)"]], "not over", id="shell-character"),
        pytest.param([["a/b/c"]], "not over", id="multiple-slashes"),
        pytest.param([["../outside"]], "not over", id="parent-path"),
        pytest.param([["same", "same"]], "repeat", id="duplicate-within-shard"),
        pytest.param([["same"], ["same"]], "repeat", id="duplicate-across-shards"),
        pytest.param([["foo/bar_baz"], ["foo_bar/baz"]], "collide", id="stored-id-collision"),
    ],
)
def test_shards_are_nonempty_safe_unique_member_groups(tmp_path, shards, reason_pattern):
    data = {**target_data(), "shards": shards}
    with pytest.raises(
        ValueError,
        match=rf"field 'shards'.*{reason_pattern}",
    ):
        read_test_targets(tmp_path, {"target": data})


def test_component_members_use_one_path_component(tmp_path):
    data = target_data(((OLMO_MODEL_ID,),), adapter="component-equivalence")
    with pytest.raises(ValueError, match=r"field 'shards'.*not over"):
        read_test_targets(tmp_path, {"target": data})


def test_one_target_shape_accepts_model_entry_and_component_member_names(tmp_path):
    parsed_targets = read_test_targets(
        tmp_path,
        {
            "hub": target_data(((OLMO_MODEL_ID,), (LLAMA_MODEL_ID,)), config="full_dense_decoder"),
            "titan": target_data(
                tuple((entry,) for entry in SAMPLE_TITAN_ENTRY_NAMES), adapter="titan"
            ),
            "component": target_data(DENSE_COMPONENT_SHARDS, adapter="component-equivalence"),
        },
    )
    assert shard_state(parsed_targets["hub"]) == [
        (0, (OLMO_MODEL_ID,)),
        (1, (LLAMA_MODEL_ID,)),
    ]
    assert shard_state(parsed_targets["titan"]) == [
        (index, (entry,)) for index, entry in enumerate(SAMPLE_TITAN_ENTRY_NAMES)
    ]
    assert shard_state(parsed_targets["component"]) == [
        (0, DENSE_COMPONENT_SHARDS[0]),
        (1, DENSE_COMPONENT_SHARDS[1]),
    ]
    for target in parsed_targets.values():
        assert target.torch_compile_enabled is False
        assert not any(
            hasattr(target, field)
            for field in (
                "kind",
                "selector",
                "models",
                "entries",
                "model_config_shards",
                "test_files",
            )
        )


def test_common_defaults_and_member_level_xfail(tmp_path):
    target = read_test_targets(
        tmp_path,
        {
            "target": target_data(
                (("A",), ("B",)),
                torch_compile_enabled=True,
                xfail={"B": "known failure"},
            )
        },
    )["target"]
    assert target.instance == TRAINIUM_INSTANCE
    assert target.adapter == TEST_ADAPTER
    assert target.config == ""
    assert target.torch_compile_enabled is True
    assert target.xfail == {"B": "known failure"}


def test_excluded_models_keep_their_listed_order_and_default_to_none(tmp_path):
    excluded_models = {PHI3_MODEL_ID: "remote code", OLMO_MODEL_ID: "no weights"}
    data = target_data(((LLAMA_MODEL_ID,),), excluded_models=excluded_models)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": {"target": data}}))
    target = targets_module.resolve_target("target", path=str(path))
    assert target.excluded_models == (PHI3_MODEL_ID, OLMO_MODEL_ID)
    assert target.members == (LLAMA_MODEL_ID,)
    assert targets_module.resolve_target(TEST_COMPONENT_EQUIVALENCE_TARGET).excluded_models == ()


def test_shard_order_is_the_only_member_order(tmp_path):
    target = read_test_targets(
        tmp_path,
        {"target": target_data((("C", "A"), ("B",)))},
    )["target"]
    assert target.shards == (
        shard_planner.Shard(0, ("C", "A")),
        shard_planner.Shard(1, ("B",)),
    )
    assert target.members == ("C", "A", "B")


@pytest.mark.parametrize(
    ("shards", "expected_rows"),
    [
        pytest.param(
            tuple((model_id,) for model_id in SAMPLE_HUGGING_FACE_MODEL_IDS),
            [
                {"index": index, "label": model_id}
                for index, model_id in enumerate(SAMPLE_HUGGING_FACE_MODEL_IDS)
            ],
            id="hub-models",
        ),
        pytest.param(
            tuple((entry,) for entry in SAMPLE_TITAN_ENTRY_NAMES),
            [
                {"index": index, "label": entry}
                for index, entry in enumerate(SAMPLE_TITAN_ENTRY_NAMES)
            ],
            id="titan-entries",
        ),
        pytest.param(DENSE_COMPONENT_SHARDS, DENSE_COMPONENT_ROWS, id="component-groups"),
    ],
)
def test_matrix_and_plan_use_the_same_generic_rows(tmp_path, shards, expected_rows):
    target = read_test_targets(tmp_path, {"target": target_data(shards)})["target"]
    assert targets_module.shard_plan(target) == list(target.shards)
    assert targets_module.shard_matrix(target) == expected_rows
    output = targets_module.plan(target, 0, 30)
    assert output == {
        "shards": expected_rows,
        "timeout_minutes": target.timeout_minutes,
        "inner_timeout_seconds": (target.timeout_minutes - 30) * 60,
        "instance": target.instance,
        "relay_artifact": "custom-test-artifacts",
    }
    assert all(set(row) == {"index", "label"} for row in output["shards"])
    assert '"kind"' not in json.dumps(output, sort_keys=True)


def test_group_members_survive_resolve_plan_matrix_and_cli():
    target = targets_module.resolve_target(TEST_COMPONENT_EQUIVALENCE_TARGET)
    assert [shard.members for shard in target.shards] == list(DENSE_COMPONENT_SHARDS)
    assert targets_module.shard_plan(target) == list(target.shards)
    assert targets_module.shard_matrix(target) == DENSE_COMPONENT_ROWS
    assert targets_module.plan(target, 0, 30) == {
        "shards": DENSE_COMPONENT_ROWS,
        "timeout_minutes": 330,
        "inner_timeout_seconds": 18000,
        "instance": TRAINIUM_INSTANCE,
        "relay_artifact": "custom-test-artifacts",
    }

    command = run_plan_command(TEST_COMPONENT_EQUIVALENCE_TARGET, 0)
    assert (command.returncode, command.stderr) == (0, "")
    output_values = dict(line.split("=", 1) for line in command.stdout.splitlines())
    assert output_values == {
        "shards": json.dumps(DENSE_COMPONENT_ROWS, separators=(",", ":")),
        "timeout_minutes": "330",
        "inner_timeout_seconds": "18000",
        "instance": TRAINIUM_INSTANCE,
        "relay_artifact": "custom-test-artifacts",
    }


def test_the_plan_contains_all_workflow_fields_and_one_relay_artifact(tmp_path):
    target = read_test_targets(
        tmp_path,
        {"test_hf": target_data(tuple((model_id,) for model_id in SAMPLE_HUGGING_FACE_MODEL_IDS))},
    )["test_hf"]
    assert targets_module.plan(target, 0, 30) == {
        "shards": targets_module.shard_matrix(target),
        "timeout_minutes": 60,
        "inner_timeout_seconds": 1800,
        "instance": TRAINIUM_INSTANCE,
        "relay_artifact": "custom-test-artifacts",
    }
    overridden = targets_module.plan(target, 120, 30)
    assert (overridden["timeout_minutes"], overridden["inner_timeout_seconds"]) == (120, 5400)


def test_the_plan_rejects_invalid_limits(tmp_path):
    target = read_test_targets(
        tmp_path,
        {"test_titan": target_data(tuple((entry,) for entry in SAMPLE_TITAN_ENTRY_NAMES))},
    )["test_titan"]
    for timeout, buffer in ((30, 30), (20, 30), (-5, 30), (0, 70), (0, -1)):
        with pytest.raises(ValueError, match="copy_buffer_minutes"):
            targets_module.plan(target, timeout, buffer)


def test_plan_cli_prints_an_error_for_a_refused_target():
    command = run_plan_command("missing-target", 0)
    assert command.returncode == 1
    assert command.stdout == ""
    assert command.stderr.startswith("::error::")
    assert "no target named" in command.stderr
