import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_op_autograd.py"
SPEC = importlib.util.spec_from_file_location("probe_op_autograd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Info:
    supports_autograd = True
    aten_name = "relu"
    aten_backward_name = "threshold_backward"


def install_fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def test_normalize_exact_operator_identity():
    assert MODULE.normalize_op("relu") == "aten::relu"
    assert MODULE.normalize_op("aten.relu") == "aten::relu"
    assert MODULE.normalize_op("aten::relu") == "aten::relu"
    assert MODULE.normalize_op("relu.out") == "aten::relu.out"


def test_operator_name_uses_schema_name():
    value = types.SimpleNamespace(_schema=types.SimpleNamespace(name="aten::relu"))
    assert MODULE.operator_name(value) == "aten::relu"

    overloaded = types.SimpleNamespace(
        _schema=types.SimpleNamespace(name="aten::add", overload_name="Tensor")
    )
    assert MODULE.operator_name(overloaded) == "aten::add.Tensor"


def test_operator_name_normalizes_packet_text():
    class Packet:
        _schema = None

        def __str__(self):
            return "aten.mul.Tensor"

    assert MODULE.operator_name(Packet()) == "aten::mul.Tensor"
    assert MODULE.operator_name(object()) is None


def test_exact_opinfo_identity_does_not_match_partial_name(monkeypatch):
    forward = types.SimpleNamespace(aten_name="relu", aten_backward_name="relu_backward")
    backward = types.SimpleNamespace(aten_name="relu_backward", aten_backward_name=None)
    module = types.ModuleType("torch.testing._internal.common_methods_invocations")
    module.op_db = [forward, backward]
    monkeypatch.setitem(sys.modules, "torch.testing._internal.common_methods_invocations", module)
    assert MODULE.find_opinfo("aten::relu") is forward
    assert MODULE.find_opinfo("aten::relu_backward") is backward
    assert MODULE.find_opinfo("aten::reluish.out") is None


def test_opinfo_base_name_is_used_for_an_exact_overload_trace(monkeypatch):
    forward = types.SimpleNamespace(aten_name="add", variant_test_name="")
    module = types.ModuleType("torch.testing._internal.common_methods_invocations")
    module.op_db = [forward]
    monkeypatch.setitem(sys.modules, "torch.testing._internal.common_methods_invocations", module)
    assert MODULE.find_opinfo("aten::add.Tensor") is forward


def test_ordinary_forward_primitive_is_not_dedicated():
    info = types.SimpleNamespace(aten_name="mul", aten_backward_name="mul")
    assert MODULE.dedicated_backward_name(info, "aten::mul", ["aten::mul"]) is None


def test_exact_distinct_opinfo_backward_is_dedicated():
    info = types.SimpleNamespace(aten_name="relu", aten_backward_name="threshold_backward")
    assert (
        MODULE.dedicated_backward_name(info, "aten::relu", ["aten::threshold_backward"])
        == "aten::threshold_backward"
    )
    assert MODULE.dedicated_backward_name(info, "aten::relu", ["aten::mul"]) is None


def test_traced_backward_name_is_dedicated_without_opinfo_metadata():
    info = types.SimpleNamespace(aten_name="relu", aten_backward_name=None)
    assert MODULE.dedicated_backward_names(
        info,
        "aten::relu",
        ["aten::detach", "aten::threshold_backward"],
    ) == ["aten::threshold_backward"]


def test_non_differentiable_op(monkeypatch):
    install_fake_torch(monkeypatch)
    info = Info()
    info.supports_autograd = False
    monkeypatch.setattr(MODULE, "find_opinfo", lambda _op: info)
    result = MODULE.probe("aten::logical_not")
    assert result["differentiable"] is False
    assert result["backward_kind"] == "none"
    assert "no autograd support" in result["note"]


def test_derived_backward_from_ordinary_primitives(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(MODULE, "find_opinfo", lambda _op: Info())
    monkeypatch.setattr(MODULE, "trace_backward_ops", lambda _info, _op: ([], ["aten::mul"]))
    result = MODULE.probe("aten::relu")
    assert result["backward_kind"] == "derived"
    assert result["backward_ops"] == []
    assert result["backward_dispatch_all"] == ["aten::mul"]


def test_dedicated_backward_uses_exact_metadata(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(MODULE, "find_opinfo", lambda _op: Info())
    monkeypatch.setattr(
        MODULE,
        "trace_backward_ops",
        lambda _info, _op: (["aten::threshold_backward"], ["aten::threshold_backward"]),
    )
    result = MODULE.probe("relu")
    assert result["backward_kind"] == "dedicated_op"
    assert result["backward_ops"] == ["aten::threshold_backward"]


def test_missing_opinfo_and_trace_failure_are_unknown(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(MODULE, "find_opinfo", lambda _op: None)
    result = MODULE.probe("aten::not_an_op")
    assert result["backward_kind"] == "unknown"
    assert "No exact matching" in result["note"]

    monkeypatch.setattr(MODULE, "find_opinfo", lambda _op: Info())
    monkeypatch.setattr(
        MODULE,
        "trace_backward_ops",
        lambda _info, _op: (_ for _ in ()).throw(RuntimeError("sample failed")),
    )
    result = MODULE.probe("relu")
    assert result["backward_kind"] == "unknown"
    assert "trace failed" in result["note"]


def import_real_opinfo_or_skip():
    try:
        import torch  # noqa: F401
        from torch.testing._internal.common_methods_invocations import op_db  # noqa: F401
    except Exception as error:
        pytest.skip(f"PyTorch OpInfo dependencies are unavailable: {error}")


@pytest.mark.parametrize("op", ["mul", "sin"])
def test_real_torch_primitive_backward_is_derived(op):
    import_real_opinfo_or_skip()
    result = MODULE.probe(op)
    assert result["backward_kind"] == "derived", result
    assert result["backward_ops"] == []


def test_real_torch_overload_uses_matching_opinfo_sample():
    import_real_opinfo_or_skip()
    result = MODULE.probe("aten::add.Tensor")
    assert result["backward_kind"] == "derived", result
    assert result["op"] == "aten::add.Tensor"


@pytest.mark.parametrize("op", ["relu", "log_softmax"])
def test_real_torch_known_backward_is_dedicated_when_available(op):
    import_real_opinfo_or_skip()
    result = MODULE.probe(op)
    if result["backward_kind"] == "unknown":
        pytest.skip(result["note"])
    assert result["backward_kind"] == "dedicated_op", result
    assert result["backward_ops"]


def test_main_success_writes_named_evidence(monkeypatch, tmp_path):
    output = tmp_path / "probe.json"
    monkeypatch.setattr(
        MODULE,
        "probe",
        lambda _op: {
            "op": "aten::relu",
            "differentiable": True,
            "backward_kind": "derived",
            "backward_ops": [],
        },
    )
    assert MODULE.main(["--op", "relu", "--out", str(output)]) == 0
    assert '"backward_kind": "derived"' in output.read_text(encoding="utf-8")


def test_cli_help(capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "probe_op_autograd.py"
    assert script == SCRIPT
    with pytest.raises(SystemExit) as result:
        MODULE.main(["--help"])
    assert result.value.code == 0
    assert "Probe whether" in capsys.readouterr().out


def test_main_rejects_empty_operator_name(capsys):
    assert MODULE.main(["--op", "   "]) == 2
    assert "nonempty string" in capsys.readouterr().err


@pytest.mark.parametrize("unsafe_kind", ["directory", "symlink"])
def test_main_rejects_nonregular_output(monkeypatch, tmp_path, unsafe_kind, capsys):
    monkeypatch.setattr(
        MODULE,
        "probe",
        lambda _op: {
            "op": "aten::relu",
            "differentiable": True,
            "backward_kind": "derived",
            "backward_ops": [],
        },
    )
    output = tmp_path / "unsafe-output"
    if unsafe_kind == "directory":
        output.mkdir()
    else:
        target = tmp_path / "regular-target"
        target.write_text("keep", encoding="utf-8")
        output.symlink_to(target)
    assert MODULE.main(["--op", "relu", "--out", str(output)]) == 2
    assert "not a regular file" in capsys.readouterr().err


def test_main_rejects_parent_swapped_for_symlink_after_preflight(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        MODULE,
        "probe",
        lambda _op: {
            "op": "aten::relu",
            "differentiable": True,
            "backward_kind": "derived",
            "backward_ops": [],
        },
    )
    output_parent = tmp_path / "output"
    moved_parent = tmp_path / "moved-output"
    outside = tmp_path / "outside"
    output_parent.mkdir()
    outside.mkdir()

    def swap_parent(_output):
        output_parent.rename(moved_parent)
        output_parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(MODULE, "_after_output_preflight", swap_parent)
    output = output_parent / "probe.json"
    assert MODULE.main(["--op", "relu", "--out", str(output)]) == 2
    assert "Cannot write the autograd result" in capsys.readouterr().err
    assert not (outside / "probe.json").exists()
    assert not (moved_parent / "probe.json").exists()


def test_probe_op_autograd_script_help_subprocess():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
