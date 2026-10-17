import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_hlo.py"
SPEC = importlib.util.spec_from_file_location("validate_hlo_test", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class FakeTorchNeuronx:
    @staticmethod
    def synchronize():
        return None


class FakeCompiler:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.executed_cache_key = None
        self.compiled_names = []

    def compile_graph(self, name, *_args):
        self.compiled_names.append(name)
        if self.error:
            raise RuntimeError(self.error)
        return "compiled-key"

    def execute_compiled_graph(self, _name, cache_key, _inputs, outputs, _collectives):
        self.executed_cache_key = cache_key
        outputs[0].copy_(self.output)


def runtime(output=None, error=None):
    return validator.Runtime(torch, FakeTorchNeuronx(), FakeCompiler(output, error))


def make_hlo(tmp_path):
    hlo = tmp_path / "module.mlir"
    hlo.write_text("module {}", encoding="utf-8")
    return hlo


def run_numeric(tmp_path, monkeypatch, actual, expected, shape, dtype, atol="0", rtol="0"):
    hlo = make_hlo(tmp_path)
    input_path = tmp_path / "input.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(torch.tensor([1.0]), input_path)
    torch.save(expected, reference_path)
    fake_runtime = runtime(actual)
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [
            str(hlo),
            "--input",
            str(input_path),
            "--out-shape",
            shape,
            "--out-dtype",
            dtype,
            "--ref",
            str(reference_path),
            "--atol",
            atol,
            "--rtol",
            rtol,
        ],
        runtime_loader=lambda: fake_runtime,
    )
    assert fake_runtime.compiler.executed_cache_key == "compiled-key"
    return code


def test_expected_compile_error_is_reproduced(tmp_path, capsys, monkeypatch):
    hlo = make_hlo(tmp_path)
    fake_runtime = runtime(error="COMPILER_123 expected failure")
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [str(hlo), "--expect-error-regex", "COMPILER_123"],
        runtime_loader=lambda: fake_runtime,
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "REPRODUCED"
    assert result["hlo_path"] == str(hlo)
    assert result["hlo_sha256"] == hashlib.sha256(hlo.read_bytes()).hexdigest()


def test_clean_compile_is_not_reproduced(tmp_path, capsys, monkeypatch):
    hlo = make_hlo(tmp_path)
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [str(hlo), "--expect-error-regex", "COMPILER_123"],
        runtime_loader=lambda: runtime(),
    )
    assert code == 1
    assert '"state": "NOT_REPRODUCED"' in capsys.readouterr().out


def test_different_error_is_not_false_success(tmp_path, capsys, monkeypatch):
    hlo = make_hlo(tmp_path)
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [str(hlo), "--expect-error-regex", "COMPILER_123"],
        runtime_loader=lambda: runtime(error="OTHER_456"),
    )
    assert code == 1
    assert "different error" in capsys.readouterr().out


def test_numerical_mode_requires_explicit_tolerances(tmp_path, capsys):
    hlo = make_hlo(tmp_path)
    value = tmp_path / "value.pt"
    torch.save(torch.tensor([1.0]), value)
    code = validator.main(
        [
            str(hlo),
            "--input",
            str(value),
            "--out-shape",
            "1",
            "--out-dtype",
            "float32",
            "--ref",
            str(value),
        ]
    )
    assert code == 2
    assert "requires both --atol and --rtol" in capsys.readouterr().out


def test_tolerance_boundary_is_close(tmp_path, monkeypatch):
    code = run_numeric(
        tmp_path,
        monkeypatch,
        torch.tensor([1.5], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        "1",
        "float64",
        atol="0.5",
    )
    assert code == 1


def test_one_sided_nan_reproduces(tmp_path, monkeypatch):
    code = run_numeric(
        tmp_path,
        monkeypatch,
        torch.tensor([float("nan")]),
        torch.tensor([1.0]),
        "1",
        "float32",
    )
    assert code == 0


def test_signed_infinity_reproduces(tmp_path, monkeypatch):
    code = run_numeric(
        tmp_path,
        monkeypatch,
        torch.tensor([float("-inf")]),
        torch.tensor([float("inf")]),
        "1",
        "float32",
    )
    assert code == 0


def test_int64_values_are_compared_exactly(tmp_path, monkeypatch):
    code = run_numeric(
        tmp_path,
        monkeypatch,
        torch.tensor([2**60 + 1], dtype=torch.int64),
        torch.tensor([2**60], dtype=torch.int64),
        "1",
        "int64",
    )
    assert code == 0


@pytest.mark.parametrize("shape, expected", [("[]", ()), ("scalar", ()), ("0x3", (0, 3))])
def test_shape_accepts_scalars_and_zero_size_dimensions(shape, expected):
    assert validator._parse_shape(shape) == expected


@pytest.mark.parametrize("shape", ["-1x3", "1xx3", "3x", "[]x1", "one"])
def test_shape_rejects_negative_or_malformed_dimensions(shape):
    with pytest.raises(ValueError, match="output shape is invalid"):
        validator._parse_shape(shape)


def test_non_tensor_reference_is_rejected(tmp_path, monkeypatch, capsys):
    hlo = make_hlo(tmp_path)
    input_path = tmp_path / "input.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(torch.tensor([1.0]), input_path)
    torch.save({"value": torch.tensor([1.0])}, reference_path)
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [
            str(hlo),
            "--input",
            str(input_path),
            "--out-shape",
            "1",
            "--out-dtype",
            "float32",
            "--ref",
            str(reference_path),
            "--atol",
            "0",
            "--rtol",
            "0",
        ],
        runtime_loader=lambda: runtime(torch.tensor([1.0])),
    )
    assert code == 2
    result = json.loads(capsys.readouterr().out)
    assert "does not contain one tensor" in result["error"]


def test_invalid_input_returns_two(tmp_path, capsys):
    code = validator.main(
        [str(tmp_path / "missing.mlir"), "--expect-error-regex", "COMPILER_123"],
        runtime_loader=lambda: runtime(error="COMPILER_123"),
    )
    assert code == 2
    assert '"state": "INVALID_ARGUMENTS"' in capsys.readouterr().out


class CachingCompiler:
    def __init__(self):
        self.cache = {}
        self.names = []

    def compile_graph(self, name, hlo, _collectives):
        self.names.append(name)
        return self.cache.setdefault(name, hashlib.sha256(hlo).hexdigest())


def test_hlo_digest_is_appended_to_default_and_user_key(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    compiler = CachingCompiler()
    fake_runtime = validator.Runtime(torch, FakeTorchNeuronx(), compiler)
    first = tmp_path / "first.mlir"
    second = tmp_path / "second.mlir"
    first.write_bytes(b"first HLO")
    second.write_bytes(b"second HLO")

    assert (
        validator.main(
            [str(first), "--expect-error-regex", "never"],
            runtime_loader=lambda: fake_runtime,
        )
        == 1
    )
    assert (
        validator.main(
            [str(second), "--expect-error-regex", "never", "--key", "custom"],
            runtime_loader=lambda: fake_runtime,
        )
        == 1
    )

    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
    assert compiler.names == [f"validate_hlo-{first_digest}", f"custom-{second_digest}"]
    assert len(compiler.cache) == 2


def test_unknown_input_type_is_rejected_before_runtime(tmp_path, capsys):
    hlo = make_hlo(tmp_path)
    input_path = tmp_path / "input.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(torch.tensor([1.0]), input_path)
    torch.save(torch.tensor([1.0]), reference_path)
    runtime_called = False

    def fail_runtime():
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not load")

    code = validator.main(
        [
            str(hlo),
            "--input",
            f"{input_path}:unknown",
            "--out-shape",
            "1",
            "--out-dtype",
            "float32",
            "--ref",
            str(reference_path),
            "--atol",
            "0",
            "--rtol",
            "0",
        ],
        runtime_loader=fail_runtime,
    )
    assert code == 2
    assert runtime_called is False
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "INVALID_ARGUMENTS"
    assert "unsupported" in result["message"]


def test_nonfinite_comparison_metrics_are_strict_json_nulls(tmp_path, monkeypatch, capsys):
    hlo = make_hlo(tmp_path)
    input_path = tmp_path / "input.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(torch.tensor([1.0]), input_path)
    torch.save(torch.tensor([float("inf"), 1.0]), reference_path)
    fake_runtime = runtime(torch.tensor([float("-inf"), float("nan")]))
    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")

    code = validator.main(
        [
            str(hlo),
            "--input",
            str(input_path),
            "--out-shape",
            "2",
            "--out-dtype",
            "float32",
            "--ref",
            str(reference_path),
            "--atol",
            "0",
            "--rtol",
            "0",
        ],
        runtime_loader=lambda: fake_runtime,
    )
    output = capsys.readouterr().out
    result = json.loads(output, parse_constant=lambda value: pytest.fail(value))
    assert code == 0
    assert result["max_abs_diffs"] == [None]
    assert result["nan_mismatch_counts"] == [1]
    assert result["inf_mismatch_counts"] == [1]
    assert "NaN" not in output and "Infinity" not in output


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--out-shape", "1xx2", "output shape is invalid"),
        ("--out-dtype", "complex64", "output data type is unsupported"),
    ],
)
def test_invalid_output_specs_are_rejected_before_runtime_and_compile(
    tmp_path, capsys, monkeypatch, option, value, message
):
    hlo = make_hlo(tmp_path)
    input_path = tmp_path / "input.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(torch.tensor([1.0]), input_path)
    torch.save(torch.tensor([1.0]), reference_path)
    compiler = FakeCompiler(torch.tensor([1.0]))
    fake_runtime = validator.Runtime(torch, FakeTorchNeuronx(), compiler)
    runtime_calls = 0

    def load_runtime():
        nonlocal runtime_calls
        runtime_calls += 1
        return fake_runtime

    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    arguments = [
        str(hlo),
        "--input",
        str(input_path),
        "--out-shape",
        "1",
        "--out-dtype",
        "float32",
        "--ref",
        str(reference_path),
        "--atol",
        "0",
        "--rtol",
        "0",
    ]
    arguments[arguments.index(option) + 1] = value

    code = validator.main(arguments, runtime_loader=load_runtime)

    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result["state"] == "INVALID_ARGUMENTS"
    assert message in result["message"].lower()
    assert runtime_calls == 0
    assert compiler.compiled_names == []


def test_script_path_points_to_validate_hlo():
    spec = importlib.util.spec_from_file_location("validate_hlo_reference_test", SCRIPT)
    assert spec is not None
    assert spec.origin == str(SCRIPT)


@pytest.fixture(autouse=True)
def required_artdir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTDIR", str(tmp_path / "validator-artifacts"))


def test_artdir_confines_runtime_outputs_before_runtime_load(tmp_path, monkeypatch):
    artdir = tmp_path / "artifacts"
    hlo = make_hlo(tmp_path)
    names = (
        "TORCH_NEURONX_DEBUG_DIR",
        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS",
        "TORCH_NEURONX_NEFF_CACHE_DIR",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    captured = {}

    def load_runtime():
        captured.update({name: validator.os.environ.get(name) for name in names})
        return runtime()

    monkeypatch.setattr(validator, "_initialize", lambda _runtime: "cpu")
    code = validator.main(
        [
            str(hlo),
            "--artdir",
            str(artdir),
            "--expect-error-regex",
            "never",
        ],
        runtime_loader=load_runtime,
    )

    assert code == 1
    assert captured == {
        "TORCH_NEURONX_DEBUG_DIR": str(artdir / "compiler_dump"),
        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(artdir / "neff_cache"),
    }
    assert (artdir / "compiler_dump").is_dir()
    assert (artdir / "neff_cache").is_dir()


def test_missing_artdir_is_rejected_before_runtime(tmp_path, monkeypatch, capsys):
    hlo = make_hlo(tmp_path)
    monkeypatch.delenv("ARTDIR")
    runtime_called = False

    def load_runtime():
        nonlocal runtime_called
        runtime_called = True
        return runtime()

    code = validator.main(
        [str(hlo), "--expect-error-regex", "never"],
        runtime_loader=load_runtime,
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result["state"] == "INVALID_ARGUMENTS"
    assert "--artdir" in result["message"]
    assert runtime_called is False


def test_artdir_symlink_is_rejected_before_runtime(tmp_path, monkeypatch, capsys):
    hlo = make_hlo(tmp_path)
    real_artdir = tmp_path / "real-artifacts"
    real_artdir.mkdir()
    artdir = tmp_path / "artifacts"
    artdir.symlink_to(real_artdir, target_is_directory=True)
    runtime_called = False

    def load_runtime():
        nonlocal runtime_called
        runtime_called = True
        return runtime()

    code = validator.main(
        [
            str(hlo),
            "--artdir",
            str(artdir),
            "--expect-error-regex",
            "never",
        ],
        runtime_loader=load_runtime,
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result["state"] == "INVALID_ENVIRONMENT"
    assert "not a real directory" in result["error"]
    assert runtime_called is False
    assert list(real_artdir.iterdir()) == []
