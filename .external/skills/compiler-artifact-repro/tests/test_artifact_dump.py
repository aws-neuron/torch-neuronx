import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "artifact_dump.py"
MLIR_BYTECODE = b"ML\xefR\x00\xff\x10"


def load_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTDIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(f"artifact_dump_test_{id(tmp_path)}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configures_directories_records_launch_blocking_and_lists_bytecode_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NEURON_LAUNCH_BLOCKING", "inherited-value")
    helper = load_helper(tmp_path, monkeypatch)
    assert os.environ["TORCH_NEURONX_DEBUG_DIR"] == str(tmp_path / "compiler_dump")
    assert os.environ["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] == "1"
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neff_cache")
    assert (tmp_path / "compiler_dump").is_dir()
    assert (tmp_path / "neff_cache").is_dir()
    assert (tmp_path / "launch_blocking.txt").read_text(encoding="ascii") == "inherited-value\n"
    assert os.environ["NEURON_LAUNCH_BLOCKING"] == "inherited-value"

    hlo = tmp_path / "compiler_dump" / "neuronx-cc" / "case" / "module.mlir"
    hlo.parent.mkdir(parents=True)
    hlo.write_bytes(MLIR_BYTECODE)
    assert helper.list_modules(tmp_path) == [str(hlo)]


def test_records_unset_launch_blocking_without_setting_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURON_LAUNCH_BLOCKING", raising=False)
    load_helper(tmp_path, monkeypatch)
    assert (tmp_path / "launch_blocking.txt").read_text(encoding="ascii") == "<unset>\n"
    assert "NEURON_LAUNCH_BLOCKING" not in os.environ


def test_replaces_preexisting_neff_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", str(tmp_path / "wrong-cache"))
    load_helper(tmp_path, monkeypatch)
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neff_cache")


def test_import_rejects_launch_blocking_symlink_without_changing_target(tmp_path, monkeypatch):
    artdir = tmp_path / "artifacts"
    artdir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="ascii")
    (artdir / "launch_blocking.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="launch blocking output is not a regular file"):
        load_helper(artdir, monkeypatch)

    assert (artdir / "launch_blocking.txt").is_symlink()
    assert outside.read_text(encoding="ascii") == "outside\n"


def test_import_rejects_compiler_dump_symlink(tmp_path, monkeypatch):
    artdir = tmp_path / "artifacts"
    artdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artdir / "compiler_dump").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="capture directory is not a real directory"):
        load_helper(artdir, monkeypatch)

    assert list(outside.iterdir()) == []


def test_import_rejects_artdir_symlink(tmp_path, monkeypatch):
    real_artdir = tmp_path / "real-artifacts"
    real_artdir.mkdir()
    artdir = tmp_path / "artifacts"
    artdir.symlink_to(real_artdir, target_is_directory=True)

    with pytest.raises(ValueError, match="artifact directory is not a real directory"):
        load_helper(artdir, monkeypatch)

    assert list(real_artdir.iterdir()) == []


def test_save_io_rejects_normalized_name_collision_before_writing(tmp_path, monkeypatch):
    helper = load_helper(tmp_path / "config", monkeypatch)
    destination = tmp_path / "io"
    with pytest.raises(ValueError, match="both normalize"):
        helper.save_io(
            destination,
            inputs={"input/a": torch.tensor([1]), "input_a": torch.tensor([2])},
        )
    assert not destination.exists()


def test_invalid_tensor_name_fails(tmp_path, monkeypatch):
    helper = load_helper(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="tensor name is invalid"):
        helper._safe_name("///")


def test_cli_reports_missing_modules(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--artdir", str(tmp_path), "--list"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "No module.mlir files exist" in result.stdout


def test_import_without_artdir_fails(monkeypatch):
    monkeypatch.delenv("ARTDIR", raising=False)
    spec = importlib.util.spec_from_file_location("artifact_dump_no_artdir", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="ARTDIR is not set"):
        spec.loader.exec_module(module)


def test_cli_creates_no_files(tmp_path):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "ARTDIR"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--artdir", str(tmp_path / "missing"), "--list"],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert list(workdir.iterdir()) == []
    assert not (tmp_path / "missing").exists()


def test_save_io_rejects_symlink_destination(tmp_path, monkeypatch):
    helper = load_helper(tmp_path / "config", monkeypatch)
    real_destination = tmp_path / "real-io"
    real_destination.mkdir()
    destination = tmp_path / "io-link"
    destination.symlink_to(real_destination, target_is_directory=True)

    with pytest.raises(ValueError, match="not a real directory"):
        helper.save_io(destination, inputs={"value": torch.tensor([1])})

    assert list(real_destination.iterdir()) == []


@pytest.mark.parametrize("node_type", ["directory", "symlink"])
def test_save_io_rejects_non_regular_output_before_any_write(tmp_path, monkeypatch, node_type):
    helper = load_helper(tmp_path / "config", monkeypatch)
    destination = tmp_path / "io"
    destination.mkdir()
    unsafe_output = destination / "out_bad.pt"
    if node_type == "directory":
        unsafe_output.mkdir()
    else:
        target = tmp_path / "target.pt"
        target.write_bytes(b"target")
        unsafe_output.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        helper.save_io(
            destination,
            inputs={"good": torch.tensor([1])},
            outputs={"bad": torch.tensor([2])},
        )

    assert not (destination / "in_good.pt").exists()


def test_save_io_rejects_target_swapped_to_symlink_after_preflight(tmp_path, monkeypatch):
    helper = load_helper(tmp_path / "config", monkeypatch)
    destination = tmp_path / "io"
    destination.mkdir()
    output = destination / "out_value.pt"
    output.write_bytes(b"original")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")

    def swap_target():
        output.unlink()
        output.symlink_to(outside)

    monkeypatch.setattr(helper, "_before_save_io_install", swap_target)
    with pytest.raises(ValueError, match="not a regular file"):
        helper.save_io(destination, outputs={"value": torch.tensor([1])})

    assert output.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert not any(".tmp-" in path.name for path in destination.iterdir())


def test_save_io_rolls_back_all_outputs_when_second_install_fails(tmp_path, monkeypatch):
    helper = load_helper(tmp_path / "config", monkeypatch)
    destination = tmp_path / "io"
    destination.mkdir()
    first = destination / "in_first.pt"
    second = destination / "out_second.pt"
    first.write_bytes(b"original first")
    second.write_bytes(b"original second")
    real_replace = helper.os.replace
    install_count = 0

    def fail_second_install(source, target, *args, **kwargs):
        nonlocal install_count
        if (
            str(source).startswith(".")
            and ".tmp-" in str(source)
            and target
            in {
                first.name,
                second.name,
            }
        ):
            install_count += 1
            if install_count == 2:
                raise OSError("injected second tensor install failure")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(helper.os, "replace", fail_second_install)
    with pytest.raises(OSError, match="injected second tensor"):
        helper.save_io(
            destination,
            inputs={"first": torch.tensor([1])},
            outputs={"second": torch.tensor([2])},
        )

    assert first.read_bytes() == b"original first"
    assert second.read_bytes() == b"original second"
    assert sorted(path.name for path in destination.iterdir()) == [first.name, second.name]


def test_save_io_second_tensor_serialization_failure_preserves_originals(tmp_path, monkeypatch):
    helper = load_helper(tmp_path / "config", monkeypatch)
    destination = tmp_path / "io"
    destination.mkdir()
    first = destination / "in_first.pt"
    second = destination / "out_second.pt"
    first.write_bytes(b"original first")
    second.write_bytes(b"original second")
    real_save = torch.save
    save_count = 0

    def fail_second_save(value, target, *args, **kwargs):
        nonlocal save_count
        save_count += 1
        if save_count == 2:
            raise OSError("injected second tensor serialization failure")
        return real_save(value, target, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_second_save)
    with pytest.raises(OSError, match="injected second tensor serialization"):
        helper.save_io(
            destination,
            inputs={"first": torch.tensor([1])},
            outputs={"second": torch.tensor([2])},
        )

    assert first.read_bytes() == b"original first"
    assert second.read_bytes() == b"original second"
    assert sorted(path.name for path in destination.iterdir()) == [first.name, second.name]
