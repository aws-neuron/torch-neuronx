#!/usr/bin/env python3
"""Configure a local torch-neuronx artifact capture directory."""

import argparse
import glob
import io
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _entry_stat(directory_fd: int, name: str) -> os.stat_result | None:
    """Read one directory entry without following a symbolic link."""
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _create_temp(directory_fd: int, label: str) -> tuple[str, int]:
    """Create an exclusive temporary file inside a pinned directory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _attempt in range(128):
        name = f".{label}.tmp-{secrets.token_hex(16)}"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
    raise OSError("Could not create an exclusive temporary file.")


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("The temporary file write did not make progress.")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _remove_matching(directory_fd: int, name: str, expected: os.stat_result) -> None:
    current = _entry_stat(directory_fd, name)
    if current is not None and _same_node(current, expected):
        os.unlink(name, dir_fd=directory_fd)


def _configure(artdir: Path) -> None:
    """Create and pin real capture directories before setting the environment."""
    try:
        artdir.mkdir(parents=True, exist_ok=True)
        root_stat = os.stat(artdir, follow_symlinks=False)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"The artifact directory is not a real directory: {artdir}")
        root_fd = os.open(artdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"The artifact directory is not a real directory: {artdir}") from error

    temporary: tuple[str, os.stat_result] | None = None
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError(f"The artifact directory is not a real directory: {artdir}")

        for name in ("compiler_dump", "neff_cache"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            except OSError as error:
                raise ValueError(
                    f"The capture directory is not a real directory: {artdir / name}"
                ) from error
            child_stat = _entry_stat(root_fd, name)
            if child_stat is None or not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"The capture directory is not a real directory: {artdir / name}")

        final_name = "launch_blocking.txt"
        expected_final = _entry_stat(root_fd, final_name)
        if expected_final is not None and not stat.S_ISREG(expected_final.st_mode):
            raise ValueError(
                f"The launch blocking output is not a regular file: {artdir / final_name}"
            )

        temporary_name, temporary_fd = _create_temp(root_fd, final_name)
        temporary_stat = os.fstat(temporary_fd)
        temporary = (temporary_name, temporary_stat)
        try:
            value = os.environ.get("NEURON_LAUNCH_BLOCKING", "<unset>") + "\n"
            _write_all(temporary_fd, value.encode("ascii"))
        finally:
            os.close(temporary_fd)

        current_final = _entry_stat(root_fd, final_name)
        if current_final is not None and not stat.S_ISREG(current_final.st_mode):
            raise ValueError(
                f"The launch blocking output is not a regular file: {artdir / final_name}"
            )
        if (expected_final is None) != (current_final is None) or (
            expected_final is not None
            and current_final is not None
            and not _same_node(expected_final, current_final)
        ):
            raise OSError(f"The launch blocking output changed: {artdir / final_name}")

        os.replace(
            temporary_name,
            final_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        temporary = None
        os.fsync(root_fd)
    except Exception:
        if temporary is not None:
            _remove_matching(root_fd, *temporary)
        raise
    finally:
        os.close(root_fd)

    os.environ["TORCH_NEURONX_DEBUG_DIR"] = str(artdir / "compiler_dump")
    os.environ["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] = "1"
    os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(artdir / "neff_cache")


_ENV_ARTDIR = os.environ.get("ARTDIR", "")
if __name__ != "__main__" and not _ENV_ARTDIR:
    raise RuntimeError("ARTDIR is not set. Export ARTDIR before you import artifact_dump.")
ARTDIR = Path(os.path.abspath(_ENV_ARTDIR or "."))
DEBUG_DIR = ARTDIR / "compiler_dump"
NEFF_CACHE_DIR = ARTDIR / "neff_cache"
if __name__ != "__main__":
    _configure(ARTDIR)


def _safe_name(name: str) -> str:
    """Return a portable file name component."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    if not result:
        raise ValueError("The tensor name is invalid. Use a visible file name component.")
    return result


def _normalized_names(values: Mapping[str, Any], kind: str) -> list[tuple[str, Any]]:
    """Normalize names and reject ambiguous destination paths before writing."""
    normalized: list[tuple[str, Any]] = []
    owners: dict[str, str] = {}
    for name, value in values.items():
        safe_name = _safe_name(name)
        if safe_name in owners:
            raise ValueError(
                f"The {kind} tensor names {owners[safe_name]!r} and {name!r} "
                f"both normalize to {safe_name!r}. Use distinct names."
            )
        owners[safe_name] = name
        normalized.append((safe_name, value))
    return normalized


def _open_destination(path: Path) -> int:
    """Create and pin a real destination directory."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"The save destination is not a real directory: {path}") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"The save destination is not a real directory: {path}")
    return descriptor


def _unused_name(directory_fd: int, label: str) -> str:
    for _attempt in range(128):
        name = f".{label}.backup-{secrets.token_hex(16)}"
        if _entry_stat(directory_fd, name) is None:
            return name
    raise OSError("Could not allocate a tensor backup name.")


def _before_save_io_install() -> None:
    """Provide a deterministic test hook after final-node preflight."""


def save_io(
    artdir: str | Path,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Save named tensors as one directory-relative transaction."""
    import torch

    normalized_inputs = _normalized_names(inputs or {}, "input")
    normalized_outputs = _normalized_names(outputs or {}, "output")
    save_values = [
        *((f"in_{name}.pt", value) for name, value in normalized_inputs),
        *((f"out_{name}.pt", value) for name, value in normalized_outputs),
    ]

    destination = Path(artdir)
    directory_fd = _open_destination(destination)
    temporaries: dict[str, tuple[str, os.stat_result]] = {}
    backups: dict[str, tuple[str, os.stat_result]] = {}
    installed: dict[str, os.stat_result] = {}
    try:
        serialized: list[tuple[str, bytes]] = []
        for name, value in save_values:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            serialized.append((name, buffer.getvalue()))

        for final_name, data in serialized:
            temporary_name, temporary_fd = _create_temp(directory_fd, final_name)
            temporary_stat = os.fstat(temporary_fd)
            temporaries[final_name] = (temporary_name, temporary_stat)
            try:
                _write_all(temporary_fd, data)
            finally:
                os.close(temporary_fd)

        expected_finals: dict[str, os.stat_result | None] = {}
        for final_name, _data in serialized:
            current = _entry_stat(directory_fd, final_name)
            if current is not None and not stat.S_ISREG(current.st_mode):
                raise ValueError(
                    f"The tensor output is not a regular file: {destination / final_name}"
                )
            expected_finals[final_name] = current

        _before_save_io_install()

        for final_name, expected in expected_finals.items():
            current = _entry_stat(directory_fd, final_name)
            if current is not None and not stat.S_ISREG(current.st_mode):
                raise ValueError(
                    f"The tensor output is not a regular file: {destination / final_name}"
                )
            if (expected is None) != (current is None) or (
                expected is not None and current is not None and not _same_node(expected, current)
            ):
                raise OSError(
                    f"The tensor output changed during publication: {destination / final_name}"
                )

        for final_name, expected in expected_finals.items():
            if expected is None:
                continue
            backup_name = _unused_name(directory_fd, final_name)
            os.rename(
                final_name,
                backup_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            backups[final_name] = (backup_name, expected)
            backup_stat = _entry_stat(directory_fd, backup_name)
            if (
                backup_stat is None
                or not stat.S_ISREG(backup_stat.st_mode)
                or not _same_node(expected, backup_stat)
            ):
                raise OSError(
                    f"The tensor output changed during backup: {destination / final_name}"
                )
            backups[final_name] = (backup_name, backup_stat)

        for final_name, _data in serialized:
            temporary_name, temporary_stat = temporaries[final_name]
            os.replace(
                temporary_name,
                final_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            installed[final_name] = temporary_stat
        os.fsync(directory_fd)
    except Exception:
        for final_name, (backup_name, _backup_stat) in reversed(backups.items()):
            if _entry_stat(directory_fd, backup_name) is not None:
                os.replace(
                    backup_name,
                    final_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
        for final_name, installed_stat in installed.items():
            if final_name not in backups:
                _remove_matching(directory_fd, final_name, installed_stat)
        for temporary_name, temporary_stat in temporaries.values():
            _remove_matching(directory_fd, temporary_name, temporary_stat)
        os.fsync(directory_fd)
        raise
    else:
        for backup_name, backup_stat in backups.values():
            _remove_matching(directory_fd, backup_name, backup_stat)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def list_modules(artdir: str | Path | None = None) -> list[str]:
    """Return dumped module.mlir paths without decoding their bytecode content."""
    root = Path(artdir).resolve() if artdir else ARTDIR
    pattern = str(root / "compiler_dump" / "neuronx-cc" / "*" / "module.mlir")
    return sorted(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artdir", default=str(ARTDIR), help="Artifact root directory.")
    parser.add_argument("--list", action="store_true", help="List dumped module.mlir files.")
    args = parser.parse_args()
    if args.list:
        modules = list_modules(args.artdir)
        if not modules:
            print("No module.mlir files exist. Run the capture before this command.")
            return 1
        for path in modules:
            print(path)
    else:
        print(f"Artifact directory: {Path(args.artdir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
