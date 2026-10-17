#!/usr/bin/env python3
"""Install files with descriptor-based path safety.

The code opens all output descriptors without following symbolic links.
All output writes use these descriptors.
The code never writes a file through a symbolic link.
The code never writes outside the directory inode that it opens at the start.
The code checks the requested path before and after the commit.
If a check finds a changed path, the code rolls back and reports an error.
POSIX path APIs cannot detect a change that is undone between two checks.
They also cannot detect a change after the final check.
In both cases, files stay only in the original directory inode.
The commit uses one rename for each file. Other processes can see a partial set.
If the process stops during the commit, hidden backup files can remain.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from redact import redact_text

DirectoryIdentity = tuple[int, int]
_TEST_HOOK: Callable[[str, int], None] | None = None
_OUTPUT_PATHS: dict[int, Path] = {}


def _directory_flags() -> int:
    try:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    except AttributeError as exc:
        raise OSError("directory descriptor safety flags are unavailable") from exc


def _identity(info: os.stat_result) -> DirectoryIdentity:
    return info.st_dev, info.st_ino


def _check_identity(fd: int, identity: DirectoryIdentity) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or _identity(info) != identity:
        raise OSError("output directory identity changed")


def open_output_dir(path: str | os.PathLike[str]) -> tuple[int, DirectoryIdentity]:
    """Open and remember a real output directory through its anchored path chain."""
    anchored = _AnchoredOutput(Path(path))
    try:
        fd = os.dup(anchored.fd)
        identity = _identity(os.fstat(fd))
        _OUTPUT_PATHS[fd] = anchored.path
        return fd, identity
    finally:
        anchored.close()


def _validate_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("output names must be plain file names")
    return name


def _stat_name(fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _check_final(fd: int, name: str) -> bool:
    info = _stat_name(fd, name)
    if info is None:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise OSError(f"{name} is not a regular file")
    return True


def check_output_names(
    fd: int,
    identity: DirectoryIdentity,
    names: Iterable[str],
    *,
    require_absent: Iterable[str] = (),
) -> None:
    """Check output names against the opened directory without changing files."""
    _check_identity(fd, identity)
    absent = {_validate_name(name) for name in require_absent}
    for raw_name in names:
        name = _validate_name(raw_name)
        exists = _check_final(fd, name)
        if exists and name in absent:
            raise FileExistsError(f"destination already exists: {name}")


def _new_name(fd: int, final_name: str, purpose: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(100):
        name = f".{final_name}.{purpose}.{secrets.token_hex(12)}"
        try:
            return os.open(name, flags, 0o600, dir_fd=fd), name
        except FileExistsError:
            continue
    raise OSError("cannot allocate an exclusive transaction file")


def _write_file(fd: int, final_name: str, data: bytes) -> str:
    output_fd, name = _new_name(fd, final_name, "new")
    try:
        view = memoryview(data)
        while view:
            written = os.write(output_fd, view)
            if written <= 0:
                raise OSError("short write while staging output")
            view = view[written:]
        os.fsync(output_fd)
        os.close(output_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(output_fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=fd)
        raise
    return name


def _reserve_backup(fd: int, final_name: str) -> str:
    backup_fd, backup_name = _new_name(fd, final_name, "backup")
    try:
        os.close(backup_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(backup_fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(backup_name, dir_fd=fd)
        raise
    return backup_name


def _unlink(fd: int, name: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(name, dir_fd=fd)


def _call_hook(stage: str, fd: int) -> None:
    if _TEST_HOOK is not None:
        _TEST_HOOK(stage, fd)


def _revalidate_requested_output(
    fd: int,
    identity: DirectoryIdentity,
    output_path: Path | None,
) -> None:
    """Reopen the requested path chain and match it to the held descriptor."""
    _check_identity(fd, identity)
    requested = output_path or _OUTPUT_PATHS.get(fd)
    if requested is None:
        raise OSError("requested output path is unavailable for revalidation")
    try:
        reopened = _AnchoredOutput(requested)
    except OSError:
        raise OSError("requested output-directory path changed") from None
    try:
        reopened.revalidate()
        if reopened.identity != identity:
            raise OSError("requested output directory identity changed")
    finally:
        reopened.close()


def _restore_backups(
    fd: int,
    backups: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Try every backup restore and retry each failed rename once."""
    errors: list[str] = []
    unrecovered: list[str] = []
    for name, backup in reversed(list(backups.items())):
        last_error: BaseException | None = None
        for _attempt in range(2):
            try:
                os.replace(backup, name, src_dir_fd=fd, dst_dir_fd=fd)
            except BaseException as exc:
                last_error = exc
                continue
            backups.pop(name, None)
            break
        else:
            unrecovered.append(name)
            errors.append(f"{name}: {last_error}")
    return errors, unrecovered


def install_files(
    fd: int,
    identity: DirectoryIdentity,
    files: Mapping[str, bytes],
    *,
    require_absent: Iterable[str] = (),
    output_path: Path | None = None,
) -> None:
    """Install all files, or roll back to the prior set on an in-process failure."""
    normalized: dict[str, bytes] = {}
    for raw_name, data in files.items():
        name = _validate_name(raw_name)
        if name in normalized:
            raise ValueError(f"duplicate output name: {name}")
        if not isinstance(data, bytes):
            raise TypeError(f"output data must be bytes: {name}")
        normalized[name] = data
    if not normalized:
        raise ValueError("at least one output file is required")

    absent = {_validate_name(name) for name in require_absent}
    if not absent.issubset(normalized):
        raise ValueError("require_absent names must be installed files")

    names = list(normalized)
    staged: dict[str, str] = {}
    backups: dict[str, str] = {}
    installed: list[str] = []
    committed = False
    first_commit_rename = True
    try:
        _check_identity(fd, identity)
        for name, data in normalized.items():
            staged[name] = _write_file(fd, name, data)
        check_output_names(fd, identity, names, require_absent=absent)
        _call_hook("after_preflight", fd)
        _revalidate_requested_output(fd, identity, output_path)

        for name in names:
            exists = _check_final(fd, name)
            if exists and name in absent:
                raise FileExistsError(f"destination already exists: {name}")
            if exists:
                backup = _reserve_backup(fd, name)
                try:
                    if first_commit_rename:
                        _revalidate_requested_output(fd, identity, output_path)
                        first_commit_rename = False
                    os.replace(name, backup, src_dir_fd=fd, dst_dir_fd=fd)
                except BaseException:
                    _unlink(fd, backup)
                    raise
                backups[name] = backup

        fail_install = os.environ.get("SAFE_OUTPUT_TEST_FAIL_INSTALL", "")
        for index, name in enumerate(names, start=1):
            _check_identity(fd, identity)
            if _stat_name(fd, name) is not None:
                raise OSError(f"destination changed during install: {name}")
            if fail_install and int(fail_install) == index:
                raise OSError("injected late install failure")
            if first_commit_rename:
                _revalidate_requested_output(fd, identity, output_path)
                first_commit_rename = False
            os.replace(staged[name], name, src_dir_fd=fd, dst_dir_fd=fd)
            installed.append(name)
            staged.pop(name)
        os.fsync(fd)
        _revalidate_requested_output(fd, identity, output_path)
        committed = True
    except BaseException as original:
        for name in reversed(installed):
            _unlink(fd, name)
        rollback_errors, unrecovered = _restore_backups(fd, backups)
        try:
            os.fsync(fd)
        except BaseException as exc:
            rollback_errors.append(f"directory fsync: {exc}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            message = f"{original}; rollback errors: {details}"
            if unrecovered:
                message += f"; unrecovered files: {', '.join(unrecovered)}"
            raise OSError(message) from original
        raise
    finally:
        for name in staged.values():
            _unlink(fd, name)
        if committed:
            for name in backups.values():
                _unlink(fd, name)


def _read_staging(path: Path) -> dict[str, bytes]:
    fd, identity = open_output_dir(path)
    try:
        _check_identity(fd, identity)
        files: dict[str, bytes] = {}
        for name in sorted(os.listdir(fd)):
            _validate_name(name)
            info = _stat_name(fd, name)
            if info is None or not stat.S_ISREG(info.st_mode):
                raise OSError(f"staging entry is not a regular file: {name}")
            source_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                    raise OSError(f"staging entry is not a regular file: {name}")
                chunks = []
                while chunk := os.read(source_fd, 1024 * 1024):
                    chunks.append(chunk)
                files[name] = b"".join(chunks)
            finally:
                os.close(source_fd)
        return files
    finally:
        os.close(fd)


class _AnchoredOutput:
    """Hold every directory in an absolute output path open by descriptor."""

    def __init__(self, path: Path):
        self.path = Path(os.path.abspath(path))
        self._parts = self.path.parts[1:]
        self._fds: list[int] = []
        self._identities: list[DirectoryIdentity] = []
        flags = _directory_flags()
        current = os.open("/", flags)
        self._fds.append(current)
        self._identities.append(_identity(os.fstat(current)))
        try:
            for index, part in enumerate(self._parts):
                is_final = index == len(self._parts) - 1
                try:
                    next_fd = os.open(part, flags, dir_fd=current)
                except FileNotFoundError:
                    if not is_final:
                        raise OSError("an output-directory ancestor does not exist") from None
                    os.mkdir(part, dir_fd=current)
                    next_fd = os.open(part, flags, dir_fd=current)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        info = None
                        with contextlib.suppress(OSError):
                            info = os.stat(part, dir_fd=current, follow_symlinks=False)
                        if is_final and info is not None and stat.S_ISLNK(info.st_mode):
                            raise OSError("output directory must not be a symlink") from None
                        label = "output directory" if is_final else "output-directory ancestor"
                        raise OSError(f"{label} must be a real directory") from None
                    raise
                current = next_fd
                self._fds.append(current)
                self._identities.append(_identity(os.fstat(current)))
        except BaseException:
            self.close()
            raise
        if not self._parts:
            self.fd = self._fds[0]
        else:
            self.fd = self._fds[-1]
        self.identity = self._identities[-1]

    def revalidate(self) -> None:
        """Fail if reopening the absolute path reaches a different directory chain."""
        flags = _directory_flags()
        current = os.open("/", flags)
        opened = [current]
        try:
            if _identity(os.fstat(current)) != self._identities[0]:
                raise OSError("output-directory root identity changed")
            for index, part in enumerate(self._parts, start=1):
                current = os.open(part, flags, dir_fd=current)
                opened.append(current)
                if _identity(os.fstat(current)) != self._identities[index]:
                    raise OSError("output-directory ancestor identity changed")
        except OSError as exc:
            if str(exc).startswith("output-directory"):
                raise
            raise OSError("output-directory path changed before install") from None
        finally:
            for descriptor in reversed(opened):
                with contextlib.suppress(OSError):
                    os.close(descriptor)

    def close(self) -> None:
        for descriptor in reversed(self._fds):
            with contextlib.suppress(OSError):
                os.close(descriptor)
        self._fds.clear()


def _redact_json(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, dict):
        return {
            redact_text(key) if isinstance(key, str) else key: _redact_json(item)
            for key, item in value.items()
        }
    return value


def _redact_files(files: Mapping[str, bytes]) -> dict[str, bytes]:
    redacted: dict[str, bytes] = {}
    for name, data in files.items():
        if name == "report.json":
            try:
                report = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot redact report.json: {exc}") from None
            redacted[name] = (
                json.dumps(_redact_json(report), indent=2, ensure_ascii=True) + "\n"
            ).encode()
        else:
            redacted[name] = redact_text(data.decode("utf-8", errors="replace")).encode()
    return redacted


def _apply_ancestor_swap_test_seam(output: Path) -> None:
    ancestor_text = os.environ.get("SAFE_OUTPUT_TEST_SWAP_ANCESTOR", "")
    if not ancestor_text:
        return
    target_text = os.environ.get("SAFE_OUTPUT_TEST_SWAP_TARGET", "")
    if not target_text:
        raise OSError("ancestor-swap test seam requires a target")
    ancestor = Path(ancestor_text).absolute()
    target = Path(target_text).absolute()
    try:
        output.relative_to(ancestor)
    except ValueError:
        raise OSError("ancestor-swap test seam is outside the output path") from None
    if not target.is_dir():
        raise OSError("ancestor-swap test seam requires a directory target")
    displaced = ancestor.with_name(ancestor.name + ".displaced")
    ancestor.rename(displaced)
    ancestor.symlink_to(target, target_is_directory=True)


def _run_transaction(args: argparse.Namespace) -> int:
    anchored = _AnchoredOutput(args.dir)
    stage = Path(tempfile.mkdtemp(prefix="pytorch-upstream-safe-output."))
    try:
        names = [_validate_name(name) for name in args.name]
        if len(set(names)) != len(names):
            raise ValueError("output names must be unique")
        check_output_names(anchored.fd, anchored.identity, names)
        command = list(args.program)
        if command[:1] == ["--"]:
            command.pop(0)
        if not command:
            raise ValueError("run requires a command after '--'")
        child_env = os.environ.copy()
        child_env["SAFE_OUTPUT_STAGE"] = str(stage)
        child_env.pop("SAFE_OUTPUT_TEST_SWAP_ANCESTOR", None)
        child_env.pop("SAFE_OUTPUT_TEST_SWAP_TARGET", None)
        completed = subprocess.run(command, env=child_env, check=False)
        staged = _read_staging(stage)
        if not staged:
            return completed.returncode
        if set(staged) != set(names):
            missing = sorted(set(names) - set(staged))
            extra = sorted(set(staged) - set(names))
            raise OSError(f"staged artifact set differs: missing={missing} extra={extra}")
        _apply_ancestor_swap_test_seam(anchored.path)
        anchored.revalidate()
        if args.redact_all:
            staged = _redact_files(staged)
        anchored.revalidate()
        install_files(
            anchored.fd,
            anchored.identity,
            staged,
            output_path=anchored.path,
        )
        return completed.returncode
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        anchored.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a command and install its complete artifact set")
    run.add_argument("--dir", required=True, type=Path, help="output directory")
    run.add_argument("--name", action="append", required=True, help="required output file name")
    run.add_argument("--redact-all", action="store_true", help="redact every staged text artifact")
    run.add_argument(
        "--error-exit",
        type=int,
        default=4,
        help="status for safe-output validation, I/O, or report errors",
    )
    run.add_argument("program", nargs=argparse.REMAINDER, help="command after '--'")
    install = subparsers.add_parser("install", help="install every regular file from staging")
    install.add_argument("--dir", required=True, type=Path, help="output directory")
    install.add_argument("--from-dir", required=True, type=Path, help="staging directory")
    check = subparsers.add_parser("check", help="validate output names without writing")
    check.add_argument("--dir", required=True, type=Path, help="output directory")
    check.add_argument("--name", action="append", required=True, help="expected output file name")
    args = parser.parse_args()

    try:
        if args.command == "run":
            return _run_transaction(args)
        fd, identity = open_output_dir(args.dir)
        try:
            if args.command == "install":
                install_files(fd, identity, _read_staging(args.from_dir))
            else:
                check_output_names(fd, identity, args.name)
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: safe output failed: {exc}", file=sys.stderr)
        return args.error_exit if args.command == "run" else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
