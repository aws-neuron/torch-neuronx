#!/usr/bin/env python3
"""Probe whether one ATen operator uses a dedicated backward operator."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any


def _after_output_preflight(_output: Path) -> None:
    """Provide a test hook at the path-validation boundary."""


def atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    try:
        mode = output.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(mode):
            raise ValueError("The output path exists and is not a regular file.")
    if not output.parent.is_dir():
        raise ValueError(f"The output parent directory does not exist: {output.parent}")
    _after_output_preflight(output)
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_name: str | None = None
    try:
        while temporary_name is None:
            candidate = f".{output.name}.{os.urandom(8).hex()}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            mode = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(mode):
                raise ValueError("The output path exists and is not a regular file.")
        os.replace(
            temporary_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


def normalize_op(op: str) -> str:
    if not isinstance(op, str) or not op.strip():
        raise ValueError("An operator name must be a nonempty string.")
    value = op.strip().replace("aten.", "aten::", 1)
    return value if value.startswith("aten::") else f"aten::{value}"


def find_opinfo(op: str) -> Any | None:
    from torch.testing._internal.common_methods_invocations import op_db

    wanted = normalize_op(op)
    fallback: list[Any] = []
    wanted_base = wanted.split(".", 1)[0]
    for info in op_db:
        aten_name = getattr(info, "aten_name", None)
        if not aten_name:
            continue
        candidate = normalize_op(str(aten_name))
        if candidate == wanted:
            return info
        if wanted != wanted_base and candidate == wanted_base:
            fallback.append(info)
    return min(
        fallback, key=lambda info: bool(getattr(info, "variant_test_name", "")), default=None
    )


def operator_name(value: Any) -> str | None:
    schema = getattr(value, "_schema", None)
    name = getattr(schema, "name", None)
    if isinstance(name, str) and name:
        overload = getattr(schema, "overload_name", None)
        if isinstance(overload, str) and overload:
            name = f"{name}.{overload}"
        return normalize_op(name)
    text = str(value)
    if text.startswith("aten."):
        return normalize_op(text.removeprefix("aten."))
    return None


def dedicated_backward_name(info: Any, op: str, all_ops: list[str]) -> str | None:
    """Return only the exact distinct OpInfo backward operator seen in the trace."""
    raw = getattr(info, "aten_backward_name", None)
    if not isinstance(raw, str) or not raw:
        return None
    backward = normalize_op(raw)
    forward_raw = getattr(info, "aten_name", None)
    forward = normalize_op(str(forward_raw)) if forward_raw else normalize_op(op)
    if backward == forward:
        return None
    return backward if backward in all_ops else None


def dedicated_backward_names(info: Any, op: str, all_ops: list[str]) -> list[str]:
    """Return exact traced operators that implement a dedicated backward."""
    names = {
        name
        for name in all_ops
        if name != normalize_op(getattr(info, "aten_name", None) or op)
        and "backward" in name.split("::", 1)[-1]
    }
    metadata_name = dedicated_backward_name(info, op, all_ops)
    if metadata_name:
        names.add(metadata_name)
    return sorted(names)


def trace_backward_ops(info: Any, op: str) -> tuple[list[str], list[str]]:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    seen: list[str] = []

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = operator_name(func)
            if name:
                seen.append(name)
            return func(*args, **(kwargs or {}))

    sample = next(info.sample_inputs("cpu", torch.float32, requires_grad=True), None)
    if sample is None:
        raise RuntimeError("The OpInfo entry has no differentiable CPU sample.")
    seen.clear()
    with Capture():
        output = info.op(sample.input, *sample.args, **sample.kwargs)
    wanted = normalize_op(op)
    forward_ops = sorted(set(seen))
    if "." in wanted.split("::", 1)[-1] and wanted not in forward_ops:
        raise RuntimeError(f"The OpInfo sample did not dispatch the requested overload {wanted}.")
    outputs = [
        value
        for value in tree_leaves(output)
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    input_values = [
        value
        for value in tree_leaves((sample.input, sample.args, sample.kwargs))
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    inputs = list({id(value): value for value in input_values}.values())
    if not outputs or not inputs:
        raise RuntimeError("The OpInfo sample has no differentiable tensor input and output.")

    seen.clear()
    with Capture():
        torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=[torch.ones_like(value) for value in outputs],
            allow_unused=True,
        )
    all_ops = sorted(set(seen))
    return dedicated_backward_names(info, op, all_ops), all_ops


def unknown_result(op: str, note: str) -> dict[str, Any]:
    return {
        "op": normalize_op(op),
        "differentiable": None,
        "backward_kind": "unknown",
        "backward_ops": [],
        "backward_dispatch_all": [],
        "method": "OpInfo and a CPU backward dispatch trace",
        "note": note,
    }


def probe(op: str) -> dict[str, Any]:
    normalized = normalize_op(op)
    try:
        import torch  # noqa: F401
    except Exception as error:
        return unknown_result(normalized, f"PyTorch import failed: {error}")
    try:
        info = find_opinfo(normalized)
    except Exception as error:
        return unknown_result(normalized, f"OpInfo import failed: {error}")
    if info is None:
        return unknown_result(
            normalized, "No exact matching OpInfo entry exists. Inspect PyTorch source."
        )

    supports_autograd = bool(getattr(info, "supports_autograd", False))
    result: dict[str, Any] = {
        "op": normalized,
        "differentiable": supports_autograd,
        "backward_kind": "none",
        "backward_ops": [],
        "backward_dispatch_all": [],
        "method": "OpInfo and a CPU backward dispatch trace",
        "note": "",
    }
    if not supports_autograd:
        result["note"] = "OpInfo reports no autograd support."
        return result
    try:
        dedicated, all_ops = trace_backward_ops(info, normalized)
    except Exception as error:
        return unknown_result(normalized, f"The CPU backward trace failed: {error}")
    result["backward_dispatch_all"] = all_ops
    if dedicated:
        result["backward_kind"] = "dedicated_op"
        result["backward_ops"] = dedicated
        result["note"] = "Lower and test each exact OpInfo backward operator."
    else:
        result["backward_kind"] = "derived"
        result["note"] = "The backward uses ordinary differentiable primitives."
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, help="An operator such as aten::relu.")
    parser.add_argument("--out", help="Write the JSON result to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = probe(args.op)
    except (TypeError, ValueError) as error:
        print(f"Invalid operator name: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        try:
            atomic_write_text(args.out, text)
        except (OSError, ValueError) as error:
            print(f"Cannot write the autograd result: {error}", file=sys.stderr)
            return 2
    print(text, end="")
    if result["backward_kind"] == "unknown":
        print("The autograd result is unknown. Inspect PyTorch source.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
