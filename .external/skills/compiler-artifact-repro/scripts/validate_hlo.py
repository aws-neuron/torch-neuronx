#!/usr/bin/env python3
"""Validate one HLO against an expected standalone compiler symptom."""

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPRODUCED = 0
NOT_REPRODUCED = 1
INVALID_ENVIRONMENT = 2
INPUT_DTYPES = (
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int8",
    "bool",
)


@dataclass
class Runtime:
    """Hold modules that require a Neuron host."""

    torch: Any
    torch_neuronx: Any
    compiler: Any


@dataclass(frozen=True)
class InputSpec:
    """Hold a validated input path and optional cast."""

    path: str
    dtype: str | None


def _configure_artdir(value: str) -> None:
    """Configure confined compiler outputs before the Neuron runtime import."""
    artdir = Path(os.path.abspath(value))
    os.environ["ARTDIR"] = str(artdir)
    runpy.run_path(str(Path(__file__).with_name("artifact_dump.py")))


def _load_runtime() -> Runtime:
    import torch

    import torch_neuronx
    from torch_neuronx import _C

    return Runtime(torch=torch, torch_neuronx=torch_neuronx, compiler=_C)


def _json_safe(value: Any) -> Any:
    """Replace nonfinite floats so JSON output stays standards compliant."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _emit(state: str, message: str, **details: Any) -> None:
    payload = _json_safe({"state": state, "message": message, **details})
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def _hlo_evidence(path: str, hlo: bytes) -> dict[str, str]:
    return {"hlo_path": path, "hlo_sha256": hashlib.sha256(hlo).hexdigest()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mlir", help="Selected module.mlir file.")
    parser.add_argument(
        "--artdir",
        default=os.environ.get("ARTDIR"),
        help="Artifact root directory. This option or ARTDIR is required.",
    )
    parser.add_argument("--input", action="append", default=[], help="FILE[:TYPE] in input order.")
    parser.add_argument("--out-shape", action="append", default=[], help="Shape such as 8x16.")
    parser.add_argument(
        "--out-dtype", action="append", default=[], help="Data type such as float32."
    )
    parser.add_argument(
        "--ref", action="append", default=[], help="Reference tensor for one output."
    )
    parser.add_argument("--atol", type=float, help="Required absolute tolerance in numerical mode.")
    parser.add_argument("--rtol", type=float, help="Required relative tolerance in numerical mode.")
    parser.add_argument("--expect-error-regex", help="Expected compile error regular expression.")
    parser.add_argument(
        "--key",
        default="validate_hlo",
        help="Compiler cache-key prefix. The HLO SHA-256 digest is always appended.",
    )
    return parser


def _parse_input_spec(value: str) -> InputSpec:
    """Parse FILE[:TYPE] exactly once and reject unknown explicit types."""
    if ":" not in value:
        return InputSpec(value, None)
    path, dtype = value.rsplit(":", 1)
    if not path:
        raise ValueError("An input file path is empty.")
    if dtype not in INPUT_DTYPES:
        raise ValueError(
            f"The input data type is unsupported: {dtype}. Use one of: {', '.join(INPUT_DTYPES)}"
        )
    return InputSpec(path, dtype)


def _validate_args(args: argparse.Namespace) -> str | None:
    if not args.artdir:
        return "The artifact directory is missing. Pass --artdir or set ARTDIR."
    if not os.path.isfile(args.mlir):
        return f"The HLO file does not exist: {args.mlir}"
    if os.path.getsize(args.mlir) == 0:
        return "The HLO file is empty. Select a valid module.mlir file."
    if not args.key:
        return "The compiler cache-key prefix is empty."
    try:
        args.parsed_inputs = [_parse_input_spec(value) for value in args.input]
    except ValueError as error:
        return str(error)
    if args.expect_error_regex:
        if (
            args.input
            or args.out_shape
            or args.out_dtype
            or args.ref
            or args.atol is not None
            or args.rtol is not None
        ):
            return "Compile-error mode has numerical arguments. Remove the numerical arguments."
        try:
            re.compile(args.expect_error_regex)
        except re.error as error:
            return f"The expected error expression is invalid: {error}"
        return None
    if args.atol is None or args.rtol is None:
        return "Numerical mode requires both --atol and --rtol."
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol):
        return "A tolerance is not finite. Use finite tolerances."
    if args.atol < 0 or args.rtol < 0:
        return "A tolerance is negative. Use nonnegative tolerances."
    counts = (len(args.out_shape), len(args.out_dtype), len(args.ref))
    if not args.parsed_inputs:
        return "Numerical mode has no input. Add at least one --input option."
    if not counts[0] or len(set(counts)) != 1:
        return "Output argument counts differ. Add one shape, data type, and reference per output."
    try:
        args.parsed_shapes = [_parse_shape(shape) for shape in args.out_shape]
    except ValueError as error:
        return str(error)
    unknown_dtypes = [dtype for dtype in args.out_dtype if dtype not in INPUT_DTYPES]
    if unknown_dtypes:
        return (
            "An output data type is unsupported: "
            + ", ".join(unknown_dtypes)
            + f". Use one of: {', '.join(INPUT_DTYPES)}"
        )
    for file_spec in args.parsed_inputs:
        if not os.path.isfile(file_spec.path):
            return f"An input file does not exist: {file_spec.path}"
    for path in args.ref:
        if not os.path.isfile(path):
            return f"A reference file does not exist: {path}"
    return None


def _dtype_map(torch: Any) -> dict[str, Any]:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def _load_tensor(path: str, torch: Any) -> Any:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise ValueError(f"The tensor file does not contain one tensor: {path}")
    return value


def _load_input(spec: InputSpec, device: str, torch: Any, dtypes: dict[str, Any]) -> Any:
    tensor = _load_tensor(spec.path, torch)
    if spec.dtype:
        tensor = tensor.to(dtypes[spec.dtype])
    return tensor.contiguous().to(device)


def _initialize(runtime: Runtime) -> str:
    warmup = runtime.torch.ones(4, device="neuron")
    runtime.torch_neuronx.synchronize()
    del warmup
    return f"neuron:{runtime.torch.neuron.current_device()}"


def _compile_error_mode(
    args: argparse.Namespace, runtime: Runtime, hlo: bytes, evidence: dict[str, str]
) -> int:
    try:
        runtime.compiler.compile_graph(args.key, hlo, False)
        runtime.torch_neuronx.synchronize()
    except Exception as error:
        text = str(error)
        if re.search(args.expect_error_regex, text):
            _emit(
                "REPRODUCED",
                "The standalone compile produced the expected error.",
                error=text[:500],
                **evidence,
            )
            return REPRODUCED
        _emit(
            "NOT_REPRODUCED",
            "The standalone compile produced a different error.",
            error=text[:500],
            **evidence,
        )
        return NOT_REPRODUCED
    _emit(
        "NOT_REPRODUCED",
        "The standalone compile completed without the expected error.",
        **evidence,
    )
    return NOT_REPRODUCED


def _parse_shape(text: str) -> tuple[int, ...]:
    normalized = text.strip().lower()
    if normalized in {"[]", "scalar"}:
        return ()
    if not normalized or not re.fullmatch(r"\d+(?:x\d+)*", normalized):
        raise ValueError(f"The output shape is invalid: {text}")
    return tuple(int(value) for value in normalized.split("x"))


def _compare_tensors(
    actual: Any, expected: Any, torch: Any, atol: float, rtol: float
) -> tuple[int, float | None, int, int]:
    if actual.is_floating_point():
        close = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
        actual_nan = torch.isnan(actual)
        expected_nan = torch.isnan(expected)
        actual_inf = torch.isinf(actual)
        expected_inf = torch.isinf(expected)
        one_sided_nan = torch.logical_xor(actual_nan, expected_nan)
        one_sided_inf = torch.logical_xor(actual_inf, expected_inf)
        opposite_inf_sign = (
            actual_inf
            & expected_inf
            & torch.logical_xor(torch.signbit(actual), torch.signbit(expected))
        )
        inf_mismatch = one_sided_inf | opposite_inf_sign
        mismatch = ~close | one_sided_nan | inf_mismatch
        finite_pairs = torch.isfinite(actual) & torch.isfinite(expected)
        max_abs_diff = None
        if finite_pairs.any().item():
            candidate = float((actual[finite_pairs] - expected[finite_pairs]).abs().max().item())
            if math.isfinite(candidate):
                max_abs_diff = candidate
        return (
            int(mismatch.sum().item()),
            max_abs_diff,
            int(one_sided_nan.sum().item()),
            int(inf_mismatch.sum().item()),
        )
    mismatch = actual != expected
    return int(mismatch.sum().item()), None, 0, 0


def _numeric_mode(
    args: argparse.Namespace,
    runtime: Runtime,
    hlo: bytes,
    device: str,
    evidence: dict[str, str],
) -> int:
    torch = runtime.torch
    dtypes = _dtype_map(torch)
    try:
        cache_key = runtime.compiler.compile_graph(args.key, hlo, False)
        runtime.torch_neuronx.synchronize()
        inputs = [_load_input(spec, device, torch, dtypes) for spec in args.parsed_inputs]
        outputs = [
            torch.empty(shape, dtype=dtypes[dtype], device=device)
            for shape, dtype in zip(args.parsed_shapes, args.out_dtype, strict=True)
        ]
        runtime.compiler.execute_compiled_graph(args.key, cache_key, inputs, outputs, False)
        runtime.torch_neuronx.synchronize()
    except Exception as error:
        _emit(
            "INVALID_ENVIRONMENT",
            "The standalone numerical compile or execution failed.",
            error=str(error)[:500],
            **evidence,
        )
        return INVALID_ENVIRONMENT

    mismatch_counts = []
    max_abs_diffs = []
    nan_mismatch_counts = []
    inf_mismatch_counts = []
    for index, (output, reference_path, dtype_name) in enumerate(
        zip(outputs, args.ref, args.out_dtype, strict=True)
    ):
        try:
            actual = output.cpu()
            expected = _load_tensor(reference_path, torch)
            if tuple(actual.shape) != tuple(expected.shape):
                raise ValueError(f"Output {index} has a different reference shape.")
            if actual.dtype != dtypes[dtype_name] or expected.dtype != dtypes[dtype_name]:
                raise ValueError(f"Output {index} has a different reference data type.")
            mismatch_count, max_abs_diff, nan_count, inf_count = _compare_tensors(
                actual, expected, torch, args.atol, args.rtol
            )
            mismatch_counts.append(mismatch_count)
            max_abs_diffs.append(max_abs_diff)
            nan_mismatch_counts.append(nan_count)
            inf_mismatch_counts.append(inf_count)
        except Exception as error:
            _emit(
                "INVALID_ENVIRONMENT",
                "Reference validation failed.",
                error=str(error)[:500],
                **evidence,
            )
            return INVALID_ENVIRONMENT

    details = {
        "atol": args.atol,
        "rtol": args.rtol,
        "mismatch_counts": mismatch_counts,
        "nan_mismatch_counts": nan_mismatch_counts,
        "inf_mismatch_counts": inf_mismatch_counts,
        "max_abs_diffs": max_abs_diffs,
        **evidence,
    }
    if any(mismatch_counts):
        _emit("REPRODUCED", "Standalone values exceeded the stated tolerances.", **details)
        return REPRODUCED
    _emit("NOT_REPRODUCED", "Standalone values stayed within the stated tolerances.", **details)
    return NOT_REPRODUCED


def _main(
    argv: Sequence[str] | None,
    runtime_loader: Callable[[], Runtime],
) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    problem = _validate_args(args)
    if problem:
        _emit("INVALID_ARGUMENTS", problem)
        return INVALID_ENVIRONMENT
    try:
        with open(args.mlir, "rb") as hlo_file:
            hlo = hlo_file.read()
        evidence = _hlo_evidence(args.mlir, hlo)
        args.key = f"{args.key}-{evidence['hlo_sha256']}"
    except Exception as error:
        _emit("INVALID_ENVIRONMENT", "The HLO setup failed.", error=str(error)[:500])
        return INVALID_ENVIRONMENT
    try:
        _configure_artdir(args.artdir)
        runtime = runtime_loader()
        device = _initialize(runtime)
    except Exception as error:
        _emit(
            "INVALID_ENVIRONMENT",
            "The artifact or Neuron setup failed.",
            error=str(error)[:500],
            **evidence,
        )
        return INVALID_ENVIRONMENT
    if args.expect_error_regex:
        return _compile_error_mode(args, runtime, hlo, evidence)
    return _numeric_mode(args, runtime, hlo, device, evidence)


def main(
    argv: Sequence[str] | None = None,
    runtime_loader: Callable[[], Runtime] = _load_runtime,
) -> int:
    """Run validation and convert unexpected failures into one JSON result."""
    try:
        return _main(argv, runtime_loader)
    except Exception as error:
        _emit("UNEXPECTED_ERROR", "Validation failed unexpectedly.", error=str(error)[:500])
        return INVALID_ENVIRONMENT


if __name__ == "__main__":
    sys.exit(main())
