"""
StableHLO Module I/O Parser
Extracts input/output shapes and dtypes from torch_mlir StableHLO modules.
"""

import hashlib
import io
from dataclasses import dataclass

import torch
from torch_mlir._mlir_libs._mlir import ir


def compute_cache_key(stablehlo_mlir) -> str:
    """
    Generate base cache key from StableHLO graph.

    Args:
        stablehlo_mlir: Compiled StableHLO MLIR module

    Returns:
        String cache key for C++ consumption

    Example:
        >>> stablehlo_mlir = convert_fx_to_stablehlo(gm, example_inputs)
        >>> cache_key = compute_cache_key(stablehlo_mlir)
    """
    try:
        io_specs = parse_module_io(stablehlo_mlir)

        # Get StableHLO bytecode content for hashing
        bytecode_buffer = io.BytesIO()
        stablehlo_mlir.operation.write_bytecode(bytecode_buffer)
        stablehlo_bytes = bytecode_buffer.getvalue()

        # Create hash components
        hasher = hashlib.sha256()

        # StableHLO bytecode content
        hasher.update(stablehlo_bytes)

        # I/O tensor specs (shapes, dtypes)
        for input_spec in io_specs.inputs:
            hasher.update(str(input_spec.shape).encode("utf-8"))
            hasher.update(str(input_spec.dtype).encode("utf-8"))

        for output_spec in io_specs.outputs:
            hasher.update(str(output_spec.shape).encode("utf-8"))
            hasher.update(str(output_spec.dtype).encode("utf-8"))

        # Function signature information (main function name)
        hasher.update(b"main")

        # Final cache key
        cache_key = hasher.hexdigest()[:16]  # Use first 16 chars for readability
        return cache_key

    except Exception as e:
        raise RuntimeError(f"Failed to generate cache key from StableHLO: {e}") from e


# MLIR element type -> torch.dtype mapping
MLIR_TO_TORCH_DTYPE = {
    "f32": torch.float32,
    "f64": torch.float64,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i64": torch.int64,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8": torch.int8,
    "i1": torch.bool,
    "ui8": torch.uint8,
    "ui16": torch.uint16,
    "ui32": torch.uint32,
    "ui64": torch.uint64,
    "complex<f32>": torch.complex64,
    "complex<f64>": torch.complex128,
}


@dataclass(frozen=True)
class TensorSpec:
    """Tensor shape and dtype specification."""

    shape: tuple[int, ...]
    dtype: torch.dtype

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) if d >= 0 else "?" for d in self.shape)
        dtype_str = str(self.dtype).replace("torch.", "")
        return f"<{shape_str}x{dtype_str}>" if shape_str else f"<{dtype_str}>"

    def to_torch_size(self) -> torch.Size:
        return torch.Size(self.shape)

    def is_dynamic(self) -> bool:
        return any(d < 0 for d in self.shape)


@dataclass(frozen=True)
class FunctionIO:
    """Input/Output specification for a function."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def __repr__(self) -> str:
        inputs_str = ", ".join(str(t) for t in self.inputs)
        outputs_str = ", ".join(str(t) for t in self.outputs)
        return f"Inputs: [{inputs_str}] → Outputs: [{outputs_str}]"


def _parse_mlir_type(mlir_type: ir.Type) -> TensorSpec:
    """Parse MLIR type into TensorSpec."""
    shape = tuple(int(d) for d in mlir_type.shape) if hasattr(mlir_type, "shape") else ()

    if hasattr(mlir_type, "element_type"):
        element_type_str = str(mlir_type.element_type)
    else:
        element_type_str = str(mlir_type).rstrip(">").split("x")[-1]

    dtype = MLIR_TO_TORCH_DTYPE.get(element_type_str)
    if dtype is None:
        raise ValueError(f"Unknown MLIR element type: {element_type_str}")
    return TensorSpec(shape=shape, dtype=dtype)


def parse_module_io(module: ir.Module, func_name: str = "main") -> FunctionIO:
    """
    Parse input/output specs from a StableHLO module's function.

    Args:
    module: torch_mlir MLIR module
    func_name: Target function name (default: "main")

    Returns:
    FunctionIO with input and output TensorSpecs
    """
    for op in module.body.operations:
        if (
            op.OPERATION_NAME in ("func.func", "builtin.func", "stablehlo.func", "main")
            and op.attributes["sym_name"].value == func_name
        ):
            func_type = op.attributes["function_type"].value
            inputs = tuple(_parse_mlir_type(i) for i in func_type.inputs)
            outputs = tuple(_parse_mlir_type(o) for o in func_type.results)
            return FunctionIO(inputs=inputs, outputs=outputs)
        raise ValueError(f"Function '{func_name}' not found in module")


def get_input_specs(module: ir.Module, func_name: str = "main") -> list[TensorSpec]:
    return list(parse_module_io(module, func_name).inputs)


def get_output_specs(module: ir.Module, func_name: str = "main") -> list[TensorSpec]:
    return list(parse_module_io(module, func_name).outputs)


def shapes_match(shape1: TensorSpec, shape2: TensorSpec):
    """Check if two shapes match exactly."""
    if shape1 is None or shape2 is None:
        return False
    return shape1 == shape2
