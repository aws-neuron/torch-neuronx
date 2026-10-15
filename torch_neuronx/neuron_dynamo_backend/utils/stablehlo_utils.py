"""
StableHLO Module I/O Parser
Extracts input/output shapes and dtypes from torch_mlir StableHLO modules.
"""

import hashlib
import io
from dataclasses import dataclass

import torch


def compute_cache_key(stablehlo_mlir) -> str:
    """Compute a cache key from a StableHLO graph.

    Args:
        stablehlo_mlir (mlir.ir.Module): StableHLO MLIR module to hash.

    Returns:
        str: 64-character SHA-256 hexadecimal digest.

    Raises:
        RuntimeError: If printing or parsing the module fails.

    Note:
        The hash includes module text without debug locations plus input and
        output shapes and dtypes. Strip nondeterministic attributes such as
        ``torch.debug_dump_path`` before calling this function.
    """
    try:
        hasher = hashlib.sha256()

        # Hash module text with location info disabled
        buf = io.StringIO()
        stablehlo_mlir.operation.print(file=buf, enable_debug_info=False)
        hasher.update(buf.getvalue().encode("utf-8"))

        # I/O tensor specs (shapes, dtypes)
        io_specs = parse_module_io(stablehlo_mlir)
        for input_spec in io_specs.inputs:
            hasher.update(str(input_spec.shape).encode("utf-8"))
            hasher.update(str(input_spec.dtype).encode("utf-8"))
        for output_spec in io_specs.outputs:
            hasher.update(str(output_spec.shape).encode("utf-8"))
            hasher.update(str(output_spec.dtype).encode("utf-8"))

        cache_key = hasher.hexdigest()[:64]
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
    "f8E5M2": torch.float8_e5m2,
    "f8E4M3FN": torch.float8_e4m3fn,
}

if hasattr(torch, "float8_e8m0fnu"):
    MLIR_TO_TORCH_DTYPE["f8E8M0FNU"] = torch.float8_e8m0fnu


@dataclass(frozen=True)
class TensorSpec:
    """Describe a tensor shape and dtype.

    Args:
        shape (tuple[int, ...]): Tensor dimensions, with negative values marking
            dynamic dimensions.
        dtype (torch.dtype): Tensor element dtype.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype

    def __repr__(self) -> str:
        shape_str = "x".join(str(d) if d >= 0 else "?" for d in self.shape)
        dtype_str = str(self.dtype).replace("torch.", "")
        return f"<{shape_str}x{dtype_str}>" if shape_str else f"<{dtype_str}>"

    def to_torch_size(self) -> torch.Size:
        """Convert the stored shape to a ``torch.Size``.

        Returns:
            torch.Size: Size containing the stored dimensions.
        """
        return torch.Size(self.shape)

    def is_dynamic(self) -> bool:
        """Check whether the shape contains a dynamic dimension.

        Returns:
            bool: ``True`` if any dimension is negative; otherwise ``False``.
        """
        return any(d < 0 for d in self.shape)


@dataclass(frozen=True)
class FunctionIO:
    """Store the input and output tensor specifications for a function.

    Args:
        inputs (tuple[TensorSpec, ...]): Input tensor specifications.
        outputs (tuple[TensorSpec, ...]): Output tensor specifications.
    """

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    def __repr__(self) -> str:
        inputs_str = ", ".join(str(t) for t in self.inputs)
        outputs_str = ", ".join(str(t) for t in self.outputs)
        return f"Inputs: [{inputs_str}] → Outputs: [{outputs_str}]"


def _parse_mlir_type(mlir_type) -> TensorSpec:
    """Parse an MLIR type into a TensorSpec.

    Extracts shape and dtype information from MLIR tensor types.

    Args:
        mlir_type (ir.Type): MLIR type to parse (e.g., tensor<2x3xf32>).

    Returns:
        TensorSpec: Parsed tensor specification with shape and dtype.

    Raises:
        ValueError: If the MLIR element type is not recognized.
    """
    shape = tuple(int(d) for d in mlir_type.shape) if hasattr(mlir_type, "shape") else ()

    if hasattr(mlir_type, "element_type"):
        element_type_str = str(mlir_type.element_type)
    else:
        element_type_str = str(mlir_type).rstrip(">").split("x")[-1]

    dtype = MLIR_TO_TORCH_DTYPE.get(element_type_str)
    if dtype is None:
        raise ValueError(f"Unknown MLIR element type: {element_type_str}")
    return TensorSpec(shape=shape, dtype=dtype)


def parse_module_io(
    module,
    func_name: str = "main",
) -> FunctionIO:
    """Parse tensor specifications from a StableHLO module function.

    Args:
        module (mlir.ir.Module): StableHLO MLIR module to inspect.
        func_name (str, optional): Target symbol name. Defaults to ``"main"``.

    Returns:
        FunctionIO | None: Parsed specifications when the first operation is the
        target function, or ``None`` when the module has no operations.

    Raises:
        ValueError: If the first operation is not the target function or an
            element dtype is unsupported.
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


def get_input_specs(module, func_name: str = "main") -> list[TensorSpec]:
    """Get input tensor specifications from a StableHLO module function.

    Args:
        module (mlir.ir.Module): StableHLO MLIR module to inspect.
        func_name (str, optional): Target symbol name. Defaults to ``"main"``.

    Returns:
        list[TensorSpec]: Input tensor specifications in parameter order.

    Raises:
        ValueError: If the target function cannot be parsed.
        AttributeError: If parsing returns no function data, including when the module has no
            operations.
    """
    return list(parse_module_io(module, func_name).inputs)


def get_output_specs(module, func_name: str = "main") -> list[TensorSpec]:
    """Get output tensor specifications from a StableHLO module function.

    Args:
        module (mlir.ir.Module): StableHLO MLIR module to inspect.
        func_name (str, optional): Target symbol name. Defaults to ``"main"``.

    Returns:
        list[TensorSpec]: Output tensor specifications in result order.

    Raises:
        ValueError: If the target function cannot be parsed.
        AttributeError: If the module has no operations.
    """
    return list(parse_module_io(module, func_name).outputs)


def shapes_match(shape1: TensorSpec, shape2: TensorSpec) -> bool:
    """Check whether two tensor specifications match exactly.

    Args:
        shape1 (TensorSpec | None): First tensor specification, or ``None``.
        shape2 (TensorSpec | None): Second tensor specification, or ``None``.

    Returns:
        bool: ``True`` if both non-``None`` specifications have equal shapes
            and dtypes; otherwise ``False``.
    """
    if shape1 is None or shape2 is None:
        return False
    return shape1 == shape2
