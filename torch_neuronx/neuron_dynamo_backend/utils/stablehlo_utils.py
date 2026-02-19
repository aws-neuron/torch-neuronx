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
    """Generate a cache key from a StableHLO graph.

    Creates a SHA-256 hash from the StableHLO module text and I/O specifications.

    Note: Non-deterministic attributes (e.g., torch.debug_dump_path) should be
    stripped before calling this function.

    Args:
        stablehlo_mlir: Compiled StableHLO MLIR module.

    Returns:
        str: 64-character hex cache key for C++ consumption.

    Raises:
        RuntimeError: If cache key generation fails.

    Example:
        >>> stablehlo_mlir = convert_fx_to_stablehlo(gm, example_inputs)
        >>> cache_key = compute_cache_key(stablehlo_mlir)
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


# =============================================================================
# Random Op Descriptors
# =============================================================================


@dataclass(frozen=True)
class NativeDropoutOp:
    """Descriptor for native_dropout operation with all info needed to generate mask.

    Attributes:
        input_position: Position in the graph's input list (after original inputs)
        shape: Shape of the mask tensor
        dtype: Data type of the mask tensor
        probability: Dropout probability (probability of zeroing an element)
        train: Whether dropout is in training mode
    """

    input_position: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    probability: float
    train: bool

    def sample(self, device: str | torch.device) -> torch.Tensor:
        """Generate a boolean dropout mask tensor.

        Args:
            device: Target device

        Returns:
            Boolean dropout mask tensor where True = keep, False = drop.
            The scaling by 1/(1-p) is done in the compiled graph, not here.
        """
        zeros = torch.zeros(self.shape, dtype=self.dtype, device="cpu")
        _, mask = torch.ops.aten.native_dropout(zeros, p=self.probability, train=self.train)
        return mask.to(device)


@dataclass
class RandomInputInfo:
    """Metadata to track random inputs added to the graph.

    Attributes:
        ops: List of random op descriptors
        original_input_count: Number of inputs before pass
        new_input_count: Number of inputs after pass
    """

    ops: list[NativeDropoutOp]
    original_input_count: int
    new_input_count: int


@dataclass(frozen=True)
class FunctionIO:
    """Input/Output specification for a function."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    random_input_info: RandomInputInfo | None = None

    def __repr__(self) -> str:
        inputs_str = ", ".join(str(t) for t in self.inputs)
        outputs_str = ", ".join(str(t) for t in self.outputs)
        base = f"Inputs: [{inputs_str}] → Outputs: [{outputs_str}]"
        if self.random_input_info:
            base += f" | {self.random_input_info}"
        return base


def _parse_mlir_type(mlir_type: ir.Type) -> TensorSpec:
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
    module: ir.Module,
    func_name: str = "main",
    random_input_info: RandomInputInfo | None = None,
) -> FunctionIO:
    """
    Parse input/output specs from a StableHLO module's function.

    Args:
        module: torch_mlir MLIR module
        func_name: Target function name (default: "main")
        random_input_info: Optional metadata about random inputs added to the graph

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
            return FunctionIO(
                inputs=inputs,
                outputs=outputs,
                random_input_info=random_input_info,
            )
        raise ValueError(f"Function '{func_name}' not found in module")


def get_input_specs(module: ir.Module, func_name: str = "main") -> list[TensorSpec]:
    """Get input tensor specifications from a StableHLO module.

    Args:
        module (ir.Module): MLIR module to parse.
        func_name (str): Target function name. Defaults to "main".

    Returns:
        list[TensorSpec]: List of input tensor specifications.
    """
    return list(parse_module_io(module, func_name).inputs)


def get_output_specs(module: ir.Module, func_name: str = "main") -> list[TensorSpec]:
    """Get output tensor specifications from a StableHLO module.

    Args:
        module (ir.Module): MLIR module to parse.
        func_name (str): Target function name. Defaults to "main".

    Returns:
        list[TensorSpec]: List of output tensor specifications.
    """
    return list(parse_module_io(module, func_name).outputs)


def shapes_match(shape1: TensorSpec, shape2: TensorSpec) -> bool:
    """Check if two tensor specifications match exactly.

    Compares both shape and dtype for equality. Used to validate
    aliasing relationships where input and output must have identical specs.

    Args:
        shape1 (TensorSpec): First tensor specification.
        shape2 (TensorSpec): Second tensor specification.

    Returns:
        bool: True if both specs are non-None and equal, False otherwise.
    """
    if shape1 is None or shape2 is None:
        return False
    return shape1 == shape2
