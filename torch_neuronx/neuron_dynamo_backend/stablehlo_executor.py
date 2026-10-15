"""StableHLO reference interpreter executor."""

import io
import logging
import re

import numpy as np
import torch
from torch_mlir import ir as mlir_ir
from torch_mlir._mlir_libs._stablehlo import eval_module

logger = logging.getLogger(__name__)

MLIR_TYPE_STR_TO_TORCH_DTYPE = {
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

MLIR_TYPE_STR_TO_NP_DTYPE = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


def _get_splat_np_value(attr, np_dtype, element_type_str):
    """Extract the scalar splat value from a DenseElementsAttr as a numpy scalar."""
    splat = attr.get_splat_value()
    if element_type_str == "i1":
        return np_dtype(bool(splat))
    if element_type_str.startswith("f"):
        return np_dtype(mlir_ir.FloatAttr(splat).value)
    return np_dtype(mlir_ir.IntegerAttr(splat).value)


def _parse_bf16_dense_attr(attr_str: str) -> torch.Tensor:
    """Parse a bf16 DenseElementsAttr string into a flat bf16 torch tensor.

    Handles both formats:
      - Raw hex: dense<"0x..."> : tensor<...xbf16>
      - Float literals: dense<[1.0, 2.0, ...]> : tensor<...xbf16>
    """
    hex_match = re.search(r'dense<"0x([0-9A-Fa-f]+)">', attr_str)
    if hex_match:
        raw = bytes.fromhex(hex_match.group(1))
        flat = np.frombuffer(raw, dtype=np.uint16)
        return torch.from_numpy(flat.copy()).view(torch.bfloat16)

    inner = re.search(r"dense<(.+)> :", attr_str).group(1)
    inner = inner.replace("[", "").replace("]", "")
    vals = [float(v) for v in inner.split(",")]
    return torch.tensor(vals, dtype=torch.float32).to(torch.bfloat16)


def dense_attr_to_torch_tensor(attr) -> torch.Tensor:
    """Convert an MLIR ``DenseElementsAttr`` into a PyTorch tensor.

    Args:
        attr (mlir_ir.DenseElementsAttr): Dense MLIR attribute to convert.

    Returns:
        torch.Tensor: Tensor with the attribute's values and element dtype. Rank-zero
            attributes are represented as one-element tensors.

    Raises:
        ValueError: If the MLIR element type is unsupported.
    """
    ranked_type = mlir_ir.RankedTensorType(attr.type)
    shape = list(ranked_type.shape)
    element_type_str = str(ranked_type.element_type)
    num_elements = max(1, int(np.prod(shape))) if shape else 1

    if element_type_str == "bf16":
        # Handle bf16 attrs separately since the underlying bytes
        # must be extracted from the attr's string representation
        if attr.is_splat:
            t = _parse_bf16_dense_attr(str(attr))
            return t.expand(shape) if shape else t
        flat = _parse_bf16_dense_attr(str(attr))
        if shape:
            flat = flat.reshape(shape)
        return flat

    np_dtype = MLIR_TYPE_STR_TO_NP_DTYPE.get(element_type_str)
    torch_dtype = MLIR_TYPE_STR_TO_TORCH_DTYPE.get(element_type_str)
    if np_dtype is None or torch_dtype is None:
        raise ValueError(f"Unsupported MLIR element type: {element_type_str}")

    if attr.is_splat:
        val = _get_splat_np_value(attr, np_dtype, element_type_str)
        flat = np.full(num_elements, val, dtype=np_dtype)
    else:
        flat = np.array(attr, dtype=np_dtype)
    if shape:
        flat = flat.reshape(shape)
    return torch.from_numpy(flat.copy()).to(torch_dtype)


def torch_tensor_to_dense_attr(val, expected_type, context):
    """Convert a PyTorch tensor or scalar into an MLIR ``DenseElementsAttr``.

    Args:
        val (torch.Tensor | Scalar): Value whose shape and contents to convert; tensors
            are copied to CPU first.
        expected_type (mlir_ir.RankedTensorType): MLIR type whose element type to use.
        context (mlir_ir.Context): MLIR context used to construct the attribute.

    Returns:
        mlir_ir.DenseElementsAttr: Dense attribute with the value's shape and the
            expected element type.
    """
    if isinstance(val, torch.Tensor):
        t = val.detach().cpu().contiguous()
        # Handle bf16 tensors separately by casting
        # the underlying bytes to np.uint16
        if t.dtype == torch.bfloat16:
            np_array = t.view(torch.uint16).numpy().view(np.uint16)
        else:
            np_array = t.numpy()
    else:
        np_array = np.array(val)

    shape = list(np_array.shape)
    element_type_str = str(expected_type.element_type)
    # Handle i1 tensors separately by bitpacking
    # 8 elements per byte
    if element_type_str == "i1":
        np_array = np.packbits(np_array.flatten(), bitorder="little")

    with context:
        return mlir_ir.DenseElementsAttr.get(
            array=np_array, type=expected_type.element_type, shape=shape
        )


def _tensorize_scalar_args(module):
    """Rewrite bare scalar function args and results to 0-d tensors."""
    func = module.body.operations[0]
    ft = mlir_ir.FunctionType(mlir_ir.TypeAttr(func.attributes["function_type"]).value)

    has_scalar_inputs = any(str(t) in MLIR_TYPE_STR_TO_TORCH_DTYPE for t in ft.inputs)
    has_scalar_results = any(str(t) in MLIR_TYPE_STR_TO_TORCH_DTYPE for t in ft.results)
    if not has_scalar_inputs and not has_scalar_results:
        return

    context = module.context
    with context, mlir_ir.Location.unknown():
        new_inputs = []
        for t in ft.inputs:
            if str(t) in MLIR_TYPE_STR_TO_TORCH_DTYPE:
                new_inputs.append(mlir_ir.RankedTensorType.get([], t))
            else:
                new_inputs.append(t)

        new_results = []
        for t in ft.results:
            if str(t) in MLIR_TYPE_STR_TO_TORCH_DTYPE:
                new_results.append(mlir_ir.RankedTensorType.get([], t))
            else:
                new_results.append(t)

        new_ft = mlir_ir.FunctionType.get(new_inputs, new_results)
        func.attributes["function_type"] = mlir_ir.TypeAttr.get(new_ft)

        entry_block = func.body.blocks[0]
        for arg in entry_block.arguments:
            if str(arg.type) in MLIR_TYPE_STR_TO_TORCH_DTYPE:
                arg.set_type(mlir_ir.RankedTensorType.get([], arg.type))


class StableHLOInterpreterExecutor:
    """Run StableHLO through the CPU reference interpreter.

    Args:
        stablehlo_mlir (mlir_ir.Module): StableHLO module to execute and mutate in place.
        io_spec (Any): Input and output specification stored but otherwise unused.
        cast_spec (Sequence): Output dtype specification stored but otherwise unused.

    Note:
        This is a drop-in replacement for the NEFF executor, and all inputs remain on CPU.
    """

    def __init__(self, stablehlo_mlir, io_spec, cast_spec):
        self.stablehlo_mlir = stablehlo_mlir
        self.io_spec = io_spec
        self.cast_spec = cast_spec
        func = stablehlo_mlir.body.operations[0]
        ft = mlir_ir.FunctionType(mlir_ir.TypeAttr(func.attributes["function_type"]).value)
        self._original_result_types = [str(t) for t in ft.results]
        _tensorize_scalar_args(self.stablehlo_mlir)

    def __call__(self, inputs):
        context = self.stablehlo_mlir.context
        func = self.stablehlo_mlir.body.operations[0]
        ft = mlir_ir.FunctionType(mlir_ir.TypeAttr(func.attributes["function_type"]).value)

        attrs = []
        for val, expected_type in zip(inputs, ft.inputs, strict=False):
            attrs.append(torch_tensor_to_dense_attr(val, expected_type, context))

        buf = io.StringIO()
        self.stablehlo_mlir.operation.print(file=buf, enable_debug_info=False)
        logger.info(f"StableHLO module:\n{buf.getvalue()}")
        logger.info(f"Inputs ({len(attrs)}):")
        for i, a in enumerate(attrs):
            logger.info(f"  arg{i}: {a.type}")

        try:
            results = eval_module(self.stablehlo_mlir, attrs)
        except Exception as e:
            raise RuntimeError(f"StableHLO interpreter failed: {e}") from e

        outputs = []
        for r, orig_type in zip(results, self._original_result_types, strict=False):
            t = dense_attr_to_torch_tensor(r)
            if t.ndim == 0 and orig_type in MLIR_TYPE_STR_TO_TORCH_DTYPE:
                outputs.append(t.item())
            else:
                outputs.append(t)
        return outputs
