"""
FX Graph to StableHLO conversion utilities
"""

import logging
from pathlib import Path

import torch
from torch._dynamo.utils import dynamo_timed
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report
from torch_mlir.ir import Module

from torch_neuronx.neuron_dynamo_backend.config import (
    get_err_mlir_path,
    get_raw_torch_path,
    get_transformed_fx_path,
)
from torch_neuronx.neuron_dynamo_backend.fx.fx_hooks import CustomFxImporterHooks
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronFxImporter
from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    RandomInputInfo,
    TensorSpec,
    parse_module_io,
    shapes_match,
)

logger = logging.getLogger(__name__)
IR_DEBUG_PRINT = logger.getEffectiveLevel() <= logging.DEBUG


def _is_tuple_return(module: Module) -> bool:
    """Check if the module's entry function returns a tuple type.

    Args:
        module (Module): MLIR module to check.

    Returns:
        bool: True if the main function returns a tuple or multiple results.
    """
    with module.context:
        for op in module.body.operations:
            if op.operation.name == "func.func":
                func_type = mlir_ir.FunctionType(op.type)
                results = func_type.results
                if len(results) == 0:
                    return False
                if len(results) > 1:
                    # Multiple results = effectively a tuple
                    return True
                # Single result - check if it's actually a tuple type
                result_type_str = str(results[0])
                return result_type_str.startswith("tuple<")
    return False


def _set_module_alias(module: Module, aliasing_info: AliasingInfo) -> None:
    """Set aliasing attributes on the MLIR module for input-output tensor aliasing.

    Attaches mhlo.input_output_alias attributes to the module operation,
    informing the compiler which output tensors should alias input tensors.

    Args:
        module (Module): MLIR Module to modify.
        aliasing_info (AliasingInfo): Aliasing relationships to apply.

    Raises:
        IndexError: If parameter_number or output_index is out of bounds.
    """
    if aliasing_info is None:
        return

    is_tuple_output = _is_tuple_return(module)
    alias_dicts = []

    with module.context:
        function_io = parse_module_io(module)
        inputs = function_io.inputs
        outputs = function_io.outputs

        for alias in aliasing_info:
            param_num = alias.parameter_number
            param_idx = alias.parameter_index
            output_idx = alias.output_index

            if param_num < 0 or param_num >= len(inputs):
                raise IndexError(
                    f"param_num {param_num} is out of bounds for inputs array "
                    f"of length {len(inputs)}"
                )
            input_shape = inputs[param_num]

            # Bounds check for outputs
            if output_idx < 0 or output_idx >= len(outputs):
                raise IndexError(
                    f"output_idx {output_idx} is out of bounds for outputs array "
                    f"of length {len(outputs)}"
                )
            output_shape = outputs[output_idx]

            # If there's a shape mismatch, skip this alias
            # StableHLO does not recognize mismatch-shaped
            # alias relationship.
            if not shapes_match(input_shape, output_shape):
                continue

            # Build inner alias dict
            param_index_attr = mlir_ir.DenseI64ArrayAttr.get(param_idx if param_idx else [])
            param_number_attr = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signed(64), param_num
            )
            kind_attr = mlir_ir.StringAttr.get("must_alias")

            alias_inner_dict = mlir_ir.DictAttr.get(
                {
                    "kind": kind_attr,
                    "parameter_index": param_index_attr,
                    "parameter_number": param_number_attr,
                }
            )

            # Build output_index based on return type
            if is_tuple_output:
                # Tuple output: index into the tuple
                # output_idx could be int or list
                output_index_attr = mlir_ir.DenseI64ArrayAttr.get(
                    [output_idx] if isinstance(output_idx, int) else output_idx
                )
            else:
                # Single tensor output: must use empty index
                output_index_attr = mlir_ir.DenseI64ArrayAttr.get([])

            alias_outer_dict = mlir_ir.DictAttr.get(
                {"alias": alias_inner_dict, "output_index": output_index_attr}
            )

            alias_dicts.append(alias_outer_dict)

        # Create array with ALL aliases at once
        final_attr = mlir_ir.ArrayAttr.get(alias_dicts)
        module.operation.attributes["mhlo.input_output_alias"] = final_attr


def convert_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    aliasing_info: AliasingInfo = None,
    preserve_artifacts: bool = False,
    random_input_info: RandomInputInfo | None = None,
) -> tuple[str, FunctionIO, list[TensorSpec]]:
    """
    Convert FX GraphModule to StableHLO using torch-mlir with collective support

    Pipeline:
    1. Apply FX graph-level collective transformations
    2. Import FX to RAW Torch dialect
    3. Apply custom Neuron transformations/lowerings
    4. Lower to Torch backend IR
    5. Lower to StableHLO IR

    Args:
        gm: FX GraphModule with Aten operations (from AOTAutograd)
        example_inputs: Example input tensors
        preserve_artifacts: Save artifacts
        random_input_info: Optional metadata about random inputs added to the graph

    Returns:
        tuple: (stablehlo_module, io_spec, cast_spec)
            - stablehlo_module (Module): StableHLO MLIR module.
            - io_spec (FunctionIO): Input/output specifications for Executor.
            - cast_spec (list[TensorSpec]): Output specifications for dtype preservation.
    """
    # Save transformed fx graph for debugging
    if preserve_artifacts:
        transformed_fx_path = get_transformed_fx_path()
        with open(transformed_fx_path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved transformed fx graph to {transformed_fx_path}")

    # Torch-MLIR lowering: FX → RAW → Torch Backend → StableHLO
    with dynamo_timed("torch_neuronx_lower"):
        logger.debug("Importing FX to RAW Torch dialect")
        raw_module = fx.stateless_fx_import(
            gm,
            output_type=OutputType.RAW,  # RAW = no backend pipeline
            fx_importer=NeuronFxImporter(hooks=CustomFxImporterHooks()),
            verbose=IR_DEBUG_PRINT,
            enable_ir_printing=IR_DEBUG_PRINT,
        )

        # Save RAW module for debugging
        if preserve_artifacts:
            raw_path = get_raw_torch_path()
            save_mlir_bytecode(raw_module, raw_path)
            logger.debug(f"Saved RAW Torch module to {raw_path}")
        raw_module.operation.attributes["torch.debug_dump_path"] = mlir_ir.StringAttr.get(
            str(get_err_mlir_path()), raw_module.context
        )

        # Custom Neuron RAW lowerings/transformations
        logger.debug("Applying Neuron lowerings/transformations")
        run_pipeline_with_repro_report(
            raw_module,
            "builtin.module(torch-raw-to-neuron-backend-pipeline)",
            "Neuron partial lowering from TorchFX IR -> StableHLO IR",
            enable_ir_printing=IR_DEBUG_PRINT,
        )

        # Lower to TORCH
        logger.debug("Lowering to TORCH")
        run_pipeline_with_repro_report(
            raw_module,
            "builtin.module(func.func(torch-match-quantized-custom-ops), "
            + "torchdynamo-export-to-torch-backend-pipeline)",
            "Lowering Torch FX IR (RAW) -> Torch Backend IR",
            enable_ir_printing=IR_DEBUG_PRINT,
        )

        # Lower to StableHLO
        logger.debug("Lowering to StableHLO")
        run_pipeline_with_repro_report(
            raw_module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO IR",
            enable_ir_printing=IR_DEBUG_PRINT,
        )

        # Out spec for dtype preservation.
        cast_spec = parse_module_io(raw_module).outputs

        # Convert f64 to f32
        raw_module = dtype_conversion_pass(raw_module)

    # Spec for executor.
    io_spec = parse_module_io(raw_module, random_input_info=random_input_info)
    # Module should have at least one output
    if len(io_spec.outputs) == 0:
        raise ValueError(
            "No outputs found in the module, expected at least one output, "
            "modify your code to have at least one output."
        )
    logger.debug("FX Graph -> StableHLO conversion successful")
    # Set the aliasing attributes on the MLIR module
    _set_module_alias(raw_module, aliasing_info)
    return raw_module, io_spec, cast_spec


def save_mlir_bytecode(mlir: mlir_ir.Module, output_path: Path) -> Path:
    """Save MLIR bytecode to file.

    Args:
        mlir (mlir_ir.Module): MLIR module to save.
        output_path (Path): Path to save the file.

    Returns:
        Path: Path to the saved file.
    """
    logger.debug(f"Saving MLIR to {output_path}")
    with open(output_path, "wb") as f:
        mlir.operation.write_bytecode(f)
    return output_path
