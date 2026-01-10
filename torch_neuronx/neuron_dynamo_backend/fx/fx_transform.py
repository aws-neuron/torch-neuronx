"""
FX Graph to StableHLO conversion utilities
"""

import logging
from pathlib import Path

import torch
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report

from torch_neuronx.neuron_dynamo_backend.config import (
    get_err_mlir_path,
    get_raw_torch_path,
    get_transformed_fx_path,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization import (
    legalize_replica_groups,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    TensorSpec,
    parse_module_io,
)

logger = logging.getLogger(__name__)
IR_DEBUG_PRINT = logger.getEffectiveLevel() <= logging.DEBUG


def convert_fx_to_stablehlo(
    gm: torch.fx.GraphModule, example_inputs, preserve_artifacts: bool = False
) -> tuple[str, FunctionIO, list[TensorSpec]]:
    """
    Convert FX GraphModule to StableHLO using torch-mlir with collective support

    1. Apply FX graph-level collective transformations
    2. Import FX to RAW Torch dialect (no backend pipeline to avoid ReduceOpVariants)
    3. Apply custom Neuron transformations/lowerings
    4. Lower to torch backend IR
    5. Lower to stablehlo IR

    Args:
        gm: FX GraphModule with Aten operations (from AOTAutograd)
        example_inputs: Example input tensors
        preserve_artifacts: Save artifacts

    Returns:
        str: StableHLO MLIR string
        io_spec: Input/output specifications for Executor.
        cast_spec: Output specifications for dtype preservation.

    Raises:
        Exception: If StableHLO conversion fails
    """
    logger.debug("FX transformation (replica_groups legalization)")

    # TODO: Manage via FX-passes.
    gm = legalize_replica_groups(gm)

    # Save transformed fx graph for debugging
    if preserve_artifacts:
        transformed_fx_path = get_transformed_fx_path()
        with open(transformed_fx_path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved transformed fx graph to {transformed_fx_path}")

    logger.debug("Importing FX to RAW Torch dialect")
    raw_module = fx.stateless_fx_import(
        gm,
        output_type=OutputType.RAW,  # RAW = no backend pipeline
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
    io_spec = parse_module_io(raw_module)
    logger.debug("FX Graph -> StableHLO conversion successful")
    return raw_module, io_spec, cast_spec


def save_mlir_bytecode(mlir: mlir_ir.Module, output_path: Path) -> Path:
    """
    Save MLIR bytecode to file

    Args:
        stablemlirhlo_mlir: MLIR Module
        output_path: Path to save the file

    Returns:
        Path: Path to the saved file
    """
    logger.debug(f"Saving MLIR to {output_path}")
    with open(output_path, "wb") as f:
        mlir.operation.write_bytecode(f)
    return output_path
