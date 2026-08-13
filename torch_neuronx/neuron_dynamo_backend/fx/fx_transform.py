"""
FX Graph to StableHLO conversion utilities
"""

import logging

import torch
from torch._dynamo.utils import dynamo_timed
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.compiler_utils import OutputType, run_pipeline_with_repro_report
from torch_mlir.ir import Module

from torch_neuronx.kernels.cache_utils import CompilationJob
from torch_neuronx.neuron_dynamo_backend.config import get_model_name
from torch_neuronx.neuron_dynamo_backend.fx.dynamic_dim_constraints import (
    DynamicDimConstraints,
    build_dynamic_dim_constraints,
)
from torch_neuronx.neuron_dynamo_backend.fx.fx_hooks import CustomFxImporterHooks
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronFxImporter
from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    TensorSpec,
    parse_module_io,
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


def _set_module_alias(module: Module, alias_buffer_map: dict[int, int] | None) -> None:
    """Set aliasing attributes on the MLIR module for input-output tensor aliasing.

    Attaches mhlo.input_output_alias attributes to the module operation,
    informing the compiler which output tensors should alias input tensors.

    Takes a pre-computed {neff_output_index: neff_input_index} map from
    get_alias_output_buffer_map().

    Args:
        module (Module): MLIR Module to modify.
        alias_buffer_map (dict[int, int]): NEFF output→input index pairs to alias.
    """
    if not alias_buffer_map:
        return

    is_tuple_output = _is_tuple_return(module)

    with module.context:
        function_io = parse_module_io(module)
        inputs = function_io.inputs
        outputs = function_io.outputs

        # MLIR output index -> (input param_num, parameter_index) to alias.
        output_to_alias: dict[int, tuple[int, list[int]]] = {}
        for output_idx, param_num in alias_buffer_map.items():
            assert (0 <= param_num < len(inputs)) and (0 <= output_idx < len(outputs)), (
                f"Alias index out of range for lowered MLIR IO: "
                f"param_num={param_num} output_idx={output_idx} "
                f"(inputs={len(inputs)}, outputs={len(outputs)})"
            )
            output_to_alias[output_idx] = (param_num, [])

        if not output_to_alias:
            return

        # Emit mhlo.input_output_alias from the validated map.
        alias_dicts = []
        for output_idx in sorted(output_to_alias):
            param_num, param_idx = output_to_alias[output_idx]
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
                output_index_attr = mlir_ir.DenseI64ArrayAttr.get([output_idx])
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


def _set_dynamic_dim_constraints(module: Module, constraints: DynamicDimConstraints) -> None:
    """Attach dynamic dimension constraint metadata as MLIR attributes on the module.

    Encodes two attributes:
      - "neuron.dynamic_dim_symbols": DictAttr {symbol_name -> IntegerAttr(arg_index)}
      - "neuron.dynamic_dim_exprs": DictAttr {arg_index -> DictAttr {dim_index -> StringAttr(expr)}}

    These survive lowering as module-level attributes and inform the downstream
    compiler which runtime scalar args provide dynamic dimension sizes.
    """
    with module.context:
        # Symbol-to-arg-index mapping: {"s0": 0, "s1": 3, ...}
        i64 = mlir_ir.IntegerType.get_signless(64)
        symbol_dict = {
            name: mlir_ir.IntegerAttr.get(i64, idx)
            for name, idx in constraints.symbol_to_arg_index.items()
        }
        module.operation.attributes["neuron.dynamic_dim_symbols"] = mlir_ir.DictAttr.get(
            symbol_dict
        )

        # Tensor-dim expressions: {arg_idx: {dim_idx: expr_str}}
        outer_dict = {}
        for arg_idx, dim_map in constraints.tensor_dim_exprs.items():
            inner_dict = {
                str(dim_idx): mlir_ir.StringAttr.get(expr_str)
                for dim_idx, expr_str in dim_map.items()
            }
            outer_dict[str(arg_idx)] = mlir_ir.DictAttr.get(inner_dict)
        module.operation.attributes["neuron.dynamic_dim_exprs"] = mlir_ir.DictAttr.get(outer_dict)


def convert_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    aliasing_info: dict[int, int] | None = None,
    *,
    job: CompilationJob | None = None,
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
        job: CompilationJob for artifact management. If None, artifact dumping is skipped.

    Returns:
        tuple: (stablehlo_module, io_spec, cast_spec)
            - stablehlo_module (Module): StableHLO MLIR module.
            - io_spec (FunctionIO): Input/output specifications for Executor.
            - cast_spec (list[TensorSpec]): Output specifications for dtype preservation.
    """
    model_name = get_model_name()

    # Save transformed fx graph for debugging
    if job is not None:
        job.save_transformed_fx(model_name, gm)

    # Torch-MLIR lowering: FX → RAW → Torch Backend → StableHLO
    with dynamo_timed("torch_neuronx_lower"):
        logger.debug("Importing FX to RAW Torch dialect")
        fx_importer = NeuronFxImporter(hooks=CustomFxImporterHooks())
        raw_module = fx.stateless_fx_import(
            gm,
            output_type=OutputType.RAW,  # RAW = no backend pipeline
            fx_importer=fx_importer,
            verbose=IR_DEBUG_PRINT,
            enable_ir_printing=IR_DEBUG_PRINT,
        )
        has_hierarchy = fx_importer.has_hierarchy

        # Save RAW module for debugging
        if job is not None:
            job.save_raw_torch(model_name, raw_module)

        if job is not None:
            err_mlir_path = job.get_err_mlir_path(model_name)
            raw_module.operation.attributes["torch.debug_dump_path"] = mlir_ir.StringAttr.get(
                str(err_mlir_path), raw_module.context
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
    # Module should have at least one output
    if len(io_spec.outputs) == 0:
        raise ValueError(
            "No outputs found in the module, expected at least one output, "
            "modify your code to have at least one output."
        )
    logger.debug("FX Graph -> StableHLO conversion successful")
    # Mark module as hierarchy-splitter compatible if hierarchy metadata was used
    if has_hierarchy:
        raw_module.operation.attributes["stablehlo.hierarchy_splitter_compatible"] = (
            mlir_ir.StringAttr.get("true", raw_module.context)
        )
    # Attach dynamic dimension constraint metadata
    dynamic_dim_constraints = build_dynamic_dim_constraints(gm.graph)
    if dynamic_dim_constraints is not None:
        _set_dynamic_dim_constraints(raw_module, dynamic_dim_constraints)
    # Set the aliasing attributes on the MLIR module
    _set_module_alias(raw_module, aliasing_info)
    return raw_module, io_spec, cast_spec
