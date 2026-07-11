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
from torch_neuronx.neuron_dynamo_backend.fx.fx_hooks import CustomFxImporterHooks
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronFxImporter
from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    ShardInputInfo,
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


def _detect_passthrough_aliases_from_module(module: Module) -> dict[int, int]:
    """Find pass-through aliases directly on the lowered StableHLO module.

    A function result that is *exactly* a block argument is an input returned
    unchanged (a pass-through). Detecting this on the final module — rather than
    trusting FX-graph placeholder/output indices — keeps the indices in the same
    space as the emitted function signature.

    This matters because lowering can drop or reorder graph inputs/outputs. For
    example, PT2.12 dynamo lifts module ``forward_pre_hook`` closure cells as
    extra FX placeholders; those are constant-folded out of the MLIR, so FX
    placeholder/output indices no longer line up with the MLIR function IO and
    the pre-computed (FX-space) alias indices land out of bounds.

    Returns:
        dict[int, int]: mapping of MLIR output index -> MLIR input (arg) index.
    """
    output_to_input: dict[int, int] = {}
    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        block = op.regions[0].blocks[0]
        for inner in block.operations:
            if inner.operation.name in ("func.return", "return"):
                for out_idx, operand in enumerate(inner.operands):
                    if mlir_ir.BlockArgument.isinstance(operand):
                        output_to_input[out_idx] = mlir_ir.BlockArgument(operand).arg_number
        break
    return output_to_input


def _set_module_alias(module: Module, aliasing_info: AliasingInfo) -> None:
    """Set aliasing attributes on the MLIR module for input-output tensor aliasing.

    Attaches mhlo.input_output_alias attributes to the module operation,
    informing the compiler which output tensors should alias input tensors.

    Aliases are always resolved against the ACTUAL lowered StableHLO IO:
      1. Pass-through aliases are read straight off the module (a result that is
         a block argument), which is inherently in the correct MLIR index space.
      2. Any explicitly-provided aliases (in-place mutation / NKI / pre-AOT
         analysis, expressed in FX-graph index space) are applied only when they
         still map validly onto the real MLIR IO. A stale / out-of-range hint is
         skipped instead of applied at the wrong index — an aliasing hint is a
         buffer-donation optimization, so a mis-indexed alias would be incorrect
         while a missing one is merely a lost optimization.

    Args:
        module (Module): MLIR Module to modify.
        aliasing_info (AliasingInfo): Aliasing relationships to apply.
    """
    is_tuple_output = _is_tuple_return(module)

    with module.context:
        function_io = parse_module_io(module)
        inputs = function_io.inputs
        outputs = function_io.outputs

        # MLIR output index -> (input param_num, parameter_index) to alias.
        output_to_alias: dict[int, tuple[int, list[int]]] = {}
        # Output indices whose alias came from an explicit in-place mutation
        # (aliasing_info), as opposed to a detected pass-through. Used to prefer
        # the mutation alias when one input parameter is claimed by both.
        mutation_output_indices: set[int] = set()

        # (1) Ground-truth pass-throughs from the lowered module.
        # Gate to the AOT/compile context (aliasing_info present) only. In pure
        # eager per-op lowering (aliasing_info is None) an output that is a bare
        # pass-through/identity of an input uses a SEPARATE output buffer at
        # runtime, so emitting mhlo.input_output_alias donates the input buffer but
        # the graph never writes the output -> it reads uninitialized memory and the
        # op returns zeros/garbage. This hit integer ceil/trunc (and other int
        # no-ops that lower to identity); float ops are unaffected because they
        # compute a new value (not a bare block-arg pass-through). The compile/AOT
        # path reuses the input buffer for pass-throughs, so the alias is required
        # there and this detection stays active for it.
        if aliasing_info is not None:
            for output_idx, param_num in _detect_passthrough_aliases_from_module(module).items():
                if 0 <= param_num < len(inputs) and 0 <= output_idx < len(outputs):
                    output_to_alias[output_idx] = (param_num, [])

        # (2) Merge explicitly-provided aliases, validated against the real IO.
        if aliasing_info is not None:
            for alias in aliasing_info:
                param_num = alias.parameter_number
                param_idx = alias.parameter_index or []
                output_idx = alias.output_index
                if not isinstance(output_idx, int):
                    output_idx = output_idx[0] if output_idx else -1

                if not (0 <= param_num < len(inputs)) or not (0 <= output_idx < len(outputs)):
                    logger.debug(
                        "Skipping alias with indices outside the lowered MLIR IO: "
                        "param_num=%s output_idx=%s (inputs=%d, outputs=%d)",
                        param_num,
                        output_idx,
                        len(inputs),
                        len(outputs),
                    )
                    continue
                if output_idx in output_to_alias:
                    # Already covered by a ground-truth pass-through.
                    continue
                if not shapes_match(inputs[param_num], outputs[output_idx]):
                    logger.warning(
                        "Dropping alias: output[%d] -> input[%d] shape mismatch "
                        "(output_shape=%s, input_shape=%s)",
                        output_idx,
                        param_num,
                        outputs[output_idx],
                        inputs[param_num],
                    )
                    continue
                output_to_alias[output_idx] = (param_num, param_idx)
                mutation_output_indices.add(output_idx)

        if not output_to_alias:
            return

        # (3) Emit mhlo.input_output_alias from the validated map.
        #
        # GATED FIX: an input buffer may be donated to at most ONE output. XLA's
        # HLO verifier rejects a second alias on the same parameter (RET_CHECK
        # param_has_seen[parameter_number] == false, at
        # hlo_input_output_alias_config.cc). This happens when one input feeds two
        # aliased outputs -- e.g. an in-place-mutated input that AOTAutograd also
        # returns as a bare pass-through (saved for backward) at a different output.
        #
        # When a parameter is claimed by both a mutation and a pass-through, the
        # MUTATION alias must win: donating the input buffer to the pass-through
        # output while the mutation's copy-back overwrites that same input would
        # corrupt the pass-through (e.g. a saved-for-backward tensor). We therefore
        # emit mutation-sourced outputs first, then skip any later alias on an
        # already-donated parameter.
        alias_dicts = []
        seen_params: set[int] = set()
        emit_order = sorted(output_to_alias, key=lambda o: (o not in mutation_output_indices, o))
        for output_idx in emit_order:
            param_num, param_idx = output_to_alias[output_idx]

            if param_num in seen_params:
                continue

            # StableHLO does not recognize mismatch-shaped alias relationships.
            if not shapes_match(inputs[param_num], outputs[output_idx]):
                continue

            seen_params.add(param_num)

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


def convert_fx_to_stablehlo(
    gm: torch.fx.GraphModule,
    example_inputs,
    aliasing_info: AliasingInfo = None,
    *,
    job: CompilationJob | None = None,
    shard_input_info: ShardInputInfo | None = None,
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
    io_spec = parse_module_io(raw_module, shard_input_info=shard_input_info)
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
    # Set the aliasing attributes on the MLIR module
    _set_module_alias(raw_module, aliasing_info)
    return raw_module, io_spec, cast_spec
