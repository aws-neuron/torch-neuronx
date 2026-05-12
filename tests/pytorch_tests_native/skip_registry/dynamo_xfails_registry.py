from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import xfail_tests as x

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[PT2.11] Autocast SDPA fails Torch-MLIR partial lowering from TorchFX"
            " IR -> StableHLO IR: 'torch.operator' explicitly marked illegal."
        ),
        tests=[
            "test_autocast_sdpa",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.11] test_inplace_ops: Numerical tolerance mismatch on Neuron"
            " hardware. Gradient metadata propagation differs from CPU reference"
            " on bfloat16/float16 for various inplace ops (fmod, lerp, etc.)."
        ),
        patterns=[
            "test_inplace_ops_propagate_requires_grad_metadata_*_neuron_bfloat16",
            "test_inplace_ops_propagate_requires_grad_metadata_*_neuron_float16",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.11] Test expects RuntimeError under FakeTensorMode that the"
            " Neuron backend does not raise (compilation succeeds instead)."
        ),
        tests=[
            "test_fake_tensor_mode_raises_error",
        ],
    ),
    # TODO: Upstream fix needed - tests assert on log messages, which are different
    #   for neuron backend because of differences in the callstack or backend function
    #   itself
    SkipGroup(
        reason=(
            "Error/log message format mismatch: Neuron backend produces "
            "extra stack frames in graph-break and error messages"
        ),
        tests=x.DYNAMO_ERROR_LOG_FORMAT_MISMATCH,
    ),
    SkipGroup(
        reason=(
            "Log message side-effects format mismatch: Neuron backend "
            "produces extra stack frames in mutation logs"
        ),
        tests=[
            "LoggingTests::test_side_effects_dict_mutations",
            "LoggingTests::test_side_effects_multiple_mutations_same_object",
            "LoggingTests::test_side_effects_nn_module_buffer",
        ],
    ),
    SkipGroup(
        reason=("Inductor-specific logging/output tests fail on " "Neuron backend"),
        tests=[
            "LoggingTests::test_inductor_error",
            "LoggingTests::test_inductor_info",
            "LoggingTests::test_kernel_code",
            "LoggingTests::test_output_code",
            "StructuredTraceTest::test_dump_file",
            "TestBackendOverrideIntegrationNEURON::test_multiple_rules_neuron",
        ],
    ),
    SkipGroup(
        reason=("Test output includes device='neuron:0' in string " "representation"),
        tests=[
            "ReorderLogsTests::test_constant_mutation",
            "ReorderLogsTests::test_reorder_print",
            "ReorderLogsTests::test_reorder_print_graph_break",
        ],
    ),
    # TODO: Upstream fix needed - test input/output devices will mismatch
    #   with the devices of results from the neuron backend
    SkipGroup(
        reason="index_put: Non-scalar tensor on cpu device, expected neuron",
        tests=[
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_put_neuron_bfloat16",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_put_neuron_float16",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_put_neuron_float32",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_put_neuron_float64",
        ],
    ),
    SkipGroup(
        reason=(
            "Device mismatch: test expects cpu output but gets neuron:0 "
            "(autocast cpu graph break)"
        ),
        tests=[
            "CtxManagerTests::test_autocast_cpu_graph_break",
            "CtxManagerTests::test_autocast_cpu_graph_break_inner_fn",
            "DynamicShapesCtxManagerTests::test_autocast_cpu_graph_break_inner_fn_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Gradient device mismatch: neuron gradient assigned " "to cpu tensor"),
        tests=x.DYNAMO_GRADIENT_NEURON_TO_CPU,
    ),
    SkipGroup(
        reason="Device mismatch: mixed neuron:0 and cpu tensors",
        tests=[
            "DynamicShapesFunctionTests::test_itertools_compress_tensors_dynamic_shapes",
        ],
    ),
    # TODO: Upstream fix needed - test hardcodes cuda-specific behavior
    #   that must be modified to be device-agnostic
    SkipGroup(
        reason=("Neuron stream/event type mismatch in CUDA " "stream/event tests"),
        tests=x.DYNAMO_STREAM_EVENT_TYPE_MISMATCH,
    ),
    SkipGroup(
        reason="Test requires CUDA driver (record_stream / stream backward)",
        tests=[
            "TestStreams::test_record_stream_problem_basic",
            "TestStreams::test_record_stream_problem_interleaved",
            "TestStreams::test_stream_backward_simple",
        ],
    ),
    SkipGroup(
        reason="Hardcoded CUDA device reference: test uses .to('cuda') directly",
        tests=[
            "DynamicShapesFunctionTests::test_get_device_properties_tensor_device_dynamic_shapes",
            "FunctionTests::test_get_device_properties_tensor_device",
        ],
    ),
    SkipGroup(
        reason="Test hardcodes 'cuda' device string comparison",
        tests=[
            "CtxManagerTests::test_cuda_amp_autocast",
        ],
    ),
    # TODO: Upstream fix needed - test expects autocast behavior matching CPU, but
    #   neuron autocast behavior is different
    SkipGroup(
        reason=("Autocast dtype mismatch: Neuron does not support " "expected autocast dtype"),
        tests=[
            "CtxManagerTests::test_autocast_device",
            "CtxManagerTests::test_autocast_float64",
            "DynamicShapesCtxManagerTests::test_autocast_float64_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=(
            "Torch-MLIR scalar-tensor type mismatch: aten.add.Tensor/aten.mul.Tensor "
            "operand must be a Torch tensor type but got a scalar"
        ),
        tests=x.DYNAMO_SCALAR_TENSOR_TYPE_MISMATCH,
    ),
    SkipGroup(
        reason=("Torch-MLIR unresolved materialization between i64 and tensor<i64> types"),
        tests=x.DYNAMO_UNRESOLVED_MATERIALIZATION_I64,
    ),
    SkipGroup(
        reason=("Torch-MLIR return type mismatch: tensor<i64> vs i64"),
        tests=[
            "DecoratorTests::test_assume_constant_result_on_user_defined_fn",
            "NbIndexTests::test_tensor_int_index",
            "DynamicShapesAotAutogradFallbackTests::test_aot_autograd_expand_mutation_backwards_dynamic_shapes",
            "DynamicShapesAotAutogradFallbackTests::test_aot_autograd_expand_mutation_functionalizes_dynamic_shapes",
            "DynamicShapesReproTests::test_intermediate_leaf_requires_grad_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR reduction region element-type mismatch in select_scatter"),
        tests=[
            "DynamicShapesReproTests::test_select_scatter_mixed_dtype_dynamic_shapes",
            "ReproTests::test_select_scatter_mixed_dtype",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR stablehlo.scatter dimension bounds mismatch"),
        tests=[
            "DynamicShapesReproTests::test_setitem_tensor_prop_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR stablehlo.concatenate inferred type incompatibility"),
        tests=[
            "MiscTests::test_scalar_tensor_is_equivalent_to_symint_list_argument",
        ],
    ),
    SkipGroup(
        reason=(
            "Gradient device mismatch: CompiledFunctionBackward returns "
            "gradients on neuron:0 but framework expects cpu"
        ),
        tests=x.DYNAMO_GRADIENT_DEVICE_MISMATCH,
    ),
    SkipGroup(
        reason=("Device mismatch: mixed neuron:0 and cpu tensors"),
        tests=x.DYNAMO_DEVICE_MISMATCH_MIXED,
    ),
    SkipGroup(
        reason=("Torch-MLIR failed to legalize 'operator'" "operator missing from Torch dialect"),
        tests=x.DYNAMO_FAILED_TO_LEGALIZE_OPERATOR,
    ),
    SkipGroup(
        reason=(
            "Torch-MLIR failed to legalize runtime.assert"
            "Runtime asserts embedded in the graph are not supported"
        ),
        tests=x.DYNAMO_FAILED_TO_LEGALIZE_RUNTIME_ASSERT,
    ),
    SkipGroup(
        reason=("Torch-MLIR failed to legalize tensor.dim"),
        tests=x.DYNAMO_FAILED_TO_LEGALIZE_TENSOR_DIM,
    ),
    SkipGroup(
        reason=("Torch-MLIR failed to legalize constant.int"),
        tests=x.DYNAMO_FAILED_TO_LEGALIZE_CONSTANT_INT,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.Int"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_INT,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.gather"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_GATHER,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.add"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_ADD,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.zeros"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_ZEROS,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.scatter_add"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_SCATTER_ADD,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.cumsum"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_CUMSUM,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.uniform"),
        tests=[
            "DynamicShapesReproTests::test_tensor_uniform_dynamic_shapes",
            "ReproTests::test_randint_out_dynamic",
            "UnspecTests::test_exponential",
            "UnspecTests::test_shape_graph_break",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.ones"),
        tests=[
            "DynamicShapesReproTests::test_list_reverse_dynamic_shapes",
            "DynamicShapesReproTests::test_ones_out_dynamic_dynamic_shapes",
            "ReproTests::test_ones_out_dynamic",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.xlogy"),
        tests=x.DYNAMO_LOWERING_FAILURE_ATEN_XLOGY,
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.Float"),
        tests=[
            "DefaultsTests::test_cast_tensor_single_elem",
            "UnspecTests::test_symfloat_to_tensor",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.squeeze"),
        tests=[
            "DynamicShapesFunctionTests::test_unary_fold_op_dynamic_shapes",
            "DynamicShapesFunctionTests::test_tensor_type5_dynamic_shapes",
            "DynamicShapesMiscTests::test_scalar_tensor_is_equivalent_to_symint_argument_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR failed to legalize constant.bool"),
        tests=[
            "DynamicShapesMiscTests::test_unique_consecutive_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.view"),
        tests=[
            "DynamicShapesReproTests::test_view_dtype_overload_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR lowering failure at aten.mul"),
        tests=[
            "SubclassTests::test_tensor_subclass_TwoTensor_return_shape",
            "DynamicShapesReproTests::test_user_defined_iter_dynamic_shapes",
            "DynamicShapesReproTests::test_as_strided_on_base_with_mutation_works_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Torch-MLIR failed to legalize: aten.eq"),
        tests=[
            "HigherOrderOpVmapGuardTests::test_vmap_guard_fail_different_state",
            "DynamicShapesMiscTests::test_ne_operator_with_custom_graphbreak_eq_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("HOP 'auto_functionalized_v2' not implemented in the FxImporter"),
        tests=[
            "DynamicShapesMiscTests::test_out_variant_custom_op_dynamic_shapes",
            "MiscTests::test_generate_trivial_abstract_impl",
            "MiscTests::test_out_variant_custom_op",
            "NestedGraphBreakLoggingTests::test_nested_generic_ctx_mgr",
        ],
    ),
    SkipGroup(
        reason=("HOP 'invoke_subgraph' not implemented in the FxImporter"),
        tests=[
            "AOTAutogradCacheBundledTests::test_invoke_subgraph",
            "AOTAutogradCacheTests::test_invoke_subgraph",
            "GraphDededuplicationTests::test_param_transfer_to_submodule",
            "GraphDededuplicationTests::test_tuple_return",
        ],
    ),
    SkipGroup(
        reason=("HOP 'control_deps' not implemented in the FxImporter"),
        tests=[
            "TestStreams::test_event_synchronize_control_deps_e2e",
            "TestStreams::test_event_synchronize_e2e",
            "TestStreams::test_event_synchronize_inductor_lowering",
            "TestStreams::test_inductor_lowering",
        ],
    ),
    SkipGroup(
        reason=("HOP 'run_and_save_rng_state' not implemented in the FxImporter"),
        tests=[
            "HOPCacheTests::test_checkpoint_with_dropout_caches_correctly",
        ],
    ),
    SkipGroup(
        reason=("HOP 'map_impl' not implemented in the FxImporter"),
        tests=[
            "DynamicShapesMiscTests::test_map_with_quantization_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Missing fx_graph_runnable payload"),
        tests=[
            "FxGraphRunnableTest::test_all_gather_collective",
            "FxGraphRunnableTest::test_basic_tensor_add",
            "FxGraphRunnableTest::test_broadcast_collective",
            "FxGraphRunnableTest::test_toy_model_basic",
        ],
    ),
    SkipGroup(
        reason=("AoTAutogradCache not supported: expected OutputCode, got function"),
        tests=x.DYNAMO_AOT_AUTOGRAD_CACHE_OUTPUT_CODE,
    ),
    SkipGroup(
        reason=("AoTAutogradCache not supported: cache hit/miss count mismatch"),
        tests=x.DYNAMO_AOT_AUTOGRAD_CACHE_HIT_MISS,
    ),
    SkipGroup(
        reason=("Tensor with empty dimensions"),
        tests=x.DYNAMO_TENSOR_EMPTY_DIMENSIONS,
    ),
    SkipGroup(
        reason=("Numerical precision mismatch on Neuron for inplace ops"),
        tests=x.DYNAMO_NUMERICAL_PRECISION_INPLACE_OPS,
    ),
    SkipGroup(
        reason="aten._to_copy.default dispatch failure for tensor subclass on Neuron",
        tests=x.DYNAMO_TO_COPY_DISPATCH_FAILURE,
    ),
    SkipGroup(
        reason=("Numerical precision or value mismatch on Neuron " "(non-inplace-op tests)"),
        tests=x.DYNAMO_NUMERICAL_PRECISION_NON_INPLACE,
    ),
    SkipGroup(
        reason="vmap recompile guard tests: RecompileError not raised on Neuron",
        tests=x.DYNAMO_VMAP_RECOMPILE_GUARD,
    ),
    SkipGroup(
        reason="vmap guard recompilation count is 0 on Neuron",
        tests=[
            "HigherOrderOpVmapGuardTests::test_vmap_grad_vmap_guard_fail",
            "HigherOrderOpVmapGuardTests::test_vmap_guard_fail",
            "HigherOrderOpVmapGuardTests::test_vmap_guard_ok",
            "HigherOrderOpVmapGuardTests::test_vmap_recompile_different_states",
        ],
    ),
    SkipGroup(
        reason="Stream sync_dealloc not emitted by Neuron backend",
        tests=[
            "TestStreams::test_del_attr_multi_stream_sync_dealloc",
            "TestStreams::test_del_multi_stream_sync_dealloc",
            "TestStreams::test_del_subscr_multi_stream_sync_dealloc",
        ],
    ),
    SkipGroup(
        reason=("Test expects BackendCompilerFailed but Neuron backend " "does not raise it"),
        tests=[
            "AOTAutogradCacheBundledTests::test_compiled_autograd_bypass",
            "AOTAutogradCacheBundledTests::test_unsafe_mark_cacheable",
            "AOTAutogradCacheTests::test_compiled_autograd_bypass",
        ],
    ),
    SkipGroup(
        reason="Neuron backend wrapper is not picklable",
        tests=[
            "OptimizedModuleTest::test_save_and_load_all_backends",
            "OptimizedModuleTest::test_save_and_load_inductor",
        ],
    ),
    SkipGroup(
        reason="Neuron backend error: name 'sparsity' is not defined",
        tests=[
            "GetItemTests::test_opaque_object_getitem",
            "HooksTests::test_functools_arg_vary",
            "HooksTests::test_intermediary_hooks_same_on_aot_eager",
            "HooksTests::test_intermediary_hooks_same_on_inductor",
            "HooksTests::test_intermediate_hook_with_closure_aot",
        ],
    ),
    SkipGroup(
        reason="Neuron tuple index out of range for masked_fill/scatter ops",
        tests=x.DYNAMO_TUPLE_INDEX_OUT_OF_RANGE,
    ),
    SkipGroup(
        reason="Neuron compilation failure for addmm op",
        tests=x.DYNAMO_ADDMM_COMPILATION_FAILURE,
    ),
    SkipGroup(
        reason="index_copy: list assignment index out of range on Neuron",
        tests=[
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_copy_neuron_bfloat16",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_copy_neuron_float16",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_copy_neuron_float32",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_index_copy_neuron_float64",
        ],
    ),
    SkipGroup(
        reason="Neuron NEFF param_num / output_idx out of bounds",
        tests=[
            "AOTAutogradCacheBundledTests::test_dtensor_cache_hit",
            "MiscTests::test_builtin_abs",
            "TestAOTCompile::test_aot_compile_with_redistribute",
            "TestNestedTensor::test_basic_autograd",
        ],
    ),
    SkipGroup(
        reason="_copy_from_and_resize size mismatch on Neuron",
        tests=[
            "DynamicShapesReproTests::test_sort_out_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason="RecursionError: maximum recursion depth exceeded on Neuron",
        tests=[
            "DynamicShapesReproTests::test_tokenization_dynamic_shapes",
            "MiscTests::test_sourceless_namedtuple",
        ],
    ),
    SkipGroup(
        reason=("conv_transpose / functorch correctness failure on Neuron"),
        tests=[
            "ContextlibContextManagerTests::test_functorch_low_level_gb_False",
            "DynamicShapesNNModuleTests::test_conv_transpose_call_forward_directly_dynamic_shapes",
            "DynamicShapesNNModuleTests::test_conv_transpose_call_super_forward_directly_dynamic_shapes",
            "NNModuleTests::test_conv_transpose_call_forward_directly",
            "NNModuleTests::test_conv_transpose_call_super_forward_directly",
        ],
    ),
    SkipGroup(
        reason=("Neuron backend returns unexpected number of outputs " "or wrong error type"),
        tests=[
            "AotAutogradFallbackTests::test_mutation",
            "DefaultsTests::test_is_vmapped_mutated_tensor_tensor",
            "DynamicShapesAotAutogradFallbackTests::test_call_fn_with_non_const_inputs_aot_unsafe_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("Neuron ShardIndexInjection pass error for " "arange with float32 dtype"),
        tests=[
            "DynamicShapesMiscTests::test_arange_length_with_float32_dtype_dynamic_shapes",
            "MiscTests::test_arange_length_with_float32_dtype",
            "UnspecTests::test_mark_unbacked_hint_consistency",
        ],
    ),
    SkipGroup(
        reason=("Autograd engine stream assertion failure on Neuron"),
        tests=[
            "AOTAutogradCacheTests::test_aot_runtime_trace_joint",
        ],
    ),
    SkipGroup(
        reason=("Double backward error message mismatch on Neuron"),
        tests=[
            "DynamicShapesAotAutogradFallbackTests::test_double_backward_with_compile_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=(
            "execnet DumpError: can't serialize tensor or " "builtin_function_or_method with xdist"
        ),
        tests=[
            "DynamicShapesFunctionTests::test_compare_constant_and_tensor_dynamic_shapes",
            "FunctionTests::test_itemgetter",
            "FunctionTests::test_methodcaller",
            "FunctionTests::test_operator_concat",
            "FunctionTests::test_unary_fold_op_seq",
        ],
    ),
    SkipGroup(
        reason=("FakeTensor / meta tensor device propagation failure " "on Neuron"),
        tests=[
            "DynamicShapesReproTests::test_inplace_add_from_meta_tensor_factory_dynamic_shapes",
            "ReproTests::test_inplace_add_from_meta_tensor_factory",
        ],
    ),
    SkipGroup(
        reason="HOP 'auto_functionalized_v2' not implemented in the FxImporter",
        tests=[
            "DynamicShapesMiscTests::test_generate_trivial_abstract_impl_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason=("CompiledFunctionBackward returns incorrect number " "of gradients on Neuron"),
        tests=[
            "DynamicShapesMiscTests::test_compile_with_userland_fake_tensor_mode_dynamic_shapes",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_addmm_decomposed_neuron_float64",
            "TestTensorMetaPropNEURON::test_inplace_ops_propagate_requires_grad_metadata_addmm_neuron_float64",
        ],
    ),
    SkipGroup(
        reason="Unsupported: Failed to convert args/kwargs to proxy",
        tests=[
            "DynamicShapesReproTests::test_issue164247_backend_eager_dynamic_shapes",
        ],
    ),
    SkipGroup(
        reason="Unsupported: Attempt to trace generator",
        tests=[
            "HooksTests::test_post_acc_grad_hook",
        ],
    ),
    SkipGroup(
        reason="AoTAutogradCache not supported: cache hit/miss count mismatch",
        tests=[
            "AOTAutogradCacheBundledTests::test_fx_graph_cache_off",
            "AOTAutogradCacheTests::test_fx_graph_cache_off",
        ],
    ),
    SkipGroup(
        reason="Distributed rank not set",
        tests=[
            "FxGraphRunnableTest::test_all_reduce_collective",
        ],
    ),
    SkipGroup(
        reason="Assertion failure on Neuron: multithread compile dynamic",
        tests=[
            "MiscTests::test_multithread_compile_dynamic",
        ],
    ),
    SkipGroup(
        reason="InternalTorchDynamoError: KeyError or AttributeError in guard/getattr",
        tests=[
            "GuardManagerTests::test_dimension_marking_guard_dependent_attrs",
            "OptimizedModuleTest::test_dict_insertion_guard_method_func",
        ],
    ),
    SkipGroup(
        reason="Expected RuntimeError not raised on Neuron",
        tests=[
            "ReproTests::test_one_hot_bounds_check_compiled",
        ],
    ),
    SkipGroup(
        reason="FunctionalTensor usage without FunctionalTensorMode",
        tests=[
            "ReproTests::test_trace_functional_tensor_with",
        ],
    ),
    SkipGroup(
        reason="Invalid device argument on Neuron",
        tests=[
            "ReproTests::test_mem_leak_guards",
        ],
    ),
    SkipGroup(
        reason="SymInt proxy tracking failure on Neuron",
        tests=[
            "SubclassTests::test_tensor_subclass_metadata_with_symint",
        ],
    ),
    SkipGroup(
        reason="BytecodeDebugger verbose mode assertion failure on Neuron",
        tests=[
            "TestBytecodeDebugger::test_verbose_mode",
        ],
    ),
    SkipGroup(
        reason="Neuron backend compilation failure for NestedTensor unbind",
        tests=[
            "TestNestedTensor::test_unbind",
        ],
    ),
    SkipGroup(
        reason="Numerical or count mismatch on Neuron",
        tests=[
            "TestPackage::test_automatic_dynamo_graph_breaks_from_print_model_as_fn_device_cpu",
            "TestPackage::test_code_with_generator_device_cpu",
        ],
    ),
    SkipGroup(
        reason="Neuron compilation failure",
        tests=[
            "DynamicShapesAotAutogradFallbackTests::test_aot_autograd_expand_mutation_error_dynamic_shapes",
            "FunctionTests::test_quantize_per_tensor",
            "MiscTests::test_itertools_infinite_count",
            "NbFloatTests::test_tensor_dunder_float",
            "NbFloatTests::test_tensor_int_dtype",
            "NbIntTests::test_tensor_dunder_int",
            "NbIntTests::test_tensor_float_dtype",
            "NbIntTests::test_tensor_int_dtype",
        ],
    ),
    SkipGroup(
        reason="autograd.Function with functorch transforms requires setup_context",
        tests=[
            "AutogradFunctionFunctorchTests::test_old_style_autograd_function_with_grad_compiled",
        ],
    ),
    SkipGroup(
        reason="Unsupported: Failed to convert args/kwargs to proxy",
        tests=[
            "ReproTests::test_issue164247_backend_eager",
        ],
    ),
    SkipGroup(
        reason="Numerical precision mismatch on Neuron for float rounding",
        tests=[
            "UnspecTests::test_unspec_roundtrip_float_input",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.compile failure in neuron's dynamo backend. The FX→StableHLO"
            " lowering drops DTensor sentinel outputs/inputs (mesh, placement,"
            " layout) from the emitted function signature, but aliasing_info"
            " still holds the pre-lowering indices. We get IndexError"
            " (param_num/output_idx out of bounds)."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_basic",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_dynamo_device_mesh_attrs",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_from_local_with_extracted_device_mesh_dce",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_from_local_with_extracted_device_mesh_no_dce",
            "test_dtensor_compile.py::TestDTensorCompile::test_dtensor_input_mutations",
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_dtensor",
            "test_dtensor_compile.py::TestDTensorCompile::test_dynamo_to_local_kwargs_forward_hook",
            "test_dtensor_compile.py::TestDTensorCompile::test_get_local_rank_compile",
            "test_dtensor_compile.py::TestDTensorCompile::test_device_mesh_input_to_dtensor_from_local",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.compile failure from torch 2.13 API changes."
            " mesh_registry.py calls .nest() on the mesh layout, but torch 2.13"
            " replaced it with _FlatLayout which no longer has that method."
            " Eager paths tolerate the failed sub-mesh registration, but"
            " torch.compile's sharding propagation needs the PG at trace time,"
            " finds nothing, and emits a no-op graph — result is 100% mismatch."
        ),
        tests=[
            "test_dtensor_compile.py::TestDTensorCompile::test_device_mesh_slice",
        ],
    ),
    SkipGroup(
        reason=(
            "Dynamo cannot reconstruct FakeIdVariable across graph breaks."
            " id() on compile-time containers produces a fake ID that is not"
            " a real Python constant, so it cannot be reconstructed when"
            " Dynamo breaks the graph."
        ),
        tests=[
            "test_id.py::IdTests::test_id_builtin_type_list",
            "test_id.py::IdTests::test_id_sourceful_dict",
            "test_id.py::IdTests::test_id_sourceful_list",
            "test_id.py::IdTests::test_id_sourceful_tuple",
            "test_id.py::IdTests::test_id_sourceless_comparison",
        ],
    ),
    SkipGroup(
        reason=(
            "Dynamo internal bug: IndexError (tuple index out of range) when"
            " id() is called with wrong argument count."
        ),
        tests=[
            "test_id.py::IdTests::test_id_wrong_arg_count",
        ],
    ),
    SkipGroup(
        reason=(
            "Test compares memory addresses (id()) which differ across runs"
            " due to Neuron backend's tensor wrapping / process layout."
        ),
        tests=[
            "test_id.py::IdTests::test_id_functools_partial",
        ],
    ),
    SkipGroup(
        reason=("DictGuardManager does not support leaf_guard on Neuron."),
        tests=[
            "test_guard_manager.py::GuardManagerTests::test_dict_guard_manager",
        ],
    ),
    SkipGroup(
        reason=(
            "torch.compile graph capture fails on Neuron for dynamic ops"
            " (randint, isin, interpolate, module.to)."
        ),
        tests=[
            "MiscTestsDeviceNEURON::test_module_to_with_shared_weights_compile_neuron",
            "MiscTestsDeviceNEURON::test_randint_no_graphbreak_neuron",
            "MiscTestsDeviceNEURON::test_scalar_isin_decomposition_neuron",
            "MiscTestsDeviceNEURON::test_module_to_move_compile_neuron",
            "MiscTestsDeviceNEURON::test_interpolate_propagate_real_tensors_neuron",
        ],
    ),
]
