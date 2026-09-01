"""Dynamo skip registry — tests that cannot run on Neuron."""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import skipped_tests as s

SKIP_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[PT2.13] Hangs indefinitely under torch.compile(neuron): one xdist worker"
            " stalls on this test forever, holding the dynamo suite at 99% until the"
            " ~70-min job timeout (KeyboardInterrupt). Deterministic — the sole"
            " started-but-never-finished test across full runs."
        ),
        tests=[
            "test_exception_contract.py::*::test_compile_preserves_index_error",
        ],
    ),
    SkipGroup(
        reason="Test subprocess timeout on Neuron (exceeds 600s limit)",
        tests=[
            "TestFxGraphRunnableMultiProcessGroup::test_multiple_process_groups",
        ],
    ),
    SkipGroup(
        reason=("[PT2.11] Worker crash on Neuron during logging source-tracing test."),
        tests=[
            "test_trace_source_if_stmt",
        ],
    ),
    SkipGroup(
        reason=(
            "StableHLO broadcast_in_dim requires static shapes,"
            " Neuron does not support unbounded dynamism"
        ),
        tests=s.DYNAMO_DYNAMIC_SHAPES_UNSUPPORTED,
    ),
    SkipGroup(
        reason=(
            "[PT2.11] Torch-MLIR lowering gap: failed to legalize unresolved"
            " materialization from 'index' to 'tensor<i64>' for dynamic shape"
            " broadcasting in Inductor repros."
        ),
        tests=[
            "test_inductor_dynamic_shapes_broadcasting",
        ],
    ),
    SkipGroup(
        reason=("Symbolic handling unsupported"),
        tests=s.DYNAMO_SYMBOLIC_HANDLING_UNSUPPORTED,
    ),
    SkipGroup(
        reason=("Higher-order operation 'cond' not implemented in the FxImporter"),
        tests=s.DYNAMO_CONDITIONAL_BRANCHING_UNSUPPORTED,
    ),
    SkipGroup(
        reason=(
            "[PT2.11] Higher-order operation 'with_effects' not implemented in the"
            " FxImporter. Inserted by Dynamo's leaf-function decorator for tracking"
            " effect tokens; needs a passthrough/strip pass or FxImporter handler."
        ),
        patterns=[
            "test_leaf_function_*",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.11] CUDA-specific tests that assume CUDA device_type or rely on"
            " CUDA-only APIs/semantics with no Neuron equivalent (torch.cuda"
            " stream/event APIs, torch.autocast(\"cuda\"), hardcoded 'cuda'"
            " assertions). Not applicable to Neuron."
        ),
        tests=s.DYNAMO_CUDA_SPECIFIC_UNSUPPORTED,
    ),
    SkipGroup(
        reason=(
            "Tests run under dynamic=True, which makes scalar operands SymInts and"
            " tensor shapes symbolic. "
        ),
        tests=[
            "*DecoratorTests::test_nonstrict_trace_tuple_and_sym_int_output*",
            "ExportTests::test_dupes_and_bypass_reorder_with_non_tensor_arg",
            "ExportTests::test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph",
            "ExportTests::test_dupes_and_bypass_with_non_tensor_arg",
            "ExportTests::test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph",
            "ExportTests::test_dupes_and_bypass_with_non_tensor_output",
            "ExportTests::test_dupes_and_bypass_with_non_tensor_output_with_aten_graph",
            "*FunctionTests::test_pos*",
            "InlineAndInstallExportTests::test_dupes_and_bypass_reorder_with_non_tensor_arg_inline_and_install",
            "InlineAndInstallExportTests::test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph_inline_and_install",
            "InlineAndInstallExportTests::test_dupes_and_bypass_with_non_tensor_arg_inline_and_install",
            "InlineAndInstallExportTests::test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph_inline_and_install",
            "InlineAndInstallExportTests::test_dupes_and_bypass_with_non_tensor_output_inline_and_install",
            "InlineAndInstallExportTests::test_dupes_and_bypass_with_non_tensor_output_with_aten_graph_inline_and_install",
            "*MiscTests::test_dynamic_shapes_as_strided*",
            "*MiscTests::test_make_contiguous_strides_for_under_compile*",
            "*ReproTests::test_add_sub_alpha_out*",
            "*ReproTests::test_addr_alpha_beta_out*",
            "*ReproTests::test_dynamic_shape_disable_duck_size*",
            "*UnspecTests::test_unspec_float_precision*",
            "SubclassTests::test_tensor_subclass_TwoTensor_nested",
            "TestTwoTensorSubclass::test_tensor_subclass_TwoTensor_nested",
        ],
    ),
    SkipGroup(
        reason=(
            "Scalar operands become SymInts via automatic dynamic (a scalar value"
            " changes across calls, triggering recompilation with a symbolic int)."
        ),
        tests=[
            "*DictTests::test_dict_tag_guard*",
            "*DictTests::test_mapping_proxy_existing*",
            "*FunctionTests::test_dict_tuple_lazy_guard*",
            "*MiscTests::test_itertools_infinite_count*",
            "*ReproTests::test_guard_with_tuple_mutation*",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.11] Segmentation fault on Neuron. These tests crash the worker"
            " process during execution."
        ),
        tests=s.DYNAMO_SEGMENTATION_FAULT,
    ),
    SkipGroup(
        reason=(
            "Dynamo regression test for tensor metadata segfault"
            " (pytorch/pytorch#180741). Test repros the segfault scenario on"
            " Neuron and crashes the worker."
        ),
        tests=[
            "test_repros.py::*ReproTests::test_guard_tag_safe_tensor_metadata_segfault*",
        ],
    ),
    SkipGroup(
        reason=(
            "Triton not available on Neuron backend. These decomposition-numerics"
            " tests call pytest.xfail('Triton not available') at runtime; Neuron"
            " has no Triton backend, so skip them rather than record a runtime"
            " xfail that can never flip to pass."
        ),
        tests=[
            "TestDynamoDecompositionsNumericsNEURON::test_add_scalar_alpha_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_add_tensor_alpha_fma_matches_aten_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_add_tensor_alpha_numerics_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcdiv__backward_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcdiv__tensor_value_backward_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcdiv_tensor_value_numerics_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul__backward_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul__tensor_value_backward_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul_scalar_value_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul_tensor_value_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul_tensor_value_numerics_neuron",
            "TestDynamoDecompositionsNumericsNEURON::test_addcmul_value_1_fma_matches_aten_neuron",
        ],
    ),
    SkipGroup(
        reason=(
            "neuronx-cc rejects a dynamic shape (error NCC_EMOD025). Each test"
            " compiles a graph that has an unbounded parameter dimension, for"
            " example f32[?] or s64[?,?]. The Neuron compiler needs static shapes."
            " These tests cannot pass, so skip them."
        ),
        tests=[
            "FunctionTests::test_foreach_norm_inf_dynamic",
            "NestedGraphBreaksFunctionTests::test_foreach_norm_inf_dynamic_nested_graph_breaks",
            "MiscTests::test_dim_order",
            "MiscTests::test_item_changes_new_shape",
            "NestedGraphBreaksMiscTests::test_item_changes_new_shape_nested_graph_breaks",
            "NestedGraphBreaksDefaultsTests::test_is_tensor_tensor_nested_graph_breaks",
            "ReproTests::test_randint_out_dynamic",
            "NestedGraphBreaksReproTests::test_randint_out_dynamic_nested_graph_breaks",
        ],
    ),
    SkipGroup(
        reason=(
            "neuronx-cc cannot convert the StableHLO module to HLO (error"
            " NCC_EMOD023). Each test builds a dynamic or data-dependent shape"
            " graph (automatic dynamic, unbacked SymInts, or a shape taken from"
            " tensor values). The StableHLO then has dynamic-shape operations that"
            " HLO does not support. The Neuron compiler needs static shapes, so"
            " these tests cannot pass. Skip them."
        ),
        tests=[
            "DefaultsTests::test_is_vmapped_mutated_tensor_tensor",
            "NestedGraphBreaksDefaultsTests::test_is_vmapped_mutated_tensor_tensor_nested_graph_breaks",
            "MiscTests::test_bound_shape_checks",
            "ReproTests::test_ones_out_dynamic",
            "NestedGraphBreaksReproTests::test_ones_out_dynamic_nested_graph_breaks",
            "ReproTests::test_zeros_out_dynamic",
            "NestedGraphBreaksReproTests::test_zeros_out_dynamic_nested_graph_breaks",
            "ReproTests::test_unsqueeze_mul_strides",
            "UnspecTests::test_exponential",
            "NestedGraphBreaksUnspecTests::test_exponential",
            "NestedGraphBreaksUnspecTests::test_exponential_nested_graph_breaks",
            "UnspecTests::test_argmin_coerces_symint_to_intlist_spec",
            "UnspecTests::test_mark_unbacked",
            "UnspecTests::test_mark_unbacked_channels_last",
            "ViewTests::test_tensor_view_with_tensor_args",
            "ViewTests::test_tensor_view_with_tensor_shape_params",
            "ViewTests::test_torch_reshape_with_tensor_shape_params",
            "ViewTests::test_view_with_tensor_shape_params",
        ],
    ),
]
