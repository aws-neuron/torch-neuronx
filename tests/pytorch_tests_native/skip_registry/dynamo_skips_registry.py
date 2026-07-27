"""Dynamo skip registry — tests that cannot run on Neuron."""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup
from tests.pytorch_tests_native.skip_registry import skipped_tests as s

SKIP_GROUPS: list[SkipGroup] = [
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
            " CUDA-only APIs/semantics with no Neuron equivalent (hardcoded 'cuda'"
            " assertions, torch.cuda.HalfTensor, torch.cuda.get_device_properties,"
            " CUDA memory-stats, CUDA stream/event graphs). Not applicable to Neuron."
        ),
        tests=s.DYNAMO_CUDA_SPECIFIC_UNSUPPORTED,
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
            "test_repros.py::ReproTests::test_guard_tag_safe_tensor_metadata_segfault",
        ],
    ),
]
