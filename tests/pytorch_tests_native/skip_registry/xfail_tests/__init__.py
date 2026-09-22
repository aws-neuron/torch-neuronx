from .accuracy.gradcheck import ACCURACY_GRADCHECK as ACCURACY_GRADCHECK
from .accuracy.jacobian_mismatch import ACCURACY_JACOBIAN_MISMATCH as ACCURACY_JACOBIAN_MISMATCH
from .accuracy.minmax_clamp_nan_propagation import (
    MINMAX_CLAMP_NAN_PROPAGATION as MINMAX_CLAMP_NAN_PROPAGATION,
)
from .accuracy.pt211_bf16_reciprocal_regression import (
    PT211_BF16_RECIPROCAL_REGRESSION as PT211_BF16_RECIPROCAL_REGRESSION,
)
from .accuracy.scalar_not_close import ACCURACY_SCALAR_NOT_CLOSE as ACCURACY_SCALAR_NOT_CLOSE
from .accuracy.scalar_not_equal import ACCURACY_SCALAR_NOT_EQUAL as ACCURACY_SCALAR_NOT_EQUAL
from .accuracy.tensor_not_close import ACCURACY_TENSOR_NOT_CLOSE as ACCURACY_TENSOR_NOT_CLOSE
from .accuracy.tensor_not_equal import ACCURACY_TENSOR_NOT_EQUAL as ACCURACY_TENSOR_NOT_EQUAL
from .assertion_misc.false_not_true import (
    ASSERTION_MISC_FALSE_NOT_TRUE as ASSERTION_MISC_FALSE_NOT_TRUE,
)
from .assertion_misc.tensor_false import ASSERTION_MISC_TENSOR_FALSE as ASSERTION_MISC_TENSOR_FALSE
from .compare_cpu_numerical import COMPARE_CPU_NUMERICAL as COMPARE_CPU_NUMERICAL
from .compiler_error.neuron import COMPILER_ERROR_NEURON as COMPILER_ERROR_NEURON
from .compiler_error.torch_mlir import COMPILER_ERROR_TORCH_MLIR as COMPILER_ERROR_TORCH_MLIR
from .decomp_logit import DECOMP_LOGIT as DECOMP_LOGIT
from .decomp_reference_precision import DECOMP_REFERENCE_PRECISION as DECOMP_REFERENCE_PRECISION
from .device_mismatch import DEVICE_MISMATCH_CPU_VS_NEURON as DEVICE_MISMATCH_CPU_VS_NEURON
from .dtensor.backend_type_missing import (
    DTENSOR_BACKEND_TYPE_MISSING as DTENSOR_BACKEND_TYPE_MISSING,
)
from .dtensor.rank_env_not_set import DTENSOR_RANK_ENV_NOT_SET as DTENSOR_RANK_ENV_NOT_SET
from .dtype_mismatch.float64_downcast import (
    DTYPE_MISMATCH_FLOAT64_DOWNCAST as DTYPE_MISMATCH_FLOAT64_DOWNCAST,
)
from .dtype_mismatch.object_comparison import (
    DTYPE_MISMATCH_OBJECT_COMPARISON as DTYPE_MISMATCH_OBJECT_COMPARISON,
)
from .dynamo_xfails import (
    DYNAMO_AOT_AUTOGRAD_CACHE_HIT_MISS as DYNAMO_AOT_AUTOGRAD_CACHE_HIT_MISS,
)
from .dynamo_xfails import (
    DYNAMO_DEVICE_MISMATCH_MIXED as DYNAMO_DEVICE_MISMATCH_MIXED,
)
from .dynamo_xfails import (
    DYNAMO_ERROR_LOG_FORMAT_MISMATCH as DYNAMO_ERROR_LOG_FORMAT_MISMATCH,
)
from .dynamo_xfails import (
    DYNAMO_FAILED_TO_LEGALIZE_CONSTANT_INT as DYNAMO_FAILED_TO_LEGALIZE_CONSTANT_INT,
)
from .dynamo_xfails import (
    DYNAMO_FAILED_TO_LEGALIZE_OPERATOR as DYNAMO_FAILED_TO_LEGALIZE_OPERATOR,
)
from .dynamo_xfails import (
    DYNAMO_FAILED_TO_LEGALIZE_RUNTIME_ASSERT as DYNAMO_FAILED_TO_LEGALIZE_RUNTIME_ASSERT,
)
from .dynamo_xfails import (
    DYNAMO_FAILED_TO_LEGALIZE_TENSOR_DIM as DYNAMO_FAILED_TO_LEGALIZE_TENSOR_DIM,
)
from .dynamo_xfails import (
    DYNAMO_GRADIENT_DEVICE_MISMATCH as DYNAMO_GRADIENT_DEVICE_MISMATCH,
)
from .dynamo_xfails import (
    DYNAMO_GRADIENT_NEURON_TO_CPU as DYNAMO_GRADIENT_NEURON_TO_CPU,
)
from .dynamo_xfails import (
    DYNAMO_GRAPH_STRUCTURE_ASSERTION_MISMATCH as DYNAMO_GRAPH_STRUCTURE_ASSERTION_MISMATCH,
)
from .dynamo_xfails import (
    DYNAMO_INPLACE_METADATA_PROPAGATION as DYNAMO_INPLACE_METADATA_PROPAGATION,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_ADD as DYNAMO_LOWERING_FAILURE_ATEN_ADD,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_CUMSUM as DYNAMO_LOWERING_FAILURE_ATEN_CUMSUM,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_GATHER as DYNAMO_LOWERING_FAILURE_ATEN_GATHER,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_INT as DYNAMO_LOWERING_FAILURE_ATEN_INT,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_SCATTER_ADD as DYNAMO_LOWERING_FAILURE_ATEN_SCATTER_ADD,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_XLOGY as DYNAMO_LOWERING_FAILURE_ATEN_XLOGY,
)
from .dynamo_xfails import (
    DYNAMO_LOWERING_FAILURE_ATEN_ZEROS as DYNAMO_LOWERING_FAILURE_ATEN_ZEROS,
)
from .dynamo_xfails import (
    DYNAMO_MISC_FAILURES as DYNAMO_MISC_FAILURES,
)
from .dynamo_xfails import (
    DYNAMO_NCC_EMOD023_COMPILATION_FAILURE as DYNAMO_NCC_EMOD023_COMPILATION_FAILURE,
)
from .dynamo_xfails import (
    DYNAMO_NESTED_GRAPH_BREAKS_WRAPPED as DYNAMO_NESTED_GRAPH_BREAKS_WRAPPED,
)
from .dynamo_xfails import (
    DYNAMO_NUMERICAL_PRECISION_INPLACE_OPS as DYNAMO_NUMERICAL_PRECISION_INPLACE_OPS,
)
from .dynamo_xfails import (
    DYNAMO_NUMERICAL_PRECISION_NON_INPLACE as DYNAMO_NUMERICAL_PRECISION_NON_INPLACE,
)
from .dynamo_xfails import (
    DYNAMO_SCALAR_TENSOR_TYPE_MISMATCH as DYNAMO_SCALAR_TENSOR_TYPE_MISMATCH,
)
from .dynamo_xfails import (
    DYNAMO_STREAM_EVENT_TYPE_MISMATCH as DYNAMO_STREAM_EVENT_TYPE_MISMATCH,
)
from .dynamo_xfails import (
    DYNAMO_TO_COPY_DISPATCH_FAILURE as DYNAMO_TO_COPY_DISPATCH_FAILURE,
)
from .dynamo_xfails import (
    DYNAMO_UNRESOLVED_MATERIALIZATION_I64 as DYNAMO_UNRESOLVED_MATERIALIZATION_I64,
)
from .dynamo_xfails import (
    DYNAMO_VMAP_RECOMPILE_GUARD as DYNAMO_VMAP_RECOMPILE_GUARD,
)
from .error_not_raised.assertion_error import (
    ERROR_NOT_RAISED_ASSERTION_ERROR as ERROR_NOT_RAISED_ASSERTION_ERROR,
)
from .error_not_raised.runtime_error import (
    ERROR_NOT_RAISED_RUNTIME_ERROR as ERROR_NOT_RAISED_RUNTIME_ERROR,
)
from .error_not_raised.warning import ERROR_NOT_RAISED_WARNING as ERROR_NOT_RAISED_WARNING
from .extremal_ref_numerics import EXTREMAL_REF_NUMERICS as EXTREMAL_REF_NUMERICS
from .foreach_different_devices import FOREACH_DIFFERENT_DEVICES as FOREACH_DIFFERENT_DEVICES
from .foreach_norm_integer import FOREACH_NORM_INTEGER as FOREACH_NORM_INTEGER
from .index_error import INDEX_ERROR as INDEX_ERROR
from .logit_integer_bool import LOGIT_INTEGER_BOOL as LOGIT_INTEGER_BOOL
from .meta_mismatch import META_MISMATCH_OUTPUT as META_MISMATCH_OUTPUT
from .other import OTHER as OTHER
from .output_stride_mismatch import OUTPUT_STRIDE_MISMATCH as OUTPUT_STRIDE_MISMATCH
from .pt214_misc_remaining import PT214_MISC_REMAINING as PT214_MISC_REMAINING
from .reduction_nan_drop import REDUCTION_NAN_DROP as REDUCTION_NAN_DROP
from .refs_executor_integer import REFS_EXECUTOR_INTEGER as REFS_EXECUTOR_INTEGER
from .runtime_misc.comparison import RUNTIME_MISC_COMPARISON as RUNTIME_MISC_COMPARISON
from .runtime_misc.module_input_count import (
    RUNTIME_MISC_MODULE_INPUT_COUNT as RUNTIME_MISC_MODULE_INPUT_COUNT,
)
from .runtime_misc.no_refs import RUNTIME_MISC_NO_REFS as RUNTIME_MISC_NO_REFS
from .runtime_misc.null_storage import RUNTIME_MISC_NULL_STORAGE as RUNTIME_MISC_NULL_STORAGE
from .runtime_misc.shape_broadcast import (
    RUNTIME_MISC_SHAPE_BROADCAST as RUNTIME_MISC_SHAPE_BROADCAST,
)
from .runtime_nrt_error import RUNTIME_NRT_ERROR as RUNTIME_NRT_ERROR
from .sqrt_not_ieee754 import SQRT_NOT_IEEE754 as SQRT_NOT_IEEE754
from .timeout import TIMEOUT as TIMEOUT
from .unknown import UNKNOWN as UNKNOWN
from .unsupported_op.cpu_fallback_disabled import (
    UNSUPPORTED_OP_CPU_FALLBACK_DISABLED as UNSUPPORTED_OP_CPU_FALLBACK_DISABLED,
)
from .unsupported_op.not_implemented import (
    UNSUPPORTED_OP_NOT_IMPLEMENTED as UNSUPPORTED_OP_NOT_IMPLEMENTED,
)
from .unsupported_op.sparse import UNSUPPORTED_OP_SPARSE as UNSUPPORTED_OP_SPARSE
