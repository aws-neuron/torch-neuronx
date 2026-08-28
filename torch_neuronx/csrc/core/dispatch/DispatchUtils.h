#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace torch_neuronx {
namespace dispatch {

/// Int64 execution mode — controls how int64 tensors are handled during dispatch.
///
/// Unifies the packed-view storage strategy (cpu_bounce / packed) with the
/// native S64 dispatch policy (native / native_strict). Integer values are
/// stable; the Python Int64Mode enum and Int64Policy mirror them.
enum class Int64Mode : uint8_t {
  kCpuBounce = 0,  // Cast int64→int32 via CPU roundtrip; op runs on device.
  kNative = 1,     // Per-op cascade (default): native ops S64; non-native packed; matmul/conv CPU.
  kPacked = 2,     // Force every int64 op through the packed view (low word).
  kNativeStrict = 3,  // S64 to compiler for natively-supported ops only; rest → CPU.
  kCpuFallback = 4,   // Run every int64 op entirely on CPU (no Neuron execution).
};

namespace detail {

/// Parse TORCH_NEURONX_INT64_MODE.
inline Int64Mode ParseInt64ModeEnv() {
  const char* mode_env = std::getenv("TORCH_NEURONX_INT64_MODE");
  if (mode_env != nullptr) {
    std::string val(mode_env);
    if (val == "packed") return Int64Mode::kPacked;
    if (val == "native") return Int64Mode::kNative;
    if (val == "native_strict") return Int64Mode::kNativeStrict;
    if (val == "cpu_bounce") return Int64Mode::kCpuBounce;
    if (val == "cpu_fallback") return Int64Mode::kCpuFallback;
    TORCH_WARN("Invalid TORCH_NEURONX_INT64_MODE='", val,
               "'. Valid: packed, native, native_strict, cpu_bounce, cpu_fallback. "
               "Defaulting to native.");
    return Int64Mode::kNative;
  }
  return Int64Mode::kNative;
}

/// Process-wide int64 mode storage. Parsed from env on first access. Mutable
/// only so the test harness can exercise each mode within one process via
/// SetInt64ModeForTesting; production code never writes it, so the value equals
/// the env parse (identical behavior to the previous cached static).
inline Int64Mode& Int64ModeStorage() {
  static Int64Mode mode = ParseInt64ModeEnv();
  return mode;
}

}  // namespace detail

/// Read the active int64 mode. Default: native (per-op cascade).
inline Int64Mode GetInt64Mode() { return detail::Int64ModeStorage(); }

/// Test-only: override the process int64 mode in-place. Lets the pytest harness
/// (conftest.py int64_mode marker) run the same test under each mode in one
/// process. NOT called by any production dispatch path — mirror it with the
/// Python reset_int64_mode_cache() so both layers agree.
inline void SetInt64ModeForTesting(Int64Mode mode) { detail::Int64ModeStorage() = mode; }

/// Test-only: re-parse TORCH_NEURONX_INT64_MODE into the mode storage.
inline void ResetInt64ModeFromEnv() { detail::Int64ModeStorage() = detail::ParseInt64ModeEnv(); }

/// True when int64 flows to the compiler as S64 (native or native_strict).
/// This is the gate the dispatch policy keys off.
inline bool IsS64Passthrough() {
  const Int64Mode m = GetInt64Mode();
  return m == Int64Mode::kNative || m == Int64Mode::kNativeStrict;
}

/// Check if native int64 (S64 HLO) is enabled. Now means "S64 reaches the
/// compiler" and covers both native and native_strict, matching the original
/// semantics ("int64 passes through as S64") for all existing call sites.
inline bool IsNativeInt64Enabled() { return IsS64Passthrough(); }

/// Check if explicit packed int64 mode is enabled.
inline bool IsPackedInt64Enabled() { return GetInt64Mode() == Int64Mode::kPacked; }

/// True when an int64 op that is being *downcast* (i.e. NOT kept as S64) should
/// use the on-device packed view rather than a CPU roundtrip. That's the
/// explicit packed mode, and the native cascade (whose non-native ops demote to
/// int32 via the packed view). cpu_bounce uses the CPU roundtrip instead.
inline bool UsePackedInt64() {
  const Int64Mode m = GetInt64Mode();
  return m == Int64Mode::kPacked || m == Int64Mode::kNative;
}

// ═══════════════════════════════════════════════════════════════
//  ForEachTensor — generic visitor over tensors in IValue args
//
//  Eliminates the duplicated "iterate args, check isTensor /
//  isTensorList, extract defined tensors" pattern that appears
//  in ExtractDevice, HasEmptyTensorInput, ExtractTensorInputs,
//  PreprocessArgs target-device detection, etc.
//
//  @param fn  Callable with signature (const at::Tensor&) → void.
//             Return type is ignored.
// ═══════════════════════════════════════════════════════════════

template <typename Fn>
void ForEachTensor(const std::vector<c10::IValue>& args, Fn&& fn) {
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      const auto& t = arg.toTensor();
      if (t.defined()) fn(t);
    } else if (arg.isTensorList()) {
      for (const auto& t_ref : arg.toTensorList()) {
        at::Tensor t = t_ref;
        if (t.defined()) fn(t);
      }
    } else if (arg.isList()) {
      // GenericList — covers Tensor?[] (e.g. _index_put_impl_ indices).
      for (const auto& elem : arg.toListRef()) {
        if (elem.isTensor()) {
          const auto& t = elem.toTensor();
          if (t.defined()) fn(t);
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  Empty Tensor Detection
// ═══════════════════════════════════════════════════════════════

/**
 * Check if any input tensor has numel() == 0.
 * Mirrors kernel._should_handle_empty() in Python.
 *
 * @param args The IValue arguments from the dispatch stack
 * @return true if any tensor input has zero elements
 */
bool HasEmptyTensorInput(const std::vector<c10::IValue>& args);

/**
 * Check if any output tensor argument (e.g., 'out' parameter) is empty.
 * Output tensors are identified by writable alias info in the schema.
 *
 * @param args The IValue arguments from the dispatch stack
 * @param schema The function schema to identify output parameters
 * @return true if any output tensor has zero elements
 */
bool HasEmptyOutputTensor(const std::vector<c10::IValue>& args, const c10::FunctionSchema& schema);

// ═══════════════════════════════════════════════════════════════
//  Complex Dtype Detection
// ═══════════════════════════════════════════════════════════════

/**
 * Check if any input tensor has a complex dtype (complex32/64/128).
 * Complex dtypes are not supported on Neuron hardware — ops with complex
 * inputs must fall back to CPU.  Mirrors Python can_handle() in base.py.
 */
bool HasComplexInput(const std::vector<c10::IValue>& args);

/** Read TORCH_NEURONX_COMPLEX_DECOMPOSE on each call (default: false). */
bool ComplexDecompositionEnabled();

/**
 * Check if any *non-output* tensor argument has a complex dtype.
 * Skips tensors at schema positions whose AliasInfo indicates a write
 * (out= params). Use for ops that produce complex from real inputs
 * (e.g., aten::complex.out).
 */
bool HasComplexNonOutputInput(const std::vector<c10::IValue>& args,
                              const c10::FunctionSchema& schema);

/**
 * Raise the standard "CPU fallback disabled for complex input" error.
 * Used at all dispatch sites that detect complex input with
 * TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1.
 */
[[noreturn]] inline void RaiseComplexFallbackDisabled(const std::string& op_name) {
  TORCH_CHECK(false,
              "CPU fallback disabled because TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. "
              "Operation ",
              op_name,
              " received a complex-dtype input; complex dtypes are not supported natively "
              "on Neuron. Unset the env var to enable CPU fallback.");
}

bool HasNativeComplexLowering(const std::string& lookup_name);

bool ShouldUseComplexDecomposition(bool has_lower_func);

// ═══════════════════════════════════════════════════════════════
//  64-bit Dtype Detection and Downcasting
// ═══════════════════════════════════════════════════════════════

/**
 * Check if a dtype is natively supported for MLIR lowering on Neuron.
 * Supported: float32, float16, bfloat16, int32, bool.
 * Unsupported dtypes (float64, int64, uint8, int16, etc.) are handled
 * via CPU bounce in DispatchCopy/DispatchToCopy and downcast in DowncastArgs.
 */
bool IsSupportedDtype(c10::ScalarType dt);

/**
 * Downcast int64→int32 and float64→float32 tensors in-place.
 * Called only on the casting path, after PreprocessArgs.
 * Uses NeuronSafeCast for CPU round-trip when needed.
 *
 * @param args   IValue args to downcast in place
 * @param schema Op schema (to iterate args correctly)
 */
void DowncastArgs(std::vector<c10::IValue>& args, const c10::FunctionSchema& schema);

/**
 * Promote the two leading tensor operands to c10::promoteTypes(a, b) for
 * binary compare/logical ops, mirroring TensorIterator's
 * promote_inputs_to_common_dtype(true) on CPU/CUDA. Must be called
 * after PreprocessArgs.
 */
void PromoteBinaryCompareOperands(std::vector<c10::IValue>& args,
                                  const c10::FunctionSchema& schema);

/**
 * Downcast a vector of tensors in-place to Neuron-supported dtypes.
 * Float64 → Float32; int64 → Int32 (always, in every mode).
 * Returns true iff any tensor was downcasted.
 *
 * Used by the C++ process-group collective path, which lacks a FunctionSchema
 * to drive DowncastArgs. Collectives are handled in int32, so int64 is demoted
 * here regardless of mode. Leaves already-supported tensors untouched (by identity).
 */
bool DowncastTensorList(std::vector<at::Tensor>& tensors);

// ═══════════════════════════════════════════════════════════════
//  Debug logging helpers
//
//  Wrap per-element debug loops so the iteration and string
//  formatting are only compiled once and the call site is a
//  single line.  The TORCH_NEURONX_DEBUG macro still controls
//  whether anything is actually emitted at runtime.
// ═══════════════════════════════════════════════════════════════

/**
 * Log each IValue arg (type, shape, dtype, value) under @p prefix.
 */
void LogArgs(const char* prefix, const std::vector<c10::IValue>& args);

/**
 * Log each tensor (data_ptr, shape, dtype, device) under @p prefix.
 */
void LogTensors(const char* prefix, const std::vector<torch::Tensor>& tensors);

// ═══════════════════════════════════════════════════════════════
//  Dispatch snapshot helpers
//
//  These capture pre-preprocessing state that is needed after
//  execution (original dtypes, out tensors, CPU fallback context).
//  Moved here from NeuronDispatcher.cpp for testability.
// ═══════════════════════════════════════════════════════════════

/**
 * Snapshot original out/inplace tensors before preprocessing.
 * Returns (return_index → original_tensor) for every schema return
 * that carries a write alias (Tensor(a!)) pointing at an argument.
 */
std::vector<std::pair<size_t, torch::Tensor>> SnapshotOutTensors(
    const c10::FunctionSchema& schema, const std::vector<c10::IValue>& args);

/**
 * Check if a 'dtype' kwarg was explicitly set (non-None).
 */
bool HasExplicitDtypeKwarg(const c10::FunctionSchema& schema, const std::vector<c10::IValue>& args);

/**
 * Build a CPUFallbackContext from the current args and schema.
 * Captures tensor metadata and data pointers for CPU fallback recovery.
 */
at::neuron::CPUFallbackContext BuildCPUFallbackContext(const std::vector<c10::IValue>& args,
                                                       const c10::FunctionSchema& schema);

// ═══════════════════════════════════════════════════════════════
//  Empty Tensor Handling — op-type configs
//
//  C++ port of Python ReductionOps/IndexOps from config.py.
//  Used by TryHandleEmptyTensors to determine correct fill values.
// ═══════════════════════════════════════════════════════════════

/**
 * Extract base operation name from full ATen name.
 * "aten::sum.dim_IntList" → "sum"
 */
std::string ExtractBaseOpName(const std::string& op_name);

/**
 * True if `src` and `dst` share storage with overlapping byte ranges.
 * Caller must pass contiguous tensors so [storage_offset, storage_offset + numel)
 * describes the actual byte footprint.
 */
bool RawNrtCopyOverlaps(const at::Tensor& src, const at::Tensor& dst);

/**
 * Reduction op configuration for empty tensor handling.
 * Mirrors Python ReductionOps in config.py.
 */
struct ReductionConfig {
  enum class FillType { kIdentity, kNaN, kError };
  FillType fill_type;
  double identity_value;  // Only used when fill_type == kIdentity
};

/**
 * Check if an op is a known reduction and get its empty-tensor config.
 * Returns nullptr if the op is not a reduction.
 */
const ReductionConfig* GetReductionConfig(const std::string& op_name);

/**
 * Index op configuration for empty tensor handling.
 * Mirrors Python IndexOps in config.py.
 */
struct IndexOpConfig {
  int index_argnum;        // Position of the index tensor argument
  int input_argnum;        // Position of the input tensor argument
  int source_argnum = -1;  // Position of the source argument, -1 if the op has none
};

/**
 * Check if an op is a known index op and get its config.
 * Returns nullptr if the op is not an index op.
 */
const IndexOpConfig* GetIndexOpConfig(const std::string& op_name);

/**
 * Matmul-accumulate op configuration for empty tensor handling.
 *
 * These ops compute ``beta * self + alpha * (matmul)``. When only the
 * *contraction* dimension is zero (e.g. baddbmm with batch1=[b,n,0],
 * batch2=[b,0,p]) the matmul term is an empty sum — zero — but the output
 * is NOT empty: the correct result is ``beta * self``, broadcast to the
 * matmul's output shape.
 *
 * The generic empty-tensor path fills such outputs with zeros, which
 * silently drops the ``beta * self`` term. Ops listed here are handled by a
 * dedicated branch in HandleEmptyTensors that fabricates ``beta * self``
 * directly (see the matmul-accumulate branch). Only ``self_argnum`` is stored;
 * the two matmul operands sit immediately after ``self`` (``self_argnum + 1``
 * and ``+ 2``) in both baddbmm and addmm, and the branch reads them there to
 * infer the output shape without a Meta kernel (baddbmm.dtype has none).
 *
 * Only ops whose result depends on a non-empty operand belong here. Plain
 * matmuls (mm/bmm) are correctly served by the zeros path — with a zero
 * contraction dim their result genuinely is all zeros.
 */
struct MatmulAccumulateConfig {
  int self_argnum;  // Position of the accumulated (`self`) tensor
};

/**
 * Check if an op is a matmul-accumulate op and get its config.
 * Returns nullptr if the op is not one.
 */
const MatmulAccumulateConfig* GetMatmulAccumulateConfig(const std::string& op_name);

/** True if any dimension of `shape` is zero (i.e. the tensor is empty). */
bool HasZeroDim(const std::vector<int64_t>& shape);

/**
 * Read a positional/keyword Scalar argument by schema name, returning
 * `default_value` when it is absent or None. Used to recover `beta` for the
 * matmul-accumulate empty-tensor branch.
 *
 * Returns the `c10::Scalar` verbatim rather than flattening to `double`: the
 * scalar's type category (integral vs floating) must survive so that
 * `beta * self` keeps an integer output integer instead of promoting it to
 * float. Flattening to double silently broke that (int32 `self` → float32).
 */
c10::Scalar ExtractScalarArg(const std::vector<c10::IValue>& args,
                             const c10::FunctionSchema& schema, const std::string& name,
                             const c10::Scalar& default_value);

// ═══════════════════════════════════════════════════════════════
//  Meta dispatch utilities — used by HandleEmptyTensors.
//
//  Uses redispatchBoxed with explicit {Meta} keyset, matching the
//  canonical PyTorch pattern (FunctionalTensorWrapper.cpp:873).
// ═══════════════════════════════════════════════════════════════

/**
 * Convert a single tensor to meta device.
 * Preserves shape, strides, dtype.  Strips requires_grad to prevent
 * AutogradMeta from entering the dispatch keyset.
 * Uses at::empty_strided for explicit stride control (empty_like infers
 * strides from memory format which adds unnecessary logic).
 * Note: storage_offset is not preserved (always 0).
 */
at::Tensor TensorToMeta(const at::Tensor& t);

/**
 * Convert a single IValue to meta device.
 * Handles: Tensor, TensorList, OptionalTensorList, GenericList (Tensor?[]).
 * Non-tensor values pass through unchanged.
 */
c10::IValue IValueToMeta(const c10::IValue& val);

/**
 * Run an op on meta-converted args via redispatchBoxed({Meta}).
 * Returns the stack containing only return values.
 *
 * @param op   The operator handle
 * @param args The original IValue args (may be on any device)
 * @return     Stack with return values from the meta kernel
 * @throws     c10::Error if the meta kernel rejects the inputs
 *             (shape mismatch, dtype mismatch, etc.) or if no
 *             meta kernel exists for this op.
 */
torch::jit::Stack RunMetaDispatch(const c10::OperatorHandle& op,
                                  const std::vector<c10::IValue>& args);

/**
 * Check if an op is an index-family op where Meta dispatch does not validate
 * shapes/bounds (e.g., index_copy, gather, index_select).
 * These ops must NOT be short-circuited by empty tensor handling because
 * Meta silently produces wrong shapes — they must go through normal lowering
 * so the compiler or Python validation catches errors.
 */
bool IsIndexOp(const std::string& op_name);

/**
 * Output specification inferred from Meta dispatch.
 */
struct MetaOutputSpec {
  std::vector<int64_t> shape;
  c10::ScalarType dtype;
  bool is_undefined;  // True for None/undefined tensor outputs
};

/**
 * Extract MetaOutputSpecs from a post-dispatch stack.
 * Used by HandleEmptyTensors which needs shapes and dtypes.
 *
 * @param stack  Stack after RunMetaDispatch (contains only return values)
 * @return       Vector of MetaOutputSpec for each tensor return value.
 *               Non-tensor returns are skipped (caller must verify schema).
 */
std::vector<MetaOutputSpec> ExtractMetaOutputSpecs(torch::jit::Stack& stack);

/**
 * Convenience: RunMetaDispatch + ExtractMetaOutputSpecs.
 * Drop-in replacement for the old InferOutputsViaMeta.
 */
std::vector<MetaOutputSpec> InferOutputsViaMeta(const c10::OperatorHandle& op,
                                                const std::vector<c10::IValue>& args);

// ═══════════════════════════════════════════════════════════════
//  IValue ↔ Python conversion helpers
// ═══════════════════════════════════════════════════════════════

/**
 * Result of converting IValue args to Python positional args + kwargs,
 * split according to the op schema's kwarg_only annotations.
 */
struct PyArgsKwargs {
  py::tuple args;
  py::dict kwargs;
};

/**
 * Convert IValue args to Python objects, splitting into positional
 * args and keyword-only kwargs according to the schema.
 *
 * Tensors are wrapped via THPVariable_Wrap; all other types go
 * through IValueToPyObjectWithSchema.  Must be called under the GIL.
 */
PyArgsKwargs IValuesToPyArgsKwargs(const std::vector<c10::IValue>& args,
                                   const c10::FunctionSchema& schema);

}  // namespace dispatch
}  // namespace torch_neuronx
