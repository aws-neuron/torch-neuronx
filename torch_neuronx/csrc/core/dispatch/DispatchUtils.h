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
enum class Int64Mode : uint8_t {
  kCpuBounce = 0,  // Move to CPU, cast to int32, move back (default)
  kNative = 1,     // Pass int64 through to compiler as S64 HLO
  kPacked = 2,     // Store as two int32 lanes; execute on low word only
};

/// Read TORCH_NEURONX_INT64_MODE (or legacy TORCH_NEURONX_NATIVE_INT64=1).
/// Cached on first call.
inline Int64Mode GetInt64Mode() {
  static const Int64Mode mode = []() {
    const char* mode_env = std::getenv("TORCH_NEURONX_INT64_MODE");
    if (mode_env != nullptr) {
      std::string val(mode_env);
      if (val == "packed") return Int64Mode::kPacked;
      if (val == "native") return Int64Mode::kNative;
      if (val == "cpu_bounce") return Int64Mode::kCpuBounce;
      TORCH_WARN("Invalid TORCH_NEURONX_INT64_MODE='", val,
                 "'. Valid: packed, native, cpu_bounce. Defaulting to cpu_bounce.");
      return Int64Mode::kCpuBounce;
    }
    // Legacy fallback
    const char* legacy = std::getenv("TORCH_NEURONX_NATIVE_INT64");
    if (legacy != nullptr && std::string(legacy) == "1") {
      return Int64Mode::kNative;
    }
    return Int64Mode::kPacked;
  }();
  return mode;
}

/// Check if native int64 (S64 HLO) is enabled.
/// Cached on first call — delegates to GetInt64Mode().
inline bool IsNativeInt64Enabled() {
  static const bool enabled = (GetInt64Mode() == Int64Mode::kNative);
  return enabled;
}

/// Check if packed int64 mode is enabled.
inline bool IsPackedInt64Enabled() {
  static const bool enabled = (GetInt64Mode() == Int64Mode::kPacked);
  return enabled;
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
 * Float64 → Float32; unsupported int types → Int32 (unless IsNativeInt64Enabled).
 * Returns true iff any tensor was downcasted.
 *
 * Used by the C++ process-group collective path, which lacks a FunctionSchema
 * to drive DowncastArgs. Leaves already-supported tensors untouched (by identity).
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
  int index_argnum;  // Position of the index tensor argument
  int input_argnum;  // Position of the input tensor argument
};

/**
 * Check if an op is a known index op and get its config.
 * Returns nullptr if the op is not an index op.
 */
const IndexOpConfig* GetIndexOpConfig(const std::string& op_name);

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
