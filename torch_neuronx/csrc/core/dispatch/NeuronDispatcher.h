#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <memory>
#include <shared_mutex>
#include <string>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"

namespace torch_neuronx {
namespace dispatch {

/**
 * Cached lowering result — C++ mirror of Python LoweringResult.
 *
 * Populated on cache miss by calling the Python lower_func, then stored
 * so that subsequent calls with the same (op, shapes, dtypes) are
 * handled entirely in C++ with no Python calls.
 */
struct DispatchCacheEntry {
  // StableHLO IR bytes for this (op, shape) combination
  std::vector<uint8_t> stablehlo_bytes;

  // Cache key for the CompilationCache (NEFF lookup)
  std::string neff_cache_key;

  // Which flattened inputs the HLO actually uses
  std::vector<int> kept_input_indices;

  // Which original outputs the HLO produces
  std::vector<int> kept_output_indices;

  // Output specification for tensor creation
  struct OutputSpec {
    std::vector<int64_t> shape;
    c10::ScalarType dtype;
    // When present, the dispatch allocates a physical-shape buffer and returns an
    // as_strided view with target_strides. Epilogue is fused into the
    // NEFF so it writes directly in the physical layout.
    std::optional<std::vector<int64_t>> target_strides;
  };
  std::vector<OutputSpec> output_specs;           // For execution (potentially downcasted)
  std::vector<OutputSpec> original_output_specs;  // For upcast (original dtypes)

  // Passthrough mappings: output_idx → input_idx
  std::unordered_map<int, int> passthrough_mappings;

  // Input-output aliasing: pairs of (ir_param_idx, output_idx) indicating
  // that the given output buffer should alias (share memory with) the given
  // input parameter in the compiled StableHLO module.
  // For inplace ops this is typically {{0, 0}} meaning output 0 aliases IR input 0.
  std::vector<std::pair<int, int>> input_output_aliases;

  // Op metadata
  std::string op_name;
  bool is_inplace = false;
  bool has_collectives = false;
  bool is_empty_tensor = false;

  // Inferred output dtypes at original (pre-downcast) precision.
  // Used by DispatchWithCasting to upcast execution outputs to user-expected dtypes.
  // Empty means no upcasting needed (specialized ops or no 64-bit inputs).
  std::vector<c10::ScalarType> inferred_output_dtypes;

  // NKI DumpedConfig fields — cached on first miss for eager bypass.
  // These allow NkiDispatchEager to skip Python get_dumped_config() on cache hit.
  struct NkiOutputSpec {
    std::string dtype_str;
    std::vector<int64_t> shape;
  };
  std::vector<NkiOutputSpec> nki_output_specs;
  std::unordered_map<int, int> nki_operand_output_aliases;  // input_idx → output_idx
  bool has_nki_config = false;
};

/**
 * Per-op registration info stored in the dispatcher.
 */
struct OpRegistration {
  // Python wrapper for full-fallback (calls the entire Python Operation path)
  py::object python_wrapper;

  // Python lowering function: kernel.lower(*args, **kwargs) → LoweringResult
  // Called on cache miss. May be py::none() for ops with deferred kernel creation.
  py::object lower_func;

  // Indices of positional args that are compile-time constants (e.g. dim, keepdim).
  std::unordered_set<size_t> static_argnums;

  // Names of keyword arguments that are compile-time constants.
  std::unordered_set<std::string> static_argnames;

  // When true, always call lower_func on every dispatch — never cache the
  // lowering result. Set for ops with uses_preprocessing=True whose
  // preprocessing reads tensor values (not just shapes/dtypes).
  bool skip_cache = false;

  // When true, this op has a lower_func and is eligible for the C++ meta
  // dtype inference path.  Set automatically at registration time when
  // lower_func is not py::none().
  bool has_lower_func = false;

  // Optional callback that adds operation-specific output metadata to a
  // LoweringResult. Called after lowering and by the empty-tensor Meta path.
  py::object lowering_result_transform = py::none();
};

/**
 * Result bundle from DispatchSdpaCore — allows SDPA handlers to do post-processing.
 */
struct SdpaCoreResult {
  std::vector<torch::Tensor> final_outputs;
  std::vector<c10::IValue> original_args;
  torch::Device device;
  bool handled;  // true if early-exit already pushed to stack
};

/**
 * NeuronDispatcher — C++ dispatch cache front for Neuron ops.
 *
 * The Dispatch() handler follows a linear pipeline:
 *
 *   1. Resolve op registration (captured at registration time — no map lookup)
 *   2. Pop args from stack, validate inputs are on Neuron
 *   3. Generate cache key (op_name + shapes + dtypes)
 *   4. GetOrLowerOperation(cache_key, lower_func, args)
 *        → returns DispatchCacheEntry (from cache or freshly lowered)
 *   5. Allocate output tensors from entry specs
 *   6. Filter inputs via kept_input_indices
 *   7. Enqueue to OperationExecutionEngine
 *   8. Push outputs onto stack
 *
 * Steps 5-8 are shared between cache hit and miss — the only
 * difference is whether step 4 reads from cache or calls Python.
 *
 * Registration captures a shared_ptr<OpRegistration> into each boxed
 * functor at registration time, so the dispatch hot path never touches
 * the registration map and never copies py::objects.
 */
class NeuronDispatcher {
 public:
  static NeuronDispatcher& Instance();

  /**
   * Register an op with a Python wrapper and optional lower function.
   *
   * Stores the registration as a shared_ptr and captures it in the
   * boxed dispatch functor, so the hot path accesses it directly
   * without map lookups or py::object copies.
   *
   * @param op_name        ATen op name without "aten::" prefix
   * @param python_wrapper Python callable (full execution fallback)
   * @param lower_func     Python callable: kernel.lower(*args) → LoweringResult
   */
  void RegisterOp(const std::string& op_name, py::object python_wrapper,
                  py::object lower_func = py::none(), std::vector<int64_t> static_argnums_vec = {},
                  std::vector<std::string> static_argnames_vec = {}, bool skip_cache = false,
                  py::object lowering_result_transform = py::none());

  /**
   * Dedicated boxed handler for aten::scalar_tensor.
   * Creates the scalar tensor on CPU (letting PyTorch handle dtype inference),
   * then copies it to the Neuron device. Avoids the full lowering pipeline
   * and prevents infinite recursion from PreprocessArgs re-entering the
   * PrivateUse1 dispatcher.
   */
  static void DispatchScalarTensor(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                   torch::jit::Stack* stack);

  // ── Accessors (for debugging) ──

  std::vector<std::string> GetRegisteredOps() const;

  /**
   * Explicitly release Python objects and torch::Library before interpreter
   * finalization.  Called from the module cleanup capsule so that the implicit
   * static destructor (which runs *after* Python is gone) finds empty
   * containers and does not segfault on py::object decref.
   *
   * Follows the same pattern as python_contiguous_op / _stream_cleanup_capsule
   * in NeuronBindings.cpp.
   */
  void Cleanup();

  /**
   * Collective lowering — cache lookup, GIL acquire on miss, call Python
   * lowering callback, parse result, cache and return.
   */
  std::shared_ptr<DispatchCacheEntry> GetOrLowerCollectiveOperation(
      const std::string& cache_key, pybind11::object& lower_callback, c10d::OpType op_type,
      const std::vector<at::Tensor>& inputs, const std::vector<at::Tensor>& outputs,
      const std::vector<std::vector<int>>& replica_groups, int reduce_op,
      bool slice_output = false);

 private:
  friend class NeuronDispatcherTestAccess;
  friend std::vector<torch::Tensor> NkiDispatchCore(const c10::List<at::Tensor>&, int64_t,
                                                    const std::string&, const c10::List<int64_t>&,
                                                    const std::string&, bool,
                                                    const c10::FunctionSchema&,
                                                    std::shared_ptr<DispatchCacheEntry>*);
  friend py::object NkiDispatchEager(std::vector<at::Tensor>, int64_t, const std::string&,
                                     std::vector<int64_t>, const std::string&, bool);

  NeuronDispatcher() = default;

  // ── Registration storage ──
  // Registrations are stored as shared_ptrs so that boxed dispatch functors
  // can hold a long-lived reference without copying py::objects on every call.
  std::unordered_map<std::string, std::shared_ptr<OpRegistration>> op_registrations_;

  // ── Dispatch cache: cache_key → DispatchCacheEntry ──
  // Protected by cache_mutex_ for thread safety (neuron::nki_call may be
  // invoked from multiple threads).
  mutable std::shared_mutex cache_mutex_;
  std::unordered_map<std::string, std::shared_ptr<DispatchCacheEntry>> dispatch_cache_;

  // ── PyTorch Library for per-op registration ──
  std::unique_ptr<torch::Library> dispatch_lib_;
  torch::Library& GetDispatchLib();

  // ════════════════════════════════════════════════════
  //  Pipeline step functions (called from Dispatch)
  // ════════════════════════════════════════════════════

  /**
   * Step 3: Generate cache key from op name + tensor shapes/dtypes.
   * Pure C++, no Python, no GIL.  Delegates to CacheKeyGenerator::Generate.
   */
  static std::string GenerateCacheKey(const std::string& op_name, const c10::FunctionSchema& schema,
                                      const std::vector<c10::IValue>& args,
                                      const OpRegistration& reg);

  /**
   * Step 4: Get cached lowering or perform lowering on miss.
   *
   * This is the ONLY place where cache hit vs miss diverges:
   *   - Hit:  Return cached entry immediately (no Python, no GIL)
   *   - Miss: Acquire GIL, call lower_func, parse result, cache it, return
   *
   * On cache miss, lower_func receives both the downcasted args (for HLO
   * compilation) and the original args (for dtype inference at original
   * precision).  The resulting DispatchCacheEntry contains
   * inferred_output_dtypes for upcasting.
   *
   * @param cache_key      The key from GenerateCacheKey
   * @param reg            The op registration (contains lower_func)
   * @param args           Downcasted/preprocessed IValue args (for lowering)
   * @param original_args  Original (pre-downcast) IValue args (for dtype inference)
   * @param schema         Op schema
   * @return               The DispatchCacheEntry (never null on success)
   */
  std::shared_ptr<DispatchCacheEntry> GetOrLowerOperation(
      const std::string& cache_key, const OpRegistration& reg, const std::vector<c10::IValue>& args,
      const std::vector<c10::IValue>& original_args, const c10::FunctionSchema& schema,
      bool use_complex_decomposition = false);

  // ════════════════════════════════════════════════════
  //  Step 7: Output processing
  //  Allocate, upcast, and reconstruct outputs.
  //  These are free functions declared in OutputProcessing.h
  //  and called directly from Dispatch().
  // ════════════════════════════════════════════════════

  /**
   * Step 8: Enqueue an XLA operation to the OperationExecutionEngine.
   * Mirrors the logic of _submit_xla_task_to_pipeline binding.
   */
  static void EnqueueXlaOperation(const std::string& op_name,
                                  const std::vector<torch::Tensor>& inputs,
                                  std::vector<torch::Tensor>& outputs,
                                  const std::vector<uint8_t>& ir_bytes,
                                  const std::string& cache_key, bool has_collectives, int device_id,
                                  at::neuron::CPUFallbackContext fallback_ctx = {},
                                  const std::string& hierarchy_string = "");

  // ── Internal helpers ──

  /** Cache lookup. Returns nullptr on miss. */
  std::shared_ptr<DispatchCacheEntry> CacheLookup(const std::string& cache_key);

  /** Cache insert. */
  void CacheInsert(const std::string& cache_key, std::shared_ptr<DispatchCacheEntry> entry);

  /** Get Neuron device from first tensor in args. */
  static torch::Device ExtractDevice(const std::vector<c10::IValue>& args);

  /**
   * Flatten preprocessed args into a tensor list (positional first, then kwargs),
   * then filter by kept_input_indices to produce the IR input tensors.
   *
   * Mirrors Python __call__'s:
   *   flattened = _flatten_tensors(processed_inputs)
   *   for value in processed_kwargs.values(): if isinstance(value, Tensor): flattened.append(value)
   *   filtered = [flattened[i] for i in kept_input_indices]
   *
   * PreprocessArgs must be called before this so scalars are already tensors.
   *
   * @param args                   Preprocessed IValue args
   * @param schema                 Op schema (to distinguish positional vs kwarg-only)
   * @param kept_input_indices     Which flattened indices the IR actually uses
   * @param op_name                Op name for error messages
   * @return                       Filtered tensors ready for IR execution
   */
  static std::vector<torch::Tensor> FlattenAndFilterInputs(
      const std::vector<c10::IValue>& args, const c10::FunctionSchema& schema,
      const std::vector<int>& kept_input_indices, const std::string& op_name,
      const std::unordered_set<size_t>& static_argnums = {},
      const std::unordered_set<std::string>& static_argnames = {});

  // ════════════════════════════════════════════════════
  //  Dispatch implementations — called with a pre-captured
  //  OpRegistration reference (no map lookup on hot path).
  // ════════════════════════════════════════════════════

  /**
   * Main dispatch pipeline implementation.
   * Called from the boxed functor with the captured registration.
   */
  static void DispatchImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                           torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler implementation for aten::copy_.
   * Handles CPU↔Neuron via NRT, Neuron→Neuron same-dtype via NRT,
   * or NRT-based CPU-bounce for unsupported dtypes (float64, bool, int64).
   * Falls through to DispatchImpl for MLIR-supported Neuron→Neuron cases.
   */
  static void DispatchCopyImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                               torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler implementation for aten::_to_copy.
   * Handles CPU↔Neuron via NRT, Neuron→Neuron dtype cast via MLIR,
   * or NRT-based CPU-bounce for unsupported dtypes (float64, bool, int32, int64).
   * Falls through to DispatchImpl for MLIR-supported Neuron→Neuron cases.
   */
  static void DispatchToCopyImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                 torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler implementation for aten::contiguous.
   * Thin wrapper around the normal dispatch pipeline that passes
   * skip_contiguous=true so PreprocessArgs does not call t.contiguous()
   * (which would re-dispatch into DispatchContiguousImpl and cause
   * infinite recursion).  ProcessInputTransformations still runs in
   * EnqueueXlaOperation to handle the non-contiguous → contiguous
   * transformation via prologue transforms.
   */
  static void DispatchContiguousImpl(const c10::OperatorHandle& op,
                                     c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack,
                                     const OpRegistration& reg);

  /**
   * Dedicated handler implementation for aten::nonzero / aten::nonzero.out.
   *
   * nonzero uses the nonzero_with_count kernel which returns (indices, count)
   * as 2 tensors. This handler runs the standard dispatch pipeline via
   * DispatchCore, then post-processes: slices indices[:count] and casts to
   * int64 to match the ATen nonzero schema (single Tensor output).
   */
  static void DispatchNonzeroImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                  torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler implementation for aten::index.Tensor / aten::index.Tensor_out.
   *
   * Boolean indices must be converted to integer indices (via at::nonzero)
   * BEFORE cache key generation, because boolean→integer conversion produces
   * data-dependent shapes. This handler preprocesses the indices list,
   * then delegates to DispatchWithCasting for the standard pipeline.
   */
  static void DispatchIndexImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler for SDPA forward.
   * Uses DispatchSdpaCore (no DowncastArgs), then applies stride fixup
   * and pushes the 9-tuple output expected by the ATen schema.
   */
  static void DispatchSdpaFwdImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                  torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Dedicated handler for SDPA backward.
   * Uses DispatchSdpaCore (no DowncastArgs), then applies grad stride fixup
   * and handles None attn_bias gradient.
   */
  static void DispatchSdpaBwdImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                  torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Shared NKI dispatch core: preamble, cache key, preprocess, GetOrLower,
   * execute, reconstruct. Bypasses DowncastArgs/UpcastOutputs entirely.
   * Returns outputs + original_args for SDPA post-processing.
   */
  static SdpaCoreResult DispatchSdpaCore(const c10::OperatorHandle& op,
                                         c10::DispatchKeySet dispatch_keys,
                                         torch::jit::Stack* stack, const OpRegistration& reg);

  /**
   * Core dispatch pipeline shared by both fast path and casting path.
   * Takes preprocessed (and optionally downcasted) args, a pre-computed
   * cache key, and runs cache lookup / execution / output reconstruction.
   * Returns final output tensors with NO dtype casting awareness.
   *
   * @param out_entry  If non-null, receives the DispatchCacheEntry used for
   *                   execution. Caller can use this to access
   *                   inferred_output_dtypes for upcasting.
   */
  static std::vector<torch::Tensor> DispatchCore(
      const std::string& op_name, const std::string& cache_key, const c10::OperatorHandle& op,
      c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack, std::vector<c10::IValue>& args,
      const c10::FunctionSchema& schema, const OpRegistration& reg, NeuronDispatcher& dispatcher,
      at::neuron::CPUFallbackContext fallback_ctx,
      const std::vector<c10::IValue>& original_args = {},
      std::shared_ptr<DispatchCacheEntry>* out_entry = nullptr,
      bool use_complex_decomposition = false);

  /**
   * Casting path: handles dtype downcasting/upcasting.
   * Downcasts args, runs DispatchCore, upcasts outputs using
   * inferred_output_dtypes from the DispatchCacheEntry, handles
   * out-tensor copy-back, pushes to stack.
   */
  static void DispatchWithCasting(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                                  torch::jit::Stack* stack, std::vector<c10::IValue>& args,
                                  const c10::FunctionSchema& schema, const OpRegistration& reg,
                                  NeuronDispatcher& dispatcher, const std::string& op_name,
                                  const std::string& cache_key, bool skip_contiguous = false,
                                  bool use_complex_decomposition = false);

  /**
   * Try to handle an op with empty tensor inputs.
   *
   * Consolidates detection (which args are empty?) and handling (what to
   * produce?) into a single function, mirroring the Python path's
   * _should_handle_empty + _handle_empty_tensors.
   *
   * Op categories:
   *   1. GetIndexOpConfig ops (scatter, scatter_add, _index_put_impl_):
   *      checks all args including lists; empty index → return self.
   *   2. IsIndexOp ops (gather, index_copy, index_select, ...):
   *      empty self → return false (normal lowering validates shapes).
   *      empty index only → Meta + zeros.
   *   3. Reductions: identity fill (0, 1, NaN, or error).
   *   4. General: Meta + zeros.
   *
   * Returns true if handled (outputs pushed to stack).
   * Returns false if caller should fall through to normal lowering.
   */
  static bool TryHandleEmptyTensors(const c10::OperatorHandle& op, torch::jit::Stack* stack,
                                    const std::vector<c10::IValue>& args,
                                    const c10::FunctionSchema& schema, const std::string& op_name,
                                    const OpRegistration& reg);
};

/**
 * Boxed dispatch functor that captures a shared_ptr<OpRegistration>
 * at registration time, eliminating per-dispatch map lookups and
 * py::object copy overhead on the hot path.
 *
 * The handler function pointer selects which implementation to call
 * (DispatchImpl, DispatchCopyImpl, DispatchToCopyImpl, DispatchContiguousImpl).
 */
class BoxedDispatchFunctor final : public c10::OperatorKernel {
 public:
  using HandlerFn = void (*)(const c10::OperatorHandle&, c10::DispatchKeySet, torch::jit::Stack*,
                             const OpRegistration&);

  BoxedDispatchFunctor(std::shared_ptr<OpRegistration> reg, HandlerFn handler)
      : reg_(std::move(reg)), handler_(handler) {}

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet ks, torch::jit::Stack* stack) {
    handler_(op, ks, stack, *reg_);
  }

 private:
  std::shared_ptr<OpRegistration> reg_;
  HandlerFn handler_;
};

}  // namespace dispatch
}  // namespace torch_neuronx
