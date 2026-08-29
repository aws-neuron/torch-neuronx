#pragma once

#include <nrt/nrt.h>
#include <torch/torch.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"
#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace at::neuron {

namespace utils {
inline void* GetBaseTensorPtr(void* ptr, size_t offset_bytes) {
  return static_cast<char*>(ptr) - offset_bytes;
}
}  // namespace utils

// Forward declarations
class CompilationCache;

enum class KernelTypeEnum {
  kHLO,
  kCollective,
  kCopy,
  kWrite,
  kRead,
  kEvent,
  kBarrier,
  kHint,  // Merely a non-functional hint
  kAlloc,
  kDealloc,
  kHost,
};

// Check if a kernel type represents a device operation (vs host-side operation).
// Device operations: kHLO, kCollective, kCopy, kWrite, kRead
// Host-side operations: kEvent, kBarrier, kHint, kAlloc, kDealloc, kHost
inline bool IsDeviceKernelType(KernelTypeEnum kernel_type) {
  switch (kernel_type) {
    case KernelTypeEnum::kHLO:
    case KernelTypeEnum::kCollective:
    case KernelTypeEnum::kCopy:
    case KernelTypeEnum::kWrite:
    case KernelTypeEnum::kRead:
      return true;
    case KernelTypeEnum::kEvent:
    case KernelTypeEnum::kBarrier:
    case KernelTypeEnum::kHint:
    case KernelTypeEnum::kAlloc:
    case KernelTypeEnum::kDealloc:
    case KernelTypeEnum::kHost:
      return false;
  }
  return false;
}

// kHint/kAlloc/kDealloc are bufferable because the merge path consumes their
// lifetime info; only kHLO is fusible. Everything else flushes the buffer.
inline bool IsConcatBufferableKernelType(KernelTypeEnum kernel_type) {
  switch (kernel_type) {
    case KernelTypeEnum::kHLO:
    case KernelTypeEnum::kHint:
    case KernelTypeEnum::kAlloc:
    case KernelTypeEnum::kDealloc:
      return true;
    case KernelTypeEnum::kCollective:
    case KernelTypeEnum::kCopy:
    case KernelTypeEnum::kWrite:
    case KernelTypeEnum::kRead:
    case KernelTypeEnum::kEvent:
    case KernelTypeEnum::kBarrier:
    case KernelTypeEnum::kHost:
      return false;
  }
  return false;
}

// Convert device KernelTypeEnum to XU index for completion/inflight tracking.
inline size_t GetXUIndex(KernelTypeEnum kernel_type) {
  switch (kernel_type) {
    case KernelTypeEnum::kHLO:
      return NRTA_XU_COMPUTE;
    case KernelTypeEnum::kCollective:
      return NRTA_XU_COLLECTIVES;
    case KernelTypeEnum::kCopy:
    case KernelTypeEnum::kWrite:
    case KernelTypeEnum::kRead:
      return NRTA_XU_TENSOR_OP;
    default:
      EXECUTION_RUNTIME_CHECK(
          false, "Invalid device kernel type for XU indexing: ", static_cast<int>(kernel_type));
  }
}

// Type alias for shorter names
using NeuronTensorPtr = c10_neuron::NeuronCachingAllocator::TensorPtr;

// Data reference for tensor pointers.
// Complementary to TensorContext: TensorDataRef owns runtime pointers (*where* data lives),
// TensorContext carries shape/offset metadata (*what region* to slice).
// Both must be propagated together during concatenation.
struct TensorDataRef {
  NeuronTensorPtr tensor_ptr;  // Shared ownership for the data pointer (Neuron-only)
  void* ptr;

  TensorDataRef() : ptr(nullptr) {}
  TensorDataRef(NeuronTensorPtr tp, void* p) : tensor_ptr(std::move(tp)), ptr(p) {}
  explicit TensorDataRef(void* p) : ptr(p) {}
};

class NeuronKernelExecution {
 public:
  virtual ~NeuronKernelExecution() = default;

  bool IsValid() const { return !op_name_.empty() && ValidateImpl(); }
  std::string GetOpName() const { return op_name_; }
  int GetDeviceId() const { return device_id_; }

  virtual bool RequiresCompilation() const = 0;
  bool RequiresExecution() const { return requires_execution_; }
  virtual KernelTypeEnum GetKernelType() const = 0;

  // Called when the op resolves without device execution.
  virtual void CompleteWithoutExecution() const {}

  const std::vector<void*>& GetSrcPtrs() const { return src_ptrs_; }
  const std::vector<void*>& GetDstPtrs() const { return dst_ptrs_; }

  // Prepares kernel state before dispatch. Override for operations requiring setup (e.g.,
  // collectives).
  virtual void Prepare() const {}

  // Schedules kernel for async execution.
  // Returns true if work was submitted to NRT, false otherwise.
  virtual bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                        c10::StreamId stream_id) const = 0;

  // Check if this operation requires Prepare() to be called before Dispatch()
  virtual bool RequiresPrepare() const { return false; }

 protected:
  NeuronKernelExecution(const std::string& op_name, int device_id)
      : NeuronKernelExecution(op_name, device_id, CaptureRequiresExecution()) {}

  NeuronKernelExecution(const std::string& op_name, const std::vector<void*>& src_ptrs,
                        const std::vector<void*>& dst_ptrs, int device_id)
      : op_name_(op_name),
        src_ptrs_(src_ptrs),
        dst_ptrs_(dst_ptrs),
        device_id_(device_id),
        requires_execution_(CaptureRequiresExecution()) {}

  NeuronKernelExecution(const std::string& op_name, int device_id, bool requires_execution)
      : op_name_(op_name), device_id_(device_id), requires_execution_(requires_execution) {}

  virtual bool ValidateImpl() const { return true; }

  bool HasInputs() const { return !src_ptrs_.empty(); }
  bool HasOutputs() const { return !dst_ptrs_.empty(); }

  std::string op_name_;
  std::vector<void*> src_ptrs_;
  std::vector<void*> dst_ptrs_;
  int device_id_;

 private:
  static bool CaptureRequiresExecution();
  const bool requires_execution_;
};

template <KernelTypeEnum KernelType, ::c10::DeviceType SrcDevice, ::c10::DeviceType DstDevice>
class NeuronKernelExecutionTyped : public NeuronKernelExecution {
 public:
  friend class CPUFallbackExecutor;

  ~NeuronKernelExecutionTyped() noexcept override { FreeSlices(); }

  KernelTypeEnum GetKernelType() const override { return KernelType; }

  bool RequiresCompilation() const override { return KernelType == KernelTypeEnum::kHLO; }

  template <c10::DeviceType Type>
  using Storage =
      typename std::conditional<Type == c10::DeviceType::PrivateUse1, nrt_tensor_t*, void*>::type;

  using SrcStorageVec = std::vector<Storage<SrcDevice>>;
  using DstStorageVec = std::vector<Storage<DstDevice>>;

  SrcStorageVec GetSrcDataPtrs() const { return ExtractDataPtrs<SrcStorageVec>(src_data_refs_); }
  DstStorageVec GetDstDataPtrs() const { return ExtractDataPtrs<DstStorageVec>(dst_data_refs_); }

  const std::vector<TensorDataRef>& GetSrcRefs() const { return src_data_refs_; }
  const std::vector<TensorDataRef>& GetDstRefs() const { return dst_data_refs_; }

  const std::vector<TensorContext>& GetSrcTensorContexts() const { return src_tensor_ctxs_; }
  const std::vector<TensorContext>& GetDstTensorContexts() const { return dst_tensor_ctxs_; }

 protected:
  // Constructor without TensorContext (for kernels that don't need slicing)
  NeuronKernelExecutionTyped(const std::string& op_name, std::vector<TensorDataRef>&& src_refs,
                             std::vector<TensorDataRef>&& dst_refs, int device_id)
      : NeuronKernelExecution(op_name, ExtractPtrs(src_refs), ExtractPtrs(dst_refs), device_id),
        src_data_refs_(std::move(src_refs)),
        dst_data_refs_(std::move(dst_refs)) {}
  // Constructor with TensorContext (for kernels that need slicing support)
  NeuronKernelExecutionTyped(const std::string& op_name, std::vector<TensorDataRef>&& src_refs,
                             std::vector<TensorDataRef>&& dst_refs,
                             const std::vector<TensorContext>& src_contexts,
                             const std::vector<TensorContext>& dst_contexts, int device_id)
      : NeuronKernelExecution(op_name, ExtractPtrs(src_refs), ExtractPtrs(dst_refs), device_id),
        src_data_refs_(std::move(src_refs)),
        dst_data_refs_(std::move(dst_refs)),
        src_tensor_ctxs_(src_contexts),
        dst_tensor_ctxs_(dst_contexts) {}

  // Contiguous slicing support for Neuron tensors
  std::vector<nrt_tensor_t*> HandleContiguousSlicing(
      const std::vector<nrt_tensor_t*>& data_ptrs,
      const std::vector<TensorContext>& tensor_ctxs) const {
    EXECUTION_RUNTIME_CHECK(data_ptrs.size() == tensor_ctxs.size(),
                            "Mismatched data_ptrs and tensor_ctxs sizes");
    std::vector<nrt_tensor_t*> sliced_data_ptrs;
    sliced_data_ptrs.reserve(data_ptrs.size());
    for (size_t i = 0; i < data_ptrs.size(); ++i) {
      const auto& tensor_ctx = tensor_ctxs[i];
      size_t offset_bytes = tensor_ctx.storage_offset * tensor_ctx.element_size;
      EXECUTION_RUNTIME_CHECK(data_ptrs[i], "Null tensor pointer at index ", i);
      size_t nrt_size = nrt_tensor_get_size(data_ptrs[i]);
      if (offset_bytes > 0 || tensor_ctx.size_bytes != nrt_size) {
        nrt_tensor_t* slice = nullptr;
        NRT_STATUS status = nrt_tensor_allocate_slice(data_ptrs[i], offset_bytes,
                                                      tensor_ctx.size_bytes, nullptr, &slice);
        EXECUTION_RUNTIME_CHECK(status == NRT_SUCCESS && slice,
                                "Failed to create slice: offset=", offset_bytes,
                                " size=", tensor_ctx.size_bytes, " nrt_size=", nrt_size);
        sliced_data_ptrs.push_back(slice);
        slice_data_ptrs_.push_back(slice);
      } else {
        sliced_data_ptrs.push_back(data_ptrs[i]);
      }
    }
    return sliced_data_ptrs;
  }

  void FreeSlices() const {
    for (auto* slice_data_ptr : slice_data_ptrs_) {
      nrt_tensor_free(&slice_data_ptr);
    }
    slice_data_ptrs_.clear();
  }

  std::vector<TensorDataRef> src_data_refs_;
  std::vector<TensorDataRef> dst_data_refs_;

 private:
  static std::vector<void*> ExtractPtrs(const std::vector<TensorDataRef>& refs) {
    std::vector<void*> ptrs;
    ptrs.reserve(refs.size());
    for (const auto& ref : refs) {
      ptrs.push_back(ref.ptr);
    }
    return ptrs;
  }

  template <typename StorageVec>
  static StorageVec ExtractDataPtrs(const std::vector<TensorDataRef>& refs) {
    using StorageType = typename StorageVec::value_type;
    StorageVec data_ptrs;
    data_ptrs.reserve(refs.size());
    if constexpr (std::is_same_v<StorageType, nrt_tensor_t*>) {
      // TODO(rpsilva): Simplify once converged
      if (c10_neuron::NeuronCachingAllocator::IsLazyAllocEnabled()) {
        for (const auto& ref : refs) {
          nrt_tensor_t* data_ptr = c10_neuron::NeuronCachingAllocator::findTensor(ref.ptr);
          EXECUTION_RUNTIME_CHECK(data_ptr,
                                  "Tensor not bound: lazy alloc kAlloc may not have executed");
          data_ptrs.push_back(data_ptr);
        }
      } else {
        for (const auto& ref : refs) {
          data_ptrs.push_back(ref.tensor_ptr.get());
        }
      }
    } else {
      for (const auto& ref : refs) {
        data_ptrs.push_back(ref.ptr);
      }
    }
    return data_ptrs;
  }

 protected:
  std::vector<TensorContext> src_tensor_ctxs_;
  std::vector<TensorContext> dst_tensor_ctxs_;
  mutable std::vector<nrt_tensor_t*> slice_data_ptrs_;
};

// Base class for all compilable kernels (XLA).
class CompilableKernelExecution {
 public:
  virtual ~CompilableKernelExecution() noexcept = default;

  // Compilation-specific interface
  virtual std::vector<uint8_t> CompileToNeff() const = 0;

  // Returns the HLO/StableHLO bytes for this kernel.
  // Used to compute hash for persistent cache keys.
  // Returns empty vector if HLO is not available (e.g., NeffDirectKernelExecution).
  virtual const std::vector<uint8_t>& GetHloBytes() const {
    static const std::vector<uint8_t> empty;
    return empty;
  }

  // Cache management methods
  const std::vector<uint8_t>& GetCachedNeff() const { return *cached_neff_ptr_; }
  void SetCachedNeff(NeffBytesPtr neff_ptr) const { cached_neff_ptr_ = std::move(neff_ptr); }
  const std::string& GetCompileKey() const { return compile_key_.GetKey(); }
  const CompilationCacheKey& GetCompilationCacheKey() const { return compile_key_; }
  void SetCompilationCacheKey(CompilationCacheKey key) { compile_key_ = std::move(key); }
  void UpdateCompileKey(const std::string& new_key) { compile_key_.SetKey(new_key); }
  bool HasCachedNeff() const { return cached_neff_ptr_ != nullptr; }
  bool HasCollectives() const { return has_collectives_; }

  // Compilation error tracking — set by CompilationCache on failure,
  // read by execution worker to report the real neuronx-cc error.
  void SetCompilationError(std::string error) const { compilation_error_ = std::move(error); }
  bool IsFailedCompilation() const { return !compilation_error_.empty(); }
  const std::string& GetCompilationError() const { return compilation_error_; }
  const std::string& GetAdditionalArgs() const { return additional_args_; }
  const std::string& GetOptimizationLevel() const { return optimization_level_; }

  // Transformation metadata accessors
  void SetInputTransformations(
      std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> transforms) {
    per_input_transformations_ = std::move(transforms);
  }

  const std::vector<std::vector<c10_neuron::lazy::TensorTransformation>>& GetInputTransformations()
      const {
    return per_input_transformations_;
  }

  bool HasInputTransformations() const { return !per_input_transformations_.empty(); }

  // Output transformation metadata accessors (for epilogue)
  void SetOutputTransformations(
      std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> transforms) {
    per_output_transformations_ = std::move(transforms);
  }

  const std::vector<std::vector<c10_neuron::lazy::TensorTransformation>>& GetOutputTransformations()
      const {
    return per_output_transformations_;
  }

  bool HasOutputTransformations() const { return !per_output_transformations_.empty(); }

  // Optional per-tensor NEFF I/O port names. When empty (the default), the
  // runtime falls back to positional ``input{i}`` / ``output{i}`` names —
  // the convention used by NEFFs produced through the torch.compile /
  // StableHLO flow. Callers that supply a NEFF with arbitrary port names
  // (e.g. a hand-authored artifact, an NKI kernel-builder NEFF with
  // ``must_alias_input`` slots) can set these to have the runtime bind
  // tensors by name.
  //
  // Precondition when non-empty: sizes must match the kernel's input /
  // output tensor counts. Ordering must be the same as the tensor
  // ordering passed to the kernel.
  void SetInputNames(std::vector<std::string> names) { input_names_ = std::move(names); }
  void SetOutputNames(std::vector<std::string> names) { output_names_ = std::move(names); }
  const std::vector<std::string>& GetInputNames() const { return input_names_; }
  const std::vector<std::string>& GetOutputNames() const { return output_names_; }

 protected:
  virtual bool ValidateImpl() const;

  // Construct with a pre-computed CompilationCacheKey.
  CompilableKernelExecution(CompilationCacheKey key, const std::string& additional_args,
                            const std::string& optimization_level, bool has_collectives)
      : compile_key_(std::move(key)),
        additional_args_(additional_args),
        optimization_level_(optimization_level),
        has_collectives_(has_collectives) {}

  // Construct from a key string
  CompilableKernelExecution(const std::string& compile_key_str, const std::string& additional_args,
                            const std::string& optimization_level, bool has_collectives)
      : compile_key_(CompilationCacheKey(compile_key_str, additional_args, optimization_level)),
        additional_args_(additional_args),
        optimization_level_(optimization_level),
        has_collectives_(has_collectives) {}

 private:
  CompilationCacheKey compile_key_;
  std::string additional_args_;
  std::string optimization_level_;
  mutable NeffBytesPtr cached_neff_ptr_;
  mutable std::string compilation_error_;
  bool has_collectives_;
  // Per-input transformation metadata (optional, for in-place transformations)
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> per_input_transformations_;
  // Per-output transformation metadata (optional, for epilogue — inverse transformations)
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> per_output_transformations_;
  // Optional NEFF I/O port names — see SetInputNames / SetOutputNames.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

// XLA-based kernels (StableHLO compilation path).
class XLACompilableKernelExecution
    : public CompilableKernelExecution,
      public NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  XLACompilableKernelExecution(const std::string& op_name, std::vector<TensorDataRef>&& input_refs,
                               std::vector<TensorDataRef>&& output_refs,
                               const std::vector<TensorContext>& input_contexts,
                               const std::vector<TensorContext>& output_contexts,
                               const std::string& base_cache_key,
                               const std::vector<uint8_t>& hlo_bytes, bool has_collectives,
                               int device_id);

  // CompilableKernelExecution interface implementation
  std::vector<uint8_t> CompileToNeff() const override;

  // Override to return HLO bytes for persistent cache key generation
  const std::vector<uint8_t>& GetHloBytes() const override { return hlo_bytes_; }

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;

  // Update methods for in-place kernel modification, used in fast path pre-processing step.
  void UpdateHloBytes(const std::vector<uint8_t>& new_hlo_bytes) {
    hlo_bytes_ = new_hlo_bytes;
    // Invalidate cached persistent key if already computed.
    GetCompilationCacheKey().InvalidateCachedStrings();
  }

  // Update source data pointers and contexts (for input deduplication)
  void UpdateSrcData(std::vector<TensorDataRef>&& new_src_refs,
                     const std::vector<TensorContext>& new_src_ctxs) {
    src_data_refs_ = std::move(new_src_refs);
    src_tensor_ctxs_ = new_src_ctxs;
    // Rebuild base class src_ptrs_ to match (used by GetSrcPtrs())
    src_ptrs_.clear();
    src_ptrs_.reserve(src_data_refs_.size());
    for (const auto& ref : src_data_refs_) {
      src_ptrs_.push_back(ref.ptr);
    }
  }

  const std::vector<TensorDataRef>& GetSrcRefs() const { return src_data_refs_; }
  const std::vector<TensorDataRef>& GetDstRefs() const { return dst_data_refs_; }

  const std::vector<TensorContext>& GetInputContexts() const { return GetSrcTensorContexts(); }
  const std::vector<TensorContext>& GetOutputContexts() const { return GetDstTensorContexts(); }

 protected:
  bool ValidateImpl() const override;

 private:
  // Returns true when any input or output context has dtype Long (i64).
  // Used to gate the int64-policy compile-flag emission so that ops
  // whose I/O has no int64 don't pick up int64-related compiler flags.
  // See implementation comment in KernelExecution.cpp for rationale.
  static bool HasInt64IO(const std::vector<TensorContext>& input_contexts,
                         const std::vector<TensorContext>& output_contexts);

 public:
  // Get XLA compilation configuration from environment.
  // ``has_int64_io`` gates int64-policy flag emission; defaults to true to
  // preserve legacy behavior for any caller that hasn't been updated to
  // pass the flag.
  static std::string GetXLAAdditionalArgs(const std::string& op_name = "",
                                          bool has_int64_io = true);
  static std::string GetXLAOptimizationLevel();

 private:
  std::vector<uint8_t> hlo_bytes_;
};

// torch.compile compilation kernels.
class CompileOnlyKernelExecution : public CompilableKernelExecution, public NeuronKernelExecution {
 public:
  // Default constructor uses GetCompileAdditionalArgs() / GetCompileOptimizationLevel().
  CompileOnlyKernelExecution(const std::string& base_cache_key,
                             const std::vector<uint8_t>& mlir_bytes, bool has_collectives,
                             bool use_tensorizer = false);

  // Constructor with explicit compiler config overrides. When additional_args/optimization_level
  // are non-empty, they replace the defaults. Use
  // XLACompilableKernelExecution::GetXLAAdditionalArgs() and GetXLAOptimizationLevel() to match the
  // eager dispatch path.
  CompileOnlyKernelExecution(const std::string& base_cache_key,
                             const std::vector<uint8_t>& mlir_bytes, bool has_collectives,
                             const std::string& additional_args,
                             const std::string& optimization_level, bool use_tensorizer = false);

  std::vector<uint8_t> CompileToNeff() const override;

  const std::vector<uint8_t>& GetHloBytes() const override { return mlir_bytes_; }

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override {
    return false;
  }

  bool RequiresCompilation() const override { return true; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kHLO; }

  bool IsValid() const { return ValidateImpl(); }

  // Get compile config from environment
  static std::string GetCompileAdditionalArgs();
  static std::string GetCompileOptimizationLevel();

  // Target captured at submit time; async workers compile for this, not live env.
  const std::string& GetCapturedPlatformTarget() const { return platform_target_; }

 protected:
  bool ValidateImpl() const override;

 private:
  std::vector<uint8_t> mlir_bytes_;
  bool use_tensorizer_;
  // Captured at submit time so async compile uses the correct target/LNC.
  std::string platform_target_;
  std::string logical_neuron_cores_;
};

// NEFF execution kernel (torch.compile executor)
// Prereq: Cache entry must be present (managed via prior compilation via CompileOnlyKernelExecution
// kernel).
class NeffDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1>,
      public CompilableKernelExecution {
 public:
  NeffDirectKernelExecution(const std::string& graph_name, const std::string& cache_key,
                            std::vector<TensorDataRef>&& input_refs,
                            std::vector<TensorDataRef>&& output_refs,
                            const std::vector<TensorContext>& input_contexts,
                            const std::vector<TensorContext>& output_contexts, int device_id,
                            bool has_collectives);

  std::vector<uint8_t> CompileToNeff() const override;

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;

  // Forces a cache-lookup.
  // TODO: Expose CompileCache to Binding / Execution engine.
  bool RequiresCompilation() const override { return true; }

 protected:
  bool ValidateImpl() const override;
};

// Direct collective operations
class CollectiveDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kCollective, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  enum class CollectiveType {
    kAllGather,
    kAllReduce,
    kReduceScatter,
    kAllToAll,
    kBroadcast,
    kReduce
  };

  CollectiveDirectKernelExecution(const std::string& op_name, CollectiveType collective_type,
                                  std::vector<TensorDataRef>&& src_refs,
                                  std::vector<TensorDataRef>&& dst_refs,
                                  const std::vector<TensorContext>& input_contexts,
                                  const std::vector<TensorContext>& output_contexts,
                                  c10::ScalarType dtype, c10d::ReduceOp reduce_op,
                                  const std::vector<std::vector<int>>& replica_groups,
                                  int device_id);

  // Collective-specific methods
  CollectiveType GetCollectiveType() const { return collective_type_; }
  c10d::ReduceOp GetReduceOp() const { return reduce_op_; }
  const std::vector<std::vector<int>>& GetReplicaGroups() const { return replica_groups_; }
  c10::ScalarType GetDtype() const { return dtype_; }
  int GetMyRank() const { return my_rank_; }
  const nrt::NRTCollectiveComm& GetComm() const { return comm_; }

  void Prepare() const override;
  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kCollective; }
  bool RequiresPrepare() const override { return true; }

 protected:
  bool ValidateImpl() const override;
  void PrepareImpl() const;

 private:
  CollectiveType collective_type_;
  c10d::ReduceOp reduce_op_;
  c10::ScalarType dtype_;
  std::vector<std::vector<int>> replica_groups_;

  // Precomputed NRT types (computed once in constructor)
  nrt::NRTDType nrt_dtype_;
  nrt::NRTReduceOp nrt_reduce_op_;
  nrt::NRTCollectiveOp nrt_collective_op_;
  std::vector<int>* my_replica_group_{nullptr};
  int my_rank_{-1};

  // Mutable state for prepare/schedule cycle
  mutable nrt::NRTCollectiveContext* cc_ctx_{nullptr};
  mutable std::vector<nrt_tensor_t*> prepared_src_data_refs_;
  mutable std::vector<nrt_tensor_t*> prepared_dst_data_refs_;
  mutable nrt::NRTTensorList input_tensor_list_{};
  mutable nrt::NRTTensorList output_tensor_list_{};
  mutable nrt::NRTCollectiveComm comm_{};
};

// Hint primitive for fusion optimizations. It merely signals lifecycle events.
class HintDirectKernelExecution : public NeuronKernelExecution {
 public:
  enum class HintType {
    kAllocation,    // Tensor was allocated
    kDeallocation,  // Tensor was deallocated
  };

  HintDirectKernelExecution(const std::string& op_name, void* ptr, size_t size_bytes, int device_id,
                            HintType hint_type);

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override {
    // No-op for hints - nothing to execute or schedule
    return false;
  }
  bool RequiresCompilation() const override { return false; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kHint; }

  HintType GetHintType() const { return hint_type_; }
  size_t GetSizeBytes() const { return size_bytes_; }
  void* GetPtr() const { return ptr_; }

 protected:
  bool ValidateImpl() const override { return size_bytes_ > 0; }

 private:
  void* ptr_;
  size_t size_bytes_;
  HintType hint_type_;
};

class EventDirectKernelExecution : public NeuronKernelExecution {
 public:
  enum class EventAction {
    kSignal,  // Signals/completes the event when executed
    kWait     // Waits for the event to be signaled
  };

  EventDirectKernelExecution(const std::string& op_name, const NeuronEvent& event,
                             EventAction action, int device_id);

  // Events are host-side operations - Schedule handles async execution
  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;
  bool RequiresCompilation() const override { return false; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kEvent; }

  void CompleteWithoutExecution() const override {
    if (action_ == EventAction::kSignal) {
      event_.complete_event();
    }
  }

  // Event-specific methods
  EventAction GetEventAction() const { return action_; }
  const NeuronEvent& GetEvent() const { return event_; }
  virtual bool IsReady() const;
  c10::StreamId GetSignalStreamId() const { return signal_stream_id_; }

 protected:
  bool ValidateImpl() const override;

 private:
  NeuronEvent event_;
  EventAction action_;
  c10::StreamId signal_stream_id_;
};

class AllocDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kAlloc, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  // One-shot forward state machine: Fresh -> Deferred -> RetryReady.
  enum class AllocState {
    Fresh,
    Deferred,    // 1st OOM; gated until stream drains
    RetryReady,  // Drain hook completed; next OOM throws (no re-defer)
  };

  AllocDirectKernelExecution(const std::string& op_name, TensorDataRef dst_ref, size_t size_bytes,
                             int device_id);
  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;
  size_t GetSizeBytes() const { return size_bytes_; }
  const NeuronEvent& GetEvent() const { return event_op_.GetEvent(); }

  AllocState GetAllocState() const { return alloc_state_.load(std::memory_order_acquire); }
  void SetAllocState(AllocState s) { alloc_state_.store(s, std::memory_order_release); }

 protected:
  bool ValidateImpl() const override { return HasOutputs() && size_bytes_ > 0; }

 private:
  size_t size_bytes_;
  EventDirectKernelExecution event_op_;
  std::atomic<AllocState> alloc_state_{AllocState::Fresh};
};

class DeallocDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kDealloc, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  DeallocDirectKernelExecution(const std::string& op_name, TensorDataRef src_ref, size_t size_bytes,
                               int device_id);
  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;
  size_t GetSizeBytes() const { return size_bytes_; }
  const NeuronEvent& GetEvent() const { return event_op_.GetEvent(); }

 protected:
  bool ValidateImpl() const override { return HasInputs() && size_bytes_ > 0; }

 private:
  size_t size_bytes_;
  EventDirectKernelExecution event_op_;
};

class CopyDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  CopyDirectKernelExecution(const std::string& op_name, TensorDataRef src_ref,
                            TensorDataRef dst_ref, size_t src_offset_bytes, size_t dst_offset_bytes,
                            size_t size_bytes, int device_id);

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t src_offset_bytes_;
  size_t dst_offset_bytes_;
  size_t size_bytes_;
};

class WriteDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kWrite, c10::DeviceType::CPU,
                                        c10::DeviceType::PrivateUse1> {
 public:
  struct alignas(64) CacheAlignedBlock {
    CacheAlignedBlock() {}
    char data[64];
  };
  using BounceBuffer = std::vector<CacheAlignedBlock>;

  WriteDirectKernelExecution(const std::string& op_name, void* src_ptr, TensorDataRef dst_ref,
                             size_t dst_offset_bytes, size_t dst_size_bytes, int device_id);

  // Scalar-specific source buffer for write primitive
  WriteDirectKernelExecution(const std::string& op_name, const c10::Scalar& scalar_value,
                             TensorDataRef dst_ref, size_t dst_offset_bytes, c10::ScalarType dtype,
                             int device_id);

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;

  // Allocate bounce buffer for non-blocking writes.
  void* AllocateBounceBuffer(size_t size_bytes);

 protected:
  bool ValidateImpl() const override;

 private:
  size_t dst_offset_bytes_;
  size_t dst_size_bytes_;
  BounceBuffer bounce_buffer_;
};

class ReadDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kRead, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::CPU> {
 public:
  ReadDirectKernelExecution(const std::string& op_name, TensorDataRef src_ref, void* dst_ptr,
                            size_t src_offset_bytes, size_t src_size_bytes, int device_id);

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;

  void CompleteWithoutExecution() const override {
    TORCH_WARN_ONCE(
        "Neuron compile-only: device-to-host read returns placeholder data because no device "
        "executed the graph. Do not use the value for data-dependent control flow.");
  }

 protected:
  bool ValidateImpl() const override;

 private:
  size_t src_offset_bytes_;
  size_t src_size_bytes_;
};

class BarrierDirectKernelExecution : public NeuronKernelExecution {
 public:
  BarrierDirectKernelExecution(int rank, int world_size, int device_id);

  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override;
  bool RequiresCompilation() const override { return false; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kBarrier; }

 protected:
  bool ValidateImpl() const override { return true; }

 private:
  int rank_;
  int world_size_;
};

// THE single host-compute kernel — runs an arbitrary closure on a
// HostWorkerPool worker thread.
class HostKernelExecution final
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kHost, c10::DeviceType::CPU,
                                        c10::DeviceType::CPU> {
 public:
  using HostFn = std::function<void(const std::vector<TensorDataRef>& srcs,
                                    const std::vector<TensorDataRef>& dsts)>;

  HostKernelExecution(const std::string& op_name, std::vector<TensorDataRef> src_refs,
                      std::vector<TensorDataRef> dst_refs, HostFn fn, int device_id,
                      std::vector<at::Tensor> keepalives = {})
      : NeuronKernelExecutionTyped(op_name, std::move(src_refs), std::move(dst_refs), device_id),
        fn_(std::move(fn)),
        keepalives_(std::move(keepalives)) {}

  // Called on a HostWorkerPool worker thread.
  void RunHost() { fn_(src_data_refs_, dst_data_refs_); }

  // Unreachable: kHost ops go to HostWorkerPool::Submit, never the device-XU
  // Schedule() path. Assert so a mis-route surfaces loudly.
  bool Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                c10::StreamId stream_id) const override {
    EXECUTION_RUNTIME_CHECK(false, "HostKernelExecution::Schedule() is unreachable");
    return false;
  }

  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kHost; }
  bool RequiresCompilation() const override { return false; }

  // Non-copyable/movable: copying would slice the fn_ closure and keepalives.
  // OperationContext owns this via unique_ptr and never copies it.
  HostKernelExecution(const HostKernelExecution&) = delete;
  HostKernelExecution& operator=(const HostKernelExecution&) = delete;
  HostKernelExecution(HostKernelExecution&&) = delete;
  HostKernelExecution& operator=(HostKernelExecution&&) = delete;

 protected:
  // Valid iff a host fn was supplied. Buffer/shape validation is the op's
  // responsibility inside its closure (the engine has no per-op knowledge here).
  bool ValidateImpl() const override { return static_cast<bool>(fn_); }

 private:
  HostFn fn_;
  std::vector<at::Tensor> keepalives_;
};

}  // namespace at::neuron
