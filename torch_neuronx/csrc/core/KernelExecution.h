#pragma once

#include <nrt/nrt.h>
#include <torch/torch.h>

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
#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
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
  kHint,  // Merely a non-functional hint
};

// Check if a kernel type represents a device operation (vs host-side operation).
// Device operations: kHLO, kCollective, kCopy, kWrite, kRead
// Host-side operations: kEvent, kHint
inline bool IsDeviceKernelType(KernelTypeEnum kernel_type) {
  switch (kernel_type) {
    case KernelTypeEnum::kHLO:
    case KernelTypeEnum::kCollective:
    case KernelTypeEnum::kCopy:
    case KernelTypeEnum::kWrite:
    case KernelTypeEnum::kRead:
      return true;
    case KernelTypeEnum::kEvent:
    case KernelTypeEnum::kHint:
      return false;
  }
  return false;
}

// Convert device KernelTypeEnum to array index (0-4) for completion tracking.
// Only valid for device kernel types
inline size_t GetDeviceKernelTypeIndex(KernelTypeEnum kernel_type) {
  switch (kernel_type) {
    case KernelTypeEnum::kHLO:
      return 0;
    case KernelTypeEnum::kCollective:
      return 1;
    case KernelTypeEnum::kCopy:
      return 2;
    case KernelTypeEnum::kWrite:
      return 3;
    case KernelTypeEnum::kRead:
      return 4;
    default:
      TORCH_CHECK(false,
                  "Invalid device kernel type for indexing: ", static_cast<int>(kernel_type));
  }
}

// Type alias for shorter names
using NeuronTensorPtr = c10_neuron::NeuronCachingAllocator::TensorPtr;

// Data reference for tensor pointers
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

  // Execute kernel synchronously. Default calls ExecuteOrSchedule with nullptr (sync mode).
  virtual void Execute() const { ExecuteOrSchedule(nullptr, nullptr, 0); }

  virtual bool RequiresCompilation() const = 0;
  virtual KernelTypeEnum GetKernelType() const = 0;

  const std::vector<void*>& GetSrcPtrs() const { return src_ptrs_; }
  const std::vector<void*>& GetDstPtrs() const { return dst_ptrs_; }

  // Prepares kernel state before dispatch. Override for operations requiring setup (e.g.,
  // collectives).
  virtual void Prepare() const {}

  // Executes or schedules kernel. When err_tracker is nullptr, executes synchronously.
  // When err_tracker is provided, schedules asynchronously and returns sequence_id.
  virtual void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                                 c10::StreamId stream_id) const = 0;

  // Check if this operation requires Prepare() to be called before Dispatch()
  virtual bool RequiresPrepare() const { return false; }

 protected:
  NeuronKernelExecution(const std::string& op_name, int device_id)
      : op_name_(op_name), device_id_(device_id) {}

  NeuronKernelExecution(const std::string& op_name, const std::vector<void*>& src_ptrs,
                        const std::vector<void*>& dst_ptrs, int device_id)
      : op_name_(op_name), src_ptrs_(src_ptrs), dst_ptrs_(dst_ptrs), device_id_(device_id) {}

  virtual bool ValidateImpl() const { return true; }

  bool HasInputs() const { return !src_ptrs_.empty(); }
  bool HasOutputs() const { return !dst_ptrs_.empty(); }

  std::string op_name_;
  std::vector<void*> src_ptrs_;
  std::vector<void*> dst_ptrs_;
  int device_id_;
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

  SrcStorageVec GetSrcDataPtrs() const { return ExtractDataPtrs<SrcStorageVec>(src_data_ptrs_); }
  DstStorageVec GetDstDataPtrs() const { return ExtractDataPtrs<DstStorageVec>(dst_data_ptrs_); }

  const std::vector<TensorContext>& GetSrcTensorContexts() const { return src_tensor_ctxs_; }
  const std::vector<TensorContext>& GetDstTensorContexts() const { return dst_tensor_ctxs_; }

 protected:
  // Constructor without TensorContext (for kernels that don't need slicing)
  NeuronKernelExecutionTyped(const std::string& op_name, std::vector<TensorDataRef>&& src_refs,
                             std::vector<TensorDataRef>&& dst_refs, int device_id)
      : NeuronKernelExecution(op_name, ExtractPtrs(src_refs), ExtractPtrs(dst_refs), device_id),
        src_data_ptrs_(std::move(src_refs)),
        dst_data_ptrs_(std::move(dst_refs)) {}
  // Constructor with TensorContext (for kernels that need slicing support)
  NeuronKernelExecutionTyped(const std::string& op_name, std::vector<TensorDataRef>&& src_refs,
                             std::vector<TensorDataRef>&& dst_refs,
                             const std::vector<TensorContext>& src_contexts,
                             const std::vector<TensorContext>& dst_contexts, int device_id)
      : NeuronKernelExecution(op_name, ExtractPtrs(src_refs), ExtractPtrs(dst_refs), device_id),
        src_data_ptrs_(std::move(src_refs)),
        dst_data_ptrs_(std::move(dst_refs)),
        src_tensor_ctxs_(src_contexts),
        dst_tensor_ctxs_(dst_contexts) {}

  // Contiguous slicing support for Neuron tensors
  std::vector<nrt_tensor_t*> HandleContiguousSlicing(
      const std::vector<nrt_tensor_t*>& data_ptrs,
      const std::vector<TensorContext>& tensor_ctxs) const {
    TORCH_CHECK(data_ptrs.size() == tensor_ctxs.size(),
                "Mismatched data_ptrs and tensor_ctxs sizes");
    std::vector<nrt_tensor_t*> sliced_data_ptrs;
    sliced_data_ptrs.reserve(data_ptrs.size());
    for (size_t i = 0; i < data_ptrs.size(); ++i) {
      const auto& tensor_ctx = tensor_ctxs[i];
      size_t offset_bytes = tensor_ctx.storage_offset * tensor_ctx.element_size;
      size_t nrt_size = nrt_tensor_get_size(data_ptrs[i]);
      if (offset_bytes > 0 || tensor_ctx.size_bytes != nrt_size) {
        nrt_tensor_t* slice = nullptr;
        NRT_STATUS status = nrt_tensor_allocate_slice(data_ptrs[i], offset_bytes,
                                                      tensor_ctx.size_bytes, nullptr, &slice);
        TORCH_CHECK(status == NRT_SUCCESS && slice, "Failed to create slice: offset=", offset_bytes,
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

  std::vector<TensorDataRef> src_data_ptrs_;
  std::vector<TensorDataRef> dst_data_ptrs_;

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
      for (const auto& ref : refs) {
        data_ptrs.push_back(ref.tensor_ptr.get());
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
  const std::string& GetCacheKey() const { return cache_key_; }
  void UpdateCacheKey(const std::string& new_cache_key) { cache_key_ = new_cache_key; }
  bool HasCachedNeff() const { return cached_neff_ptr_ != nullptr; }
  bool HasCollectives() const { return has_collectives_; }
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

 protected:
  virtual bool ValidateImpl() const;

  // Helper to build cache key that is aware of different compilation arguments.
  static std::string GenerateCacheKey(const std::string& base_key,
                                      const std::string& additional_args,
                                      const std::string& optimization_level);

  CompilableKernelExecution(const std::string& cache_key, const std::string& additional_args,
                            const std::string& optimization_level, bool has_collectives)
      : cache_key_(cache_key),
        additional_args_(additional_args),
        optimization_level_(optimization_level),
        has_collectives_(has_collectives) {}

 private:
  // Creates a compiler-specific extension hash from compilation parameters.
  static std::string CreateCompilerExtensionHash(const std::string& additional_args,
                                                 const std::string& optimization_level);

  std::string cache_key_;
  std::string additional_args_;
  std::string optimization_level_;
  mutable NeffBytesPtr cached_neff_ptr_;
  bool has_collectives_;
  // Per-input transformation metadata (optional, for in-place transformations)
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> per_input_transformations_;
};

// XLA-based kernels (JAX/HLO compilation path).
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

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override;

  // Update methods for in-place kernel modification, used in fast path pre-processing step.
  void UpdateHloBytes(const std::vector<uint8_t>& new_hlo_bytes) { hlo_bytes_ = new_hlo_bytes; }

  const std::vector<TensorContext>& GetInputContexts() const { return GetSrcTensorContexts(); }
  const std::vector<TensorContext>& GetOutputContexts() const { return GetDstTensorContexts(); }

 protected:
  bool ValidateImpl() const override;

 private:
  // Get XLA compilation configuration from environment
  static std::string GetXLAAdditionalArgs();
  static std::string GetXLAOptimizationLevel();

  std::vector<uint8_t> hlo_bytes_;
};

// torch.compile compilation kernels (StableHLO)
class CompileOnlyKernelExecution : public CompilableKernelExecution, public NeuronKernelExecution {
 public:
  CompileOnlyKernelExecution(const std::string& base_cache_key,
                             const std::vector<uint8_t>& stablehlo_bytes, bool has_collectives);

  std::vector<uint8_t> CompileToNeff() const override;

  // Override to return StableHLO bytes for persistent cache key generation
  const std::vector<uint8_t>& GetHloBytes() const override { return stablehlo_bytes_; }

  // No-op. Currently in place due to OpExecutionEngine semantics.
  // TODO: Dispatch reliably to Compile worker.
  void Execute() const override {}
  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override {}

  bool RequiresCompilation() const override { return true; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kHLO; }

  bool IsValid() const { return ValidateImpl(); }

  // Get compile config from environment
  static std::string GetCompileAdditionalArgs();
  static std::string GetCompileOptimizationLevel();

 protected:
  bool ValidateImpl() const override;

 private:
  std::vector<uint8_t> stablehlo_bytes_;
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

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
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

  void Execute() const override;
  void Prepare() const override;
  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override;
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kCollective; }
  bool RequiresPrepare() const override { return true; }

 protected:
  bool ValidateImpl() const override;

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
  mutable std::vector<nrt_tensor_t*> prepared_src_data_ptrs_;
  mutable std::vector<nrt_tensor_t*> prepared_dst_data_ptrs_;
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

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override {
    // No-op for hints - nothing to execute or schedule
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

class CopyDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  CopyDirectKernelExecution(const std::string& op_name, TensorDataRef src_ref,
                            TensorDataRef dst_ref, size_t src_offset_bytes, size_t dst_offset_bytes,
                            size_t size_bytes, int device_id);

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
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

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
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

  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t src_offset_bytes_;
  size_t src_size_bytes_;
};

class EventDirectKernelExecution : public NeuronKernelExecution {
 public:
  enum class EventAction {
    kSignal,  // Signals/completes the event when executed
    kWait     // Waits for the event to be signaled
  };

  EventDirectKernelExecution(const std::string& op_name, const NeuronEvent& event,
                             EventAction action, int device_id);

  // Events are host-side operations - ExecuteOrSchedule handles sync execution
  void ExecuteOrSchedule(nrt::ErrorTrackerHandle* err_tracker, nrt::SequenceId* sequence_id,
                         c10::StreamId stream_id) const override;
  bool RequiresCompilation() const override { return false; }
  KernelTypeEnum GetKernelType() const override { return KernelTypeEnum::kEvent; }

  // Event-specific methods
  EventAction GetEventAction() const { return action_; }
  const NeuronEvent& GetEvent() const { return event_; }
  bool IsReady() const;

 protected:
  bool ValidateImpl() const override;

 private:
  NeuronEvent event_;
  EventAction action_;
};

}  // namespace at::neuron
