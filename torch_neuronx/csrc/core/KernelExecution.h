#pragma once

#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationTypes.h"

// Forward declaration for NRT types
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

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
  kAlloc,
  kDealloc,
  kCopy,
  kWrite,
  kRead,
  kSlice,
  kEvent,
};

class NeuronKernelExecution {
 public:
  virtual ~NeuronKernelExecution() = default;

  bool IsValid() const { return !op_name_.empty() && ValidateImpl(); }
  std::string GetOpName() const { return op_name_; }

  virtual void Execute() const = 0;
  virtual bool RequiresCompilation() const = 0;
  virtual KernelTypeEnum GetKernelType() const = 0;

 protected:
  NeuronKernelExecution(const std::string& op_name, const std::vector<void*>& src_ptrs,
                        const std::vector<void*>& dst_ptrs, int device_id)
      : op_name_(op_name), src_ptrs_(src_ptrs), dst_ptrs_(dst_ptrs), device_id_(device_id) {}

  virtual bool ValidateImpl() const { return true; }
  int GetDeviceId() const { return device_id_; }

  const std::vector<void*>& GetSrcPtrs() const { return src_ptrs_; }
  const std::vector<void*>& GetDstPtrs() const { return dst_ptrs_; }

  bool HasInputs() const { return !src_ptrs_.empty(); }
  bool HasOutputs() const { return !dst_ptrs_.empty(); }

 private:
  std::string op_name_;
  std::vector<void*> src_ptrs_;
  std::vector<void*> dst_ptrs_;
  int device_id_;
};

template <KernelTypeEnum KernelType, ::c10::DeviceType SrcDevice, ::c10::DeviceType DstDevice>
class NeuronKernelExecutionTyped : public NeuronKernelExecution {
 public:
  friend class CPUFallbackExecutor;

  KernelTypeEnum GetKernelType() const override { return KernelType; }

  template <c10::DeviceType Type>
  using DataPtrType =
      typename std::conditional<Type == c10::DeviceType::CPU, void*, nrt_tensor_t*>::type;

  using SrcDataPtrTypes = std::vector<DataPtrType<SrcDevice>>;
  using DstDataPtrTypes = std::vector<DataPtrType<DstDevice>>;

  bool RequiresCompilation() const override {
    if constexpr (KernelType == KernelTypeEnum::kHLO) {
      return true;
    }
    return false;
  }

 protected:
  NeuronKernelExecutionTyped(const std::string& op_name, const std::vector<void*>& src_ptrs,
                             const std::vector<void*>& dst_ptrs, int device_id)
      : NeuronKernelExecution(op_name, src_ptrs, dst_ptrs, device_id) {}

  SrcDataPtrTypes GetSrcDataPtrs() const {
    const auto& src_ptrs = GetSrcPtrs();
    if constexpr (SrcDevice == ::c10::DeviceType::CPU) {
      return src_ptrs;
    } else if constexpr (SrcDevice == ::c10::DeviceType::PrivateUse1) {
      std::vector<nrt_tensor_t*> nrt_ptrs;
      nrt_ptrs.reserve(src_ptrs.size());
      for (auto* ptr : src_ptrs) {
        nrt_tensor_t* data_ptr = c10_neuron::NeuronCachingAllocator::findTensor(ptr);
        if (!data_ptr) {
          throw std::runtime_error("Invalid input data for " + GetOpName());
        }
        nrt_ptrs.push_back(data_ptr);
      }
      return nrt_ptrs;
    }
    return {};
  }

  DstDataPtrTypes GetDstDataPtrs() const {
    const auto& dst_ptrs = GetDstPtrs();
    if constexpr (DstDevice == ::c10::DeviceType::CPU) {
      return dst_ptrs;
    } else if constexpr (DstDevice == ::c10::DeviceType::PrivateUse1) {
      std::vector<nrt_tensor_t*> nrt_ptrs;
      nrt_ptrs.reserve(dst_ptrs.size());
      for (auto* data_ptr : dst_ptrs) {
        nrt_tensor_t* nrt_ptr = c10_neuron::NeuronCachingAllocator::findTensor(data_ptr);
        if (!nrt_ptr) {
          throw std::runtime_error("Invalid output data for " + GetOpName());
        }
        nrt_ptrs.push_back(nrt_ptr);
      }
      return nrt_ptrs;
    }
    return {};
  }
};

// Base class for all compilable kernels (XLA).
class CompilableKernelExecution {
 public:
  virtual ~CompilableKernelExecution() = default;

  // Compilation-specific interface
  virtual std::vector<uint8_t> CompileToNeff() const = 0;

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
  XLACompilableKernelExecution(const std::string& op_name, const std::vector<void*>& src_ptrs,
                               const std::vector<void*>& dst_ptrs,
                               const std::string& base_cache_key,
                               const std::vector<uint8_t>& hlo_bytes, bool has_collectives,
                               int device_id);

  // CompilableKernelExecution interface implementation
  std::vector<uint8_t> CompileToNeff() const override;

  // XLA-specific methods
  const std::vector<uint8_t>& GetHloBytes() const { return hlo_bytes_; }

  // Public accessors for input/output pointers (returns const references)
  const std::vector<void*>& GetInputPtrs() const { return GetSrcPtrs(); }
  const std::vector<void*>& GetOutputPtrs() const { return GetDstPtrs(); }

  void Execute() const override;
  // Update methods for in-place kernel modification, used in fast path pre-processing step.
  void UpdateHloBytes(const std::vector<uint8_t>& new_hlo_bytes) { hlo_bytes_ = new_hlo_bytes; }

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

  // No-op. Currently in place due to OpExecutionEngine semantics.
  // TODO: Dispatch reliably to Compile worker.
  void Execute() const override {}

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
                            const std::vector<void*>& input_ptrs,
                            const std::vector<void*>& output_ptrs, int device_id,
                            bool has_collectives);

  std::vector<uint8_t> CompileToNeff() const override;

  void Execute() const override;

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
                                  const std::vector<void*>& src_ptrs,
                                  const std::vector<void*>& dst_ptrs, c10d::ReduceOp reduce_op,
                                  int device_id);

  // Collective-specific methods
  CollectiveType GetCollectiveType() const { return collective_type_; }
  c10d::ReduceOp GetReduceOp() const { return reduce_op_; }

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  CollectiveType collective_type_;
  c10d::ReduceOp reduce_op_;
};

class AllocDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kAlloc, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  AllocDirectKernelExecution(const std::string& op_name, void* ptr, size_t size_bytes,
                             int device_id);

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t size_bytes_;
};

class DeallocDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kDealloc, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  DeallocDirectKernelExecution(const std::string& op_name, void* src_ptr, size_t size_bytes,
                               int device_id);

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t size_bytes_;
};

class CopyDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  CopyDirectKernelExecution(const std::string& op_name, void* src_ptr, void* dst_ptr,
                            size_t src_offset_bytes, size_t dst_offset_bytes, size_t size_bytes,
                            int device_id);

  void Execute() const override;

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
  // Default write primitive
  WriteDirectKernelExecution(const std::string& op_name, void* src_ptr, void* dst_ptr,
                             size_t dst_offset_bytes, size_t dst_size_bytes, int device_id);

  // Scalar-specific source buffer for write primitive
  WriteDirectKernelExecution(const std::string& op_name, const c10::Scalar& scalar_value,
                             void* dst_ptr, size_t dst_offset_bytes, c10::ScalarType dtype,
                             int device_id);

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t dst_offset_bytes_;
  size_t dst_size_bytes_;
  alignas(16) uint8_t inline_buffer_[16] = {};  // 16 bytes for complex128
};

class ReadDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kRead, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::CPU> {
 public:
  ReadDirectKernelExecution(const std::string& op_name, void* src_ptr, void* dst_ptr,
                            size_t src_offset_bytes, size_t src_size_bytes, int device_id);

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t src_offset_bytes_;
  size_t src_size_bytes_;
};

class SliceDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kSlice, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  SliceDirectKernelExecution(const std::string& op_name, void* src_ptr, void* dst_ptr,
                             size_t src_offset_bytes, size_t dst_size_bytes, int device_id);

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

 private:
  size_t src_offset_bytes_;
  size_t dst_size_bytes_;
};

class EventDirectKernelExecution
    : public NeuronKernelExecutionTyped<KernelTypeEnum::kEvent, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  enum class EventAction {
    kSignal,  // Signals/completes the event when executed
    kWait     // Waits for the event to be signaled
  };

  EventDirectKernelExecution(const std::string& op_name, const NeuronEvent& event,
                             EventAction action, int device_id);

  void Execute() const override;

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
