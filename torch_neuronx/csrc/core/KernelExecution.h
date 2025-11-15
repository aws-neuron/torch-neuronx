#pragma once

#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"

namespace at::neuron {

// Forward declarations
class CompilationCache;

// Base class for all kernel execution types in the Neuron backend.
// Provides common interface for different execution strategies.
class NeuronKernelExecution {
 public:
  virtual ~NeuronKernelExecution() = default;

  virtual bool RequiresCompilation() const = 0;

  // Common methods for all kernel types
  virtual bool IsValid() const { return !op_name_.empty() && ValidateImpl(); }

  // Common accessors
  const std::vector<torch::Tensor>& GetInputs() const { return inputs_; }
  const std::vector<torch::Tensor>& GetOutputs() const { return outputs_; }

  // Tensor output mapping, in case we have an internal tensor due to an implicit dtype conversion.
  const std::vector<torch::Tensor>& GetInternalOutputs() const { return internal_outputs_; }

  // Type conversion methods for Neuron compatibility - common to all kernels
  torch::Tensor ConvertToNeuronDtype(const torch::Tensor& tensor) const;
  static constexpr bool NeedsDtypeConversion(torch::ScalarType dtype) noexcept {
    return dtype == torch::kInt64 || dtype == torch::kFloat64;
  }

  std::string GetOpName() const { return op_name_; }

  virtual void Execute() const = 0;

 protected:
  NeuronKernelExecution(const std::string& op_name, const std::vector<torch::Tensor>& inputs,
                        const std::vector<torch::Tensor>& outputs);

  virtual bool ValidateImpl() const { return true; }

 private:
  std::string op_name_;
  std::vector<torch::Tensor> inputs_;
  std::vector<torch::Tensor> outputs_;
  // A mapping of original to internal tensor, in case we are implicitly treating a tensor
  // with a different dtype.
  std::vector<torch::Tensor> internal_outputs_;
};

// Abstract base for kernels that need compilation.
class CompilableKernelExecution : public NeuronKernelExecution {
 public:
  bool RequiresCompilation() const override { return true; }

  // Compilation-specific interface
  virtual std::vector<uint8_t> CompileToNeff() const = 0;

  // Cache management methods
  const std::vector<uint8_t>& GetCachedNeff() const { return *cached_neff_ptr_; }
  void SetCachedNeff(NeffBytesPtr neff_ptr) const { cached_neff_ptr_ = std::move(neff_ptr); }
  const std::string& GetCacheKey() const { return cache_key_; }
  bool HasCachedNeff() const { return cached_neff_ptr_ != nullptr; }

  bool HasCollectives() const { return has_collectives_; }

  // Access to compilation configurations
  const std::string& GetAdditionalArgs() const { return additional_args_; }
  const std::string& GetOptimizationLevel() const { return optimization_level_; }

  void Execute() const override;

 protected:
  bool ValidateImpl() const override;

  // Helper to build cache key that is aware of different compilation arguments.
  static std::string GenerateCacheKey(const std::string& base_key,
                                      const std::string& additional_args,
                                      const std::string& optimization_level);

 private:
  // Creates a compiler-specific extension hash from compilation parameters.
  // This hash captures compiler settings (target, cores, optimization, args)
  // and can be appended to a kernel execution's cache key to create a
  // complete compilation cache key.
  static std::string CreateCompilerExtensionHash(const std::string& additional_args,
                                                 const std::string& optimization_level);

 protected:
  CompilableKernelExecution(const std::string& op_name, const std::vector<torch::Tensor>& inputs,
                            const std::vector<torch::Tensor>& outputs, const std::string& cache_key,
                            const std::string& additional_args,
                            const std::string& optimization_level, bool has_collectives)
      : NeuronKernelExecution(op_name, inputs, outputs),
        cache_key_(cache_key),
        additional_args_(additional_args),
        optimization_level_(optimization_level),
        has_collectives_(has_collectives) {}

 private:
  std::string cache_key_;           // Cache key with compiler extension (built at construction)
  std::string additional_args_;     // Compiler args (cached at construction)
  std::string optimization_level_;  // Optimization level (cached at construction)
  // Smart pointer to cached NEFF bytes with automatic cleanup
  mutable NeffBytesPtr cached_neff_ptr_;
  bool has_collectives_;
};

// Abstract base for kernels that execute directly (no compilation needed).
class DirectKernelExecution : public NeuronKernelExecution {
 public:
  enum class KernelType {
    kCollective,
  };

  bool RequiresCompilation() const override { return false; }

  // Get the type of direct kernel
  virtual KernelType GetKernelType() const = 0;

 protected:
  DirectKernelExecution(const std::string& op_name, const std::vector<torch::Tensor>& inputs,
                        const std::vector<torch::Tensor>& outputs)
      : NeuronKernelExecution(op_name, inputs, outputs) {}
};

// XLA-based kernels (JAX/HLO compilation path).
class XLACompilableKernelExecution : public CompilableKernelExecution {
 public:
  XLACompilableKernelExecution(const std::string& op_name, const std::vector<torch::Tensor>& inputs,
                               const std::vector<torch::Tensor>& outputs,
                               const std::string& base_cache_key,
                               const std::vector<uint8_t>& hlo_bytes, bool has_collectives);

  // Base class implementations
  std::vector<uint8_t> CompileToNeff() const override;

  // XLA-specific methods
  const std::vector<uint8_t>& GetHloBytes() const { return hlo_bytes_; }

 protected:
  bool ValidateImpl() const override;

 private:
  // Get XLA compilation configuration from environment
  static std::string GetXLAAdditionalArgs();
  static std::string GetXLAOptimizationLevel();

  std::vector<uint8_t> hlo_bytes_;
};

// NKI-based kernels (direct kernel compilation).
class NKICompilableKernelExecution : public CompilableKernelExecution {
 public:
  NKICompilableKernelExecution(const std::string& op_name, const std::string& kernel_name,
                               const std::string& module_path,
                               const std::vector<torch::Tensor>& inputs,
                               const std::vector<torch::Tensor>& outputs,
                               const std::string& base_cache_key, bool has_collectives);

  // Base class implementations
  std::vector<uint8_t> CompileToNeff() const override;

  // NKI-specific methods
  const std::string& GetKernelName() const { return nki_kernel_name_; }
  const std::string& GetModulePath() const { return nki_module_path_; }

 protected:
  bool ValidateImpl() const override;

 private:
  // Get NKI compilation configuration from environment
  static std::string GetNKIAdditionalArgs();
  static std::string GetNKIOptimizationLevel();

  std::string nki_kernel_name_;
  std::string nki_module_path_;
};

// Direct collective operations (no compilation needed).
class CollectiveDirectKernelExecution : public DirectKernelExecution {
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
                                  const std::vector<torch::Tensor>& inputs,
                                  const std::vector<torch::Tensor>& outputs,
                                  c10d::ReduceOp reduce_op);

  // Base class implementations
  KernelType GetKernelType() const override { return KernelType::kCollective; }

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

}  // namespace at::neuron
