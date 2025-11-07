#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuron/csrc/core/NeuronLogging.h"
#include "torch_neuron/csrc/core/OperationContext.h"

namespace at::neuron {

/**
 * CPU Fallback Executor - handles execution of operations on CPU when Neuron execution fails
 */
class CPUFallbackExecutor {
 public:
  CPUFallbackExecutor();

  /**
   * Execute an operation on CPU as fallback
   *
   * @param op The operation context containing all necessary information
   * @return OperationContextResult with CPU execution results
   */
  OperationContextResult ExecuteCpuFallback(OperationContext* op);

  /**
   * Check if an operation can be executed on CPU
   *
   * @param op_name The operation name
   * @return true if CPU fallback is possible
   */
  bool CanExecuteOnCpu(const std::string& op_name) const;

  /**
   * Enable/disable CPU fallback globally
   */
  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

  /**
   * Get statistics about CPU fallback usage
   */
  struct Stats {
    uint64_t total_fallback_attempts = 0;
    uint64_t successful_fallbacks = 0;
    uint64_t failed_fallbacks = 0;
    std::chrono::microseconds total_cpu_execution_time{0};
    std::chrono::microseconds total_transfer_time{0};
  };

  Stats GetStatistics() const;
  void ResetStatistics();

  ~CPUFallbackExecutor() = default;

  CPUFallbackExecutor(const CPUFallbackExecutor&) = delete;
  CPUFallbackExecutor& operator=(const CPUFallbackExecutor&) = delete;

 private:
  /**
   * Transfer tensors from Neuron device to CPU
   */
  std::vector<torch::Tensor> TransferTensorsToCpu(const std::vector<torch::Tensor>& neuron_tensors);

  /**
   * Execute the actual operation on CPU using PyTorch dispatcher
   */
  std::vector<torch::Tensor> ExecuteOperationOnCpu(const std::string& op_name,
                                                   const std::vector<torch::Tensor>& cpu_inputs,
                                                   const std::vector<torch::Tensor>& cpu_outputs);

  /**
   * Parse operation name to get the actual ATen operator
   */
  c10::optional<c10::OperatorHandle> GetCpuOperator(const std::string& op_name) const;

  /**
   * Create CPU tensors with the same properties as Neuron tensors
   */
  torch::Tensor CreateCpuTensorLike(const torch::Tensor& neuron_tensor);

  /**
   * Load configuration from environment variables
   */
  void LoadConfiguration();

  /**
   * Implementation method for CPU fallback execution
   */
  void ExecuteCpuFallbackImpl(OperationContext* op);

  /**
   * Stats counter increment
   */
  void IncrementStatsCounter(uint64_t Stats::* counter);

  /**
   * Extract results from the PyTorch stack to vector
   */
  std::vector<torch::Tensor> ExtractResultsFromStack(
      const std::vector<c10::IValue>& stack, const std::vector<torch::Tensor>& expected_outputs,
      const std::string& op_name);

  /**
   * Check if operation has output parameters using PyTorch schema
   */
  bool OperationHasOutParameters(const std::string& op_name) const;

  /**
   * Check if schema has output parameters
   */
  bool SchemaHasOutParameters(const c10::FunctionSchema& schema) const;

  /**
   * Construct native PyTorch stack with proper argument order
   */
  std::vector<c10::IValue> ConstructNativeStack(const std::vector<torch::Tensor>& cpu_inputs,
                                                const std::vector<torch::Tensor>& cpu_outputs,
                                                const c10::FunctionSchema& schema,
                                                bool has_out_params) const;

  /**
   * Handle reusing original Neuron output tensors
   */
  void HandleNeuronOutputTensorReuse(const std::vector<torch::Tensor>& cpu_results,
                                     OperationContext* op);

  // Configuration
  bool enabled_;

  // Statistics
  mutable std::mutex stats_mutex_;
  Stats stats_;
};

}  // namespace at::neuron
