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

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/OperationContext.h"

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
   * @param kernel The compilable kernel context containing all necessary information
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

  ~CPUFallbackExecutor() = default;

  CPUFallbackExecutor(const CPUFallbackExecutor&) = delete;
  CPUFallbackExecutor& operator=(const CPUFallbackExecutor&) = delete;

 protected:
  /**
   * Transfer tensors from CPU device to Neuron
   */
  virtual void TransferTensorsFromCpuToNeuron(const std::vector<torch::Tensor>& cpu_results,
                                              const std::vector<nrt_tensor_t*>& data_ptrs,
                                              const std::vector<TensorContext>& tensor_contexts);

  /**
   * Transfer tensors from Neuron device to CPU
   */
  std::vector<torch::Tensor> TransferTensorsToCpu(
      const std::vector<nrt_tensor_t*>& data_ptrs,
      const std::vector<TensorContext>& tensor_contexts);

 private:
  /**
   * Execute the actual operation on CPU using PyTorch dispatcher
   */
  std::vector<torch::Tensor> ExecuteOperationOnCpu(const std::string& op_name,
                                                   const std::vector<torch::Tensor>& cpu_inputs,
                                                   const std::vector<torch::Tensor>& cpu_outputs,
                                                   const OperationContext* op);

  /**
   * Parse operation name to get the actual ATen operator
   */
  c10::optional<c10::OperatorHandle> GetCpuOperator(const std::string& op_name) const;

  /**
   * Create CPU tensors with the same properties from TensorContext
   */
  torch::Tensor CreateCpuTensorLike(const TensorContext& tensor_context);

  /**
   * Load configuration from environment variables
   */
  void LoadConfiguration();

  /**
   * Implementation method for CPU fallback execution
   */
  void ExecuteCpuFallbackImpl(OperationContext* op);

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
                                                bool has_out_params,
                                                const OperationContext* op) const;

  bool enabled_;
};

}  // namespace at::neuron
