#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <string>

#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"

// Forward declarations
class CompilableKernelExecution;

// Forward declarations
namespace at::neuron {
class StreamImpl;
}

namespace at::neuron {

/**
 * NRTHandler encapsulates all NRT (Neuron Runtime) execution logic,
 * providing a clean separation between stream management and NRT operations.
 *
 * This class handles:
 * - Tensor set preparation for NRT execution
 * - Model loading and caching via NeuronResourceManager
 * - Resource management (semaphore acquisition/release)
 * - NRT execution
 * - Error handling and recovery
 */
class NRTHandler {
 public:
  /**
   * Result of NRT execution attempt
   */
  struct ExecutionResult {
    bool success;
    std::string error_message;
    std::chrono::milliseconds execution_time;

    static ExecutionResult Success(std::chrono::milliseconds exec_time) {
      return {true, "", exec_time};
    }

    static ExecutionResult Error(const std::string& error) {
      return {false, error, std::chrono::milliseconds(0)};
    }
  };

  /**
   * Configuration for NRT execution
   */
  struct ExecutionConfig {
    int device_id = 0;
    int num_cores = 1;

    ExecutionConfig() = default;
    ExecutionConfig(int dev_id, int cores) : device_id(dev_id), num_cores(cores) {}
  };

  // Delete constructors - this is a static utility class
  NRTHandler() = delete;
  ~NRTHandler() = delete;
  NRTHandler(const NRTHandler&) = delete;
  NRTHandler& operator=(const NRTHandler&) = delete;
  NRTHandler(NRTHandler&&) = delete;
  NRTHandler& operator=(NRTHandler&&) = delete;

  /**
   * Execute a compilable kernel using NRT
   *
   * @param compilable_kernel The compilable kernel to execute
   * @param op_name Operation name for logging
   * @param config Execution configuration
   * @return ExecutionResult indicating success/failure and timing
   */
  static ExecutionResult ExecuteCompilableKernel(const CompilableKernelExecution& compilable_kernel,
                                                 const std::string& op_name,
                                                 const ExecutionConfig& config);

 private:
  /**
   * Prepare input tensor set from compilable kernel
   */
  static void PrepareInputTensorSet(const CompilableKernelExecution& compilable_kernel,
                                    nrt::TensorSet& input_set);

  /**
   * Prepare output tensor set from compilable kernel
   */
  static void PrepareOutputTensorSet(const CompilableKernelExecution& compilable_kernel,
                                     nrt::TensorSet& output_set);

  /**
   * Get or load model from cache
   */
  static std::shared_ptr<nrt::Model> GetOrLoadModel(
      const CompilableKernelExecution& compilable_kernel, const ExecutionConfig& config);

  /**
   * Execute model with proper resource management
   */
  static ExecutionResult ExecuteModel(const std::string& op_name, std::shared_ptr<nrt::Model> model,
                                      nrt::TensorSet& input_set, nrt::TensorSet& output_set,
                                      const ExecutionConfig& config);
};

}  // namespace at::neuron
