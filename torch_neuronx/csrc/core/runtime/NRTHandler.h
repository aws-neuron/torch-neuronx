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
   * @param hlo_kernel The compilable kernel to execute
   * @param op_name Operation name for logging
   * @param config Execution configuration
   * @return ExecutionResult indicating success/failure and timing
   */
  static void PrepareAndExecuteModel(nrt::Model* model, const std::vector<nrt_tensor_t*>& inputs,
                                     const std::vector<nrt_tensor_t*>& outputs,
                                     const ExecutionConfig& config);

  /**
   * Get or load model from cache
   */
  static nrt::Model* GetOrLoadModel(const CompilableKernelExecution& compilable_kernel,
                                    const ExecutionConfig& config);

 private:
  /**
   * Prepare input tensor set from compilable kernel
   */
  static void PrepareInputTensorSet(const std::vector<nrt_tensor_t*>& inputs,
                                    nrt::TensorSet& input_set);

  /**
   * Prepare output tensor set from compilable kernel
   */
  static void PrepareOutputTensorSet(const std::vector<nrt_tensor_t*>& outputs,
                                     nrt::TensorSet& output_set);

  /**
   * Execute model with proper resource management
   */
  static void ExecuteModel(const std::shared_ptr<nrt::Model>& model, nrt::TensorSet& input_set,
                           nrt::TensorSet& output_set, const ExecutionConfig& config);
};

}  // namespace at::neuron
