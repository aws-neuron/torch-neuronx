#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
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
   * Dispatch a compilable kernel using NRT for execution
   *
   * @param model The loaded NRT model
   * @param inputs Input tensor pointers
   * @param outputs Output tensor pointers
   * @param config Execution configuration
   * @param err_tracker Error tracker pointer for nrt-async mode (nullptr for nrt-sync)
   * @param sequence_id Output sequence number for nrt-async mode (nullptr for nrt-sync)
   * @param stream_id Stream ID for queue selection in nrt-async mode
   */
  static void DispatchModelExecution(const std::shared_ptr<nrt::ModelHandle>& model,
                                     const std::vector<nrt_tensor_t*>& inputs,
                                     const std::vector<nrt_tensor_t*>& outputs,
                                     const ExecutionConfig& config,
                                     nrt::ErrorTrackerHandle* err_tracker = nullptr,
                                     nrt::SequenceId* sequence_id = nullptr,
                                     c10::StreamId stream_id = 0);

  /**
   * Get or load model from cache
   */
  static std::shared_ptr<nrt::ModelHandle> GetOrLoadModel(
      const CompilableKernelExecution& compilable_kernel, const ExecutionConfig& config);

 private:
  /**
   * Prepare input tensor set from compilable kernel
   */
  static void PrepareInputTensorSet(const std::vector<nrt_tensor_t*>& inputs,
                                    nrt::TensorSetPtr& input_set);

  /**
   * Prepare output tensor set from compilable kernel
   */
  static void PrepareOutputTensorSet(const std::vector<nrt_tensor_t*>& outputs,
                                     nrt::TensorSetPtr& output_set);
};

}  // namespace at::neuron
