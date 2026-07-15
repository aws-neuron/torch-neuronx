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
   * @param nrt_status NRT status
   * @param sequence_id Output sequence number for nrt-async mode (nullptr for nrt-sync)
   * @param stream_id Stream ID for queue selection in nrt-async mode
   */
  static void DispatchModelExecution(
      const std::shared_ptr<nrt::ModelHandle>& model, const std::vector<nrt_tensor_t*>& inputs,
      const std::vector<nrt_tensor_t*>& outputs, const ExecutionConfig& config,
      NRT_STATUS* nrt_status = nullptr, nrt::SequenceId* sequence_id = nullptr,
      c10::StreamId stream_id = 0, const std::vector<std::string>& input_names = {},
      const std::vector<std::string>& output_names = {});

  /**
   * Get or load model from cache
   */
  static std::shared_ptr<nrt::ModelHandle> GetOrLoadModel(
      const CompilableKernelExecution& compilable_kernel, const ExecutionConfig& config);

  /**
   * Allocate a slice of an existing NRT tensor, throwing on failure.
   *
   * This wraps nrt::AllocateSliceTensor so callers don't have to thread the
   * NRT_STATUS through their own error handling. Throws
   * torch_neuronx::ExecutionRuntimeException on non-success status.
   *
   * @param base_tensor Base tensor to slice from.
   * @param offset      Byte offset in base tensor.
   * @param size        Byte size of the slice.
   * @param name        Optional tensor name (may be nullptr).
   * @return Newly-allocated slice tensor (non-null).
   */
  static nrt_tensor_t* AllocateSliceTensor(nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                           const char* name);

  /**
   * Allocate an NRT tensor, throwing on failure.
   *
   * Wraps nrt::AllocateTensor so callers don't thread NRT_STATUS through
   * their own error handling. On non-success status (including NRT_RESOURCE
   * for OOM) throws torch_neuronx::ExecutionRuntimeException carrying the
   * status - the runtime worker's ErrorRecovery catches NRT_RESOURCE and
   * drives cache drain + retry.
   *
   * @param placement Tensor placement (device/host).
   * @param vnc_id    Virtual NeuronCore id.
   * @param size      Byte size.
   * @param name      Optional tensor name (may be nullptr).
   * @return Newly-allocated tensor (non-null).
   */
  static nrt_tensor_t* AllocateTensor(nrt_tensor_placement_t placement, int vnc_id, size_t size,
                                      const char* name);

 private:
  /**
   * Prepare input tensor set from compilable kernel.
   * ``names`` overrides the default ``input{i}`` port names when non-empty;
   * its size must equal ``inputs.size()`` in that case.
   */
  static void PrepareInputTensorSet(const std::vector<nrt_tensor_t*>& inputs,
                                    nrt::TensorSetPtr& input_set,
                                    const std::vector<std::string>& names = {});

  /**
   * Prepare output tensor set from compilable kernel.
   * ``names`` overrides the default ``output{i}`` port names when non-empty;
   * its size must equal ``outputs.size()`` in that case.
   */
  static void PrepareOutputTensorSet(const std::vector<nrt_tensor_t*>& outputs,
                                     nrt::TensorSetPtr& output_set,
                                     const std::vector<std::string>& names = {});
};

}  // namespace at::neuron
