#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Types.hpp>

// Include NRT headers
extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>
}

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Forward declarations
namespace c10_neuron {
namespace NeuronCachingAllocator {
nrt_tensor_t* findTensor(void* ptr);
}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron

namespace at::neuron::nrt {

// NRT Sequence ID for async operations
using SequenceId = nrta_seq_t;
using ModelHandle = nrt_model_t;
using TensorSetHandle = nrt_tensor_set_t;

// Collective communication types
using NRTCollectiveContext = nrt_cc_context_t;
using NRTCollectiveComm = nrt_cc_comm_t;
using NRTTensorList = nrt_tensor_list_t;

// Collective operation types
enum class NRTCollectiveOp {
  kAllGather = NRT_CC_ALLGATHER,
  kReduceScatter = NRT_CC_REDUCESCATTER,
  kAllReduce = NRT_CC_ALLREDUCE,
};

// Reduce operation types
enum class NRTReduceOp {
  kAdd = NRT_OP_ADD,
  kMax = NRT_OP_MAX,
  kMin = NRT_OP_MIN,
};

// NRT Data types
enum class NRTDType {
  kFloat32 = NRT_DTYPE_FLOAT32,
  kFloat16 = NRT_DTYPE_FLOAT16,
  kBFloat16 = NRT_DTYPE_BFLOAT16,
  kInt32 = NRT_DTYPE_INT32,
  kInt64 = NRT_DTYPE_INT64,
  kInt16 = NRT_DTYPE_INT16,
  kInt8 = NRT_DTYPE_INT8,
  kUInt8 = NRT_DTYPE_UINT8,
  kUInt16 = NRT_DTYPE_UINT16,
  kUInt32 = NRT_DTYPE_UINT32,
  kUInt64 = NRT_DTYPE_UINT64,
};

inline NRTDType MapScalarTypeToNRT(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return NRTDType::kFloat32;
    case c10::ScalarType::Half:
      return NRTDType::kFloat16;
    case c10::ScalarType::BFloat16:
      return NRTDType::kBFloat16;
    case c10::ScalarType::Int:
      return NRTDType::kInt32;
    case c10::ScalarType::Long:
      return NRTDType::kInt64;
    case c10::ScalarType::Short:
      return NRTDType::kInt16;
    case c10::ScalarType::Char:
      return NRTDType::kInt8;
    case c10::ScalarType::Byte:
      return NRTDType::kUInt8;
    case c10::ScalarType::UInt16:
      return NRTDType::kUInt16;
    case c10::ScalarType::UInt32:
      return NRTDType::kUInt32;
    case c10::ScalarType::UInt64:
      return NRTDType::kUInt64;
    default:
      TORCH_CHECK(false, "Unsupported dtype for NRT operation: ", c10::toString(dtype));
  }
}

inline NRTReduceOp MapReduceOpToNRT(c10d::ReduceOp op) {
  switch (op) {
    case c10d::ReduceOp::SUM:
      return NRTReduceOp::kAdd;
    case c10d::ReduceOp::MAX:
      return NRTReduceOp::kMax;
    case c10d::ReduceOp::MIN:
      return NRTReduceOp::kMin;
    default:
      TORCH_CHECK(false, "Unsupported reduce op for NRT: ", static_cast<int>(op));
  }
}

// State for tracking async NRT operation scheduling
struct AsyncSchedulingState {
  SequenceId sequence_id{0};      // Sequence number from Schedule()
  bool is_scheduled{false};       // Whether async scheduling was called (in-flight)
  bool is_pending_signal{false};  // kSignal event waiting for prior op to complete

  void Reset() {
    sequence_id = 0;
    is_scheduled = false;
    is_pending_signal = false;
  }
};

struct TensorSetDeleter {
  void operator()(TensorSetHandle* raw) const;
};

using TensorSetPtr = std::unique_ptr<TensorSetHandle, TensorSetDeleter>;

TensorSetPtr CreateTensorSet();
NRT_STATUS AddTensorToSet(const TensorSetPtr& set, const std::string& name, nrt_tensor_t* data_ptr);

struct ModelDeleter {
  void operator()(ModelHandle* raw) const;
};

using ModelPtr = std::unique_ptr<ModelHandle, ModelDeleter>;

NRT_STATUS LoadModel(ModelPtr& model, const std::vector<uint8_t>& neff_bytes, int vnc_id,
                     int num_cores);
NRT_STATUS LoadModelCollectives(ModelPtr& model, const std::vector<uint8_t>& neff_bytes, int vnc_id,
                                int num_cores, int rank, int world_size);

NRT_STATUS DispatchExecution(ModelHandle* model, TensorSetHandle* input_set,
                             TensorSetHandle* output_set, int queue, NRT_STATUS* ret,
                             SequenceId* req_sequence);

/**
 * Allocate an NRT tensor
 * @param placement Tensor placement (device or host)
 * @param vnc_id Virtual NeuronCore ID
 * @param size_bytes Size in bytes
 * @param name Tensor name for debugging
 * @param tensor Output pointer to allocated tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateTensor(nrt_tensor_placement_t placement, int vnc_id, size_t size_bytes,
                                 const char* name, nrt_tensor_t** tensor) {
  return nrt_tensor_allocate(placement, vnc_id, size_bytes, name, tensor);
}

/**
 * Free an NRT tensor
 * @param tensor Pointer to tensor pointer (will be set to nullptr)
 */
inline void FreeTensor(nrt_tensor_t** tensor) { nrt_tensor_free(tensor); }

/**
 * Copy data between NRT tensors
 * @param src Source tensor
 * @param src_offset Offset in source tensor
 * @param dst Destination tensor
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to copy
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS CopyTensor(nrt_tensor_t* src, size_t src_offset, nrt_tensor_t* dst,
                             size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_copy(src, src_offset, dst, dst_offset, size_bytes);
}

/**
 * Write data from CPU memory to NRT tensor
 * @param dst Destination NRT tensor
 * @param src Source CPU memory pointer
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to write
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS WriteTensor(nrt_tensor_t* dst, void* src, size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_write(dst, src, dst_offset, size_bytes);
}

/**
 * Read data from NRT tensor to CPU memory
 * @param src Source NRT tensor
 * @param dst Destination CPU memory pointer
 * @param src_offset Offset in source tensor
 * @param size_bytes Number of bytes to read
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ReadTensor(nrt_tensor_t* src, void* dst, size_t src_offset, size_t size_bytes) {
  return nrt_tensor_read(src, dst, src_offset, size_bytes);
}

/**
 * Execute a barrier across devices
 * @param vnc Virtual neuron core id
 * @param global_device_id Global device id for barrier coordination
 * @param global_device_count Total number of devices participating
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS Barrier(int32_t vnc, uint32_t global_device_id, uint32_t global_device_count) {
  return nrt_barrier(vnc, global_device_id, global_device_count);
}

/**
 * Get the size of an NRT tensor
 * @param tensor Tensor to query
 * @return Size in bytes
 */
inline size_t GetTensorSize(nrt_tensor_t* tensor) { return nrt_tensor_get_size(tensor); }

/**
 * Allocate a slice of an existing NRT tensor
 * @param base_tensor Base tensor to slice from
 * @param offset Offset in base tensor
 * @param size Size of slice
 * @param name Name for the slice tensor
 * @param slice_tensor Output pointer to slice tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateSliceTensor(nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                      const char* name, nrt_tensor_t** slice_tensor) {
  return nrt_tensor_allocate_slice(base_tensor, offset, size, name, slice_tensor);
}

/**
 * Get virtual address of a tensor
 */
inline void* GetTensorVA(nrt_tensor_t* tensor) { return nrt_tensor_get_va(tensor); }

// NRT Profiling/Tracing Functions
// These provide access to NRT inspect functionality for comprehensive profiling

/**
 * Stop tracing/profiling and dump profile data
 * Dumps captured profile data to the configured output directory
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS StopInspect() { return nrt_inspect_stop(); }

// NRT Profiling with Options Functions
// These provide wrapper functions for the advanced NRT inspect configuration APIs

/**
 * Allocate memory for the options structure for configurable profiling
 * @param options Output pointer to allocated config structure
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateInspectConfig(nrt_inspect_config_t** options) {
  return nrt_inspect_config_allocate(options);
}

/**
 * Free memory for the options structure
 * @param options Pointer to config structure to free
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS FreeInspectConfig(nrt_inspect_config_t* options) {
  return nrt_inspect_config_free(options);
}

/**
 * Set all fields of the config structure to their default values
 * @param options Pointer to config structure
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigDefaults(nrt_inspect_config_t* options) {
  return nrt_inspect_config_set_defaults(options);
}

/**
 * Set the output directory for profiling results
 * @param options Pointer to config structure
 * @param output_dir Path to the output directory
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigOutputDir(nrt_inspect_config_t* options, const char* output_dir) {
  return nrt_inspect_config_set_output_dir(options, output_dir);
}

/**
 * Enable or disable inspect profiling for normal execution
 * @param options Pointer to config structure
 * @param enable_inspect Boolean to enable or disable profiling
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigEnableInspect(nrt_inspect_config_t* options,
                                                bool enable_inspect) {
  return nrt_inspect_config_set_enable_inspect(options, enable_inspect);
}

/**
 * Set system trace capture enabled for a specific NeuronCore
 * @param options Pointer to config structure
 * @param nc_idx Index of the NeuronCore
 * @param enabled Boolean to enable or disable capture
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigCaptureEnabledForNC(nrt_inspect_config_t* options,
                                                      uint32_t nc_idx, bool enabled) {
  return nrt_inspect_config_set_capture_enabled_for_nc(options, nc_idx, enabled);
}

/**
 * Set system trace capture enabled for a specific event type
 * @param options Pointer to config structure
 * @param event_type String name of the event type
 * @param enabled Boolean to enable or disable capture
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigCaptureEnabledForEventType(nrt_inspect_config_t* options,
                                                             const char* event_type, bool enabled) {
  // TODO: linking to this API fails fix merged update once cherry-picked to 2.30
  //  return nrt_inspect_config_set_capture_enabled_for_event_type_string(options, event_type,
  //  enabled); (SIM ID: P372975680)
  return NRT_STATUS(0);
}

/**
 * Set the device profile mode for inspect profiling
 * @param options Pointer to config structure
 * @param device_profile_mode Device profile mode (0=disabled, 1=model, 2=session)
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigDeviceProfileMode(nrt_inspect_config_t* options,
                                                    int device_profile_mode) {
  return nrt_inspect_config_set_inspect_device_profile_mode(
      options, static_cast<nrt_inspect_device_profile_mode_t>(device_profile_mode));
}

/**
 * Get all available activity type strings
 * @param activity_types Output pointer to array of activity type strings
 * @param count Output pointer to number of activity types
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS GetAllActivityTypes(const char*** activity_types, size_t* count) {
  return nrt_inspect_config_get_all_activity_types(activity_types, count);
}

/**
 * Free activity types array allocated by GetAllActivityTypes
 * @param activity_types Pointer to activity types array
 * @param count Number of activity types
 */
inline void FreeActivityTypes(const char** activity_types, size_t count) {
  nrt_inspect_config_free_activity_types(activity_types, count);
}

/**
 * Begin tracing/profiling with configurable options
 * @param options Pointer to config structure with profiling options
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS BeginInspectWithOptions(nrt_inspect_config_t* options) {
  return nrt_inspect_begin_with_options(options);
}

/**
 * Get the instance-specific output directory path used by NRT profiling
 * @param output_dir Base output directory
 * @param instance_dir_buf Buffer to store the instance directory path
 * @param buf_size Size of the buffer
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS GetInstanceOutputDir(const char* output_dir, char* instance_dir_buf,
                                       size_t buf_size) {
  return nrt_inspect_get_instance_output_dir(output_dir, instance_dir_buf, buf_size);
}

// ============================================================================
// Asynchronous NRT operations
// ============================================================================

/**
 * Enqueue an asynchronous tensor write request
 * @param tensor Destination tensor
 * @param buf Host buffer containing source data
 * @param offset Offset into the tensor
 * @param size Number of bytes to write
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorWrite(nrt_tensor_t* tensor, const void* buf, uint64_t offset,
                                      uint64_t size, int lnc_idx, int queue, NRT_STATUS* ret,
                                      SequenceId* req_sequence) {
  return nrta_tensor_write(tensor, buf, offset, size, lnc_idx, queue, ret, req_sequence);
}

/**
 * Enqueue an asynchronous tensor read request
 * @param buf Destination host buffer
 * @param tensor Source tensor
 * @param offset Offset into the tensor
 * @param size Number of bytes to read
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorRead(void* buf, nrt_tensor_t* tensor, uint64_t offset,
                                     uint64_t size, int lnc_idx, int queue, NRT_STATUS* ret,
                                     SequenceId* req_sequence) {
  return nrta_tensor_read(buf, tensor, offset, size, lnc_idx, queue, ret, req_sequence);
}

/**
 * Enqueue an asynchronous tensor copy request
 * @param src Source tensor
 * @param src_offset Offset in source tensor
 * @param dst Destination tensor
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to copy
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorCopy(nrt_tensor_t* src, uint64_t src_offset, nrt_tensor_t* dst,
                                     uint64_t dst_offset, uint64_t size_bytes, int lnc_idx,
                                     int queue, NRT_STATUS* ret, SequenceId* req_sequence) {
  return nrta_tensor_copy(src, src_offset, dst, dst_offset, size_bytes, lnc_idx, queue, ret,
                          req_sequence);
}

/**
 * Prepare a collective communication operation (blocking)
 * @param comm Communication handle (contains replica group info)
 * @param input Input tensor list
 * @param output Output tensor list
 * @param dtype Data type
 * @param op Operation type (for reduce operations)
 * @param cc_op Collective communication operation type
 * @param cc_ctx Output collective communication context
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS CollectivesPrepare(NRTCollectiveComm* comm, NRTTensorList* input,
                                     NRTTensorList* output, NRTDType dtype, NRTReduceOp op,
                                     NRTCollectiveOp cc_op, NRTCollectiveContext** cc_ctx) {
  return nrta_cc_prepare(comm, input, output, nrt_dtype_t(dtype), nrt_op_type_t(op),
                         nrt_cc_op_type_t(cc_op), cc_ctx);
}

/**
 * Schedule an asynchronous collective communication operation
 * @param cc_ctx Collective communication context (from CollectivesPrepare)
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS CollectivesSchedule(nrt_cc_context_t** cc_ctx, int queue, NRT_STATUS* ret,
                                      nrta_seq_t* req_sequence) {
  return nrta_cc_schedule(cc_ctx, queue, ret, req_sequence);
}

/**
 * Check completion status of a scheduled async request
 * @param sequence_id Scheduled request sequence id
 * @param is_completed Output: true if the request is completed, false otherwise
 * @return NRT_SUCCESS if the request is valid, NRT_INVALID if the sequence_id is not valid
 */
inline NRT_STATUS IsNRTAsyncRequestCompleted(SequenceId sequence_id, bool* is_completed) {
  return nrta_is_completed(sequence_id, is_completed);
}

/**
 * Get sequence number of the last completed request for an execution unit queue
 * @param lnc LNC id
 * @param xu XU type
 * @param queue XU's queue
 * @param sequence_id Output: last completed sequence number
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS GetLastCompletedRequest(uint32_t lnc, nrta_xu_t xu, int queue,
                                          SequenceId* sequence_id) {
  return nrta_get_sequence(lnc, xu, queue, sequence_id);
}

// Hardware queue counts per XU type.
constexpr int kTensorOpQueues = 2;
constexpr int kComputeQueues = 1;
constexpr int kCollectivesQueues = 1;

inline int GetNumQueuesForXU(nrta_xu_t xu) {
  switch (xu) {
    case NRTA_XU_TENSOR_OP:
      return kTensorOpQueues;
    case NRTA_XU_COMPUTE:
      return kComputeQueues;
    case NRTA_XU_COLLECTIVES:
      return kCollectivesQueues;
    case NRTA_XU_TYPE_NUM:
      break;
  }
  TORCH_CHECK(false, "Unknown XU type: ", static_cast<int>(xu));
}

inline void HostCollectivesEnable(int stream_pool_size) { nrt_cc_create_stream(stream_pool_size); }

/**
 * Map stream_id to hardware queue for a given XU.
 * Queue 0 is reserved for the default stream (stream_id == 0).
 * Non-default streams are distributed round-robin across queues 1..N-1.
 */
inline int GetHardwareQueue(nrta_xu_t xu, c10::StreamId stream_id) {
  int num_queues = GetNumQueuesForXU(xu);
  if (num_queues <= 1) return 0;
  if (stream_id == 0) return 0;
  return 1 + ((stream_id - 1) % (num_queues - 1));
}

}  // namespace at::neuron::nrt
