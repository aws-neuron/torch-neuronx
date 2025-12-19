#include "KernelExecution.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

namespace at::neuron {

// CompilableKernelExecution implementation

std::string CompilableKernelExecution::CreateCompilerExtensionHash(
    const std::string& additional_args, const std::string& optimization_level) {
  // Build extension from compiler-specific parameters (no hashing for readability)
  std::string platform_target = utils::GetPlatformTarget();
  std::string logical_cores = utils::GetLogicalNeuronCores();

  // Remove leading dash from optimization level for cleaner key
  std::string opt_level_suffix = optimization_level;
  if (!opt_level_suffix.empty() && opt_level_suffix[0] == '-') {
    opt_level_suffix = opt_level_suffix.substr(1);
  }

  // Sanitize additional_args for use in cache key (replace spaces with underscores)
  std::string sanitized_args = additional_args;
  std::replace(sanitized_args.begin(), sanitized_args.end(), ' ', '_');
  std::replace(sanitized_args.begin(), sanitized_args.end(), '=', '-');

  // Build extension: <target>_<lnc>_<opt_level>_<additional_args>
  std::string extension = platform_target + "_" + logical_cores + "_" + opt_level_suffix;

  if (!sanitized_args.empty()) {
    extension += "_" + sanitized_args;
  }

  return extension;
}

std::string CompilableKernelExecution::GenerateCacheKey(const std::string& base_key,
                                                        const std::string& additional_args,
                                                        const std::string& optimization_level) {
  std::string compiler_extension = CreateCompilerExtensionHash(additional_args, optimization_level);
  return base_key + "_" + compiler_extension;
}

bool CompilableKernelExecution::ValidateImpl() const {
  return !cache_key_.empty() && cache_key_[0] != '_';
}

// XLACompilableKernelExecution implementation

std::string XLACompilableKernelExecution::GetXLAAdditionalArgs() {
  // Base XLA compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided XLA flags from environment
  if (const char* cc_flags_env = std::getenv("NEURON_CC_FLAGS")) {
    args += " " + std::string(cc_flags_env);
  }

  return args;
}

std::string XLACompilableKernelExecution::GetXLAOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O1";
}

XLACompilableKernelExecution::XLACompilableKernelExecution(const std::string& op_name,
                                                           const std::vector<void*>& src_ptrs,
                                                           const std::vector<void*>& dst_ptrs,
                                                           const std::string& base_cache_key,
                                                           const std::vector<uint8_t>& hlo_bytes,
                                                           bool has_collectives, int device_id)
    : CompilableKernelExecution(
          GenerateCacheKey(base_cache_key, GetXLAAdditionalArgs(), GetXLAOptimizationLevel()),
          GetXLAAdditionalArgs(), GetXLAOptimizationLevel(), has_collectives),
      NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, src_ptrs, dst_ptrs,
                                                               device_id),
      hlo_bytes_(hlo_bytes) {}

bool XLACompilableKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !hlo_bytes_.empty();
}

void XLACompilableKernelExecution::Execute() const {
  const std::string& operation_name = GetOpName();
  TORCH_NEURONX_DEBUG("Executing compilable kernel", "operation=", operation_name);

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1 /* num_cores */);
  auto* model = NRTHandler::GetOrLoadModel(compilable_kernel, config);

  const auto& src_data_ptrs = GetSrcDataPtrs();
  const auto& dst_data_ptrs = GetDstDataPtrs();
  // Execute using static method
  NRTHandler::PrepareAndExecuteModel(model, src_data_ptrs, dst_data_ptrs, config);
  TORCH_NEURONX_DEBUG("NRT execution completed", "operation=", operation_name);
}

std::vector<uint8_t> XLACompilableKernelExecution::CompileToNeff() const {
  if (hlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no HLO bytes available");
  }

  try {
    // Determine IR type: collectives always use XLA (HLO protobuf),
    // non-collectives may use StableHLO (MLIR) if environment variable is set
    std::string ir_type =
        (!HasCollectives() && NeuronCompiler::IsStableHLOEnabled()) ? "StableHLO" : "XLA";

    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(hlo_bytes_, config, ir_type);
  } catch (const std::exception& e) {
    throw std::runtime_error("HLO to NEFF compilation failed for op '" + GetOpName() +
                             "': " + e.what());
  }
}

// CompileOnlyKernelExecution implementation
std::string CompileOnlyKernelExecution::GetCompileAdditionalArgs() {
  // Base torch.compile compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided flags from environment
  if (const char* cc_flags_env = std::getenv("NEURON_CC_FLAGS")) {
    args += " " + std::string(cc_flags_env);
  }

  return args;
}

std::string CompileOnlyKernelExecution::GetCompileOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O2";
}

CompileOnlyKernelExecution::CompileOnlyKernelExecution(const std::string& base_cache_key,
                                                       const std::vector<uint8_t>& stablehlo_bytes,
                                                       bool has_collectives)
    : CompilableKernelExecution(GenerateCacheKey(base_cache_key, GetCompileAdditionalArgs(),
                                                 GetCompileOptimizationLevel()),
                                GetCompileAdditionalArgs(), GetCompileOptimizationLevel(),
                                has_collectives),
      NeuronKernelExecution("torch_compile", {}, {}, -1),  // Compilation-only, no actual execution
      stablehlo_bytes_(stablehlo_bytes) {}

bool CompileOnlyKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !stablehlo_bytes_.empty();
}

std::vector<uint8_t> CompileOnlyKernelExecution::CompileToNeff() const {
  if (stablehlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no StableHLO bytes available");
  }

  try {
    // torch.compile always uses StableHLO (MLIR) format
    std::string ir_type = "StableHLO";

    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(stablehlo_bytes_, config, ir_type);
  } catch (const std::exception& e) {
    throw std::runtime_error("StableHLO to NEFF compilation failed for torch.compile: " +
                             std::string(e.what()));
  }
}

// NeffDirectKernelExecution implementation
NeffDirectKernelExecution::NeffDirectKernelExecution(const std::string& graph_name,
                                                     const std::string& cache_key,
                                                     const std::vector<void*>& input_ptrs,
                                                     const std::vector<void*>& output_ptrs,
                                                     int device_id, bool has_collectives)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(graph_name, input_ptrs, output_ptrs,
                                                               device_id),
      CompilableKernelExecution(cache_key, "", "", has_collectives) {}

bool NeffDirectKernelExecution::ValidateImpl() const { return HasOutputs(); }

std::vector<uint8_t> NeffDirectKernelExecution::CompileToNeff() const {
  // NEFF is pre-compiled. Must always result in a cache hit.
  throw std::logic_error("NeffDirectKernelExecution expects pre-compiled graphs.");
}

void NeffDirectKernelExecution::Execute() const {
  const auto& src_data_ptrs = GetSrcDataPtrs();
  const auto& dst_data_ptrs = GetDstDataPtrs();

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1);
  auto* model = NRTHandler::GetOrLoadModel(compilable_kernel, config);
  NRTHandler::PrepareAndExecuteModel(model, src_data_ptrs, dst_data_ptrs, config);
}

// CollectiveDirectKernelExecution implementation

CollectiveDirectKernelExecution::CollectiveDirectKernelExecution(
    const std::string& op_name, CollectiveType collective_type, const std::vector<void*>& src_ptrs,
    const std::vector<void*>& dst_ptrs, c10d::ReduceOp reduce_op, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCollective, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, src_ptrs, dst_ptrs,
                                                               device_id),
      collective_type_(collective_type),
      reduce_op_(reduce_op) {}

bool CollectiveDirectKernelExecution::ValidateImpl() const { return HasInputs() && HasOutputs(); }

void CollectiveDirectKernelExecution::Execute() const {
  TORCH_CHECK(false, "CollectiveDirectKernelExecution::Execute() not yet implemented");
}

AllocDirectKernelExecution::AllocDirectKernelExecution(const std::string& op_name, void* dst_ptr,
                                                       size_t size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kAlloc, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {}, {dst_ptr}, device_id),
      size_bytes_(size_bytes) {}

bool AllocDirectKernelExecution::ValidateImpl() const { return size_bytes_ > 0 && HasOutputs(); }

void AllocDirectKernelExecution::Execute() const {
  int device_id = GetDeviceId();
  nrt_tensor_t* data_ptr =
      c10_neuron::NeuronCachingAllocator::getCachedBlock(device_id, size_bytes_);
  if (!data_ptr) {
    int vnc_id = c10_neuron::get_vnc_id(device_id);
    NRT_STATUS status = at::neuron::nrt::AllocateTensor(NRT_TENSOR_PLACEMENT_DEVICE, vnc_id,
                                                        size_bytes_, nullptr, &data_ptr);
    if (status != NRT_SUCCESS || !data_ptr) {
      std::string error_msg = "Failed to allocate NRT tensor. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  }

  void* dst_ptr = GetDstPtrs()[0];
  c10_neuron::NeuronCachingAllocator::recordTensor(dst_ptr, data_ptr);
  TORCH_NEURONX_DEBUG("Allocated NRT tensor", "virtual_addr=", dst_ptr, "nrt_tensor=", data_ptr,
                      "size_bytes=", size_bytes_);
}

DeallocDirectKernelExecution::DeallocDirectKernelExecution(const std::string& op_name,
                                                           void* src_ptr, size_t size_bytes,
                                                           int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kDealloc, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {src_ptr}, {}, device_id),
      size_bytes_(size_bytes) {}

bool DeallocDirectKernelExecution::ValidateImpl() const { return size_bytes_ > 0; }

void DeallocDirectKernelExecution::Execute() const {
  void* src_ptr = GetSrcPtrs()[0];
  nrt_tensor_t* src_data_ptr = GetSrcDataPtrs()[0];

  auto& slice_ptrs = c10_neuron::NeuronCachingAllocator::findTensorSlices(src_ptr);
  for (void* slice_ptr : slice_ptrs) {
    nrt_tensor_t* slice_tensor = c10_neuron::NeuronCachingAllocator::findTensor(slice_ptr);
    c10_neuron::NeuronCachingAllocator::remove(slice_ptr);
    c10_neuron::NeuronCachingAllocator::ReleaseVirtualAddress(slice_ptr);
    at::neuron::nrt::FreeTensor(&slice_tensor);
  }

  // Add to the device-specific cache instead of freeing
  // TODO(Streams): Move the stream ownership to the cache.
  int device_id = GetDeviceId();
  c10_neuron::NeuronCachingAllocator::cacheBlock(device_id, src_data_ptr, size_bytes_);
  c10_neuron::NeuronCachingAllocator::remove(src_ptr);
  c10_neuron::NeuronCachingAllocator::ReleaseVirtualAddress(src_ptr);
}

CopyDirectKernelExecution::CopyDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                     void* dst_ptr, size_t src_offset_bytes,
                                                     size_t dst_offset_bytes, size_t size_bytes,
                                                     int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {src_ptr}, {dst_ptr},
                                                               device_id),
      src_offset_bytes_(src_offset_bytes),
      dst_offset_bytes_(dst_offset_bytes),
      size_bytes_(size_bytes) {}

bool CopyDirectKernelExecution::ValidateImpl() const { return size_bytes_ > 0; }

void CopyDirectKernelExecution::Execute() const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  nrt_tensor_t* dst_ptr = GetDstDataPtrs()[0];
  NRT_STATUS status =
      nrt::CopyTensor(src_ptr, src_offset_bytes_, dst_ptr, dst_offset_bytes_, size_bytes_);
  if (status != NRT_SUCCESS) {
    std::string error_msg = "Failed to copy tensor data. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
}

WriteDirectKernelExecution::WriteDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                       void* dst_ptr, size_t dst_offset_bytes,
                                                       size_t dst_size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kWrite, c10::DeviceType::CPU,
                                 c10::DeviceType::PrivateUse1>(op_name, {src_ptr}, {dst_ptr},
                                                               device_id),
      dst_offset_bytes_(dst_offset_bytes),
      dst_size_bytes_(dst_size_bytes) {}

bool WriteDirectKernelExecution::ValidateImpl() const { return dst_size_bytes_ > 0; }

void WriteDirectKernelExecution::Execute() const {
  void* src_data_ptr = GetSrcDataPtrs()[0];
  nrt_tensor_t* dst_data_ptr = GetDstDataPtrs()[0];
  NRT_STATUS status =
      nrt::WriteTensor(dst_data_ptr, src_data_ptr, dst_offset_bytes_, dst_size_bytes_);
  if (status != NRT_SUCCESS) {
    std::string error_msg = "Failed to write tensor data. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
}

ReadDirectKernelExecution::ReadDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                     void* dst_ptr, size_t src_offset_bytes,
                                                     size_t src_size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kRead, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::CPU>(op_name, {src_ptr}, {dst_ptr}, device_id),
      src_offset_bytes_(src_offset_bytes),
      src_size_bytes_(src_size_bytes) {}

bool ReadDirectKernelExecution::ValidateImpl() const { return src_size_bytes_ > 0; }

void ReadDirectKernelExecution::Execute() const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  void* dst_ptr = GetDstDataPtrs()[0];
  NRT_STATUS status = nrt::ReadTensor(src_ptr, dst_ptr, src_offset_bytes_, src_size_bytes_);
  if (status != NRT_SUCCESS) {
    std::string error_msg = "Failed to read tensor data. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
}

SliceDirectKernelExecution::SliceDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                       void* dst_ptr, size_t src_offset_bytes,
                                                       size_t dst_size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kSlice, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {src_ptr}, {dst_ptr},
                                                               device_id),
      src_offset_bytes_(src_offset_bytes),
      dst_size_bytes_(dst_size_bytes) {}

bool SliceDirectKernelExecution::ValidateImpl() const {
  return dst_size_bytes_ > 0 && HasInputs() && HasOutputs();
}

void SliceDirectKernelExecution::Execute() const {
  nrt_tensor_t* src_data_ptr = GetSrcDataPtrs()[0];

  // Validate slice bounds
  size_t base_size_bytes = nrt::GetTensorSize(src_data_ptr);
  if (src_offset_bytes_ > base_size_bytes ||
      base_size_bytes - src_offset_bytes_ < dst_size_bytes_) {
    throw std::runtime_error("Slice bounds exceed base tensor size");
  }

  nrt_tensor_t* slice_data_ptr = nullptr;
  NRT_STATUS status = nrt::AllocateSliceTensor(src_data_ptr, src_offset_bytes_, dst_size_bytes_,
                                               nullptr, &slice_data_ptr);
  if (status != NRT_SUCCESS || !slice_data_ptr) {
    std::string error_msg = "Failed to allocate slice tensor. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
  void* dst_ptr = GetDstPtrs()[0];
  // Record the slice tensor as its standalone reference.
  c10_neuron::NeuronCachingAllocator::recordTensor(dst_ptr, slice_data_ptr);
}

EventDirectKernelExecution::EventDirectKernelExecution(const std::string& op_name,
                                                       const NeuronEvent& event, EventAction action,
                                                       int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kEvent, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {}, {}, device_id),
      event_(event),
      action_(action) {}

bool EventDirectKernelExecution::ValidateImpl() const {
  // Event primitives don't require inputs/outputs
  return true;
}

bool EventDirectKernelExecution::IsReady() const {
  if (action_ == EventAction::kSignal) {
    return true;
  }
  return event_.query();
}

void EventDirectKernelExecution::Execute() const {
  if (action_ == EventAction::kWait) {
    // At this point, the event is assumed to have completed
    TORCH_CHECK(IsReady(), "Event not ready for wait");
    return;
  }
  event_.complete_event();
}

}  // namespace at::neuron
