#include "KernelExecution.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/compilation/experimental/TensorizerCompiler.h"
#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/NonTemporalMemcpy.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"

namespace at::neuron {

bool NeuronKernelExecution::CaptureRequiresExecution() { return !c10_neuron::is_compile_only(); }

// CompilableKernelExecution implementation

bool CompilableKernelExecution::ValidateImpl() const { return !compile_key_.IsEmpty(); }

// XLACompilableKernelExecution implementation

bool XLACompilableKernelExecution::HasInt64IO(const std::vector<TensorContext>& input_contexts,
                                              const std::vector<TensorContext>& output_contexts) {
  // Returns true when any input or output tensor has dtype Long (i64).
  // Used to gate int64-policy compile-flag emission so that ops whose
  // compiled HLO has nothing to do with int64 (e.g. pow(float,
  // int_scalar) where the scalar is converted to i32 before reaching
  // the kernel) don't pick up int64-related compiler flags. See
  // .internal_docs/native_inplace_add_investigation.md "Stage 2" for
  // the broader follow-up that pushes this check earlier in the
  // dispatch path.
  for (const auto& ctx : input_contexts) {
    if (ctx.options.dtype().toScalarType() == c10::ScalarType::Long) {
      return true;
    }
  }
  for (const auto& ctx : output_contexts) {
    if (ctx.options.dtype().toScalarType() == c10::ScalarType::Long) {
      return true;
    }
  }
  return false;
}

std::string XLACompilableKernelExecution::GetXLAAdditionalArgs(const std::string& op_name,
                                                               bool has_int64_io) {
  // Base XLA compilation arguments
  std::string args = "--model-type transformer --auto-cast=none --internal-disable-fma-on-ios";

  // Per-op int64 compiler flags based on Int64Policy.
  // See torch_neuronx/int64_policy.py::get_compiler_flags for the source of
  // truth — this block must mirror it.
  //
  // The block is gated on ``has_int64_io``: we only consult the int64
  // policy and emit int64-related flags when the kernel's input/output
  // dtypes actually contain Long. For everything else (e.g.
  // ``pow(float, int_scalar)``) the policy is bypassed entirely so the
  // compile command line matches the legacy non-int64 stack.
  if (has_int64_io && torch_neuronx::dispatch::IsS64Passthrough()) {
    static const bool full_cast = [] {
      const char* env = std::getenv("TORCH_NEURONX_INT64_FULL_CAST");
      return env != nullptr && std::string(env) == "1";
    }();
    auto decision = torch_neuronx::dispatch::DecideInt64(op_name);
    if (decision == torch_neuronx::dispatch::Int64Decision::NATIVE) {
      if (full_cast) {
        args += " --native-int64=disabled --implicit-integer-downcast=all";
      } else {
        args += " --native-int64=enabled --implicit-integer-downcast=all";
      }
    } else if (decision == torch_neuronx::dispatch::Int64Decision::DOWNCAST) {
      if (full_cast) {
        args += " --native-int64=disabled";
      }
      args +=
          " --implicit-integer-downcast=all"
          " --disable-hlo-operand-type-check=evrf_035";
    }
  }

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

XLACompilableKernelExecution::XLACompilableKernelExecution(
    const std::string& op_name, std::vector<TensorDataRef>&& input_refs,
    std::vector<TensorDataRef>&& output_refs, const std::vector<TensorContext>& input_contexts,
    const std::vector<TensorContext>& output_contexts, const std::string& base_cache_key,
    const std::vector<uint8_t>& hlo_bytes, bool has_collectives, int device_id)
    : CompilableKernelExecution(
          CompilationCacheKey(
              base_cache_key,
              GetXLAAdditionalArgs(op_name, HasInt64IO(input_contexts, output_contexts)),
              GetXLAOptimizationLevel(), &hlo_bytes_, CacheKeySource::EAGER),
          GetXLAAdditionalArgs(op_name, HasInt64IO(input_contexts, output_contexts)),
          GetXLAOptimizationLevel(), has_collectives),
      NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(
          op_name, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
          device_id),
      hlo_bytes_(hlo_bytes) {}

bool XLACompilableKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !hlo_bytes_.empty();
}
bool XLACompilableKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                            c10::StreamId stream_id) const {
  const std::string& operation_name = GetOpName();
  TORCH_NEURONX_DEBUG("Scheduling compilable kernel", "operation=", operation_name);

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1 /* num_cores */);
  auto model = NRTHandler::GetOrLoadModel(compilable_kernel, config);

  // Create slices for tensors with storage_offset > 0
  const auto src_data_ptrs = HandleContiguousSlicing(GetSrcDataPtrs(), src_tensor_ctxs_);
  const auto dst_data_ptrs = HandleContiguousSlicing(GetDstDataPtrs(), dst_tensor_ctxs_);

  NRTHandler::DispatchModelExecution(model, src_data_ptrs, dst_data_ptrs, config, nrt_status,
                                     sequence_id, stream_id, GetInputNames(), GetOutputNames());

  TORCH_NEURONX_DEBUG("NRT execution completed", "operation=", operation_name);
  return true;
}

std::vector<uint8_t> XLACompilableKernelExecution::CompileToNeff() const {
  if (hlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no HLO bytes available");
  }

  try {
    std::string ir_type = "StableHLO";

    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(hlo_bytes_, config, ir_type, GetOpName(),
                                            GetCompileKey(), "XLACompilableKernelExecution");
  } catch (const std::exception& e) {
    throw std::runtime_error("HLO to NEFF compilation failed for op '" + GetOpName() +
                             "': " + e.what());
  }
}

// CompileOnlyKernelExecution implementation
std::string CompileOnlyKernelExecution::GetCompileAdditionalArgs() {
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
                                                       const std::vector<uint8_t>& mlir_bytes,
                                                       bool has_collectives, CompileMode mode)
    // WLO identity rides on the "|wlo" suffix of base_cache_key (added by the frontend), which
    // CompilationCacheKey inspects; no separate flag is threaded through the key.
    : CompilableKernelExecution(CompilationCacheKey(base_cache_key, GetCompileAdditionalArgs(),
                                                    GetCompileOptimizationLevel(), &mlir_bytes_,
                                                    CacheKeySource::TORCH_COMPILE),
                                GetCompileAdditionalArgs(), GetCompileOptimizationLevel(),
                                has_collectives),
      NeuronKernelExecution("torch_compile", -1, false),
      mlir_bytes_(mlir_bytes),
      use_tensorizer_(mode == CompileMode::kTensorizer),
      wlo_enabled_(mode == CompileMode::kWlo),
      platform_target_(utils::GetPlatformTarget()),
      logical_neuron_cores_(utils::GetLogicalNeuronCores()) {}

CompileOnlyKernelExecution::CompileOnlyKernelExecution(
    const std::string& base_cache_key, const std::vector<uint8_t>& mlir_bytes, bool has_collectives,
    const std::string& additional_args, const std::string& optimization_level, bool use_tensorizer)
    : CompilableKernelExecution(
          CompilationCacheKey(base_cache_key, additional_args, optimization_level, &mlir_bytes_,
                              CacheKeySource::EAGER),
          additional_args, optimization_level, has_collectives),
      NeuronKernelExecution("precompile_eager", -1, false),
      mlir_bytes_(mlir_bytes),
      use_tensorizer_(use_tensorizer),
      platform_target_(utils::GetPlatformTarget()),
      logical_neuron_cores_(utils::GetLogicalNeuronCores()) {}

bool CompileOnlyKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !mlir_bytes_.empty();
}

std::vector<uint8_t> CompileOnlyKernelExecution::CompileToNeff() const {
  if (mlir_bytes_.empty()) throw std::invalid_argument("Cannot compile: no MLIR bytes available");

  try {
    if (use_tensorizer_)
      return TensorizerCompiler::CompileLinalgToNeff(mlir_bytes_, platform_target_);

    CompilationConfig config{.framework = "XLA",
                             .platform_target = platform_target_,
                             .logical_neuron_cores = logical_neuron_cores_,
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(mlir_bytes_, config, "StableHLO", "torch_compile",
                                            GetCompileKey(), "CompileOnlyKernelExecution");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(use_tensorizer_ ? "Linalg" : "StableHLO") +
                             " to NEFF compilation failed: " + std::string(e.what()));
  }
}

CompileResult CompileOnlyKernelExecution::CompileToNeffResult() const {
  // Non-WLO kernels: NEFF only, empty wrapped_neff.hlo.
  if (!wlo_enabled_) {
    return CompileResult{CompileToNeff(), /*wrapped_neff_hlo=*/{}};
  }

  if (mlir_bytes_.empty()) throw std::invalid_argument("Cannot compile: no MLIR bytes available");

  // WLO does not support the tensorizer/Linalg route; that path emits no wrapped_neff.hlo.
  if (use_tensorizer_) {
    throw std::runtime_error(
        "WLO (weight_layout_optimization) is not supported with the tensorizer backend");
  }

  try {
    CompilationConfig config{.framework = "XLA",
                             .platform_target = platform_target_,
                             .logical_neuron_cores = logical_neuron_cores_,
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeffWithWlo(mlir_bytes_, config, "StableHLO",
                                                   "torch_compile_wlo", GetCompileKey(),
                                                   "CompileOnlyKernelExecution");
  } catch (const std::exception& e) {
    throw std::runtime_error("StableHLO to NEFF (WLO) compilation failed: " +
                             std::string(e.what()));
  }
}

// NeffDirectKernelExecution implementation
NeffDirectKernelExecution::NeffDirectKernelExecution(
    const std::string& graph_name, const std::string& cache_key,
    std::vector<TensorDataRef>&& input_refs, std::vector<TensorDataRef>&& output_refs,
    const std::vector<TensorContext>& input_contexts,
    const std::vector<TensorContext>& output_contexts, int device_id, bool has_collectives)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(
          graph_name, std::move(input_refs), std::move(output_refs), input_contexts,
          output_contexts, device_id),
      // No IR here, so this kernel cannot compute the content-addressed persistent key.
      // Keep CompileOnlyKernelExecution's compiler config, and let CompilationCache
      // resolve the key from the hint that the compile step recorded.
      CompilableKernelExecution(
          CompilationCacheKey(cache_key, CompileOnlyKernelExecution::GetCompileAdditionalArgs(),
                              CompileOnlyKernelExecution::GetCompileOptimizationLevel(),
                              /*hlo_bytes_ptr=*/nullptr, CacheKeySource::TORCH_COMPILE),
          CompileOnlyKernelExecution::GetCompileAdditionalArgs(),
          CompileOnlyKernelExecution::GetCompileOptimizationLevel(), has_collectives) {}

bool NeffDirectKernelExecution::ValidateImpl() const { return HasOutputs(); }

std::vector<uint8_t> NeffDirectKernelExecution::CompileToNeff() const {
  // NEFF is pre-compiled. Must always result in a cache hit.
  throw std::logic_error("NeffDirectKernelExecution expects pre-compiled graphs.");
}

bool NeffDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                         c10::StreamId stream_id) const {
  // Create slices for tensors with storage_offset > 0
  const auto src_data_ptrs = HandleContiguousSlicing(GetSrcDataPtrs(), src_tensor_ctxs_);
  const auto dst_data_ptrs = HandleContiguousSlicing(GetDstDataPtrs(), dst_tensor_ctxs_);

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1);
  auto model = NRTHandler::GetOrLoadModel(compilable_kernel, config);
  NRTHandler::DispatchModelExecution(model, src_data_ptrs, dst_data_ptrs, config, nrt_status,
                                     sequence_id, stream_id, GetInputNames(), GetOutputNames());
  return true;
}

// CollectiveDirectKernelExecution implementation

CollectiveDirectKernelExecution::CollectiveDirectKernelExecution(
    const std::string& op_name, CollectiveType collective_type,
    std::vector<TensorDataRef>&& src_refs, std::vector<TensorDataRef>&& dst_refs,
    const std::vector<TensorContext>& input_contexts,
    const std::vector<TensorContext>& output_contexts, c10::ScalarType dtype,
    c10d::ReduceOp reduce_op, const std::vector<std::vector<int>>& replica_groups, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCollective, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, std::move(src_refs),
                                                               std::move(dst_refs), input_contexts,
                                                               output_contexts, device_id),
      collective_type_(collective_type),
      reduce_op_(reduce_op),
      dtype_(dtype),
      replica_groups_(replica_groups),
      nrt_dtype_(nrt::MapScalarTypeToNRT(dtype)),
      nrt_reduce_op_(nrt::MapReduceOpToNRT(reduce_op)) {
  // Map collective type to NRT
  switch (collective_type) {
    case CollectiveType::kAllGather:
      nrt_collective_op_ = nrt::NRTCollectiveOp::kAllGather;
      break;
    case CollectiveType::kReduceScatter:
      nrt_collective_op_ = nrt::NRTCollectiveOp::kReduceScatter;
      break;
    case CollectiveType::kAllReduce:
      nrt_collective_op_ = nrt::NRTCollectiveOp::kAllReduce;
      break;
    default:
      TORCH_CHECK(false, "Unsupported collective type: ", static_cast<int>(collective_type));
  }

  // Find the replica group that contains the current device
  int global_rank = c10_neuron::get_rank();
  TORCH_CHECK(!replica_groups_.empty(), "Replica groups must not be empty");
  for (auto& group : replica_groups_) {
    for (size_t i = 0; i < group.size(); ++i) {
      if (group[i] == global_rank) {
        my_replica_group_ = &group;
        my_rank_ = static_cast<int>(i);
        break;
      }
    }
    if (my_replica_group_ != nullptr) {
      break;
    }
  }
  TORCH_CHECK(my_replica_group_ != nullptr && my_rank_ >= 0, "Global rank ", global_rank,
              " (device_id=", device_id, ") not found in any replica group");

  // Build comm struct (all fields are known at construction time)
  int vnc_id = c10_neuron::get_vnc_id(device_id);
  comm_ = {.replica_group = reinterpret_cast<uint32_t*>(my_replica_group_->data()),
           .rank = static_cast<uint32_t>(my_rank_),
           .rank_n = static_cast<uint32_t>(my_replica_group_->size()),
           .ctx_device_id = static_cast<uint32_t>(global_rank),
           .ctx_device_count = static_cast<uint32_t>(c10_neuron::get_world_size()),
           .vnc = static_cast<uint32_t>(vnc_id)};
}

bool CollectiveDirectKernelExecution::ValidateImpl() const {
  return HasInputs() && HasOutputs() && !replica_groups_.empty();
}

void CollectiveDirectKernelExecution::PrepareImpl() const {
  TORCH_NEURONX_DEBUG("CollectiveDirectKernelExecution::PrepareImpl()",
                      "collective_type=", static_cast<int>(collective_type_),
                      "device_id=", GetDeviceId());
  prepared_src_data_refs_ = HandleContiguousSlicing(GetSrcDataPtrs(), src_tensor_ctxs_);
  prepared_dst_data_refs_ = HandleContiguousSlicing(GetDstDataPtrs(), dst_tensor_ctxs_);

  EXECUTION_RUNTIME_CHECK(!prepared_src_data_refs_.empty(),
                          "Collective operation requires at least one input tensor");
  EXECUTION_RUNTIME_CHECK(!prepared_dst_data_refs_.empty(),
                          "Collective operation requires at least one output tensor");

  input_tensor_list_ = {.tensors = prepared_src_data_refs_.data(),
                        .num_tensors = prepared_src_data_refs_.size()};

  output_tensor_list_ = {.tensors = prepared_dst_data_refs_.data(),
                         .num_tensors = prepared_dst_data_refs_.size()};

  NRT_STATUS status =
      nrt::CollectivesPrepare(&comm_, &input_tensor_list_, &output_tensor_list_, nrt_dtype_,
                              nrt_reduce_op_, nrt_collective_op_, &cc_ctx_);
  EXECUTION_RUNTIME_CHECK(status == NRT_SUCCESS || status == NRT_COLL_PENDING,
                          "Collective prepare failed with status: ", status);
  TORCH_NEURONX_DEBUG("CollectiveDirectKernelExecution::PrepareImpl() completed",
                      "device_id=", GetDeviceId(), "rank=", my_rank_, "cc_ctx=", cc_ctx_);
}

void CollectiveDirectKernelExecution::Prepare() const {
  if (c10_neuron::NeuronCachingAllocator::IsLazyAllocEnabled()) {
    return;
  }
  PrepareImpl();
}

bool CollectiveDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                               c10::StreamId stream_id) const {
  TORCH_NEURONX_DEBUG("CollectiveDirectKernelExecution::Schedule()",
                      "collective_type=", static_cast<int>(collective_type_));

  if (c10_neuron::NeuronCachingAllocator::IsLazyAllocEnabled()) {
    PrepareImpl();
  } else {
    // Check that Prepare() was called (cc_ctx_ should be set)
    EXECUTION_RUNTIME_CHECK(cc_ctx_ != nullptr,
                            "CollectiveDirectKernelExecution::Schedule() called before Prepare()");
  }

  int queue = nrt::GetHardwareQueue(NRTA_XU_COLLECTIVES, stream_id);
  NRT_STATUS status = nrt::CollectivesSchedule(&cc_ctx_, queue, nrt_status, sequence_id);
  EXECUTION_RUNTIME_CHECK(status == NRT_SUCCESS,
                          "Collective schedule failed with status: ", status);

  TORCH_NEURONX_DEBUG("CollectiveDirectKernelExecution::Schedule() completed", "cc_ctx=", cc_ctx_,
                      "seq=", *sequence_id);
  return true;
}

HintDirectKernelExecution::HintDirectKernelExecution(const std::string& op_name, void* ptr,
                                                     size_t size_bytes, int device_id,
                                                     HintType hint_type)
    : NeuronKernelExecution(op_name, device_id),
      ptr_(ptr),
      size_bytes_(size_bytes),
      hint_type_(hint_type) {}

AllocDirectKernelExecution::AllocDirectKernelExecution(const std::string& op_name,
                                                       TensorDataRef dst_ref, size_t size_bytes,
                                                       int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kAlloc, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {}, {std::move(dst_ref)},
                                                               device_id),
      size_bytes_(size_bytes),
      event_op_(op_name + "::event", NeuronEvent(),
                EventDirectKernelExecution::EventAction::kSignal, device_id) {}

bool AllocDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                          c10::StreamId stream_id) const {
  try {
    c10_neuron::NeuronCachingAllocator::lazy::AllocBind(dst_ptrs_[0], size_bytes_, device_id_);
  } catch (const torch_neuronx::ExecutionRuntimeException& e) {
    // Surface NRT_RESOURCE via out-param so ScheduleOperation can defer-and-retry
    // without unwinding through the error handler on the first hit.
    if (e.HasNRTStatus() && e.GetNRTStatus() == NRT_RESOURCE) {
      *nrt_status = NRT_RESOURCE;
      return false;
    }
    throw;
  }
  event_op_.Schedule(nrt_status, sequence_id, stream_id);
  return false;
}

DeallocDirectKernelExecution::DeallocDirectKernelExecution(const std::string& op_name,
                                                           TensorDataRef src_ref, size_t size_bytes,
                                                           int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kDealloc, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {std::move(src_ref)}, {},
                                                               device_id),
      size_bytes_(size_bytes),
      event_op_(op_name + "::event", NeuronEvent(),
                EventDirectKernelExecution::EventAction::kSignal, device_id) {}

bool DeallocDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                            c10::StreamId stream_id) const {
  c10_neuron::NeuronCachingAllocator::lazy::DeallocRelease(src_ptrs_[0], size_bytes_, device_id_,
                                                           stream_id);
  event_op_.Schedule(nrt_status, sequence_id, stream_id);
  return false;
}

CopyDirectKernelExecution::CopyDirectKernelExecution(const std::string& op_name,
                                                     TensorDataRef src_ref, TensorDataRef dst_ref,
                                                     size_t src_offset_bytes,
                                                     size_t dst_offset_bytes, size_t size_bytes,
                                                     int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {std::move(src_ref)},
                                                               {std::move(dst_ref)}, device_id),
      src_offset_bytes_(src_offset_bytes),
      dst_offset_bytes_(dst_offset_bytes),
      size_bytes_(size_bytes) {}

bool CopyDirectKernelExecution::ValidateImpl() const { return size_bytes_ > 0; }

bool CopyDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                         c10::StreamId stream_id) const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  nrt_tensor_t* dst_ptr = GetDstDataPtrs()[0];

  int lnc_idx = c10_neuron::get_vnc_id(device_id_);
  EXECUTION_RUNTIME_CHECK(lnc_idx >= 0, "Invalid lnc_idx ", lnc_idx, " for device_id ", device_id_);
  int queue = nrt::GetHardwareQueue(NRTA_XU_TENSOR_OP, stream_id);
  NRT_STATUS status =
      nrt::ScheduleTensorCopy(src_ptr, src_offset_bytes_, dst_ptr, dst_offset_bytes_, size_bytes_,
                              lnc_idx, queue, nrt_status, sequence_id);
  if (status != NRT_SUCCESS) {
    std::string error_msg =
        "Failed to schedule async tensor copy. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
  return true;
}

CastDirectKernelExecution::CastDirectKernelExecution(
    const std::string& op_name, TensorDataRef src_ref, TensorDataRef dst_ref,
    uint64_t src_offset_elements, uint64_t dst_offset_elements, c10::ScalarType src_dtype,
    c10::ScalarType dst_dtype, uint64_t num_elements, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCast, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {std::move(src_ref)},
                                                               {std::move(dst_ref)}, device_id),
      src_offset_elements_(src_offset_elements),
      dst_offset_elements_(dst_offset_elements),
      src_dtype_(src_dtype),
      dst_dtype_(dst_dtype),
      num_elements_(num_elements) {}

bool CastDirectKernelExecution::ValidateImpl() const { return num_elements_ > 0; }

bool CastDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                         c10::StreamId stream_id) const {
  const NRTHandler::TensorCastOperand src{
      GetSrcDataPtrs()[0],
      src_offset_elements_,
      src_dtype_,
  };
  const NRTHandler::TensorCastOperand dst{
      GetDstDataPtrs()[0],
      dst_offset_elements_,
      dst_dtype_,
  };

  int lnc_idx = c10_neuron::get_vnc_id(device_id_);
  EXECUTION_RUNTIME_CHECK(lnc_idx >= 0, "Invalid lnc_idx ", lnc_idx, " for device_id ", device_id_);
  int queue = nrt::GetHardwareQueue(NRTA_XU_TENSOR_OP, stream_id);
  NRT_STATUS status = NRTHandler::ScheduleTensorCast(src, dst, num_elements_, lnc_idx, queue,
                                                     nrt_status, sequence_id);
  if (status != NRT_SUCCESS) {
    throw torch_neuronx::ExecutionRuntimeException(
        "Failed to schedule async tensor cast. Status: " + std::to_string(status), status);
  }
  return true;
}

WriteDirectKernelExecution::WriteDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                       TensorDataRef dst_ref,
                                                       size_t dst_offset_bytes,
                                                       size_t dst_size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kWrite, c10::DeviceType::CPU,
                                 c10::DeviceType::PrivateUse1>(op_name, {TensorDataRef{src_ptr}},
                                                               {std::move(dst_ref)}, device_id),
      dst_offset_bytes_(dst_offset_bytes),
      dst_size_bytes_(dst_size_bytes) {}

bool WriteDirectKernelExecution::ValidateImpl() const { return dst_size_bytes_ > 0; }

void* WriteDirectKernelExecution::AllocateBounceBuffer(size_t size_bytes) {
  bounce_buffer_.resize((size_bytes + 63) / 64);
  void* bounce = bounce_buffer_.data()->data;
  torch_neuronx::utils::non_temporal_memcpy(bounce, src_ptrs_[0], size_bytes);
  torch_neuronx::utils::non_temporal_sfence();
  src_ptrs_[0] = bounce;
  src_data_refs_[0].ptr = bounce;
  return bounce;
}

bool WriteDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                          c10::StreamId stream_id) const {
  nrt_tensor_t* dst_data_ptr = GetDstDataPtrs()[0];
  void* src_data_ptr =
      bounce_buffer_.empty() ? GetSrcDataPtrs()[0] : const_cast<char*>(bounce_buffer_.data()->data);

  int lnc_idx = c10_neuron::get_vnc_id(device_id_);
  EXECUTION_RUNTIME_CHECK(lnc_idx >= 0, "Invalid lnc_idx ", lnc_idx, " for device_id ", device_id_);
  int queue = nrt::GetHardwareQueue(NRTA_XU_TENSOR_OP, stream_id);
  NRT_STATUS status =
      nrt::ScheduleTensorWrite(dst_data_ptr, src_data_ptr, dst_offset_bytes_, dst_size_bytes_,
                               lnc_idx, queue, nrt_status, sequence_id);
  if (status != NRT_SUCCESS) {
    std::string error_msg =
        "Failed to schedule async tensor write. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
  return true;
}

ReadDirectKernelExecution::ReadDirectKernelExecution(const std::string& op_name,
                                                     TensorDataRef src_ref, void* dst_ptr,
                                                     size_t src_offset_bytes, size_t src_size_bytes,
                                                     int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kRead, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::CPU>(op_name, {std::move(src_ref)},
                                                       {TensorDataRef{dst_ptr}}, device_id),
      src_offset_bytes_(src_offset_bytes),
      src_size_bytes_(src_size_bytes) {}

bool ReadDirectKernelExecution::ValidateImpl() const { return src_size_bytes_ > 0; }

bool ReadDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                         c10::StreamId stream_id) const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  void* dst_ptr = GetDstDataPtrs()[0];

  int lnc_idx = c10_neuron::get_vnc_id(device_id_);
  EXECUTION_RUNTIME_CHECK(lnc_idx >= 0, "Invalid lnc_idx ", lnc_idx, " for device_id ", device_id_);
  int queue = nrt::GetHardwareQueue(NRTA_XU_TENSOR_OP, stream_id);
  NRT_STATUS status = nrt::ScheduleTensorRead(dst_ptr, src_ptr, src_offset_bytes_, src_size_bytes_,
                                              lnc_idx, queue, nrt_status, sequence_id);
  if (status != NRT_SUCCESS) {
    std::string error_msg =
        "Failed to schedule async tensor read. Status: " + std::to_string(status);
    throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
  }
  return true;
}

EventDirectKernelExecution::EventDirectKernelExecution(const std::string& op_name,
                                                       const NeuronEvent& event, EventAction action,
                                                       int device_id)
    : NeuronKernelExecution(op_name, device_id),
      event_(event),
      action_(action),
      signal_stream_id_(event.recorded_stream_id()) {}

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

bool EventDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                          c10::StreamId stream_id) const {
  // Events are host-side operations - always execute synchronously
  if (action_ == EventAction::kWait) {
    // At this point, the event is assumed to have completed
    EXECUTION_RUNTIME_CHECK(IsReady(), "Event not ready for wait");
    return false;
  }
  event_.complete_event();
  return false;
}

// BarrierDirectKernelExecution
BarrierDirectKernelExecution::BarrierDirectKernelExecution(int rank, int world_size, int device_id)
    : NeuronKernelExecution("barrier", device_id), rank_(rank), world_size_(world_size) {}

bool BarrierDirectKernelExecution::Schedule(NRT_STATUS* nrt_status, nrt::SequenceId* sequence_id,
                                            c10::StreamId stream_id) const {
  *nrt_status = at::neuron::nrt::Barrier(0, rank_, world_size_);
  return false;  // synchronous, no async tracking needed
}

}  // namespace at::neuron
