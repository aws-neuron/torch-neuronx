#include "torch_neuronx/csrc/distributed/NeuronBackend.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mutex>
#include <numeric>
#include <unordered_set>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronBarrier.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronOpsApi.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/distributed/DistributedUtils.h"

namespace py = pybind11;

namespace torch_neuronx {

at::neuron::CollectiveDirectKernelExecution::CollectiveType ConvertOpTypeToCollectiveType(
    c10d::OpType op_type);

namespace distributed {

// Static shared callback — same function for all NeuronBackend instances
py::object NeuronBackend::lower_callback_;

NeuronBackend::NeuronBackend(const c10::intrusive_ptr<c10d::Backend::Options>& options, int rank,
                             int size, int global_world_size, py::object lower_callback)
    : c10d::Backend(rank, size), timeout_ms_(options->timeout.count()) {
  static std::atomic<uint64_t> uid_counter{0};
  pg_uid_ = std::to_string(uid_counter.fetch_add(1));

  if (!lower_callback_) {
    lower_callback_ = std::move(lower_callback);
  }

  const auto& global_ranks = options->global_ranks_in_group;
  if (!global_ranks.empty()) {
    group_ranks_.assign(global_ranks.begin(), global_ranks.end());
    global_world_size_ = global_world_size;
  } else {
    group_ranks_.resize(size);
    std::iota(group_ranks_.begin(), group_ranks_.end(), 0);
    global_world_size_ = size;
  }
  watchdog_.start();
}

NeuronBackend::~NeuronBackend() { watchdog_.stop(); }

void NeuronBackend::Cleanup() { lower_callback_ = py::object(); }

c10::intrusive_ptr<NeuronWork> NeuronBackend::CreateWork(const std::vector<at::Tensor>& outputs,
                                                         const std::vector<at::Tensor>& inputs,
                                                         c10d::OpType op_type, bool async_op) {
  // Derive device from tensors (Python passes device.index from the tensors)
  int device_idx = 0;
  if (!outputs.empty()) {
    device_idx = outputs[0].device().index();
  } else if (!inputs.empty()) {
    device_idx = inputs[0].device().index();
  } else {
    TORCH_CHECK(op_type == c10d::OpType::BARRIER,
                "CreateWork called with empty inputs and outputs for non-barrier op");
    device_idx = c10_neuron::current_device();
  }
  auto stream = at::neuron::getCurrentNeuronStream(device_idx);
  auto work = c10::make_intrusive<NeuronWork>(pg_uid_, getGroupDesc(),
                                              at::Device(c10::DeviceType::PrivateUse1, device_idx),
                                              rank_, OpTypeToString(op_type), seq_num_.fetch_add(1),
                                              outputs, enable_timing_, timeout_ms_, &stream);
  if (async_op) {
    work->stash(inputs);
    work->stash(outputs);
  }
  return work;
}

bool NeuronBackend::UseHostCC() const {
  static const bool hostcc_enabled = []() {
    const char* val = std::getenv("TORCH_NEURONX_ENABLE_HOST_CC");
    return val != nullptr && val[0] != '\0' && std::string(val) != "0";
  }();
  auto current = at::neuron::getCurrentNeuronStream();
  auto default_stream = at::neuron::getDefaultNeuronStream();
  return (current != default_stream) && hostcc_enabled;
}

}  // namespace distributed
}  // namespace torch_neuronx

namespace {

int64_t CollectiveBucketSizeBytes() {
  static const int64_t val = []() {
    const char* env = std::getenv("COLLECTIVE_BUCKETSIZE_IN_MB");
    int64_t mb = 512;
    if (env && env[0] != '\0') {
      try {
        mb = std::stoll(env);
      } catch (const std::exception&) {
        // Fall back to default
      }
    }
    return mb * 1024 * 1024;
  }();
  return val;
}

bool UseSplitAllgather() {
  static const bool val = []() {
    const char* env = std::getenv("NEURON_USE_SPLIT_ALLGATHER");
    return env == nullptr || std::string(env) == "1";
  }();
  return val;
}

bool UseSplitReduceScatter() {
  static const bool val = []() {
    const char* env = std::getenv("NEURON_USE_SPLIT_REDUCE_SCATTER");
    return env == nullptr || std::string(env) == "1";
  }();
  return val;
}

std::vector<std::vector<int>> CreateGlobalReplicaGroups(
    int world_size, const std::vector<int>& current_group_ranks) {
  return torch_neuronx::distributed::CreateGlobalReplicaGroups(world_size, current_group_ranks);
}

}  // namespace

namespace torch_neuronx {
namespace distributed {

void NeuronBackend::SubmitInNeffCollective(c10d::OpType op_type,
                                           const std::vector<at::Tensor>& inputs,
                                           const std::vector<at::Tensor>& outputs,
                                           const std::vector<std::vector<int>>& replica_groups,
                                           int reduce_op, bool slice_output) {
  const std::vector<at::Tensor> original_outputs = outputs;
  std::vector<at::Tensor> working_inputs = inputs;
  std::vector<at::Tensor> working_outputs = outputs;
  const bool inputs_downcasted = dispatch::DowncastTensorList(working_inputs);
  const bool outputs_downcasted = dispatch::DowncastTensorList(working_outputs);
  (void)inputs_downcasted;  // Input downcast is permanent for this call; no copy-back.

  // For AVG, append divisor tensor to inputs. Build it at the (possibly
  // downcasted) working-input dtype so the compiled graph stays dtype-consistent.
  std::vector<at::Tensor> effective_inputs = working_inputs;
  if (reduce_op == static_cast<int>(c10d::ReduceOp::AVG) && !working_inputs.empty()) {
    auto divisor = torch::tensor(
        static_cast<double>(replica_groups[0].size()),
        torch::TensorOptions().dtype(working_inputs[0].dtype()).device(working_inputs[0].device()));
    effective_inputs.push_back(divisor);
  }

  const auto& op_str = OpTypeToString(op_type);
  auto cache_key = at::neuron::CacheKeyGenerator::GenerateCollective(
      op_str, effective_inputs, replica_groups, reduce_op, slice_output);
  auto entry = dispatch::NeuronDispatcher::Instance().GetOrLowerCollectiveOperation(
      cache_key, lower_callback_, op_type, effective_inputs, working_outputs, replica_groups,
      reduce_op, slice_output);

  auto transform_result = torch_neuronx::ProcessInputTransformations(op_str, effective_inputs,
                                                                     /*has_collectives=*/true);
  int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs, working_outputs);
  auto [input_refs, input_contexts] =
      torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
  auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(working_outputs);

  auto xla_kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
      op_str, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
      entry->neff_cache_key, entry->stablehlo_bytes, /*has_collectives=*/true, device_id);

  if (transform_result.has_input_transforms) {
    xla_kernel->SetInputTransformations(std::move(transform_result.kernel_transforms));
  }

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(xla_kernel), "");
  auto stream = at::neuron::getCurrentNeuronStream();
  at::neuron::SubmitOperationContext(stream, std::move(context));

  if (outputs_downcasted) {
    for (size_t i = 0; i < original_outputs.size(); ++i) {
      auto orig = original_outputs[i];
      if (!orig.defined()) continue;
      if (i >= working_outputs.size() || !working_outputs[i].defined()) continue;
      if (orig.scalar_type() == working_outputs[i].scalar_type()) continue;
      dispatch::CopyBackToOriginal(
          orig, working_outputs[i],
          std::string("SubmitInNeffCollective upcast idx=") + std::to_string(i));
    }
  }
}

void NeuronBackend::SubmitHostCCCollective(c10d::OpType op_type,
                                           const std::vector<at::Tensor>& inputs,
                                           const std::vector<at::Tensor>& outputs,
                                           const std::vector<std::vector<int>>& replica_groups,
                                           int reduce_op) {
  const std::vector<at::Tensor> original_outputs = outputs;
  std::vector<at::Tensor> working_inputs = inputs;
  std::vector<at::Tensor> working_outputs = outputs;
  const bool inputs_downcasted = dispatch::DowncastTensorList(working_inputs);
  const bool outputs_downcasted = dispatch::DowncastTensorList(working_outputs);
  (void)inputs_downcasted;

  const auto& op_str = OpTypeToString(op_type);
  auto transform_result =
      torch_neuronx::ProcessInputTransformations(op_str, working_inputs, /*has_collectives=*/true);
  TORCH_CHECK(!transform_result.final_inputs.empty(),
              "Collective operation requires at least one input tensor");
  int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs, working_outputs);
  auto [input_refs, input_contexts] =
      torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
  auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(working_outputs);

  auto collective_type = torch_neuronx::ConvertOpTypeToCollectiveType(op_type);
  c10::ScalarType dtype = transform_result.final_inputs[0].scalar_type();

  // NRT does not support AVG natively; convert to SUM and post-divide by group size
  int effective_reduce_op = reduce_op;
  bool avg_post_divide = (reduce_op == static_cast<int>(c10d::ReduceOp::AVG));
  if (avg_post_divide) {
    effective_reduce_op = static_cast<int>(c10d::ReduceOp::SUM);
  }

  auto collective_kernel = std::make_unique<at::neuron::CollectiveDirectKernelExecution>(
      op_str, collective_type, std::move(input_refs), std::move(output_refs), input_contexts,
      output_contexts, dtype,
      c10d::ReduceOp(static_cast<c10d::ReduceOp::RedOpType>(effective_reduce_op)), replica_groups,
      device_id);

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(collective_kernel), "");
  auto stream = at::neuron::getCurrentNeuronStream();
  at::neuron::SubmitOperationContext(stream, std::move(context));

  if (avg_post_divide) {
    for (auto& t : working_outputs) t.div_(static_cast<double>(replica_groups[0].size()));
  }

  if (outputs_downcasted) {
    for (size_t i = 0; i < original_outputs.size(); ++i) {
      auto orig = original_outputs[i];
      if (!orig.defined()) continue;
      if (i >= working_outputs.size() || !working_outputs[i].defined()) continue;
      if (orig.scalar_type() == working_outputs[i].scalar_type()) continue;
      dispatch::CopyBackToOriginal(
          orig, working_outputs[i],
          std::string("SubmitHostCCCollective upcast idx=") + std::to_string(i));
    }
  }
}

std::vector<std::vector<int>> NeuronBackend::GetReplicaGroups() const { return {group_ranks_}; }

std::vector<std::vector<int>> NeuronBackend::GetFullReplicaGroups() const {
  return CreateGlobalReplicaGroups(global_world_size_, group_ranks_);
}

void NeuronBackend::FinishWork(c10::intrusive_ptr<NeuronWork>& work, bool async_op) {
  auto stream = const_cast<at::neuron::NeuronStream*>(&work->getStream());
  work->recordEndEvent(stream, async_op);
  work->registerWithTensors();
  watchdog_.enqueueWork(work);
}

std::vector<at::Tensor> NeuronBackend::ReduceScatterSubmit(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const std::vector<std::vector<int>>& replica_groups, int red_op) {
  int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
  int64_t input_bytes = input_tensor.numel() * input_tensor.element_size();
  std::vector<at::Tensor> intermediates;

  if (UseSplitReduceScatter() && input_bytes > CollectiveBucketSizeBytes() &&
      input_tensor.size(0) == output_tensor.size(0) * world_size) {
    int64_t element_size = input_tensor.element_size();
    int64_t bucket_elements = CollectiveBucketSizeBytes() / element_size;
    int64_t elems_per_block = input_tensor.numel() / world_size;
    int64_t pick_per_block = bucket_elements / world_size;
    if (pick_per_block < 1) pick_per_block = 1;

    auto input_flat = input_tensor.reshape({-1});
    auto output_flat = output_tensor.reshape({-1});

    int64_t num_chunks = elems_per_block / pick_per_block;
    int64_t remainder = elems_per_block % pick_per_block;
    int64_t out_offset = 0;

    auto do_bucket = [&](int64_t chunk_idx, int64_t chunk_elems) {
      std::vector<at::Tensor> slices;
      slices.reserve(world_size);
      for (int64_t r = 0; r < world_size; ++r) {
        int64_t start = r * elems_per_block + chunk_idx * pick_per_block;
        slices.push_back(input_flat.narrow(0, start, chunk_elems));
      }
      auto bucket_in = torch::cat(slices, 0);
      auto bucket_out = output_flat.narrow(0, out_offset, chunk_elems);
      SubmitCollective(c10d::OpType::REDUCE_SCATTER, {bucket_in}, {bucket_out}, replica_groups,
                       red_op);
      intermediates.push_back(bucket_in);
      out_offset += chunk_elems;
    };

    for (int64_t c = 0; c < num_chunks; ++c) {
      do_bucket(c, pick_per_block);
    }
    if (remainder > 0) {
      std::vector<at::Tensor> slices;
      slices.reserve(world_size);
      for (int64_t r = 0; r < world_size; ++r) {
        int64_t start = r * elems_per_block + num_chunks * pick_per_block;
        slices.push_back(input_flat.narrow(0, start, remainder));
      }
      auto bucket_in = torch::cat(slices, 0);
      auto bucket_out = output_flat.narrow(0, out_offset, remainder);
      SubmitCollective(c10d::OpType::REDUCE_SCATTER, {bucket_in}, {bucket_out}, replica_groups,
                       red_op);
      intermediates.push_back(bucket_in);
    }
  } else {
    SubmitCollective(c10d::OpType::REDUCE_SCATTER, {input_tensor}, {output_tensor}, replica_groups,
                     red_op);
  }
  return intermediates;
}

void NeuronBackend::SubmitCollective(c10d::OpType op_type, const std::vector<at::Tensor>& inputs,
                                     const std::vector<at::Tensor>& outputs,
                                     const std::vector<std::vector<int>>& replica_groups,
                                     int reduce_op, bool slice_output) {
  if (UseHostCC()) {
    SubmitHostCCCollective(op_type, inputs, outputs, replica_groups, reduce_op);
  } else {
    SubmitInNeffCollective(op_type, inputs, outputs, replica_groups, reduce_op, slice_output);
  }
}

int NeuronBackend::ResolveReduceOp(const c10d::ReduceOp& reduce_op,
                                   std::vector<at::Tensor>& tensors) {
  int red_op = static_cast<int>(reduce_op.operator c10d::ReduceOp::RedOpType());

  // PREMUL_SUM: scale inputs in-place before collective, then use SUM
  if (reduce_op == c10d::ReduceOp::PREMUL_SUM && reduce_op.supplement_) {
    auto* supplement = dynamic_cast<c10d::NCCLPreMulSumSupplement*>(reduce_op.supplement_.get());
    TORCH_CHECK(supplement, "PREMUL_SUM supplement is not NCCLPreMulSumSupplement");
    if (supplement->tensor_factor.defined()) {
      for (auto& t : tensors) t.mul_(supplement->tensor_factor);
    } else if (supplement->double_factor != 1.0) {
      for (auto& t : tensors) t.mul_(supplement->double_factor);
    }
    red_op = static_cast<int>(c10d::ReduceOp::SUM);
  }
  return red_op;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::allreduce(std::vector<at::Tensor>& tensors,
                                                        const c10d::AllreduceOptions& opts) {
  ValidateNeuronDevice(tensors, "allreduce");
  ValidateNonEmpty(tensors, "allreduce");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  // Resolve reduce op (scales `tensors` in-place for PREMUL_SUM, then maps to SUM).
  std::vector<at::Tensor> allreduce_inputs = tensors;
  int red_op = ResolveReduceOp(opts.reduceOp, tensors);

  auto work = CreateWork(tensors, allreduce_inputs, c10d::OpType::ALLREDUCE, async_op);
  try {
    SubmitCollective(c10d::OpType::ALLREDUCE, allreduce_inputs, tensors, replica_groups, red_op);
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::_allgather_base(at::Tensor& output_tensor,
                                                              at::Tensor& input_tensor,
                                                              const c10d::AllgatherOptions& opts) {
  ValidateNeuronDevice({input_tensor}, "allgather");
  TORCH_CHECK(input_tensor.numel() > 0,
              "tensors cannot have zero size in dimension 0, found: [0 (shape: ",
              input_tensor.sizes(), ")] in allgather");
  TORCH_CHECK(output_tensor.numel() == input_tensor.numel() * static_cast<int64_t>(size_),
              "output shape mismatch, expected numel=", input_tensor.numel() * size_, " got ",
              output_tensor.numel());
  auto replica_groups = GetFullReplicaGroups();
  int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
  bool async_op = opts.asyncOp;

  // Handle dimension mismatch (matching Python PG behavior)
  if (output_tensor.dim() != input_tensor.dim()) {
    output_tensor = output_tensor.squeeze();
    input_tensor = input_tensor.squeeze();
  }

  int64_t output_bytes = output_tensor.numel() * output_tensor.element_size();

  if (UseSplitAllgather() && output_bytes > CollectiveBucketSizeBytes()) {
    auto input_flat = input_tensor.flatten();
    int64_t input_numel = input_flat.numel();
    int64_t total_output_bytes = input_numel * input_tensor.element_size() * world_size;
    int64_t num_splits =
        (total_output_bytes + CollectiveBucketSizeBytes() - 1) / CollectiveBucketSizeBytes();
    int64_t split_size = (input_numel + num_splits - 1) / num_splits;
    auto splits = input_flat.split(split_size, 0);
    auto output_flat = output_tensor.view(-1);

    auto work =
        CreateWork({output_tensor}, {input_tensor}, c10d::OpType::_ALLGATHER_BASE, async_op);
    std::vector<at::Tensor> intermediates;
    try {
      int64_t split_offset = 0;
      for (const auto& split : splits) {
        int64_t ss = split.numel();
        std::vector<at::Tensor> rank_views;
        for (int64_t r = 0; r < world_size; ++r) {
          rank_views.push_back(output_flat.narrow(0, r * input_numel + split_offset, ss));
        }
        SubmitCollective(c10d::OpType::ALLGATHER, {split}, rank_views, replica_groups, 0,
                         /*slice_output=*/true);
        intermediates.push_back(split);
        split_offset += ss;
      }
    } catch (...) {
      work->setException(std::current_exception());
      if (!intermediates.empty()) work->stash(intermediates);
      FinishWork(work, async_op);
      throw;
    }
    if (!intermediates.empty()) work->stash(intermediates);
    FinishWork(work, async_op);
    return work;
  }

  auto work = CreateWork({output_tensor}, {input_tensor}, c10d::OpType::_ALLGATHER_BASE, async_op);
  try {
    SubmitCollective(c10d::OpType::ALLGATHER, {input_tensor}, {output_tensor}, replica_groups, 0);
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors_list,
    std::vector<at::Tensor>& input_tensors, const c10d::AllgatherOptions& opts) {
  ValidateNeuronDevice(input_tensors, "allgather");
  ValidateNonZeroDim0(input_tensors, "allgather");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;
  auto& output_list = output_tensors_list[0];

  // Pre-allocate stacked output for DDP comm hooks
  std::vector<int64_t> stacked_shape = {static_cast<int64_t>(output_list.size())};
  for (auto d : output_list[0].sizes()) stacked_shape.push_back(d);
  auto stacked = torch::empty(stacked_shape, output_list[0].options());

  // Create work with stacked tensor as output (matching Python)
  auto work = CreateWork({stacked}, input_tensors, c10d::OpType::ALLGATHER, async_op);
  try {
    if (UseHostCC()) {
      // Host-CC path: NRT's native allgather expects a single flattened output
      // (world_size * per_rank_shape) rather than a list of per-rank tensors.
      // Allgather into `stacked` directly, then copy slices back into the
      // user's output_list on the same stream.
      SubmitCollective(c10d::OpType::ALLGATHER, input_tensors, {stacked}, replica_groups, 0,
                       /*slice_output=*/false);
      for (size_t i = 0; i < output_list.size(); ++i) {
        output_list[i].copy_(stacked[static_cast<int64_t>(i)]);
      }
    } else {
      // In-NEFF path: the compiled HLO writes into each per-rank slice directly.
      SubmitCollective(c10d::OpType::ALLGATHER, input_tensors, output_list, replica_groups, 0,
                       /*slice_output=*/true);
      for (size_t i = 0; i < output_list.size(); ++i) {
        stacked[static_cast<int64_t>(i)].copy_(output_list[i]);
      }
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::_reduce_scatter_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor, const c10d::ReduceScatterOptions& opts) {
  ValidateNeuronDevice({input_tensor}, "reduce_scatter");
  auto replica_groups = GetFullReplicaGroups();
  int64_t group_size = static_cast<int64_t>(replica_groups[0].size());

  // Validate input size is divisible by group size
  TORCH_CHECK(input_tensor.numel() % group_size == 0, "scatter dimension 0: input size (",
              input_tensor.numel(), ") must be evenly divisible by replica group size (",
              group_size, "). Expected output size ", input_tensor.numel() / group_size,
              ", actual output size ", output_tensor.numel());
  // Validate output size matches
  int64_t expected_output = input_tensor.numel() / group_size;
  TORCH_CHECK(output_tensor.numel() == expected_output, "scatter dimension 0: input size (",
              input_tensor.numel(), ") must be evenly divisible by replica group size (",
              group_size, "). Expected output size ", expected_output, ", actual output size ",
              output_tensor.numel());
  // Validate supported reduce ops
  auto red_op_type = opts.reduceOp.operator c10d::ReduceOp::RedOpType();
  TORCH_CHECK(red_op_type == c10d::ReduceOp::SUM || red_op_type == c10d::ReduceOp::AVG ||
                  red_op_type == c10d::ReduceOp::MIN || red_op_type == c10d::ReduceOp::MAX ||
                  opts.reduceOp == c10d::ReduceOp::PREMUL_SUM,
              "unsupported reduce operation: ", static_cast<int>(red_op_type),
              ". supported ops: SUM, AVG, MIN, MAX, PREMUL_SUM");
  bool async_op = opts.asyncOp;

  // Resolve reduce op (scales `input_tensor` in-place for PREMUL_SUM, then maps to SUM).
  // `rs_input` aliases `input_tensor`, so it picks up the scaled values.
  at::Tensor rs_input = input_tensor;
  std::vector<at::Tensor> rs_scale_list{input_tensor};
  int red_op = ResolveReduceOp(opts.reduceOp, rs_scale_list);

  auto work = CreateWork({output_tensor}, {rs_input}, c10d::OpType::_REDUCE_SCATTER_BASE, async_op);
  try {
    auto intermediates = ReduceScatterSubmit(output_tensor, rs_input, replica_groups, red_op);
    if (!intermediates.empty()) work->stash(intermediates);
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::reduce_scatter(
    std::vector<at::Tensor>& output_tensors, std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  ValidateNeuronDevice(output_tensors, "reduce_scatter");
  ValidateNeuronDevice(input_tensors[0], "reduce_scatter");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  auto& output_tensor = output_tensors[0];
  auto& input_list = input_tensors[0];

  // Validate input tensors have non-zero dim0
  for (size_t i = 0; i < input_list.size(); ++i) {
    TORCH_CHECK(input_list[i].dim() == 0 || input_list[i].size(0) != 0,
                "all input tensors must have non-zero first dimension, "
                "but tensor ",
                i, " has shape ", input_list[i].sizes());
  }

  // Resolve reduce op (scales `input_list` in-place for PREMUL_SUM, then maps to SUM).
  int red_op = ResolveReduceOp(opts.reduceOp, input_list);

  at::Tensor concat_input =
      (input_list.size() == 1) ? input_list[0] : torch::cat(input_list, /*dim=*/0);

  // Validate scatter dimension (matches XLA layer: compare concatenated input to output)
  if (concat_input.dim() == output_tensor.dim()) {
    auto in_sizes = concat_input.sizes();
    auto out_sizes = output_tensor.sizes();
    int diff_count = 0;
    int64_t scatter_dim = -1;
    for (int64_t d = 0; d < concat_input.dim(); ++d) {
      if (in_sizes[d] != out_sizes[d]) {
        ++diff_count;
        scatter_dim = d;
      }
    }
    TORCH_CHECK(diff_count <= 1,
                "input and output shapes can only differ in one dimension, "
                "but found input shape ",
                in_sizes, " and output shape ", out_sizes);
    int64_t group_size = static_cast<int64_t>(replica_groups[0].size());
    if (diff_count == 0) {
      TORCH_CHECK(group_size == 1, "when input and output shapes are identical (", in_sizes,
                  "), replica group must have size 1, got ", group_size);
    } else {
      TORCH_CHECK(in_sizes[scatter_dim] == out_sizes[scatter_dim] * group_size,
                  "scatter dimension ", scatter_dim, ": input size (", in_sizes[scatter_dim],
                  ") must be evenly divisible by replica group size (", group_size,
                  "). Expected output size ", in_sizes[scatter_dim] / group_size,
                  ", actual output size ", out_sizes[scatter_dim]);
    }
  }

  auto work = CreateWork(output_tensors, input_list, c10d::OpType::REDUCE_SCATTER, async_op);
  try {
    SubmitCollective(c10d::OpType::REDUCE_SCATTER, {concat_input}, {output_tensors[0]},
                     replica_groups, red_op);
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::broadcast(std::vector<at::Tensor>& tensors,
                                                        const c10d::BroadcastOptions& opts) {
  ValidateNeuronDevice(tensors, "broadcast");
  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_, "Invalid rootRank ", opts.rootRank,
              " for broadcast with world size ", size_);
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;
  auto& input_tensor = tensors[0];

  if (input_tensor.numel() == 0) {
    auto work = CreateWork(tensors, {}, c10d::OpType::BROADCAST, async_op);
    FinishWork(work, async_op);
    return work;
  }

  // Zero non-root ranks, then allreduce SUM to broadcast root's data
  if (rank_ != opts.rootRank) {
    torch::NoGradGuard no_grad;
    input_tensor.zero_();
  }

  auto work = CreateWork(tensors, tensors, c10d::OpType::BROADCAST, async_op);
  try {
    SubmitCollective(c10d::OpType::ALLREDUCE, tensors, tensors, replica_groups,
                     static_cast<int>(c10d::ReduceOp::SUM));
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::reduce(std::vector<at::Tensor>& tensors,
                                                     const c10d::ReduceOptions& opts) {
  ValidateNeuronDevice(tensors, "reduce");
  ValidateNonEmpty(tensors, "reduce");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_, "Invalid rootRank ", opts.rootRank,
              " for reduce with world size ", size_);

  // Non-root ranks use temporary output tensors (discard result)
  std::vector<at::Tensor> output_tensors;
  if (rank_ == opts.rootRank) {
    output_tensors = tensors;
  } else {
    for (const auto& t : tensors) output_tensors.push_back(torch::empty_like(t));
  }

  int red_op = static_cast<int>(opts.reduceOp.operator c10d::ReduceOp::RedOpType());

  auto work = CreateWork(rank_ == opts.rootRank ? tensors : output_tensors, tensors,
                         c10d::OpType::REDUCE, async_op);
  try {
    SubmitCollective(c10d::OpType::ALLREDUCE, tensors, output_tensors, replica_groups, red_op);
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::barrier(const c10d::BarrierOptions& opts) {
  TORCH_CHECK(global_world_size_ == size_, "Barrier is implemented only for default group");
  torch_neuronx::maybe_lazy_init();

  at::neuron::drain_visible_device_streams("barrier");

  // Rendezvous all ranks via nrt_barrier. This call blocks on the caller
  // thread until every rank has arrived.
  // TODO: Introduce kBarrier async kernel path
  NRT_STATUS status =
      at::neuron::nrt::Barrier(0, static_cast<uint32_t>(rank_), static_cast<uint32_t>(size_));
  TORCH_CHECK(status == NRT_SUCCESS, "NeuronBackend::barrier: nrt_barrier failed with status ",
              status);

  // Return an already-complete Work: the barrier has rendezvoused before
  // this function returns, so .wait() is a no-op.
  auto work = CreateWork({}, {}, c10d::OpType::BARRIER, /*async_op=*/false);
  FinishWork(work, /*async_op=*/false);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::alltoall(std::vector<at::Tensor>& output_tensors,
                                                       std::vector<at::Tensor>& input_tensors,
                                                       const c10d::AllToAllOptions& opts) {
  ValidateNeuronDevice(input_tensors, "alltoall");
  TORCH_CHECK(static_cast<int>(input_tensors.size()) == size_, "number of input tensors (",
              input_tensors.size(), ") must equal world size (", size_, ")");
  if (input_tensors.size() > 1) {
    auto ref_shape = input_tensors[0].sizes();
    for (size_t i = 1; i < input_tensors.size(); ++i) {
      TORCH_CHECK(input_tensors[i].sizes() == ref_shape,
                  "all input tensors must have same shape, but found ", ref_shape, " and ",
                  input_tensors[i].sizes());
    }
  }
  bool async_op = opts.asyncOp;

  // Early return for all-empty inputs
  bool all_empty = true;
  for (const auto& t : input_tensors) {
    if (t.numel() > 0) {
      all_empty = false;
      break;
    }
  }
  if (all_empty) {
    auto work = CreateWork(output_tensors, {}, c10d::OpType::ALLTOALL, async_op);
    FinishWork(work, async_op);
    return work;
  }

  auto current_ranks = GetReplicaGroups()[0];
  int64_t group_size = static_cast<int64_t>(current_ranks.size());
  // For alltoall, we need global replica groups
  auto replica_groups = CreateGlobalReplicaGroups(global_world_size_, current_ranks);

  TORCH_CHECK(group_size == 4 || group_size == 8 || group_size == 16 || group_size % 32 == 0,
              "unsupported world size ", group_size,
              ", supported sizes: 4, 8, 16, or multiples of 32");
  // Each replica group start rank must be a multiple of group size.
  for (size_t i = 0; i < replica_groups.size(); ++i) {
    const auto& g = replica_groups[i];
    TORCH_CHECK(!g.empty() && g[0] % group_size == 0,
                "replica group start ranks must be multiples of world size (", group_size,
                "), found: group[", i, "] starts at rank ", g.empty() ? -1 : g[0]);
  }
  auto input_shape = input_tensors[0].sizes();

  // Allocate temp concatenated result
  std::vector<int64_t> concat_shape = {group_size * input_shape[0]};
  for (int64_t i = 1; i < static_cast<int64_t>(input_shape.size()); ++i) {
    concat_shape.push_back(input_shape[i]);
  }
  auto temp_result = torch::empty(concat_shape, input_tensors[0].options());

  auto work = CreateWork(output_tensors, input_tensors, c10d::OpType::ALLTOALL, async_op);
  try {
    SubmitCollective(c10d::OpType::ALLTOALL, input_tensors, {temp_result}, replica_groups, 0);
    // Split result back into output tensors
    for (int64_t i = 0; i < group_size; ++i) {
      output_tensors[i].copy_(temp_result.slice(0, i * input_shape[0], (i + 1) * input_shape[0]));
    }
    work->stash({temp_result});
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::alltoall_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor, std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes, const c10d::AllToAllOptions& opts) {
  int64_t group_size = static_cast<int64_t>(size_);

  // Empty tensors: nothing to exchange
  if (input_tensor.numel() == 0 && output_tensor.numel() == 0) {
    auto work = CreateWork({output_tensor}, {input_tensor}, c10d::OpType::ALLTOALL_BASE, false);
    FinishWork(work, false);
    return work;
  }

  // Normalize split sizes: if empty/None, compute equal splits; else validate length.
  auto normalize = [&](const std::vector<int64_t>& sizes, int64_t total_size,
                       const char* name) -> std::vector<int64_t> {
    if (sizes.empty()) {
      TORCH_CHECK(total_size % group_size == 0, "alltoall_base: ", name, " size ", total_size,
                  " is not divisible by group_size ", group_size, " for equal split");
      return std::vector<int64_t>(group_size, total_size / group_size);
    }
    TORCH_CHECK(static_cast<int64_t>(sizes.size()) == group_size, "alltoall_base: ", name,
                "_split_sizes length (", sizes.size(), ") must match group_size (", group_size,
                ")");
    return sizes;
  };
  auto in_sizes = normalize(input_split_sizes, input_tensor.size(0), "input");
  auto out_sizes = normalize(output_split_sizes, output_tensor.size(0), "output");

  // Fast path: even splits (all sizes equal on both sides) — delegate to alltoall.
  bool even_in = std::all_of(in_sizes.begin() + 1, in_sizes.end(),
                             [&](int64_t s) { return s == in_sizes[0]; });
  bool even_out = std::all_of(out_sizes.begin() + 1, out_sizes.end(),
                              [&](int64_t s) { return s == out_sizes[0]; });
  if (even_in && even_out) {
    auto input_list = input_tensor.split(in_sizes[0], 0);
    auto output_list = output_tensor.split(out_sizes[0], 0);
    std::vector<at::Tensor> in_vec(input_list.begin(), input_list.end());
    std::vector<at::Tensor> out_vec(output_list.begin(), output_list.end());
    return alltoall(out_vec, in_vec, opts);
  }

  // Uneven splits: pad each sender's chunks to max_in_size, do an even alltoall via the
  // existing alltoall() method, then unpad into output. The returned Work covers the
  // inner collective; the unpad copies are enqueued on the same NeuronStream and rely
  // on stream-ordering for correctness. HW capability checks live in alltoall() itself.
  int64_t max_in_size = *std::max_element(in_sizes.begin(), in_sizes.end());

  // Build padded_input: single allocation, copy each non-empty chunk into its slot.
  // Shorter chunks / empty slots are implicitly zero-padded (torch::zeros init).
  auto padded_input = torch::zeros({group_size * max_in_size}, input_tensor.options());
  int64_t in_offset = 0;
  for (int64_t i = 0; i < group_size; ++i) {
    int64_t size = in_sizes[i];
    if (size > 0) {
      padded_input.slice(0, i * max_in_size, i * max_in_size + size)
          .copy_(input_tensor.slice(0, in_offset, in_offset + size));
    }
    in_offset += size;
  }
  // padded_output uses max_in_size as the uniform recv stride: every sender pads to the
  // same max_in_size, so every receiver slot is max_in_size.
  auto padded_output = torch::zeros({group_size * max_in_size}, output_tensor.options());

  auto padded_in_list = padded_input.split(max_in_size, 0);
  auto padded_out_list = padded_output.split(max_in_size, 0);
  std::vector<at::Tensor> in_vec(padded_in_list.begin(), padded_in_list.end());
  std::vector<at::Tensor> out_vec(padded_out_list.begin(), padded_out_list.end());

  auto work = alltoall(out_vec, in_vec, opts);

  // Unpad: copy actual-size slices from padded_output into output_tensor. Enqueued on
  // the same NeuronStream as the collective, so stream-ordering guarantees correctness
  // for in-stream readers.
  int64_t out_offset = 0;
  for (int64_t i = 0; i < group_size; ++i) {
    int64_t actual_size = out_sizes[i];
    if (actual_size > 0) {
      output_tensor.slice(0, out_offset, out_offset + actual_size)
          .copy_(padded_output.slice(0, i * max_in_size, i * max_in_size + actual_size));
    }
    out_offset += actual_size;
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceCoalescedOptions& opts) {
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  std::vector<at::Tensor> non_empty;
  for (auto& t : tensors) {
    if (t.numel() > 0) non_empty.push_back(t);
  }
  if (non_empty.empty()) {
    auto work = CreateWork(tensors, {}, c10d::OpType::ALLREDUCE_COALESCED, async_op);
    FinishWork(work, async_op);
    return work;
  }

  int red_op = ResolveReduceOp(opts.reduceOp, non_empty);

  auto work = CreateWork(tensors, non_empty, c10d::OpType::ALLREDUCE_COALESCED, async_op);
  try {
    if (UseHostCC()) {
      // Host-CC path: NRT's native allreduce handles one tensor per call.
      bool same_shape = true;
      for (size_t i = 1; i < non_empty.size(); ++i) {
        if (non_empty[i].sizes() != non_empty[0].sizes() ||
            non_empty[i].dtype() != non_empty[0].dtype()) {
          same_shape = false;
          break;
        }
      }
      if (same_shape && non_empty.size() > 1) {
        auto stacked = torch::stack(non_empty, /*dim=*/0);
        SubmitCollective(c10d::OpType::ALLREDUCE, {stacked}, {stacked}, replica_groups, red_op);
        for (size_t i = 0; i < non_empty.size(); ++i) {
          non_empty[i].copy_(stacked[static_cast<int64_t>(i)]);
        }
        work->stash({stacked});
      } else {
        for (auto& t : non_empty) {
          SubmitCollective(c10d::OpType::ALLREDUCE, {t}, {t}, replica_groups, red_op);
        }
      }
    } else {
      SubmitCollective(c10d::OpType::ALLREDUCE, non_empty, non_empty, replica_groups, red_op);
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& output_tensors_list,
    std::vector<at::Tensor>& input_tensors, const c10d::AllgatherOptions& opts) {
  auto replica_groups = GetFullReplicaGroups();
  int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
  bool async_op = opts.asyncOp;

  // Validate output shapes match input shapes
  TORCH_CHECK(output_tensors_list.size() == input_tensors.size(), "output_tensors_list size (",
              output_tensors_list.size(), ") must match input_tensors size (", input_tensors.size(),
              ")");
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    for (size_t j = 0; j < output_tensors_list[i].size(); ++j) {
      TORCH_CHECK(output_tensors_list[i][j].sizes() == input_tensors[i].sizes(),
                  "output tensor shape ", output_tensors_list[i][j].sizes(), " at [", i, "][", j,
                  "] does not match input tensor shape ", input_tensors[i].sizes());
    }
  }

  std::vector<at::Tensor> flattened_outputs;
  for (auto& output_list : output_tensors_list) {
    for (auto& t : output_list) flattened_outputs.push_back(t);
  }

  // Bucketing: split into per-tensor allgather calls when over threshold
  int64_t total_output_bytes = 0;
  for (const auto& t : input_tensors) {
    total_output_bytes += t.numel() * t.element_size() * world_size;
  }

  auto work =
      CreateWork(flattened_outputs, input_tensors, c10d::OpType::ALLGATHER_COALESCED, async_op);
  try {
    if (UseHostCC()) {
      // Host-CC path: NRT's native allgather expects a single flattened output
      // per call. Issue coalesced batches as a sequence of per-tensor submissions,
      // each writing into a contiguous scratch that's sliced back into the
      // user's per-rank outputs. Scratches are stashed on the Work so they
      // outlive the async collective.
      std::vector<at::Tensor> scratch_tensors;
      scratch_tensors.reserve(input_tensors.size());
      for (size_t i = 0; i < input_tensors.size(); ++i) {
        auto& out_list = output_tensors_list[i];
        std::vector<int64_t> stacked_shape = {static_cast<int64_t>(out_list.size())};
        for (auto d : out_list[0].sizes()) stacked_shape.push_back(d);
        auto stacked = torch::empty(stacked_shape, out_list[0].options());
        SubmitCollective(c10d::OpType::ALLGATHER, {input_tensors[i]}, {stacked}, replica_groups, 0,
                         /*slice_output=*/false);
        for (size_t j = 0; j < out_list.size(); ++j) {
          out_list[j].copy_(stacked[static_cast<int64_t>(j)]);
        }
        scratch_tensors.push_back(std::move(stacked));
      }
      if (!scratch_tensors.empty()) work->stash(scratch_tensors);
    } else if (UseSplitAllgather() && total_output_bytes > CollectiveBucketSizeBytes()) {
      for (size_t i = 0; i < input_tensors.size(); ++i) {
        SubmitCollective(c10d::OpType::ALLGATHER, {input_tensors[i]}, output_tensors_list[i],
                         replica_groups, 0, /*slice_output=*/true);
      }
    } else {
      SubmitCollective(c10d::OpType::ALLGATHER, input_tensors, flattened_outputs, replica_groups, 0,
                       /*slice_output=*/true);
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::AllgatherOptions& opts) {
  ValidateNeuronDevice(inputs, "allgather_into_tensor_coalesced");
  ValidateNeuronDevice(outputs, "allgather_into_tensor_coalesced");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  std::vector<at::Tensor> non_empty_inputs, non_empty_outputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].numel() > 0) {
      non_empty_inputs.push_back(inputs[i]);
      non_empty_outputs.push_back(outputs[i]);
    }
  }
  if (non_empty_inputs.empty()) {
    auto work = CreateWork(outputs, {}, c10d::OpType::COALESCED, async_op);
    FinishWork(work, async_op);
    return work;
  }

  // Bucketing check
  int64_t total_output_bytes = 0;
  for (const auto& o : non_empty_outputs) {
    total_output_bytes += o.numel() * o.element_size();
  }

  auto work = CreateWork(outputs, inputs, c10d::OpType::COALESCED, async_op);
  try {
    if (UseHostCC()) {
      // Host-CC path: NRT's native allgather handles one (input, output) pair
      // per call. Issue coalesced batches as a sequence of per-pair submissions.
      // Outputs are already single flattened tensors (one per input), so no
      // scratch allocation is needed.
      for (size_t i = 0; i < non_empty_inputs.size(); ++i) {
        SubmitCollective(c10d::OpType::ALLGATHER, {non_empty_inputs[i]}, {non_empty_outputs[i]},
                         replica_groups, 0);
      }
    } else if (UseSplitAllgather() && total_output_bytes > CollectiveBucketSizeBytes()) {
      for (size_t i = 0; i < non_empty_inputs.size(); ++i) {
        SubmitCollective(c10d::OpType::ALLGATHER, {non_empty_inputs[i]}, {non_empty_outputs[i]},
                         replica_groups, 0);
      }
    } else {
      SubmitCollective(c10d::OpType::ALLGATHER, non_empty_inputs, non_empty_outputs, replica_groups,
                       0);
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::ReduceScatterOptions& opts) {
  ValidateNeuronDevice(inputs, "reduce_scatter_tensor_coalesced");
  ValidateNeuronDevice(outputs, "reduce_scatter_tensor_coalesced");
  auto replica_groups = GetFullReplicaGroups();
  int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
  bool async_op = opts.asyncOp;

  // Resolve reduce op (scales `inputs` in-place for PREMUL_SUM, then maps to SUM).
  int red_op = ResolveReduceOp(opts.reduceOp, inputs);

  // Filter empty pairs
  std::vector<at::Tensor> ne_in, ne_out;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].numel() > 0) {
      ne_in.push_back(inputs[i]);
      ne_out.push_back(outputs[i]);
    }
  }
  if (ne_in.empty()) {
    auto work = CreateWork(outputs, {}, c10d::OpType::COALESCED, async_op);
    FinishWork(work, async_op);
    return work;
  }
  if (ne_in.size() == 1) {
    return _reduce_scatter_base(ne_out[0], ne_in[0], opts);
  }

  for (size_t i = 1; i < ne_in.size(); ++i) {
    TORCH_CHECK(ne_in[i].sizes().slice(1) == ne_in[0].sizes().slice(1),
                "reduce_scatter_tensor_coalesced: all tensors must have same shape except dim 0");
    TORCH_CHECK(ne_in[i].dtype() == ne_in[0].dtype(),
                "reduce_scatter_tensor_coalesced: all tensors must have same dtype");
  }

  // Bucketing check
  int64_t input_bytes = 0;
  for (const auto& t : ne_in) {
    input_bytes += t.numel() * t.element_size();
  }

  if (UseSplitReduceScatter() && input_bytes > CollectiveBucketSizeBytes()) {
    // Split path: route each tensor pair through ReduceScatterSubmit for sub-bucketing
    auto work = CreateWork(outputs, inputs, c10d::OpType::COALESCED, async_op);
    try {
      std::vector<at::Tensor> all_intermediates;
      for (size_t i = 0; i < ne_in.size(); ++i) {
        auto intermediates = ReduceScatterSubmit(ne_out[i], ne_in[i], replica_groups, red_op);
        all_intermediates.insert(all_intermediates.end(), intermediates.begin(),
                                 intermediates.end());
      }
      if (!all_intermediates.empty()) work->stash(all_intermediates);
    } catch (...) {
      work->setException(std::current_exception());
      FinishWork(work, async_op);
      throw;
    }
    FinishWork(work, async_op);
    return work;
  }

  // Non-bucketed path: chunk interleaving for multiple tensors
  std::vector<int64_t> chunk_sizes;
  int64_t total_chunk = 0;
  for (const auto& t : ne_in) {
    TORCH_CHECK(t.size(0) % world_size == 0, "reduce_scatter_tensor_coalesced: input dim0 (",
                t.size(0), ") must be divisible by world_size (", world_size, ")");
    int64_t cs = t.size(0) / world_size;
    chunk_sizes.push_back(cs);
    total_chunk += cs;
  }

  auto work = CreateWork(outputs, inputs, c10d::OpType::COALESCED, async_op);
  try {
    // Build interleaved tensor
    int64_t total_size = total_chunk * world_size;
    std::vector<int64_t> cat_shape = {total_size};
    for (int64_t i = 1; i < ne_in[0].dim(); ++i) cat_shape.push_back(ne_in[0].size(i));
    auto concatenated = torch::empty(cat_shape, ne_in[0].options());

    int64_t offset = 0;
    for (int64_t r = 0; r < world_size; ++r) {
      for (size_t t = 0; t < ne_in.size(); ++t) {
        int64_t cs = chunk_sizes[t];
        concatenated.narrow(0, offset, cs).copy_(ne_in[t].narrow(0, r * cs, cs));
        offset += cs;
      }
    }

    // Single reduce_scatter
    std::vector<int64_t> out_shape = {total_chunk};
    for (int64_t i = 1; i < ne_in[0].dim(); ++i) out_shape.push_back(ne_in[0].size(i));
    auto scattered = torch::empty(out_shape, ne_in[0].options());

    SubmitCollective(c10d::OpType::REDUCE_SCATTER, {concatenated}, {scattered}, replica_groups,
                     red_op);

    // Copy chunks to outputs
    offset = 0;
    for (size_t i = 0; i < ne_out.size(); ++i) {
      ne_out[i].copy_(scattered.narrow(0, offset, chunk_sizes[i]));
      offset += chunk_sizes[i];
    }
    work->stash({concatenated, scattered});
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::gather(
    std::vector<std::vector<at::Tensor>>& output_tensors, std::vector<at::Tensor>& input_tensors,
    const c10d::GatherOptions& opts) {
  ValidateNeuronDevice(input_tensors, "gather");
  ValidateNonZeroDim0(input_tensors, "gather");
  auto replica_groups = GetFullReplicaGroups();
  bool async_op = opts.asyncOp;

  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_, "Invalid rootRank ", opts.rootRank,
              " for world size ", size_);

  // Non-root ranks need dummy output tensors
  std::vector<std::vector<at::Tensor>> actual_outputs;
  if (rank_ != opts.rootRank) {
    std::vector<at::Tensor> dummy;
    for (int i = 0; i < size_; ++i) dummy.push_back(torch::empty_like(input_tensors[0]));
    actual_outputs.push_back(std::move(dummy));
  } else {
    actual_outputs = output_tensors;
  }

  std::vector<at::Tensor> all_outputs;
  for (auto& sub : actual_outputs) {
    for (auto& t : sub) all_outputs.push_back(t);
  }

  auto work = CreateWork(all_outputs, input_tensors, c10d::OpType::GATHER, async_op);
  try {
    if (UseHostCC()) {
      // Host-CC path: NRT's native allgather expects a single flattened output
      // rather than a list of per-rank tensors. Gather into a contiguous scratch
      // and copy slices back into the user's output list.
      auto& out_list = actual_outputs[0];
      std::vector<int64_t> stacked_shape = {static_cast<int64_t>(out_list.size())};
      for (auto d : out_list[0].sizes()) stacked_shape.push_back(d);
      auto stacked = torch::empty(stacked_shape, out_list[0].options());
      // Keep `stacked` alive for the duration of the async collective so the
      // caching allocator doesn't reclaim it before NRT writes.
      work->stash({stacked});
      SubmitCollective(c10d::OpType::ALLGATHER, input_tensors, {stacked}, replica_groups, 0,
                       /*slice_output=*/false);
      for (size_t i = 0; i < out_list.size(); ++i) {
        out_list[i].copy_(stacked[static_cast<int64_t>(i)]);
      }
    } else {
      SubmitCollective(c10d::OpType::ALLGATHER, input_tensors, actual_outputs[0], replica_groups, 0,
                       /*slice_output=*/true);
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::scatter(
    std::vector<at::Tensor>& output_tensors, std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ScatterOptions& opts) {
  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_, "Invalid rootRank ", opts.rootRank,
              " for world size ", size_);

  std::vector<std::vector<at::Tensor>> actual_inputs;
  if (rank_ == opts.rootRank) {
    TORCH_CHECK(!input_tensors.empty() && static_cast<int>(input_tensors[0].size()) == size_,
                "Incorrect number of inputs for scatter");
    actual_inputs = input_tensors;
  } else {
    std::vector<at::Tensor> zeros;
    for (int i = 0; i < size_; ++i) zeros.push_back(torch::zeros_like(output_tensors[0]));
    actual_inputs.push_back(std::move(zeros));
  }

  c10d::ReduceScatterOptions rs_opts;
  rs_opts.reduceOp = c10d::ReduceOp::SUM;
  return reduce_scatter(output_tensors, actual_inputs, rs_opts);
}

}  // namespace distributed
}  // namespace torch_neuronx
