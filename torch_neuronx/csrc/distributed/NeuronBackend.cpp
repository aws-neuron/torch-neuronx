#include "torch_neuronx/csrc/distributed/NeuronBackend.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <unordered_map>
#include <unordered_set>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/NeuronBarrier.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpsApi.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/OperationExecutionEngine.h"  // kHostWorkerThreads
#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"
#include "torch_neuronx/csrc/core/rdma/RdmaManager.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/streams/StreamConfig.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/distributed/DistributedUtils.h"
#include "torch_neuronx/csrc/distributed/DynamicCollectiveTiling.h"

namespace py = pybind11;

namespace torch_neuronx {

at::neuron::CollectiveDirectKernelExecution::CollectiveType ConvertOpTypeToCollectiveType(
    c10d::OpType op_type);

namespace distributed {

// Static shared callback — same function for all NeuronBackend instances
py::object NeuronBackend::lower_callback_;

NeuronBackend::NeuronBackend(const c10::intrusive_ptr<c10d::Store>& store,
                             const c10::intrusive_ptr<c10d::Backend::Options>& options, int rank,
                             int size, int global_world_size, py::object lower_callback)
    : c10d::Backend(rank, size), timeout_ms_(options->timeout.count()), store_(store) {
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
  // Cache global rank for RDMA P2P key indexing (rdma_ep_<global>,
  // rdma_<src>_<dst>_<tag>_<seq>). See send/recv overrides.
  global_rank_ = group_ranks_[rank];
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
    // BARRIER and COALESCED legitimately have no tensors (BARRIER never;
    // COALESCED when every op in the batch was an intra-rank pair resolved via
    // tensor.copy_). Fall back to the calling thread's current device.
    TORCH_CHECK(op_type == c10d::OpType::BARRIER || op_type == c10d::OpType::COALESCED,
                "CreateWork called with empty inputs and outputs for non-barrier/non-coalesced op");
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
  auto current = at::neuron::getCurrentNeuronStream();
  auto default_stream = at::neuron::getDefaultNeuronStream();
  return current != default_stream && at::neuron::config::IsHostCCEnabled();
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

// Default policy (in-NEFF flow only).
bool UseSplitDefaultForCurrentStream() {
  if (!at::neuron::config::IsHostCCEnabled()) return true;
  auto current = at::neuron::getCurrentNeuronStream();
  auto default_stream = at::neuron::getDefaultNeuronStream();
  return current == default_stream;
}

// Sentinel for "user did not set the env var explicitly".
constexpr int kSplitEnvUnset = -1;

int ReadSplitEnvOnce(const char* env_name) {
  const char* env = std::getenv(env_name);
  if (env == nullptr) return kSplitEnvUnset;
  return std::string(env) == "1" ? 1 : 0;
}

bool UseSplitAllgather() {
  static const int env_override = ReadSplitEnvOnce("NEURON_USE_SPLIT_ALLGATHER");
  if (env_override != kSplitEnvUnset) return env_override == 1;
  return UseSplitDefaultForCurrentStream();
}

bool UseSplitReduceScatter() {
  static const int env_override = ReadSplitEnvOnce("NEURON_USE_SPLIT_REDUCE_SCATTER");
  if (env_override != kSplitEnvUnset) return env_override == 1;
  return UseSplitDefaultForCurrentStream();
}

// Dynamic multi-tier AllGather gate
bool UseDynamicNkiCollectives() {
  static const int env = ReadSplitEnvOnce("NEURON_DYN_NKI_COLLECTIVES");
  return env == 1;
}

// Cache of runtime scalar tensors ([1,1] int32) keyed by (value, device index).
// This skips the host fill + 4-byte H2D on repeated/recurring shapes
at::Tensor GetOrCreateRuntimeScalar(int64_t value, const at::Device& device) {
  static std::mutex mtx;
  static std::unordered_map<std::string, at::Tensor> cache;
  static constexpr size_t kMaxEntries = 256;
  const std::string key =
      std::to_string(value) + "@" + std::to_string(static_cast<int>(device.index()));
  std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  // Build on the HOST (CPU fill, no dispatch) and transfer to the device
  at::Tensor scalar =
      torch::full({1, 1}, value, torch::TensorOptions().dtype(torch::kInt32)).to(device);
  if (cache.size() >= kMaxEntries) cache.clear();  // in-flight users keep their own ref via stash
  cache.emplace(key, scalar);
  return scalar;
}

// Route uneven all-to-all through native NKI all_to_all_v by default. Setting
// NEURON_USE_NKI_ALLTOALLV=0 selects the compatibility padding fallback.
// Read per-call (not cached) — tests toggle the flag via a context manager wrapping
// individual all_to_all_single calls at dispatch time.
bool UseNkiAllToAllV() {
  const char* env = std::getenv("NEURON_USE_NKI_ALLTOALLV");
  return env == nullptr || std::string(env) == "1";
}

}  // namespace

namespace torch_neuronx {
namespace distributed {

void NeuronBackend::SubmitInNeffCollective(c10d::OpType op_type,
                                           const std::vector<at::Tensor>& inputs,
                                           const std::vector<at::Tensor>& outputs,
                                           const std::vector<std::vector<int>>& replica_groups,
                                           int reduce_op, bool slice_output,
                                           const std::string& cache_key_override) {
  const std::vector<at::Tensor> original_outputs = outputs;
  std::vector<at::Tensor> working_inputs = inputs;
  std::vector<at::Tensor> working_outputs = outputs;
  const bool inputs_downcasted = dispatch::DowncastTensorList(working_inputs);
  const bool outputs_downcasted = dispatch::DowncastTensorList(working_outputs);
  (void)inputs_downcasted;  // Input downcast is permanent for this call; no copy-back.

  const auto& op_str = OpTypeToString(op_type);
  // A non-empty override (dynamic allgather) supplies a BUCKETED key so
  // every runtime numel in the same bucket (trace_numel x unroll) reuses one dispatch-cache
  // entry; otherwise the key is shape-specific (default collective behavior).
  auto cache_key = cache_key_override.empty()
                       ? at::neuron::CacheKeyGenerator::GenerateCollective(
                             op_str, working_inputs, replica_groups, reduce_op, slice_output)
                       : cache_key_override;
  auto entry = dispatch::NeuronDispatcher::Instance().GetOrLowerCollectiveOperation(
      cache_key, lower_callback_, op_type, working_inputs, working_outputs, replica_groups,
      reduce_op, slice_output);

  const bool fuse_input_transforms = torch_neuronx::IsPrologueEnabled();
  auto transform_result =
      torch_neuronx::ProcessInputTransformations(op_str, working_inputs, fuse_input_transforms);

  auto output_transform_result = torch_neuronx::ProcessOutputTransformations(
      op_str, working_outputs, /*fuse_transforms=*/true);

  // Unlike the eager ATen path, collectives have no AllocateExecutionOutputs
  // staging buffer — the kernel writes directly into the caller's tensors. When
  // epilogue is disabled (kill switch), ProcessOutputTransformations passes a
  // non-contiguous output straight through, which PrepareKernelTensors rejects.
  // Stage a contiguous temp for any non-epilogue-handled non-contiguous output;
  // the copy-back loop after submit scatters it into the original tensor.
  for (size_t i = 0; i < output_transform_result.final_outputs.size(); ++i) {
    const bool epilogue_handled = (i < output_transform_result.kernel_transforms.size() &&
                                   !output_transform_result.kernel_transforms[i].empty());
    auto& final_output = output_transform_result.final_outputs[i];
    if (!epilogue_handled && final_output.defined() && !final_output.is_contiguous()) {
      final_output =
          torch::empty(final_output.sizes().vec(),
                       final_output.options().memory_format(c10::MemoryFormat::Contiguous));
    }
  }

  int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs,
                                                output_transform_result.final_outputs);
  auto [input_refs, input_contexts] =
      torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
  auto [output_refs, output_contexts] =
      torch_neuronx::PrepareKernelTensors(output_transform_result.final_outputs);

  auto xla_kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
      op_str, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
      entry->neff_cache_key, entry->stablehlo_bytes, /*has_collectives=*/true, device_id);

  if (transform_result.has_input_transforms) {
    xla_kernel->SetInputTransformations(std::move(transform_result.kernel_transforms));
  }
  if (output_transform_result.has_output_transforms) {
    xla_kernel->SetOutputTransformations(std::move(output_transform_result.kernel_transforms));
  }

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(xla_kernel), "");
  auto stream = at::neuron::getCurrentNeuronStream();
  at::neuron::SubmitOperationContext(stream, std::move(context));

  // Epilogue-unsupported / kill-switch fallback: scatter the staged contiguous
  // output back into the caller's original (non-contiguous) tensor. Skipped when
  // epilogue fused the inverse transform (kernel writes the base storage the
  // caller's view aliases) or when the output was already contiguous (data_ptr
  // identity short-circuits inside CopyBackToOriginal anyway).
  for (size_t i = 0; i < working_outputs.size() && i < output_transform_result.final_outputs.size();
       ++i) {
    const bool epilogue_handled = (i < output_transform_result.kernel_transforms.size() &&
                                   !output_transform_result.kernel_transforms[i].empty());
    if (epilogue_handled) continue;
    if (!working_outputs[i].defined() || !output_transform_result.final_outputs[i].defined())
      continue;
    if (output_transform_result.final_outputs[i].data_ptr() == working_outputs[i].data_ptr())
      continue;
    dispatch::CopyBackToOriginal(
        working_outputs[i], output_transform_result.final_outputs[i],
        std::string("SubmitInNeffCollective epilogue fallback idx=") + std::to_string(i));
  }

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
      torch_neuronx::ProcessInputTransformations(op_str, working_inputs, /*fuse_transforms=*/false);
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

// ─────────────────────────────────────────────────────────────────────────
// Point-to-point (RDMA over EFA)
// ─────────────────────────────────────────────────────────────────────────
namespace {

// Sub-PGs (TP/DP/PP) wrap the same world TCPStore in distinct PrefixStores;
// RdmaManager is process-global, so all PGs must agree on rdma_ep_<global>
// keys. Walk PrefixStore -> root once.
c10::intrusive_ptr<c10d::Store> UnwrapToRootStore(c10::intrusive_ptr<c10d::Store> s) {
  if (auto* prefix = dynamic_cast<c10d::PrefixStore*>(s.get())) {
    return prefix->getUnderlyingNonPrefixStore();
  }
  return s;
}

// Submit an RDMA P2P batch as a kHost op. efa-direct has no usable CQ wait fd,
// so completions are busy-polled; running the whole exchange (PrepareBatch plus
// the DrainAndPending poll loop) on the HostWorkerPool keeps both the network
// round-trip and the spin off the engine RuntimeWorker.
//
// A single host worker (static_assert below) is load-bearing: RdmaManager's
// mutable state (av_cache_, seq_counters_, registered_memory_, send_contexts_,
// completed_recv_ids_, libfabric handles) is then touched by one thread and
// needs no lock. findTensor (in PrepareBatch) is safe off the engine because the
// producing AllocBind is ordered before this op's enqueue (per-stream FIFO, or
// the caller's event/wait cross-stream) and Submit→WorkerLoop is a
// release/acquire edge, so the bind happens-before the read; NeuronWork holds
// the tensor's ref until the op retires, so it stays alive across the transfer.
static_assert(at::neuron::kHostWorkerThreads == 1,
              "RDMA P2P relies on a single-threaded host pool to serialize RdmaManager's "
              "unguarded state; add locking before allowing >1 host worker.");
void SubmitP2POp(std::vector<::neuron::P2PEntry> entries, int device_idx, int64_t timeout_ms) {
  auto entries_owned = std::make_shared<std::vector<::neuron::P2PEntry>>(std::move(entries));
  const auto timeout = std::chrono::milliseconds(timeout_ms);

  // Host closure: PrepareBatch once (resolve VAs, exchange descriptors via the
  // Store, post fi_writedata), then poll DrainAndPending until every entry
  // completes. efa-direct gives the receiver no fabric event when a sender dies,
  // so a per-batch deadline (armed on the first pending poll; timeout == 0
  // disables it) bounds the poll. On expiry it throws: the WorkerLoop try/catch
  // still fires AdvanceHostCompletion so a gated device op unblocks, and the
  // error surfaces at the next Synchronize. In-flight SendContexts are left in
  // the pool on throw — libfabric may still deliver their completions (carrying
  // SendContext* as op_context), so freeing here would be a use-after-free; they
  // are reclaimed in ~RdmaManager.
  at::neuron::HostKernelExecution::HostFn drain_fn =
      [entries_owned, timeout](const std::vector<at::neuron::TensorDataRef>&,
                               const std::vector<at::neuron::TensorDataRef>&) {
        auto& mgr = ::neuron::RdmaManager::Instance();
        // Pass the batch timeout: the mailbox descriptor exchange inside
        // PrepareBatch polls for peer writes and must be bounded too (a dead peer
        // never writes its descriptor), not just the data-drain loop below.
        mgr.PrepareBatch(*entries_owned,
                         timeout.count());  // idempotent; early-returns once prepared
        std::optional<std::chrono::steady_clock::time_point> deadline;
        while (true) {
          if (!mgr.DrainAndPending(*entries_owned)) return;  // all entries done
          if (timeout.count() > 0) {
            if (!deadline) deadline = std::chrono::steady_clock::now() + timeout;
            if (std::chrono::steady_clock::now() > *deadline) {
              // Name the unfinished transfers so a cross-rank hang is
              // diagnosable without re-running with traces.
              std::string stuck;
              for (const auto& e : *entries_owned) {
                bool unfinished = (e.kind == ::neuron::P2PEntry::Kind::kRecv)
                                      ? !e.recv_consumed
                                      : (e.send_ctx != nullptr);
                if (!unfinished) continue;
                if (!stuck.empty()) stuck += ", ";
                stuck += (e.kind == ::neuron::P2PEntry::Kind::kRecv ? "recv<-" : "send->") +
                         std::to_string(e.peer) + " tag=" + std::to_string(e.tag);
              }
              throw std::runtime_error("RDMA P2P timeout after " + std::to_string(timeout.count()) +
                                       "ms; unfinished transfers [" + stuck +
                                       "] (peer may have failed)");
            }
          }
          // No CQ wait fd on efa-direct; yield to avoid a 100% hot spin.
          std::this_thread::yield();
        }
      };

  auto kernel = std::make_unique<at::neuron::HostKernelExecution>(
      "p2p::batch", std::vector<at::neuron::TensorDataRef>{},
      std::vector<at::neuron::TensorDataRef>{}, std::move(drain_fn), device_idx);

  auto context = std::make_unique<at::neuron::OperationContext>(std::move(kernel), "");
  auto stream = at::neuron::getCurrentNeuronStream(device_idx);
  at::neuron::SubmitOperationContext(stream, std::move(context));
}

// Address a tensor for RDMA as storage base VA + byte offset.
::neuron::P2PEntry MakeP2PEntry(::neuron::P2PEntry::Kind kind, const at::Tensor& tensor, int peer,
                                int tag) {
  return {kind,
          const_cast<void*>(tensor.storage().data()),
          static_cast<size_t>(tensor.numel() * tensor.element_size()),
          peer,
          tag,
          static_cast<size_t>(tensor.storage_offset() * tensor.element_size())};
}

}  // namespace

void NeuronBackend::EnsureRdmaInit(at::Tensor& /*tensor*/, int /*peer_global*/) {
  // Caller holds p2p_mutex_. Just hand RdmaManager the root store + this PG's
  // global rank once; libfabric init, local-endpoint publish, and peer AV
  // inserts all run on the host worker thread inside PrepareBatch where we have
  // a real HBM VA (resolved via findTensor + nrt_tensor_get_va) needed to query
  // NRT for the attached EFA BDF.
  if (rdma_setup_done_) return;
  auto& mgr = ::neuron::RdmaManager::Instance();
  auto root = UnwrapToRootStore(store_);
  TORCH_NEURONX_RDMA_INFO("EnsureRdmaInit: pg_uid=", pg_uid_, " global_rank=", global_rank_,
                          " group_size=", group_ranks_.size(), " root_store=", root.get(), " (was ",
                          store_.get(), ")");
  mgr.SetStore(root);
  mgr.SetRank(global_rank_);
  rdma_setup_done_ = true;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::send(std::vector<at::Tensor>& tensors, int dstRank,
                                                   int tag) {
  TORCH_CHECK(tensors.size() == 1, "NeuronBackend::send takes exactly one tensor");
  auto& tensor = tensors[0];
  TORCH_CHECK(tensor.is_contiguous(), "NeuronBackend::send requires a contiguous tensor");
  ValidateNeuronDevice(tensors, "send");

  const int dst_global = ToGlobal(dstRank);
  TORCH_NEURONX_RDMA_INFO(
      "send: pg_uid=", pg_uid_, " self_global=", global_rank_, " dst_group=", dstRank,
      " dst_global=", dst_global, " tag=", tag, " tensor.device=", tensor.device(),
      " tensor.data_ptr=", tensor.data_ptr(), " bytes=", tensor.numel() * tensor.element_size(),
      " coalescing=", coalescing_);

  // Single critical section: coalescing-flag check + lazy AV setup. Closes a
  // TOCTOU window with concurrent startCoalescing. Same-rank P2P is allowed
  // inside coalescing (endCoalescing pairs SEND/RECV via at::Tensor::copy_)
  // but rejected on the direct path — eager dist.send to self has no peer.
  {
    std::lock_guard<std::mutex> lock(p2p_mutex_);
    if (coalescing_) {
      coalesced_ops_.push_back({c10d::OpType::SEND, tensor, dst_global, tag});
      return nullptr;
    }
    TORCH_CHECK(dst_global != global_rank_,
                "NeuronBackend::send to self is not supported outside batch_isend_irecv. "
                "Use batch_isend_irecv (which pairs SEND/RECV via at::Tensor::copy_) "
                "or call tensor.copy_() directly.");
    EnsureRdmaInit(tensor, dst_global);
  }

  std::vector<::neuron::P2PEntry> entries{
      MakeP2PEntry(::neuron::P2PEntry::Kind::kSend, tensor, dst_global, tag)};

  auto work = CreateWork(/*outputs=*/{tensor}, /*inputs=*/{tensor}, c10d::OpType::SEND,
                         /*async_op=*/true);
  SubmitP2POp(std::move(entries), tensor.device().index(), timeout_ms_);
  FinishWork(work, /*async_op=*/true);
  return work;
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::recv(std::vector<at::Tensor>& tensors, int srcRank,
                                                   int tag) {
  TORCH_CHECK(tensors.size() == 1, "NeuronBackend::recv takes exactly one tensor");
  auto& tensor = tensors[0];
  TORCH_CHECK(tensor.is_contiguous(), "NeuronBackend::recv requires a contiguous tensor");
  ValidateNeuronDevice(tensors, "recv");

  const int src_global = ToGlobal(srcRank);
  TORCH_NEURONX_RDMA_INFO(
      "recv: pg_uid=", pg_uid_, " self_global=", global_rank_, " src_group=", srcRank,
      " src_global=", src_global, " tag=", tag, " tensor.device=", tensor.device(),
      " tensor.data_ptr=", tensor.data_ptr(), " bytes=", tensor.numel() * tensor.element_size(),
      " coalescing=", coalescing_);

  // Same-rank P2P is allowed inside coalescing (endCoalescing pairs SEND/RECV
  // via at::Tensor::copy_) but rejected on the direct path.
  {
    std::lock_guard<std::mutex> lock(p2p_mutex_);
    if (coalescing_) {
      coalesced_ops_.push_back({c10d::OpType::RECV, tensor, src_global, tag});
      return nullptr;
    }
    TORCH_CHECK(src_global != global_rank_,
                "NeuronBackend::recv from self is not supported outside batch_isend_irecv. "
                "Use batch_isend_irecv (which pairs SEND/RECV via at::Tensor::copy_) "
                "or call tensor.copy_() directly.");
    EnsureRdmaInit(tensor, src_global);
  }

  std::vector<::neuron::P2PEntry> entries{
      MakeP2PEntry(::neuron::P2PEntry::Kind::kRecv, tensor, src_global, tag)};

  auto work = CreateWork(/*outputs=*/{tensor}, /*inputs=*/{tensor}, c10d::OpType::RECV,
                         /*async_op=*/true);
  SubmitP2POp(std::move(entries), tensor.device().index(), timeout_ms_);
  FinishWork(work, /*async_op=*/true);
  return work;
}

void NeuronBackend::startCoalescing() {
  std::lock_guard<std::mutex> lock(p2p_mutex_);
  TORCH_NEURONX_RDMA_INFO("startCoalescing: pg_uid=", pg_uid_, " self_global=", global_rank_);
  coalescing_ = true;
  coalesced_ops_.clear();
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::endCoalescing() {
  std::vector<CoalescedOp> ops;
  {
    std::lock_guard<std::mutex> lock(p2p_mutex_);
    coalescing_ = false;
    ops = std::move(coalesced_ops_);
    coalesced_ops_.clear();
  }
  TORCH_NEURONX_RDMA_INFO("endCoalescing: pg_uid=", pg_uid_, " self_global=", global_rank_,
                          " ops=", ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    TORCH_NEURONX_RDMA_DEBUG(
        "endCoalescing:   op[", i, "] kind=", (op.op == c10d::OpType::SEND ? "SEND" : "RECV"),
        " peer_global=", op.peer_global, " tag=", op.tag, " tensor.device=", op.tensor.device(),
        " tensor.data_ptr=", op.tensor.data_ptr());
  }

  // Same-rank (peer == self) pairs in V-shaped schedules (e.g. ZBV) become
  // local copies; never reach RdmaManager. Match by tag since peer collapses.
  std::unordered_map<int, at::Tensor> local_sends;
  std::vector<::neuron::P2PEntry> remote_entries;
  remote_entries.reserve(ops.size());
  std::vector<at::Tensor> remote_tensors;

  for (const auto& op : ops) {
    if (op.op == c10d::OpType::SEND && op.peer_global == global_rank_) {
      // Two same-rank sends sharing a tag would silently drop one (emplace is a
      // no-op on collision). That is a caller bug — fail loudly rather than
      // lose data.
      auto [unused, inserted] = local_sends.emplace(op.tag, op.tensor);
      (void)unused;
      TORCH_CHECK(inserted, "NeuronBackend::endCoalescing: duplicate same-rank send tag ", op.tag,
                  " in one batch; tags must be unique per coalesced group");
    }
  }
  for (const auto& op : ops) {
    if (op.op == c10d::OpType::RECV && op.peer_global == global_rank_) {
      auto it = local_sends.find(op.tag);
      // A same-rank recv with no matching same-rank send at this tag would
      // leave the recv tensor with stale data — surface it as a caller bug.
      TORCH_CHECK(it != local_sends.end(), "NeuronBackend::endCoalescing: same-rank recv tag ",
                  op.tag, " has no matching same-rank send in this batch");
      auto recv_tensor = op.tensor;
      // NoGradGuard required, since a grad-requiring leaf recv tensor cannot be used in an in-place
      // operation. Raw P2P data movement, not a differentiable op: pipeline autograd flows through
      // the explicit send/recv backward.
      {
        torch::NoGradGuard no_grad;
        recv_tensor.copy_(it->second);
      }
      local_sends.erase(it);
      continue;
    }
    if (op.peer_global == global_rank_) continue;  // SEND with same-peer (paired above)

    remote_entries.push_back(MakeP2PEntry(op.op == c10d::OpType::SEND
                                              ? ::neuron::P2PEntry::Kind::kSend
                                              : ::neuron::P2PEntry::Kind::kRecv,
                                          op.tensor, op.peer_global, op.tag));
    remote_tensors.push_back(op.tensor);
  }

  // Every same-rank send must have been consumed by a matching recv above; a
  // leftover means a send-to-self with no receiver in this group (caller bug).
  TORCH_CHECK(local_sends.empty(), "NeuronBackend::endCoalescing: ", local_sends.size(),
              " same-rank send(s) had no matching same-rank recv in this batch");

  TORCH_NEURONX_RDMA_INFO("endCoalescing: partitioned ops=", ops.size(),
                          " -> remote_entries=", remote_entries.size(),
                          " local_pairs_resolved=", (ops.size() - remote_entries.size()) / 2);

  if (remote_entries.empty()) {
    auto work = CreateWork({}, {}, c10d::OpType::COALESCED, /*async_op=*/true);
    FinishWork(work, /*async_op=*/true);
    return work;
  }

  {
    std::lock_guard<std::mutex> lock(p2p_mutex_);
    for (size_t i = 0; i < remote_entries.size(); ++i) {
      EnsureRdmaInit(remote_tensors[i], remote_entries[i].peer);
    }
  }

  const int device_idx = remote_tensors.front().device().index();
  TORCH_NEURONX_RDMA_INFO("endCoalescing: submitting batched kP2P op (", remote_entries.size(),
                          " entries, device_idx=", device_idx, ")");
  auto work = CreateWork(remote_tensors, remote_tensors, c10d::OpType::COALESCED,
                         /*async_op=*/true);
  SubmitP2POp(std::move(remote_entries), device_idx, timeout_ms_);
  FinishWork(work, /*async_op=*/true);
  return work;
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
    int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
    // Dynamic NKI all-reduce: single-tensor only.
    const bool dynamic_handled =
        UseDynamicNkiCollectives() && tensors.size() == 1 &&
        TryDynamicAllreduce(tensors[0], replica_groups, world_size, red_op, work);
    if (!dynamic_handled) {
      SubmitCollective(c10d::OpType::ALLREDUCE, allreduce_inputs, tensors, replica_groups, red_op);
    }
  } catch (...) {
    work->setException(std::current_exception());
    FinishWork(work, async_op);
    throw;
  }
  FinishWork(work, async_op);
  return work;
}

// Attempt dynamic multi-tier NKI all-gather
bool NeuronBackend::TryDynamicAllgather(at::Tensor& input_tensor, at::Tensor& output_tensor,
                                        const std::vector<std::vector<int>>& replica_groups,
                                        int world_size, c10::intrusive_ptr<NeuronWork>& work) {
  if (!UseDynamicNkiCollectives()) return false;
  int64_t numel = input_tensor.numel();
  int64_t dsize = input_tensor.element_size();
  int lnc = std::atoi(at::neuron::utils::GetLogicalNeuronCores().c_str());  // = nprg (kernel grid)
  int64_t nprg = lnc >= 1 ? lnc : 1;
  auto plan = collective_tiling::ComputePlan(numel, world_size, nprg, dsize,
                                             at::neuron::utils::GetPlatformTarget());
  bool contig = input_tensor.is_contiguous() && output_tensor.is_contiguous();
  // Eligibility: (1) every rank-major element offset must fit a signed int32 which
  // requires world_size * numel < 2^31; (2) dtype is bf16/fp16/fp32.
  auto st = input_tensor.scalar_type();
  bool native_dtype = (st == at::kBFloat16 || st == at::kHalf || st == at::kFloat);
  constexpr int64_t kInt32Extent = int64_t{1} << 31;
  bool offsets_fit = world_size > 0 && numel <= (kInt32Extent - 1) / world_size;
  if (!(plan.eligible && contig && native_dtype && offsets_fit)) return false;
  // Bind the shard flat
  at::Tensor input_bound = input_tensor.reshape({numel, 1});
  at::Tensor output_bound = output_tensor.reshape({world_size * numel, 1});
  auto scalar = GetOrCreateRuntimeScalar(numel, input_tensor.device());
  work->stash({scalar});  // keep alive until the collective completes
  // Bucketed key (NEFF identity, not runtime numel) so the whole bucket reuses one NEFF: keyed on
  // trace_numel (rung) + tiles (regime) -- both are needed for a unique NEFF (see the header).
  auto cache_key = at::neuron::CacheKeyGenerator::GenerateDynamicCollectiveKey(
      "allgather", plan.trace_numel, plan.large, plan.medium, plan.small, replica_groups,
      input_bound.scalar_type(), plan.unroll);
  SubmitInNeffCollective(c10d::OpType::ALLGATHER, {input_bound, scalar}, {output_bound},
                         replica_groups, /*reduce_op=*/0, /*slice_output=*/false,
                         /*cache_key_override=*/cache_key);
  return true;
}

// Attempt dynamic multi-tier NKI all-reduce
bool NeuronBackend::TryDynamicAllreduce(at::Tensor& io_tensor,
                                        const std::vector<std::vector<int>>& replica_groups,
                                        int world_size, int reduce_op,
                                        c10::intrusive_ptr<NeuronWork>& work) {
  if (!UseDynamicNkiCollectives()) return false;
  // The NKI all_reduce supports SUM/MIN/MAX (c10d RedOpType 0/3/4). PREMUL_SUM is already
  // resolved to SUM (inputs pre-scaled) upstream in ResolveReduceOp; AVG/PRODUCT/bitwise ops
  // are unsupported -> fall back to the native collective.
  if (!(reduce_op == 0 || reduce_op == 3 || reduce_op == 4)) return false;
  int64_t numel = io_tensor.numel();
  int64_t dsize = io_tensor.element_size();
  int lnc = std::atoi(at::neuron::utils::GetLogicalNeuronCores().c_str());  // = nprg (kernel grid)
  int64_t nprg = lnc >= 1 ? lnc : 1;
  auto plan = collective_tiling::ComputePlan(numel, world_size, nprg, dsize,
                                             at::neuron::utils::GetPlatformTarget(),
                                             collective_tiling::Collective::kAllReduce);
  bool contig = io_tensor.is_contiguous();
  auto st = io_tensor.scalar_type();
  bool native_dtype = (st == at::kBFloat16 || st == at::kHalf || st == at::kFloat);
  // Eligibility: (1) all-reduce output is the same size as its input, so every element offset is
  // in [0, numel) and must fit a signed int32 (numel < 2^31); (2) dtype is bf16/fp16/fp32.
  constexpr int64_t kInt32Extent = int64_t{1} << 31;
  bool offsets_fit = numel > 0 && numel < kInt32Extent;
  if (!(plan.eligible && contig && native_dtype && offsets_fit)) return false;
  // The NKI kernel is functional (reads the input, writes a fresh output, no alias metadata).
  // Binding the same tensor as both the input and the output port makes the reduced result land
  // in place. Its forward OOB-skipped tail never reads an element written by an earlier tier, so
  // no pristine-input snapshot is needed.
  at::Tensor output_bound = io_tensor.reshape({numel, 1});
  auto scalar = GetOrCreateRuntimeScalar(numel, io_tensor.device());
  work->stash({scalar});  // keep alive until the collective completes
  // Bucketed key: keyed on trace_numel (rung) + tiles (regime) + unroll + reduce_op, so the whole
  // bucket reuses one NEFF and add/min/max get distinct NEFFs.
  auto cache_key = at::neuron::CacheKeyGenerator::GenerateDynamicCollectiveKey(
      "allreduce", plan.trace_numel, plan.large, plan.medium, plan.small, replica_groups,
      output_bound.scalar_type(), plan.unroll, reduce_op);
  SubmitInNeffCollective(c10d::OpType::ALLREDUCE, {output_bound, scalar}, {output_bound},
                         replica_groups, reduce_op, /*slice_output=*/false,
                         /*cache_key_override=*/cache_key);
  return true;
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
    // Dynamic NKI all-gather.
    const bool dynamic_handled =
        UseDynamicNkiCollectives() &&
        TryDynamicAllgather(input_tensor, output_tensor, replica_groups, world_size, work);
    if (!dynamic_handled) {
      SubmitCollective(c10d::OpType::ALLGATHER, {input_tensor}, {output_tensor}, replica_groups, 0);
    }
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

std::vector<std::vector<int>> NeuronBackend::ValidateNkiAllToAllV() {
  auto current_ranks = GetReplicaGroups()[0];
  int64_t rg_size = static_cast<int64_t>(current_ranks.size());
  auto replica_groups = CreateGlobalReplicaGroups(global_world_size_, current_ranks);

  TORCH_CHECK(rg_size == 4 || rg_size == 8 || rg_size == 16 || rg_size % 32 == 0,
              "all_to_all_v: unsupported world size ", rg_size,
              ", supported sizes: 4, 8, 16, or multiples of 32");
  for (size_t i = 0; i < replica_groups.size(); ++i) {
    const auto& g = replica_groups[i];
    TORCH_CHECK(!g.empty() && g[0] % rg_size == 0,
                "all_to_all_v: replica group start ranks must be multiples of world size (",
                rg_size, "), found: group[", i, "] starts at rank ", g.empty() ? -1 : g[0]);
  }
  // TRN3 NeuronSwitch fabric requires LNC=2 (host-side guard; see api doc restrictions).
  const std::string lnc = at::neuron::utils::GetLogicalNeuronCores();
  const std::string instance_type = at::neuron::utils::GetInstanceType();
  const bool is_switch_fabric = (instance_type == "trn3" || instance_type == "trn3u");
  TORCH_CHECK(!is_switch_fabric || lnc == "2",
              "all_to_all_v requires LNC=2 on NeuronSwitch (Trn3) instances, got LNC=", lnc);
  return replica_groups;
}

void NeuronBackend::SubmitNkiAllToAllV(at::Tensor& output_tensor, at::Tensor& input_tensor,
                                       const std::vector<int64_t>& output_split_sizes,
                                       const std::vector<int64_t>& input_split_sizes) {
  auto replica_groups = ValidateNkiAllToAllV();
  // Reentrant-safe: this method is entered from Python with the GIL held, so nested
  // acquisition here does not deadlock. One driver serves both backends.
  py::gil_scoped_acquire gil;
  py::object fn = py::module_::import("torch_neuronx.distributed.ops.all_to_all_v")
                      .attr("nki_all_to_all_v_single");
  fn(output_tensor, input_tensor, py::cast(output_split_sizes), py::cast(input_split_sizes),
     py::cast(replica_groups));
}

void NeuronBackend::SubmitNkiAllToAllV(std::vector<at::Tensor>& output_tensors,
                                       at::Tensor& input_tensor,
                                       const std::vector<int64_t>& output_split_sizes,
                                       const std::vector<int64_t>& input_split_sizes) {
  auto replica_groups = ValidateNkiAllToAllV();
  py::gil_scoped_acquire gil;
  py::object fn = py::module_::import("torch_neuronx.distributed.ops.all_to_all_v")
                      .attr("nki_all_to_all_v_single");
  fn(py::cast(output_tensors), input_tensor, py::cast(output_split_sizes),
     py::cast(input_split_sizes), py::cast(replica_groups));
}

c10::intrusive_ptr<c10d::Work> NeuronBackend::alltoall(std::vector<at::Tensor>& output_tensors,
                                                       std::vector<at::Tensor>& input_tensors,
                                                       const c10d::AllToAllOptions& opts) {
  ValidateNeuronDevice(input_tensors, "alltoall");
  TORCH_CHECK(static_cast<int>(input_tensors.size()) == size_, "number of input tensors (",
              input_tensors.size(), ") must equal world size (", size_, ")");
  bool async_op = opts.asyncOp;

  // Detect unequal per-rank chunk shapes. Historically the list form required every
  // input chunk to share one shape (even all-to-all). MoE token dispatch instead sends
  // a variable number of rows per destination rank, so dim 0 legitimately varies across
  // chunks. Dims >= 1 (the per-row feature size) MUST still match across every input AND
  // output chunk — only dim 0 may differ. When any dim 0 differs we route to the native
  // NKI all_to_all_v path (below), mirroring alltoall_base's uneven-split handling.
  bool unequal_chunks = false;
  if (!input_tensors.empty()) {
    auto ref_trailing = input_tensors[0].sizes().slice(1);
    int64_t ref_dim0 = input_tensors[0].size(0);
    auto check_trailing = [&](const at::Tensor& t, const char* which, size_t idx) {
      TORCH_CHECK(t.sizes().slice(1) == ref_trailing, "alltoall: all ", which,
                  " tensors must have identical shape in dims >= 1 (only dim 0 may vary for "
                  "variable-split all-to-all), but ",
                  which, "[", idx, "] has shape ", t.sizes(), " vs expected trailing dims ",
                  ref_trailing);
    };
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      check_trailing(input_tensors[i], "input", i);
      if (input_tensors[i].size(0) != ref_dim0) unequal_chunks = true;
    }
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      check_trailing(output_tensors[i], "output", i);
      if (output_tensors[i].size(0) != ref_dim0) unequal_chunks = true;
    }
  }

  // Early return for all-empty inputs
  bool all_empty = true;
  for (const auto& t : input_tensors) {
    if (t.numel() > 0) {
      all_empty = false;
      break;
    }
  }
  if (all_empty && !UseNkiAllToAllV()) {
    auto work = CreateWork(output_tensors, {}, c10d::OpType::ALLTOALL, async_op);
    FinishWork(work, async_op);
    return work;
  }
  // An empty local rank may still have peers exchanging data with each other. It must
  // enter the same variable-count collective rather than returning locally.
  if (all_empty) unequal_chunks = true;

  // Unequal per-rank chunk shapes: convert the list form to the split-size representation
  // and delegate to the same variable-split NKI all_to_all_v machinery alltoall_base uses.
  // The even list-form path below is untouched (bit-identical for the common equal-chunk
  // case). This enables torchtitan's standard MoE token dispatcher (dist.all_to_all with
  // per-rank-variable input/output chunks) on Neuron.
  if (unequal_chunks) {
    TORCH_CHECK(UseNkiAllToAllV(),
                "alltoall: input/output chunks have unequal shapes (variable-split all-to-all), "
                "which requires the native NKI all_to_all_v path. Remove "
                "NEURON_USE_NKI_ALLTOALLV=0 or set it to 1; this requires NRT runtime-lib >= "
                "2.x.62579 and "
                "its matched collectives >= 2.x.63596. The equal-shape list form does not support "
                "per-rank variable counts.");

    std::vector<int64_t> in_sizes(input_tensors.size());
    std::vector<int64_t> out_sizes(output_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) in_sizes[i] = input_tensors[i].size(0);
    for (size_t i = 0; i < output_tensors.size(); ++i) out_sizes[i] = output_tensors[i].size(0);

    // Concat inputs into one contiguous, destination-packed buffer along dim 0. The Python
    // driver gathers each received rank slot directly into the caller's output tensor.
    at::Tensor concat_in =
        (input_tensors.size() == 1) ? input_tensors[0] : torch::cat(input_tensors, /*dim=*/0);

    auto work = CreateWork(output_tensors, input_tensors, c10d::OpType::ALLTOALL, async_op);
    try {
      SubmitNkiAllToAllV(output_tensors, concat_in, out_sizes, in_sizes);
      work->stash({concat_in});
    } catch (...) {
      work->setException(std::current_exception());
      FinishWork(work, async_op);
      throw;
    }
    FinishWork(work, async_op);
    return work;
  }

  auto current_ranks = GetReplicaGroups()[0];
  int64_t group_size = static_cast<int64_t>(current_ranks.size());
  // For alltoall, we need global replica groups
  auto replica_groups = CreateGlobalReplicaGroups(global_world_size_, current_ranks);

  TORCH_CHECK(group_size == 2 || group_size == 4 || group_size == 8 || group_size == 16 ||
                  group_size % 32 == 0,
              "unsupported world size ", group_size,
              ", supported sizes: 2, 4, 8, 16, or multiples of 32");
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

  // Normalize split sizes: if empty/None, compute equal splits; else validate length.
  const bool has_explicit_splits = !input_split_sizes.empty() || !output_split_sizes.empty();
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
  const bool locally_empty = input_tensor.numel() == 0 && output_tensor.numel() == 0;

  // With implicit equal splits, a locally empty rank implies that every peer also has
  // empty equal splits. Explicit zero splits can instead describe a rank that sends and
  // receives nothing while other peers exchange data, so that rank must participate.
  if (locally_empty && !has_explicit_splits) {
    auto work = CreateWork({output_tensor}, {input_tensor}, c10d::OpType::ALLTOALL_BASE, false);
    FinishWork(work, false);
    return work;
  }

  // Fast path: even splits (all sizes equal on both sides) — delegate to alltoall.
  bool even_in = std::all_of(in_sizes.begin() + 1, in_sizes.end(),
                             [&](int64_t s) { return s == in_sizes[0]; });
  bool even_out = std::all_of(out_sizes.begin() + 1, out_sizes.end(),
                              [&](int64_t s) { return s == out_sizes[0]; });
  if (even_in && even_out && !locally_empty) {
    auto input_list = input_tensor.split(in_sizes[0], 0);
    auto output_list = output_tensor.split(out_sizes[0], 0);
    std::vector<at::Tensor> in_vec(input_list.begin(), input_list.end());
    std::vector<at::Tensor> out_vec(output_list.begin(), output_list.end());
    return alltoall(out_vec, in_vec, opts);
  }

  // Uneven splits with the NKI all_to_all_v kernel. Sends exactly the requested
  // per-rank counts with no send-side padding. The NKI branch bypasses alltoall(),
  // so we re-check the
  // world-size / replica-group constraints enforced by alltoall(), plus an LNC=2
  // guard for the NeuronSwitch (Trn3) fabric.
  if (UseNkiAllToAllV()) {
    auto work =
        CreateWork({output_tensor}, {input_tensor}, c10d::OpType::ALLTOALL_BASE, opts.asyncOp);
    try {
      SubmitNkiAllToAllV(output_tensor, input_tensor, out_sizes, in_sizes);
    } catch (...) {
      work->setException(std::current_exception());
      FinishWork(work, opts.asyncOp);
      throw;
    }
    FinishWork(work, opts.asyncOp);
    return work;
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
      // Dynamic NKI all-reduce is single-tensor only.
      int64_t world_size = static_cast<int64_t>(replica_groups[0].size());
      const bool dynamic_handled =
          UseDynamicNkiCollectives() && non_empty.size() == 1 &&
          TryDynamicAllreduce(non_empty[0], replica_groups, world_size, red_op, work);
      if (!dynamic_handled) {
        SubmitCollective(c10d::OpType::ALLREDUCE, non_empty, non_empty, replica_groups, red_op);
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
