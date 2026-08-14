#pragma once

#include <atomic>
#include <mutex>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"
#include "torch_neuronx/csrc/distributed/NeuronWatchdog.h"
#include "torch_neuronx/csrc/distributed/NeuronWork.h"

namespace torch_neuronx {
namespace distributed {

class NeuronBackend : public c10d::Backend {
 public:
  NeuronBackend(const c10::intrusive_ptr<c10d::Store>& store,
                const c10::intrusive_ptr<c10d::Backend::Options>& options, int rank, int size,
                int global_world_size, pybind11::object lower_callback);
  ~NeuronBackend() override;

  // Release Python callback before interpreter finalization.
  // Called from module cleanup capsule (same pattern as NeuronDispatcher).
  static void Cleanup();

  const std::string getBackendName() const override { return "neuron"; }

  // Advertise that this backend implements startCoalescing / endCoalescing.
  // Without this, dist.batch_isend_irecv falls through to per-op dispatch and
  // deadlocks the single engine thread on per-op Store I/O.
  bool supportsCoalescing() const override { return true; }

  // ── Core collectives ──
  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors, std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output_tensor, at::Tensor& input_tensor,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors, std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor, at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors, std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor, at::Tensor& input_tensor, std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors, std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors, std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  // ── Coalesced + bucketing ──
  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_tensors_list,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  // ── Point-to-point (RDMA over EFA) ──
  // Single-tensor send/recv. tag is a user-provided int that combines with
  // (rank, peer) to form the c10d::Store key for descriptor exchange.
  c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor>& tensors, int dstRank,
                                      int tag) override;
  c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor>& tensors, int srcRank,
                                      int tag) override;

  // batch_isend_irecv coalescing: queues send/recv ops between start/end, then
  // submits them with a single Store multi-set/multi-get round-trip and detects
  // same-rank pairs (V-shaped schedules e.g. ZBV) for local copy short-circuit.
  void startCoalescing() override;
  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

 private:
  // Create NeuronWork, stash tensors, record event, enqueue to watchdog.
  c10::intrusive_ptr<NeuronWork> CreateWork(const std::vector<at::Tensor>& outputs,
                                            const std::vector<at::Tensor>& inputs,
                                            c10d::OpType op_type, bool async_op);

  // Returns true if hostCC should be used.
  bool UseHostCC() const;

  // Submit via inNEFF path
  void SubmitInNeffCollective(c10d::OpType op_type, const std::vector<at::Tensor>& inputs,
                              const std::vector<at::Tensor>& outputs,
                              const std::vector<std::vector<int>>& replica_groups, int reduce_op,
                              bool slice_output = false,
                              const std::string& cache_key_override = "");

  // Submit via hostCC path
  void SubmitHostCCCollective(c10d::OpType op_type, const std::vector<at::Tensor>& inputs,
                              const std::vector<at::Tensor>& outputs,
                              const std::vector<std::vector<int>>& replica_groups, int reduce_op);

  // Get replica groups for this process group
  std::vector<std::vector<int>> GetReplicaGroups() const;

  // Get full-world replica groups (partitioning all ranks across subgroups) for HLO lowering.
  std::vector<std::vector<int>> GetFullReplicaGroups() const;

  // Complete work after collective submission: record event, register, enqueue.
  void FinishWork(c10::intrusive_ptr<NeuronWork>& work, bool async_op);

  // Core reduce_scatter submit logic with optional bucketing.
  // Returns intermediate tensors that must be kept alive until async completion.
  std::vector<at::Tensor> ReduceScatterSubmit(at::Tensor& output_tensor, at::Tensor& input_tensor,
                                              const std::vector<std::vector<int>>& replica_groups,
                                              int red_op);

  // Route collective to inNEFF or hostCC path.
  void SubmitCollective(c10d::OpType op_type, const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs,
                        const std::vector<std::vector<int>>& replica_groups, int reduce_op,
                        bool slice_output = false);

  // Attempt the dynamic multi-tier NKI all-gather (gated by NEURON_DYN_NKI_COLLECTIVES).
  // On an eligible call it binds {input[numel,1], numel} to the NEFF and
  // returns true; returns false when dynami all-gather is ineligible.
  bool TryDynamicAllgather(at::Tensor& input_tensor, at::Tensor& output_tensor,
                           const std::vector<std::vector<int>>& replica_groups, int world_size,
                           c10::intrusive_ptr<NeuronWork>& work);

  // Submit a variable-split (uneven) all-to-all via the native NKI all_to_all_v kernel.
  // Enforces the world-size / replica-group start-rank / LNC=2 guards, then invokes the
  // Python driver, which sends exactly the per-rank row counts in `input_split_sizes` and
  // receives `output_split_sizes` (no send-side padding). `input_tensor` must be contiguously
  // packed by destination rank along dim 0. The output may be one rank-ordered tensor or the
  // caller's per-rank output list.
  // Throws on guard failure or kernel error; does NOT create or finish the Work. Shared by
  // both `alltoall` (list form, unequal chunks) and `alltoall_base` (uneven splits).
  void SubmitNkiAllToAllV(at::Tensor& output_tensor, at::Tensor& input_tensor,
                          const std::vector<int64_t>& output_split_sizes,
                          const std::vector<int64_t>& input_split_sizes);
  void SubmitNkiAllToAllV(std::vector<at::Tensor>& output_tensors, at::Tensor& input_tensor,
                          const std::vector<int64_t>& output_split_sizes,
                          const std::vector<int64_t>& input_split_sizes);
  std::vector<std::vector<int>> ValidateNkiAllToAllV();

  // Resolve a `c10d::ReduceOp` to the integer reduce code used by the submit path.
  // For `PREMUL_SUM`, scales the provided tensors in-place by the supplement's factor
  // and returns `SUM`. For all other ops, returns the underlying `RedOpType` unchanged
  // and leaves tensors untouched.
  int ResolveReduceOp(const c10d::ReduceOp& reduce_op, std::vector<at::Tensor>& tensors);

  // Get the global rank of a group-local rank. Mirrors Python's
  // `dist.get_process_group_ranks(self)[group_rank]`. RDMA keys (rdma_ep_<global>,
  // rdma_<src>_<dst>_<tag>_<seq>) are always indexed by global rank.
  int ToGlobal(int group_rank) const { return group_ranks_[group_rank]; }

  // Per-PG one-time setup: walk PrefixStore -> root and hand it + this PG's
  // global rank to RdmaManager. Libfabric init, local-endpoint publish, and
  // peer AV inserts run lazily on the engine thread inside PrepareBatch.
  // Caller must hold p2p_mutex_.
  void EnsureRdmaInit(at::Tensor& tensor, int peer_global);

  std::string pg_uid_;
  int64_t timeout_ms_;
  bool enable_timing_ = false;
  std::atomic<uint64_t> seq_num_{0};

  std::vector<int> group_ranks_;
  int global_world_size_;

  // Global rank of THIS PG member (equal to group_ranks_[rank()]).
  // Stored explicitly because RDMA send/recv keys (rdma_<src>_<dst>_<tag>_<seq>)
  // distinguish "self" (this member) from "peer" (resolved via ToGlobal).
  int global_rank_{0};

  // Store carried into NeuronBackend so RDMA send/recv can publish and resolve
  // descriptor keys via c10d::Store. Typically a PrefixStore wrapping the world
  // TCPStore; EnsureRdmaInit walks getUnderlyingNonPrefixStore() to obtain the
  // root for cross-PG agreement on rdma_ep_<global> keys.
  c10::intrusive_ptr<c10d::Store> store_;

  // Per-PG P2P state. RdmaManager itself is process-global. Guarded by
  // p2p_mutex_; lock is dropped before submitting ops to the engine.
  bool rdma_setup_done_{false};
  bool coalescing_{false};
  struct CoalescedOp {
    c10d::OpType op;  // SEND or RECV
    at::Tensor tensor;
    int peer_global;
    int tag;
  };
  std::vector<CoalescedOp> coalesced_ops_;
  mutable std::mutex p2p_mutex_;

  NeuronWatchdog watchdog_;

  static pybind11::object lower_callback_;
};

}  // namespace distributed
}  // namespace torch_neuronx
