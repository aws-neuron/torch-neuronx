#pragma once

#include <atomic>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"
#include "torch_neuronx/csrc/distributed/NeuronWatchdog.h"
#include "torch_neuronx/csrc/distributed/NeuronWork.h"

namespace torch_neuronx {
namespace distributed {

class NeuronBackend : public c10d::Backend {
 public:
  NeuronBackend(const c10::intrusive_ptr<c10d::Backend::Options>& options, int rank, int size,
                int global_world_size, pybind11::object lower_callback);
  ~NeuronBackend() override;

  // Release Python callback before interpreter finalization.
  // Called from module cleanup capsule (same pattern as NeuronDispatcher).
  static void Cleanup();

  const std::string getBackendName() const override { return "neuron"; }

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
                              bool slice_output = false);

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

  // Resolve a `c10d::ReduceOp` to the integer reduce code used by the submit path.
  // For `PREMUL_SUM`, scales the provided tensors in-place by the supplement's factor
  // and returns `SUM`. For all other ops, returns the underlying `RedOpType` unchanged
  // and leaves tensors untouched.
  int ResolveReduceOp(const c10d::ReduceOp& reduce_op, std::vector<at::Tensor>& tensors);

  std::string pg_uid_;
  int64_t timeout_ms_;
  bool enable_timing_ = false;
  std::atomic<uint64_t> seq_num_{0};

  std::vector<int> group_ranks_;
  int global_world_size_;

  NeuronWatchdog watchdog_;

  static pybind11::object lower_callback_;
};

}  // namespace distributed
}  // namespace torch_neuronx
