#pragma once

#include <ATen/Tensor.h>

#include <string>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch_neuronx {
namespace distributed {

inline const std::string& OpTypeToString(c10d::OpType op) {
  static const std::unordered_map<c10d::OpType, std::string> kMap = {
      {c10d::OpType::ALLREDUCE, "allreduce"},
      {c10d::OpType::ALLGATHER, "allgather"},
      {c10d::OpType::_ALLGATHER_BASE, "allgather"},
      {c10d::OpType::REDUCE_SCATTER, "reduce_scatter"},
      {c10d::OpType::_REDUCE_SCATTER_BASE, "reduce_scatter"},
      {c10d::OpType::ALLTOALL, "alltoall"},
      {c10d::OpType::BROADCAST, "broadcast"},
      {c10d::OpType::REDUCE, "reduce"},
      {c10d::OpType::BARRIER, "barrier"},
  };
  static const std::string kUnknown = "unknown";
  auto it = kMap.find(op);
  return it != kMap.end() ? it->second : kUnknown;
}

inline std::vector<std::vector<int>> CreateGlobalReplicaGroups(
    int world_size, const std::vector<int>& current_group_ranks) {
  int group_size = static_cast<int>(current_group_ranks.size());
  TORCH_CHECK(world_size % group_size == 0, "Cannot create global replica groups: world_size (",
              world_size, ") not divisible by group_size (", group_size, ")");
  std::unordered_set<int> used(current_group_ranks.begin(), current_group_ranks.end());
  std::vector<int> unused_ranks;
  for (int i = 0; i < world_size; ++i) {
    if (used.find(i) == used.end()) unused_ranks.push_back(i);
  }
  std::vector<std::vector<int>> groups;
  groups.push_back(current_group_ranks);
  for (size_t j = 0; j < unused_ranks.size(); j += group_size) {
    groups.emplace_back(unused_ranks.begin() + j,
                        unused_ranks.begin() + std::min(j + group_size, unused_ranks.size()));
  }
  return groups;
}

inline void ValidateNeuronDevice(const std::vector<at::Tensor>& tensors,
                                 const std::string& op_name) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    TORCH_CHECK(tensors[i].device().type() == c10::DeviceType::PrivateUse1,
                "Expected neuron device, got ", tensors[i].device().type(), " for tensor ", i,
                " in ", op_name);
  }
}

inline void ValidateNonEmpty(const std::vector<at::Tensor>& tensors, const std::string& op_name) {
  std::string empty_indices;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() == 0) {
      if (!empty_indices.empty()) empty_indices += ", ";
      empty_indices += std::to_string(i);
    }
  }
  TORCH_CHECK(empty_indices.empty(), "tensors cannot be empty, found empty tensors at indices: [",
              empty_indices, "] in ", op_name);
}

inline void ValidateNonZeroDim0(const std::vector<at::Tensor>& tensors,
                                const std::string& op_name) {
  std::string zero_indices;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].dim() > 0 && tensors[i].size(0) == 0) {
      if (!zero_indices.empty()) zero_indices += ", ";
      zero_indices += std::to_string(i) + " (shape: " + c10::str(tensors[i].sizes()) + ")";
    }
  }
  TORCH_CHECK(zero_indices.empty(), "tensors cannot have zero size in dimension 0, found: [",
              zero_indices, "] in ", op_name);
}

}  // namespace distributed
}  // namespace torch_neuronx
