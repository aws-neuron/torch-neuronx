#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "torch_neuronx/csrc/distributed/DistributedUtils.h"

using namespace torch_neuronx::distributed;

TEST(CreateGlobalReplicaGroups, BasicContiguous) {
  auto groups = CreateGlobalReplicaGroups(4, {0, 1, 2, 3});
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0], (std::vector<int>{0, 1, 2, 3}));
}

TEST(CreateGlobalReplicaGroups, TwoGroupsContiguous) {
  auto groups = CreateGlobalReplicaGroups(8, {0, 1, 2, 3});
  ASSERT_EQ(groups.size(), 2);
  EXPECT_EQ(groups[0], (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(groups[1], (std::vector<int>{4, 5, 6, 7}));
}

TEST(CreateGlobalReplicaGroups, NonContiguousRanks) {
  auto groups = CreateGlobalReplicaGroups(8, {0, 2, 4, 6});
  ASSERT_EQ(groups.size(), 2);
  EXPECT_EQ(groups[0], (std::vector<int>{0, 2, 4, 6}));
  EXPECT_EQ(groups[1], (std::vector<int>{1, 3, 5, 7}));
}

TEST(CreateGlobalReplicaGroups, SingleRankGroups) {
  auto groups = CreateGlobalReplicaGroups(4, {2});
  ASSERT_EQ(groups.size(), 4);
  EXPECT_EQ(groups[0], (std::vector<int>{2}));
  EXPECT_EQ(groups[1], (std::vector<int>{0}));
  EXPECT_EQ(groups[2], (std::vector<int>{1}));
  EXPECT_EQ(groups[3], (std::vector<int>{3}));
}

TEST(CreateGlobalReplicaGroups, FullWorld) {
  auto groups = CreateGlobalReplicaGroups(4, {0, 1, 2, 3});
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0], (std::vector<int>{0, 1, 2, 3}));
}

TEST(CreateGlobalReplicaGroups, ReversedRanks) {
  auto groups = CreateGlobalReplicaGroups(4, {3, 2, 1, 0});
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0], (std::vector<int>{3, 2, 1, 0}));
}

TEST(CreateGlobalReplicaGroups, NotDivisible) {
  EXPECT_THROW(CreateGlobalReplicaGroups(5, {0, 1}), c10::Error);
}

TEST(OpTypeToString, AllReduce) { EXPECT_EQ(OpTypeToString(c10d::OpType::ALLREDUCE), "allreduce"); }

TEST(OpTypeToString, AllGather) { EXPECT_EQ(OpTypeToString(c10d::OpType::ALLGATHER), "allgather"); }

TEST(OpTypeToString, AllGatherBase) {
  EXPECT_EQ(OpTypeToString(c10d::OpType::_ALLGATHER_BASE), "allgather");
}

TEST(OpTypeToString, ReduceScatter) {
  EXPECT_EQ(OpTypeToString(c10d::OpType::REDUCE_SCATTER), "reduce_scatter");
}

TEST(OpTypeToString, ReduceScatterBase) {
  EXPECT_EQ(OpTypeToString(c10d::OpType::_REDUCE_SCATTER_BASE), "reduce_scatter");
}

TEST(OpTypeToString, AllToAll) { EXPECT_EQ(OpTypeToString(c10d::OpType::ALLTOALL), "alltoall"); }

TEST(OpTypeToString, Broadcast) { EXPECT_EQ(OpTypeToString(c10d::OpType::BROADCAST), "broadcast"); }

TEST(OpTypeToString, Reduce) { EXPECT_EQ(OpTypeToString(c10d::OpType::REDUCE), "reduce"); }

TEST(OpTypeToString, Barrier) { EXPECT_EQ(OpTypeToString(c10d::OpType::BARRIER), "barrier"); }

TEST(OpTypeToString, Unknown) {
  EXPECT_EQ(OpTypeToString(static_cast<c10d::OpType>(999)), "unknown");
}

TEST(ValidateNeuronDevice, CPUTensors) {
  std::vector<at::Tensor> tensors = {at::zeros({2, 3})};
  EXPECT_THROW(
      {
        try {
          ValidateNeuronDevice(tensors, "test_op");
        } catch (const c10::Error& e) {
          EXPECT_NE(std::string(e.what()).find("test_op"), std::string::npos);
          throw;
        }
      },
      c10::Error);
}

TEST(ValidateNeuronDevice, EmptyList) {
  std::vector<at::Tensor> tensors;
  EXPECT_NO_THROW(ValidateNeuronDevice(tensors, "test_op"));
}

TEST(ValidateNeuronDevice, MixedDevices) {
  // Both CPU — first one would fail if it were Neuron, second is CPU
  // Since we can't create PrivateUse1 tensors without device registration,
  // test that multiple CPU tensors throw on the first one
  std::vector<at::Tensor> tensors = {at::zeros({2, 3}), at::ones({2, 3})};
  EXPECT_THROW(ValidateNeuronDevice(tensors, "test_op"), c10::Error);
}

TEST(ValidateNonEmpty, NonEmptyTensors) {
  std::vector<at::Tensor> tensors = {at::zeros({2, 3}), at::ones({4, 5})};
  EXPECT_NO_THROW(ValidateNonEmpty(tensors, "test_op"));
}

TEST(ValidateNonEmpty, EmptyTensor) {
  std::vector<at::Tensor> tensors = {at::empty({0})};
  EXPECT_THROW(
      {
        try {
          ValidateNonEmpty(tensors, "test_op");
        } catch (const c10::Error& e) {
          EXPECT_NE(std::string(e.what()).find("test_op"), std::string::npos);
          throw;
        }
      },
      c10::Error);
}

TEST(ValidateNonEmpty, EmptyList) {
  std::vector<at::Tensor> tensors;
  EXPECT_NO_THROW(ValidateNonEmpty(tensors, "test_op"));
}

TEST(ValidateNonZeroDim0, ValidTensors) {
  std::vector<at::Tensor> tensors = {at::zeros({2, 3}), at::ones({4, 5})};
  EXPECT_NO_THROW(ValidateNonZeroDim0(tensors, "test_op"));
}

TEST(ValidateNonZeroDim0, ZeroDim0) {
  std::vector<at::Tensor> tensors = {at::empty({0, 3})};
  EXPECT_THROW(
      {
        try {
          ValidateNonZeroDim0(tensors, "test_op");
        } catch (const c10::Error& e) {
          EXPECT_NE(std::string(e.what()).find("test_op"), std::string::npos);
          throw;
        }
      },
      c10::Error);
}

TEST(ValidateNonZeroDim0, ScalarTensor) {
  // 0-dim tensor: dim()==0, so the check (dim() > 0 && size(0) == 0) is false
  std::vector<at::Tensor> tensors = {at::tensor(42.0)};
  EXPECT_NO_THROW(ValidateNonZeroDim0(tensors, "test_op"));
}
