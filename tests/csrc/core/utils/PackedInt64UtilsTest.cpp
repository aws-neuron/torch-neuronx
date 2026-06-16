/**
 * Unit tests for PackedInt64Utils — layout computation and view creation.
 * Tests run on CPU tensors (no Neuron device required).
 */

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/PackedInt64Utils.h"

namespace torch_neuronx {
namespace packed_int64 {
namespace {

TEST(PackedInt64Utils, PhysicalSizes_1D) {
  auto sizes = physical_sizes({3});
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 3);
  EXPECT_EQ(sizes[1], 2);
}

TEST(PackedInt64Utils, PhysicalSizes_2D) {
  auto sizes = physical_sizes({3, 4});
  ASSERT_EQ(sizes.size(), 3);
  EXPECT_EQ(sizes[0], 3);
  EXPECT_EQ(sizes[1], 4);
  EXPECT_EQ(sizes[2], 2);
}

TEST(PackedInt64Utils, PhysicalSizes_Scalar) {
  auto sizes = physical_sizes({});
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 1);
  EXPECT_EQ(sizes[1], 2);
}

TEST(PackedInt64Utils, PhysicalStrides_1D) {
  auto strides = physical_strides({1});
  ASSERT_EQ(strides.size(), 2);
  EXPECT_EQ(strides[0], 2);
  EXPECT_EQ(strides[1], 1);
}

TEST(PackedInt64Utils, PhysicalStrides_2D_Contiguous) {
  auto strides = physical_strides({4, 1});
  ASSERT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 8);
  EXPECT_EQ(strides[1], 2);
  EXPECT_EQ(strides[2], 1);
}

TEST(PackedInt64Utils, PhysicalStrides_2D_Transposed) {
  auto strides = physical_strides({1, 3});
  ASSERT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 2);
  EXPECT_EQ(strides[1], 6);
  EXPECT_EQ(strides[2], 1);
}

TEST(PackedInt64Utils, PhysicalView_Shape) {
  auto t = torch::zeros({3, 4}, torch::dtype(torch::kInt64));
  auto view = physical_view(t);

  EXPECT_EQ(view.scalar_type(), c10::ScalarType::Int);
  ASSERT_EQ(view.dim(), 3);
  EXPECT_EQ(view.size(0), 3);
  EXPECT_EQ(view.size(1), 4);
  EXPECT_EQ(view.size(2), 2);
}

TEST(PackedInt64Utils, PhysicalView_SharesStorage) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt64));
  auto view = physical_view(t);

  EXPECT_EQ(view.storage().data_ptr().get(), t.storage().data_ptr().get());
}

TEST(PackedInt64Utils, PhysicalView_CorrectValues) {
  // 7 as int64 on little-endian = [7, 0] as int32 pair
  auto t = torch::tensor({7L}, torch::dtype(torch::kInt64));
  auto view = physical_view(t);

  EXPECT_EQ(view[0][0].item<int32_t>(), 7);
  EXPECT_EQ(view[0][1].item<int32_t>(), 0);
}

TEST(PackedInt64Utils, PhysicalView_NegativeValue) {
  // -1 as int64 = 0xFFFFFFFF_FFFFFFFF = [-1, -1] as int32 pair
  auto t = torch::tensor({-1L}, torch::dtype(torch::kInt64));
  auto view = physical_view(t);

  EXPECT_EQ(view[0][0].item<int32_t>(), -1);
  EXPECT_EQ(view[0][1].item<int32_t>(), -1);
}

TEST(PackedInt64Utils, LowWordAlias_Shape) {
  auto t = torch::zeros({3, 4}, torch::dtype(torch::kInt64));
  auto low = low_word_alias(t);

  EXPECT_EQ(low.scalar_type(), c10::ScalarType::Int);
  ASSERT_EQ(low.dim(), 2);
  EXPECT_EQ(low.size(0), 3);
  EXPECT_EQ(low.size(1), 4);
}

TEST(PackedInt64Utils, LowWordAlias_ExtractsLowBits) {
  // 0x0000000200000001 → low=1, high=2
  auto t = torch::tensor({0x0000000200000001L}, torch::dtype(torch::kInt64));
  auto low = low_word_alias(t);

  EXPECT_EQ(low[0].item<int32_t>(), 1);
}

TEST(PackedInt64Utils, ExecutionView_CPU_Contiguous) {
  auto t = torch::tensor({1L, -1L, 42L}, torch::dtype(torch::kInt64));
  auto ev = execution_view(t);

  EXPECT_EQ(ev.scalar_type(), c10::ScalarType::Int);
  EXPECT_TRUE(ev.is_contiguous());
  ASSERT_EQ(ev.size(0), 3);
  EXPECT_EQ(ev[0].item<int32_t>(), 1);
  EXPECT_EQ(ev[1].item<int32_t>(), -1);
  EXPECT_EQ(ev[2].item<int32_t>(), 42);
}

TEST(PackedInt64Utils, PhysicalView_WrongDtype_Throws) {
  auto t = torch::zeros({3}, torch::dtype(torch::kFloat32));
  EXPECT_THROW(physical_view(t), c10::Error);
}

}  // namespace
}  // namespace packed_int64
}  // namespace torch_neuronx
