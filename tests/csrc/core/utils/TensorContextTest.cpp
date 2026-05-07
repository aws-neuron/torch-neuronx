#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/TensorContext.h"

using namespace at::neuron;

class TensorContextTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(TensorContextTest, FromTensorBasicProperties) {
  auto tensor = torch::randn({2, 3, 4});
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_EQ(ctx.get_numel(), 24);
  EXPECT_EQ(ctx.get_element_size(), sizeof(float));
  EXPECT_EQ(ctx.get_size_bytes(), 24 * sizeof(float));
  EXPECT_EQ(ctx.get_dtype(), c10::ScalarType::Float);
  EXPECT_EQ(ctx.get_storage_offset(), 0);
  EXPECT_FALSE(ctx.get_requires_grad());
}

TEST_F(TensorContextTest, FromTensorShape) {
  auto tensor = torch::randn({5, 10, 15});
  auto ctx = TensorContext::FromTensor(tensor);

  auto shape = ctx.get_shape();
  ASSERT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 5);
  EXPECT_EQ(shape[1], 10);
  EXPECT_EQ(shape[2], 15);
}

TEST_F(TensorContextTest, FromTensorDifferentDtypes) {
  auto float_tensor = torch::randn({10}, torch::kFloat32);
  auto int_tensor = torch::randint(0, 100, {10}, torch::kInt64);
  auto double_tensor = torch::randn({10}, torch::kFloat64);

  auto float_ctx = TensorContext::FromTensor(float_tensor);
  auto int_ctx = TensorContext::FromTensor(int_tensor);
  auto double_ctx = TensorContext::FromTensor(double_tensor);

  EXPECT_EQ(float_ctx.get_dtype(), c10::ScalarType::Float);
  EXPECT_EQ(float_ctx.get_element_size(), sizeof(float));

  EXPECT_EQ(int_ctx.get_dtype(), c10::ScalarType::Long);
  EXPECT_EQ(int_ctx.get_element_size(), sizeof(int64_t));

  EXPECT_EQ(double_ctx.get_dtype(), c10::ScalarType::Double);
  EXPECT_EQ(double_ctx.get_element_size(), sizeof(double));
}

TEST_F(TensorContextTest, FromTensorWithRequiresGrad) {
  auto tensor = torch::randn({5, 5}, torch::TensorOptions().requires_grad(true));
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_TRUE(ctx.get_requires_grad());
}

TEST_F(TensorContextTest, FromTensorWithStorageOffset) {
  auto base_tensor = torch::randn({10});
  auto sliced_tensor = base_tensor.slice(0, 2, 8);
  auto ctx = TensorContext::FromTensor(sliced_tensor);

  EXPECT_EQ(ctx.get_storage_offset(), 2);
  EXPECT_EQ(ctx.get_numel(), 6);
}

TEST_F(TensorContextTest, FromTensorDevice) {
  auto cpu_tensor = torch::randn({5, 5}, torch::kCPU);
  auto ctx = TensorContext::FromTensor(cpu_tensor);

  EXPECT_EQ(ctx.get_device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(ctx.get_device().index(), -1);
}

TEST_F(TensorContextTest, FromTensorLayout) {
  auto tensor = torch::randn({5, 5});
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_EQ(ctx.get_layout(), c10::Layout::Strided);
}

TEST_F(TensorContextTest, SizeBytesCalculation) {
  auto tensor = torch::randn({3, 4, 5}, torch::kFloat32);
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_EQ(ctx.get_size_bytes(), 3 * 4 * 5 * sizeof(float));
  EXPECT_EQ(ctx.get_size_bytes(), ctx.get_numel() * ctx.get_element_size());
}

TEST_F(TensorContextTest, EmptyTensor) {
  auto tensor = torch::empty({0});
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_EQ(ctx.get_numel(), 0);
  EXPECT_EQ(ctx.get_size_bytes(), 0);
  EXPECT_EQ(ctx.get_shape().size(), 1);
  EXPECT_EQ(ctx.get_shape()[0], 0);
}

TEST_F(TensorContextTest, ScalarTensor) {
  auto tensor = torch::tensor(42.0f);
  auto ctx = TensorContext::FromTensor(tensor);

  EXPECT_EQ(ctx.get_numel(), 1);
  EXPECT_EQ(ctx.get_size_bytes(), sizeof(float));
  EXPECT_EQ(ctx.get_shape().size(), 0);
}

TEST_F(TensorContextTest, TensorOptionsAccess) {
  auto tensor =
      torch::randn({3, 4}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto ctx = TensorContext::FromTensor(tensor);

  auto options = ctx.get_options();
  EXPECT_EQ(options.dtype().toScalarType(), c10::ScalarType::Float);
  EXPECT_EQ(options.device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(options.layout(), c10::Layout::Strided);
  EXPECT_TRUE(options.requires_grad());
}

TEST_F(TensorContextTest, CopyConstructor) {
  auto tensor = torch::randn({3, 4});
  auto ctx1 = TensorContext::FromTensor(tensor);
  auto ctx2 = ctx1;

  EXPECT_EQ(ctx2.get_numel(), ctx1.get_numel());
  EXPECT_EQ(ctx2.get_dtype(), ctx1.get_dtype());
  EXPECT_EQ(ctx2.get_shape(), ctx1.get_shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
