/**
 * Unit tests for DtypeUtils — NeuronSafeCast and CopyBackToOriginal.
 *
 * Tests use CPU tensors only (no Neuron hardware required).
 * The CPU path exercises the direct .to() cast; the CPU round-trip path
 * (device → CPU → cast → device) is not tested here since it requires
 * a PrivateUse1 device.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

// ═══════════════════════════════════════════════════════════════
//  NeuronSafeCast — CPU tensor paths
// ═══════════════════════════════════════════════════════════════

TEST(NeuronSafeCastTest, SameDtypeReturnsIdentical) {
  auto t = torch::ones({2, 3}, torch::kFloat32);
  auto result = NeuronSafeCast(t, c10::ScalarType::Float);
  EXPECT_EQ(result.data_ptr(), t.data_ptr());
}

TEST(NeuronSafeCastTest, Float32ToFloat64) {
  auto t = torch::tensor({1.5f, 2.5f}, torch::kFloat32);
  auto result = NeuronSafeCast(t, c10::ScalarType::Double);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Double);
  EXPECT_NEAR(result[0].item<double>(), 1.5, 1e-6);
}

TEST(NeuronSafeCastTest, Float64ToFloat32) {
  auto t = torch::tensor({1.5, 2.5}, torch::kFloat64);
  auto result = NeuronSafeCast(t, c10::ScalarType::Float);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Float);
  EXPECT_NEAR(result[0].item<float>(), 1.5f, 1e-6);
}

TEST(NeuronSafeCastTest, Int64ToInt32) {
  auto t = torch::tensor({10L, 20L}, torch::kInt64);
  auto result = NeuronSafeCast(t, c10::ScalarType::Int);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Int);
  EXPECT_EQ(result[0].item<int>(), 10);
}

TEST(NeuronSafeCastTest, Int32ToInt64) {
  auto t = torch::tensor({10, 20}, torch::kInt32);
  auto result = NeuronSafeCast(t, c10::ScalarType::Long);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(result[1].item<int64_t>(), 20);
}

TEST(NeuronSafeCastTest, Float32ToHalf) {
  auto t = torch::tensor({1.0f, 2.0f}, torch::kFloat32);
  auto result = NeuronSafeCast(t, c10::ScalarType::Half);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Half);
}

TEST(NeuronSafeCastTest, BoolToInt) {
  auto t = torch::tensor({true, false}, torch::kBool);
  auto result = NeuronSafeCast(t, c10::ScalarType::Int);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Int);
  EXPECT_EQ(result[0].item<int>(), 1);
  EXPECT_EQ(result[1].item<int>(), 0);
}

TEST(NeuronSafeCastTest, PreservesShape) {
  auto t = torch::zeros({3, 4, 5}, torch::kFloat64);
  auto result = NeuronSafeCast(t, c10::ScalarType::Float);
  EXPECT_EQ(result.sizes(), t.sizes());
}

// ═══════════════════════════════════════════════════════════════
//  CopyBackToOriginal
// ═══════════════════════════════════════════════════════════════

TEST(CopyBackToOriginalTest, SameStorageNoOp) {
  auto t = torch::ones({2, 3});
  auto orig = t;  // Same storage
  CopyBackToOriginal(orig, t, "test");
  EXPECT_EQ(orig.data_ptr(), t.data_ptr());
}

TEST(CopyBackToOriginalTest, CopiesData) {
  auto orig = torch::zeros({2, 3});
  auto result = torch::ones({2, 3});
  CopyBackToOriginal(orig, result, "test");
  EXPECT_TRUE(torch::allclose(orig, result));
}

TEST(CopyBackToOriginalTest, HandlesShapeMismatch) {
  auto orig = torch::zeros({2});
  auto result = torch::ones({4});
  CopyBackToOriginal(orig, result, "test");
  EXPECT_EQ(orig.sizes(), result.sizes());
  EXPECT_TRUE(torch::allclose(orig, result));
}

TEST(CopyBackToOriginalTest, HandlesDtypeMismatch) {
  auto orig = torch::zeros({2, 3}, torch::kFloat64);
  auto result = torch::ones({2, 3}, torch::kFloat32);
  CopyBackToOriginal(orig, result, "test");
  EXPECT_EQ(orig.scalar_type(), c10::ScalarType::Double);
  EXPECT_NEAR(orig[0][0].item<double>(), 1.0, 1e-6);
}

TEST(CopyBackToOriginalTest, UndefinedOrigSkipped) {
  auto orig = torch::Tensor();
  auto result = torch::ones({2, 3});
  EXPECT_NO_THROW(CopyBackToOriginal(orig, result, "test"));
}

TEST(CopyBackToOriginalTest, UndefinedResultSkipped) {
  auto orig = torch::zeros({2, 3});
  auto result = torch::Tensor();
  auto orig_ptr = orig.data_ptr();
  CopyBackToOriginal(orig, result, "test");
  EXPECT_EQ(orig.data_ptr(), orig_ptr);  // Unchanged
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
