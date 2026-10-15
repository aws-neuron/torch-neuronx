/**
 * Unit tests for DtypeUtils — NeuronSafeCast and CopyBackToOriginal.
 *
 * Tests use CPU tensors only (no Neuron hardware required).
 * The CPU path exercises the direct .to() cast; the CPU round-trip path
 * (device → CPU → cast → device) is not tested here since it requires
 * a PrivateUse1 device.
 */

#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
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

TEST(NeuronSafeCastTest, Int64KeptUnderPackedMode) {
  // In packed mode, int64 is kept as Long and materialized on-device as packed
  // int64 (int32[..., 2]) — not downcast to int32. (Default mode is now native,
  // so packed must be selected explicitly.)
  SetInt64ModeForTesting(Int64Mode::kPacked);
  auto t = torch::tensor({10L, 20L}, torch::kInt64);
  auto result = NeuronSafeCast(t, c10::ScalarType::Int);
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(result[0].item<int64_t>(), 10);
  SetInt64ModeForTesting(Int64Mode::kNative);
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

// A non-contiguous same-numel orig must keep its view shape, not resize. This
// is the torch.arange(4, out=x.narrow(1, 1, 2)) case (a [4] result into a
// strided [2,2] view). PyTorch scatters per the stride.
TEST(CopyBackToOriginalTest, NonContiguousSameNumelKeepsViewShape) {
  auto base = torch::zeros({2, 3});
  auto orig = base.narrow(1, 1, 2);  // shape [2,2], stride [3,1], offset 1
  ASSERT_FALSE(orig.is_contiguous());
  auto result = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f});  // shape [4]

  CopyBackToOriginal(orig, result, "test");

  EXPECT_EQ(orig.sizes(), (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(orig.strides(), (std::vector<int64_t>{3, 1}));
  auto expected = torch::tensor({{0.0f, 0.0f, 1.0f}, {0.0f, 2.0f, 3.0f}});
  EXPECT_TRUE(torch::equal(base, expected));
}

// A non-contiguous orig with a different numel still follows the resize contract.
TEST(CopyBackToOriginalTest, NonContiguousDifferentNumelResizes) {
  auto base = torch::zeros({2, 3});
  auto orig = base.narrow(1, 1, 2);  // 4 elements
  auto result = torch::ones({6});    // different numel -> resize

  CopyBackToOriginal(orig, result, "test");

  EXPECT_EQ(orig.sizes(), result.sizes());
  EXPECT_TRUE(torch::allclose(orig, result));
}

// A contiguous orig keeps the ordinary resize behavior (the non-contiguous
// branch must not capture it).
TEST(CopyBackToOriginalTest, ContiguousSameNumelStillResizes) {
  auto orig = torch::zeros({2, 2});
  auto result = torch::ones({4});  // same numel, but orig is contiguous

  CopyBackToOriginal(orig, result, "test");

  EXPECT_EQ(orig.sizes(), result.sizes());  // resized to [4], not kept as [2,2]
}

// Non-contiguous orig plus a dtype mismatch: cast and stride-respecting scatter
// must compose.
TEST(CopyBackToOriginalTest, NonContiguousWithDtypeMismatch) {
  auto base = torch::zeros({2, 3}, torch::kFloat64);
  auto orig = base.narrow(1, 1, 2);
  auto result = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f}, torch::kFloat32);

  CopyBackToOriginal(orig, result, "test");

  EXPECT_EQ(orig.scalar_type(), c10::ScalarType::Double);
  EXPECT_EQ(orig.sizes(), (std::vector<int64_t>{2, 2}));
  auto expected = torch::tensor({{0.0, 0.0, 1.0}, {0.0, 2.0, 3.0}}, torch::kFloat64);
  EXPECT_TRUE(torch::allclose(base, expected));
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
