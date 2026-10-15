/**
 * Unit tests for Int64Policy (per-op int64 decision making).
 *
 * No Python or hardware required — tests the C++ policy lookup directly.
 *
 * The active mode lives in the canonical DispatchUtils.h GetInt64Mode() storage
 * and is overridden in-process here via SetInt64ModeForTesting(). InitInt64Policy
 * only carries the per-op native/unsupported tables (mode is read separately).
 */

#include <gtest/gtest.h>

#include <unordered_set>

#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

const std::unordered_set<std::string> kNativeOps = {
    "aten::add", "aten::sub", "aten::mul", "aten::eq", "aten::sum",
};
const std::unordered_set<std::string> kUnsupportedOps = {
    "aten::mm",
    "aten::matmul",
    "aten::relu",
};

class Int64PolicyTest : public ::testing::Test {
 protected:
  void SetUp() override { InitInt64Policy(kNativeOps, kUnsupportedOps); }
  void TearDown() override { SetInt64ModeForTesting(Int64Mode::kNative); }

  static void SetMode(Int64Mode m) { SetInt64ModeForTesting(m); }
};

// ═══════════════════════════════════════════════════════════════
//  DecideInt64 — per-mode decisions
// ═══════════════════════════════════════════════════════════════

TEST_F(Int64PolicyTest, UnsupportedOpsCpuFallbackInNativeModes) {
  // matmul/conv etc. would produce garbage as S64, so the native dispatch
  // modes route them to CPU. (In packed / cpu_bounce they are downcast to
  // int32 instead — see PackedMode / CpuBounceMode below.)
  for (auto m : {Int64Mode::kNative, Int64Mode::kNativeStrict}) {
    SetMode(m);
    EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::UNSUPPORTED);
    EXPECT_EQ(DecideInt64("aten::matmul"), Int64Decision::UNSUPPORTED);
    EXPECT_EQ(DecideInt64("aten::relu"), Int64Decision::UNSUPPORTED);
  }
}

TEST_F(Int64PolicyTest, NativeMode) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::sum"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::pow"), Int64Decision::DOWNCAST);  // non-native → packed view
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::UNSUPPORTED);
  EXPECT_TRUE(IsS64Passthrough());
  EXPECT_TRUE(ShouldCpuFallback("aten::mm"));
  EXPECT_FALSE(ShouldCpuFallback("aten::add"));
  EXPECT_FALSE(ShouldCpuFallback("aten::pow"));
}

TEST_F(Int64PolicyTest, NativeStrictMode) {
  SetMode(Int64Mode::kNativeStrict);
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::pow"), Int64Decision::UNSUPPORTED);  // CPU fallback
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::UNSUPPORTED);
  EXPECT_TRUE(ShouldCpuFallback("aten::pow"));
  EXPECT_FALSE(ShouldCpuFallback("aten::add"));
  EXPECT_TRUE(IsS64Passthrough());
}

TEST_F(Int64PolicyTest, PackedMode) {
  SetMode(Int64Mode::kPacked);
  // No S64 passthrough — every int64 op is demoted to int32 (packed view),
  // including native-listed ops. Packed mode never forces a whole-op CPU
  // fallback, so even unsupported ops resolve to DOWNCAST here.
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::DOWNCAST);
  EXPECT_EQ(DecideInt64("aten::pow"), Int64Decision::DOWNCAST);
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::DOWNCAST);
  EXPECT_FALSE(IsS64Passthrough());
  EXPECT_FALSE(ShouldCpuFallback("aten::add"));
  EXPECT_FALSE(ShouldCpuFallback("aten::mm"));
}

TEST_F(Int64PolicyTest, CpuBounceMode) {
  SetMode(Int64Mode::kCpuBounce);
  // Same per-op decisions as packed (int64 → int32, on device); the difference
  // is the cast mechanism (CPU roundtrip vs packed view), not the policy.
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::DOWNCAST);
  EXPECT_EQ(DecideInt64("aten::pow"), Int64Decision::DOWNCAST);
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::DOWNCAST);
  EXPECT_FALSE(IsS64Passthrough());
  EXPECT_FALSE(ShouldCpuFallback("aten::add"));
}

TEST_F(Int64PolicyTest, CpuFallbackMode) {
  SetMode(Int64Mode::kCpuFallback);
  // Every int64 op runs entirely on CPU → ShouldCpuFallback is true for all,
  // and DecideInt64 reports UNSUPPORTED for everything.
  EXPECT_TRUE(ShouldCpuFallback("aten::add"));
  EXPECT_TRUE(ShouldCpuFallback("aten::pow"));
  EXPECT_TRUE(ShouldCpuFallback("aten::mm"));
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::UNSUPPORTED);
  EXPECT_EQ(DecideInt64("aten::pow"), Int64Decision::UNSUPPORTED);
  EXPECT_FALSE(IsS64Passthrough());
}

// ═══════════════════════════════════════════════════════════════
//  Overload stripping
// ═══════════════════════════════════════════════════════════════

TEST_F(Int64PolicyTest, OverloadStrippedForNativeOp) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("aten::add.Tensor"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::add.Scalar"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::add.out"), Int64Decision::NATIVE);
}

TEST_F(Int64PolicyTest, OverloadStrippedForUnsupportedOp) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("aten::mm.out"), Int64Decision::UNSUPPORTED);
  EXPECT_EQ(DecideInt64("aten::matmul.out"), Int64Decision::UNSUPPORTED);
}

// ═══════════════════════════════════════════════════════════════
//  Convenience functions
// ═══════════════════════════════════════════════════════════════

TEST_F(Int64PolicyTest, SupportsNativeInt64Convenience) {
  SetMode(Int64Mode::kNative);
  EXPECT_TRUE(SupportsNativeInt64("aten::add"));
  EXPECT_FALSE(SupportsNativeInt64("aten::mm"));
  EXPECT_FALSE(SupportsNativeInt64("aten::unknown"));
}

TEST_F(Int64PolicyTest, DecideInt64ReturnsDowncastForUnknownOps) {
  // Unknown ops in NATIVE mode resolve to DOWNCAST (packed view). Native ops
  // resolve to NATIVE; unsupported ops to UNSUPPORTED.
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("aten::add"), Int64Decision::NATIVE);
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::UNSUPPORTED);
  EXPECT_EQ(DecideInt64("aten::unknown"), Int64Decision::DOWNCAST);
}

// ═══════════════════════════════════════════════════════════════
//  Edge cases
// ═══════════════════════════════════════════════════════════════

TEST_F(Int64PolicyTest, EmptyOpName) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64(""), Int64Decision::DOWNCAST);
}

TEST_F(Int64PolicyTest, NoNamespace) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("add"), Int64Decision::DOWNCAST);
}

TEST_F(Int64PolicyTest, UnsupportedTakesPriorityOverNative) {
  SetMode(Int64Mode::kNative);
  EXPECT_EQ(DecideInt64("aten::mm"), Int64Decision::UNSUPPORTED);
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
