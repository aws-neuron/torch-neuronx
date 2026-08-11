#include <gtest/gtest.h>

#include "torch_neuronx/csrc/distributed/DynamicAllGatherTiling.h"

using namespace torch_neuronx::distributed::ag_tiling;

namespace {
constexpr int64_t RANKS = 4, NPRG = 2, DS = 2;
// Small-bucket helper (RANKS=4 <= 8 -> kDefaultSmall).
TilingPlan plan(int64_t numel) { return ComputePlan(numel, RANKS, NPRG, DS, "trn2"); }
// Large-bucket helper (ranks=32 > 8 -> kDefaultLarge).
TilingPlan plan_large(int64_t numel) { return ComputePlan(numel, 32, NPRG, DS, "trn2"); }
}  // namespace

TEST(DynamicAllGatherTilingTest, Pow2Floor) {
  EXPECT_EQ(Pow2Floor(1), 1);
  EXPECT_EQ(Pow2Floor(9), 8);
  EXPECT_EQ(Pow2Floor(2047), 1024);
  EXPECT_EQ(Pow2Floor(2048), 2048);
}

TEST(DynamicAllGatherTilingTest, PickUnrollDeepestFitting) {
  // Deepest pow2 in {1,2,4} fitting primary = extent / trace.
  EXPECT_EQ(PickUnroll(100, 100), 1);
  EXPECT_EQ(PickUnroll(200, 100), 2);
  EXPECT_EQ(PickUnroll(400, 100), 4);
  EXPECT_EQ(PickUnroll(800, 100), 4);
  EXPECT_EQ(PickUnroll(50, 100), 1);  // primary 0 -> floor 1
  EXPECT_EQ(PickUnroll(50, 0), 1);    // trace 0 guard
}

TEST(DynamicAllGatherTilingTest, PlanBehavior) {
  // Behavioral checks unique to ComputePlan. Exhaustive tile-shape invariants (pow2 / nested /
  // descending floors / disjoint traces) are owned by the Python validate_table over every
  // regime of every table, and C++==Python is proven by test_allgather_plan_parity.

  // (1) Multi-rung regime (D): its 2MB gathered floor is below its 4MB large tile, so
  // trace_numel = largest tile <= numel and steps large -> medium as numel shrinks.
  const auto d = plan(600000);  // gathered ~4.6MB -> regime D (small table)
  EXPECT_TRUE(d.eligible);
  const int64_t L = d.large, M = d.medium;
  EXPECT_EQ(plan(L).trace_numel, L);                 // numel >= large_D -> large rung
  if (M < L) EXPECT_EQ(plan(L - 1).trace_numel, M);  // medium <= numel < large -> medium rung

  // (2) Single-rung top regime (A): every numel >= large_A traces at large_A, so that rung is
  // one bucket / NEFF no matter how much bigger numel gets.
  const int64_t LA = plan(1 << 23).large;
  EXPECT_EQ(plan(LA).trace_numel, LA);
  EXPECT_EQ(plan(LA * 4).trace_numel, LA);  // bigger numel, same bucket
  EXPECT_EQ(plan(LA).large, plan(LA * 4).large);

  // (3) Group-size routing: ranks=32 (> kGroupSmallMax) selects kDefaultLarge, whose regime-A
  // large-tile target is 64MB gathered -> 64MiB/(dtype*ranks) elements.
  auto lg = plan_large(1 << 23);  // gathered 512MB -> large-table regime A
  EXPECT_TRUE(lg.eligible);
  EXPECT_EQ(lg.large, 64 * kMiB / (DS * 32));

  // (4) unroll folded into the plan: deepest pow2 in {1,2,4} fitting numel/trace; 0 if ineligible.
  for (int64_t mult : {1, 2, 4, 8}) {
    int64_t expect = mult >= 4 ? 4 : (mult >= 2 ? 2 : 1);
    EXPECT_EQ(plan(LA * mult).unroll, expect) << "mult=" << mult;
  }
  EXPECT_EQ(plan(1).unroll, 0);  // ineligible -> unroll 0
}

TEST(DynamicAllGatherTilingTest, IneligibleInputs) {
  EXPECT_FALSE(ComputePlan(/*numel=*/0, RANKS, NPRG, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, /*ranks=*/0, NPRG, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, RANKS, /*nprg=*/0, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, /*ranks=*/3, /*nprg=*/2, DS, "trn2").eligible);  // ranks%nprg
  auto below = ComputePlan(/*numel=*/1, RANKS, NPRG, DS, "trn2");  // far below the small-tile floor
  EXPECT_FALSE(below.eligible);
  EXPECT_EQ(below.trace_numel, 0);
}
