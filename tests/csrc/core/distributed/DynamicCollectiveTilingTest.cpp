#include <gtest/gtest.h>

#include "torch_neuronx/csrc/distributed/DynamicCollectiveTiling.h"

using namespace torch_neuronx::distributed::collective_tiling;

namespace {
constexpr int64_t RANKS = 4, NPRG = 2, DS = 2;
// Small-bucket helper (RANKS=4 <= 8). Platform "trn2" -> kTrn2AgSmall.
TilingPlan allgather_plan(int64_t numel) { return ComputePlan(numel, RANKS, NPRG, DS, "trn2"); }
// Large-bucket helper (ranks=32 > 8). Platform "trn2" -> kTrn2AgLarge.
TilingPlan allgather_plan_large(int64_t numel) { return ComputePlan(numel, 32, NPRG, DS, "trn2"); }
}  // namespace

TEST(DynamicCollectiveTilingTest, Pow2Floor) {
  EXPECT_EQ(Pow2Floor(1), 1);
  EXPECT_EQ(Pow2Floor(9), 8);
  EXPECT_EQ(Pow2Floor(2047), 1024);
  EXPECT_EQ(Pow2Floor(2048), 2048);
}

TEST(DynamicCollectiveTilingTest, PickUnrollDeepestFitting) {
  // Deepest pow2 in {1,2,4} fitting primary = extent / trace.
  EXPECT_EQ(PickUnroll(100, 100), 1);
  EXPECT_EQ(PickUnroll(200, 100), 2);
  EXPECT_EQ(PickUnroll(400, 100), 4);
  EXPECT_EQ(PickUnroll(800, 100), 4);
  EXPECT_EQ(PickUnroll(50, 100), 1);  // primary 0 -> floor 1
  EXPECT_EQ(PickUnroll(50, 0), 1);    // trace 0 guard
}

TEST(DynamicCollectiveTilingTest, AllGatherPlanBehavior) {
  // Behavioral checks unique to ComputePlan on the trn2 profile (3-regime tables). Exhaustive
  // tile-shape invariants (pow2 / nested / descending floors) are owned by the Python
  // validate_table over every regime of every table; the Python parity tests cover C++==Python.

  // (1) Multi-rung regime: at world=4 a ~4.6MB gathered shard lands in trn2 small regime B
  // (1MB floor, below its 4MB gathered large tile), so trace_numel = largest tile <= numel and
  // steps large -> medium as numel shrinks.
  const auto d = allgather_plan(600000);  // gathered ~4.6MB -> trn2 small regime B
  EXPECT_TRUE(d.eligible);
  const int64_t L = d.large, M = d.medium;
  EXPECT_EQ(allgather_plan(L).trace_numel, L);  // numel >= large -> large rung
  if (M < L) {
    // medium <= numel < large -> medium rung
    EXPECT_EQ(allgather_plan(L - 1).trace_numel, M);
  }

  // (2) The top regime's large rung is terminal: every numel >= large_A traces at large_A, so
  // that rung is one bucket / NEFF no matter how much bigger numel gets.
  const int64_t LA = allgather_plan(1 << 23).large;
  EXPECT_EQ(allgather_plan(LA).trace_numel, LA);
  EXPECT_EQ(allgather_plan(LA * 4).trace_numel, LA);  // bigger numel, same bucket
  EXPECT_EQ(allgather_plan(LA).large, allgather_plan(LA * 4).large);

  // (3) Group-size routing: ranks=32 (> kGroupSmallMax) selects the large-group table
  // (trn2 -> kTrn2AgLarge), whose regime-A large-tile target is 64MB gathered ->
  // 64MiB/(dtype*ranks) elements.
  auto lg = allgather_plan_large(1 << 23);  // gathered 512MB -> large-table regime A
  EXPECT_TRUE(lg.eligible);
  EXPECT_EQ(lg.large, 64 * kMiB / (DS * 32));

  // (4) unroll folded into the plan: deepest pow2 in {1,2,4} fitting numel/trace; 0 if ineligible.
  for (int64_t mult : {1, 2, 4, 8}) {
    int64_t expect = mult >= 4 ? 4 : (mult >= 2 ? 2 : 1);
    EXPECT_EQ(allgather_plan(LA * mult).unroll, expect) << "mult=" << mult;
  }
  EXPECT_EQ(allgather_plan(1).unroll, 0);  // ineligible -> unroll 0
}

TEST(DynamicCollectiveTilingTest, AllReducePlanBehavior) {
  // AllReduce selects its per-rank-byte table rather than the AllGather gathered-byte table.
  constexpr int64_t kLargeRanks = 32;
  constexpr int64_t kLargeNumel = int64_t{1} << 26;
  const auto ar = ComputePlan(kLargeNumel, kLargeRanks, NPRG, DS, "trn2", Collective::kAllReduce);
  EXPECT_TRUE(ar.eligible);
  EXPECT_EQ(ar.large, 16 * kMiB / DS);
  EXPECT_NE(ar.large, allgather_plan_large(kLargeNumel).large);
}

TEST(DynamicCollectiveTilingTest, IneligibleInputs) {
  EXPECT_FALSE(ComputePlan(/*numel=*/0, RANKS, NPRG, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, /*ranks=*/0, NPRG, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, RANKS, /*nprg=*/0, DS, "trn2").eligible);
  EXPECT_FALSE(ComputePlan(1 << 23, /*ranks=*/3, /*nprg=*/2, DS, "trn2").eligible);  // ranks%nprg
  auto below = ComputePlan(/*numel=*/1, RANKS, NPRG, DS, "trn2");  // far below the small-tile floor
  EXPECT_FALSE(below.eligible);
  EXPECT_EQ(below.trace_numel, 0);
}
