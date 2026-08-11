#pragma once

#include <cstdint>
#include <string>

// Host-side tiling math for the dynamic multi-tier AllGather

namespace torch_neuronx {
namespace distributed {
namespace ag_tiling {

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * 1024;
// Tiling tables keyed by (tiling profile, replica-group-size bucket); mirror of the Python
// _TILING_TABLES. A regime = {floor_gathered_bytes, large, medium, small gathered-byte targets},
// pick = first regime whose floor <= gathered bytes. See the Python module for the full contract +
// invariants.
struct Regime {
  int64_t floor;
  int64_t large;
  int64_t medium;
  int64_t small;
};

// A tiling profile is a group-bucketed pair of regime tables. Platforms map to a profile
// (ProfileForPlatform); TRN2 and TRN3 use kDefault today.
// To give a platform its own tiling: add a profile enumerator, its k<Profile>Small/Large
// table pair, and a case in ResolveTable.
enum class TilingProfile { kDefault };

// Small replica groups (size <= kGroupSmallMax).
constexpr Regime kDefaultSmall[] = {
    {32 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},    {16 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},
    {8 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},       {2 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},
    {512 * kKiB, 1 * kMiB, 512 * kKiB, 256 * kKiB}, {0, 256 * kKiB, 128 * kKiB, 64 * kKiB},
};
// Large replica groups (size > kGroupSmallMax).
constexpr Regime kDefaultLarge[] = {
    {64 * kMiB, 64 * kMiB, 32 * kMiB, 16 * kMiB}, {32 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},
    {16 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},   {4 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},
    {1 * kMiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},   {0, 512 * kKiB, 256 * kKiB, 128 * kKiB},
};
constexpr int64_t kGroupSmallMax = 8;

struct TableSpan {
  const Regime* regimes;
  int count;
};

template <int N>
constexpr TableSpan Span(const Regime (&regimes)[N]) {
  return {regimes, N};
}

// Map a Neuron platform target (e.g. "trn2"/"trn3") to a tiling profile.
inline TilingProfile ProfileForPlatform(const std::string& platform) {
  if (platform == "trn2" || platform == "trn3") return TilingProfile::kDefault;
  return TilingProfile::kDefault;  // fallback: unknown platforms use the default profile
}

// Resolve (platform, replica-group size) -> tiling table, via (profile, group bucket).
inline TableSpan ResolveTable(const std::string& platform, int64_t group_size) {
  const bool small = group_size <= kGroupSmallMax;
  switch (ProfileForPlatform(platform)) {
    case TilingProfile::kDefault:
      return small ? Span(kDefaultSmall) : Span(kDefaultLarge);
  }
  return small ? Span(kDefaultSmall) : Span(kDefaultLarge);  // unreachable; silences -Werror
}

// Tiling plan for a runtime numel. All sizes are element counts.
//   large / medium / small : pow2 tile sizes
//   trace_numel   : tile the NEFF is traced at + dispatch-bucket identity; the largest of
//                   {large, medium, small} that is <= numel.
//   unroll        : tiles per dynamic-loop iteration (1/2/4); 0 when ineligible.
//   eligible      : numel >= small (else caller falls back to the native collective).
struct TilingPlan {
  int64_t large = 0;
  int64_t medium = 0;
  int64_t small = 0;
  int64_t trace_numel = 0;
  int64_t unroll = 0;
  bool eligible = false;
};

// Largest power of two <= x (>= 1).
inline int64_t Pow2Floor(int64_t x) {
  if (x < 1) return 1;
  int64_t p = 1;
  while ((p << 1) <= x) p <<= 1;
  return p;
}

// Elements whose gathered size ~= target_bytes, as a power of two (>= nprg). Mirrors
// Python _tile_for.
inline int64_t TileFor(int64_t target_bytes, int64_t ranks, int64_t nprg, int64_t dtype_size) {
  int64_t bpe = dtype_size * ranks;  // gathered bytes per element (across all ranks)
  if (bpe <= 0) return nprg;
  int64_t r = (target_bytes + bpe / 2) / bpe;  // round-nearest
  if (r < 1) r = 1;
  int64_t p = Pow2Floor(r);
  return p < nprg ? nprg : p;
}

// Unroll factor in {1,2,4}: the deepest pow2 that fits the primary tile count
// (extent / trace). Mirrors Python _pick_unroll.
inline int64_t PickUnroll(int64_t extent, int64_t trace) {
  int64_t primary = trace > 0 ? extent / trace : 1;
  if (primary >= 4) return 4;
  if (primary >= 2) return 2;
  return 1;
}

// Mirrors Python plan_dynamic_allgather. Ineligible when numel < small or ranks % nprg != 0.
// (platform, ranks) selects the tiling table.
inline TilingPlan ComputePlan(int64_t numel, int64_t ranks, int64_t nprg, int64_t dtype_size,
                              const std::string& platform) {
  TilingPlan plan;
  if (numel <= 0 || ranks <= 0 || nprg <= 0 || ranks % nprg != 0) return plan;

  int64_t gathered = numel * dtype_size * ranks;
  TableSpan tbl = ResolveTable(platform, ranks);
  const Regime* rg = &tbl.regimes[tbl.count - 1];  // catch-all (floor 0)
  for (int i = 0; i < tbl.count; ++i) {
    if (gathered >= tbl.regimes[i].floor) {
      rg = &tbl.regimes[i];
      break;
    }
  }

  // Non-increasing pow2 tiles, clamped so each is a multiple of the previous (mirrors the
  // Python regime_tiles min() chain).
  int64_t large = TileFor(rg->large, ranks, nprg, dtype_size);
  int64_t medium = TileFor(rg->medium, ranks, nprg, dtype_size);
  if (medium > large) medium = large;
  int64_t small = TileFor(rg->small, ranks, nprg, dtype_size);
  if (small > medium) small = medium;
  plan.large = large;
  plan.medium = medium;
  plan.small = small;

  // trace_numel = largest tile that fits (large -> medium -> small).
  if (numel >= large) {
    plan.trace_numel = large;
    plan.eligible = true;
  } else if (numel >= medium) {
    plan.trace_numel = medium;
    plan.eligible = true;
  } else if (numel >= small) {
    plan.trace_numel = small;
    plan.eligible = true;
  }  // else: below floor -> ineligible
  if (plan.eligible) plan.unroll = PickUnroll(numel, plan.trace_numel);
  return plan;
}

}  // namespace ag_tiling
}  // namespace distributed
}  // namespace torch_neuronx
