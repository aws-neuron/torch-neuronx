#pragma once

#include <cstdint>
#include <string>

// Host-side tiling math for the dynamic multi-tier collectives

namespace torch_neuronx {
namespace distributed {
namespace collective_tiling {

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * 1024;
// Tiling tables keyed by (tiling profile, collective, replica-group-size bucket); mirror of
// the Python _TILING_TABLES. A regime contains a message-byte floor and large/medium/small
// byte targets; the first regime whose floor is <= the collective message size is selected.
// See the Python module for the full contract and invariants.
struct Regime {
  int64_t floor;
  int64_t large;
  int64_t medium;
  int64_t small;
};

// A tiling profile contains collective-specific small/large regime tables. Platforms map to a
// profile through ProfileForPlatform.
// To give a platform its own tiling, add a profile enumerator, its AllGather/AllReduce
// small/large table pairs, and a case in ResolveTable.
enum class TilingProfile { kDefault, kTrn2, kTrn3 };

// Which collective's tiling to use. All-gather counts gathered bytes; all-reduce counts per-rank
// bytes (see MsgBytesPerElem). Mirror of the Python collective key.
enum class Collective { kAllGather, kAllReduce };
inline bool PerRank(Collective collective) { return collective == Collective::kAllReduce; }

// Default AllGather table for small replica groups (size <= kGroupSmallMax).
constexpr Regime kDefaultAgSmall[] = {
    {32 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},    {16 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},
    {8 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},       {2 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},
    {512 * kKiB, 1 * kMiB, 512 * kKiB, 256 * kKiB}, {0, 256 * kKiB, 128 * kKiB, 64 * kKiB},
};
// Default AllGather table for large replica groups (size > kGroupSmallMax).
constexpr Regime kDefaultAgLarge[] = {
    {64 * kMiB, 64 * kMiB, 32 * kMiB, 16 * kMiB}, {32 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},
    {16 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},   {4 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},
    {1 * kMiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},   {0, 512 * kKiB, 256 * kKiB, 128 * kKiB},
};
// All-reduce tables: PER-RANK message-byte targets (all-reduce output == input size), so the
// element tile = target/dtype (world-independent). Mirror of the Python _DEFAULT_AR_* tables.
constexpr Regime kDefaultArSmall[] = {
    {8 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},        {4 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},
    {2 * kMiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},      {512 * kKiB, 1 * kMiB, 512 * kKiB, 256 * kKiB},
    {128 * kKiB, 256 * kKiB, 128 * kKiB, 64 * kKiB}, {0, 64 * kKiB, 32 * kKiB, 16 * kKiB},
};
constexpr Regime kDefaultArLarge[] = {
    {16 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},       {8 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},
    {4 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},         {1 * kMiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},
    {256 * kKiB, 512 * kKiB, 256 * kKiB, 128 * kKiB}, {0, 128 * kKiB, 64 * kKiB, 32 * kKiB},
};
// ── TRN2-tuned tables (3-regime design) ───────────────────────────────────────
constexpr Regime kTrn2AgSmall[] = {
    {8 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},
    {1 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},
    {0, 512 * kKiB, 256 * kKiB, 128 * kKiB},
};
constexpr Regime kTrn2AgLarge[] = {
    {64 * kMiB, 64 * kMiB, 32 * kMiB, 16 * kMiB},
    {8 * kMiB, 16 * kMiB, 8 * kMiB, 8 * kMiB},
    {0, 2 * kMiB, 1 * kMiB, 512 * kKiB},
};
// ── TRN3-tuned tables (3-regime design) ───────────────────────────────────────
constexpr Regime kTrn3AgSmall[] = {
    {8 * kMiB, 32 * kMiB, 16 * kMiB, 8 * kMiB},
    {1 * kMiB, 4 * kMiB, 2 * kMiB, 1 * kMiB},
    {0, 512 * kKiB, 256 * kKiB, 128 * kKiB},
};
constexpr Regime kTrn3AgLarge[] = {
    {16 * kMiB, 64 * kMiB, 32 * kMiB, 16 * kMiB},
    {4 * kMiB, 8 * kMiB, 8 * kMiB, 4 * kMiB},
    {0, 2 * kMiB, 1 * kMiB, 512 * kKiB},
};
constexpr Regime kTrn3ArSmall[] = {
    {2 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},
    {256 * kKiB, 1 * kMiB, 512 * kKiB, 256 * kKiB},
    {0, 128 * kKiB, 32 * kKiB, 16 * kKiB},
};
constexpr Regime kTrn3ArLarge[] = {
    {4 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},
    {512 * kKiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},
    {0, 128 * kKiB, 64 * kKiB, 32 * kKiB},
};
constexpr Regime kTrn2ArSmall[] = {
    {2 * kMiB, 8 * kMiB, 4 * kMiB, 2 * kMiB},
    {256 * kKiB, 1 * kMiB, 512 * kKiB, 256 * kKiB},
    {0, 128 * kKiB, 32 * kKiB, 16 * kKiB},
};
constexpr Regime kTrn2ArLarge[] = {
    {4 * kMiB, 16 * kMiB, 8 * kMiB, 4 * kMiB},
    {512 * kKiB, 2 * kMiB, 1 * kMiB, 512 * kKiB},
    {0, 128 * kKiB, 64 * kKiB, 32 * kKiB},
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
  if (platform == "trn3") return TilingProfile::kTrn3;
  if (platform == "trn2") return TilingProfile::kTrn2;
  return TilingProfile::kDefault;
}

// Resolve (platform, replica-group size, collective) -> tiling table, via (profile, collective,
// group bucket).
inline TableSpan ResolveTable(const std::string& platform, int64_t group_size,
                              Collective collective = Collective::kAllGather) {
  const bool small = group_size <= kGroupSmallMax;
  switch (ProfileForPlatform(platform)) {
    case TilingProfile::kDefault:
      if (collective == Collective::kAllReduce)
        return small ? Span(kDefaultArSmall) : Span(kDefaultArLarge);
      return small ? Span(kDefaultAgSmall) : Span(kDefaultAgLarge);
    case TilingProfile::kTrn2:
      if (collective == Collective::kAllReduce)
        return small ? Span(kTrn2ArSmall) : Span(kTrn2ArLarge);
      return small ? Span(kTrn2AgSmall) : Span(kTrn2AgLarge);
    case TilingProfile::kTrn3:
      if (collective == Collective::kAllReduce)
        return small ? Span(kTrn3ArSmall) : Span(kTrn3ArLarge);
      return small ? Span(kTrn3AgSmall) : Span(kTrn3AgLarge);
  }
  return small ? Span(kDefaultAgSmall) : Span(kDefaultAgLarge);  // unreachable; silences -Werror
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

// Bytes of collective traffic per logical element -- the divisor that turns a regime's byte
// target into an element tile. All-gather counts GATHERED bytes across all ranks
// (dtype * world); all-reduce counts PER-RANK bytes (dtype). Mirrors Python _msg_bytes_per_elem.
inline int64_t MsgBytesPerElem(int64_t ranks, int64_t dtype_size, bool per_rank) {
  return dtype_size * (per_rank ? 1 : ranks);
}

// Elements whose message size ~= target_bytes, as a power of two (>= nprg). `per_rank` selects
// the byte accounting (see MsgBytesPerElem). Mirrors Python _tile_for.
inline int64_t TileFor(int64_t target_bytes, int64_t ranks, int64_t nprg, int64_t dtype_size,
                       bool per_rank = false) {
  int64_t bpe = MsgBytesPerElem(ranks, dtype_size, per_rank);
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

// Three non-increasing pow2 element tiles for a matched regime.
struct RegimeTiles {
  int64_t large;
  int64_t medium;
  int64_t small;
};

// Shared tiling core (mirrors Python compute_tiling): walk `tbl` (regimes ordered by DESCENDING
// floor, last floor == 0) by this shard's message bytes, then convert the matched regime's byte
// targets into three non-increasing pow2 element tiles (each clamped to a multiple of the
// previous). All-gather passes its gathered-byte table with per_rank=false; all-reduce passes
// its per-rank-byte table with per_rank=true.
inline RegimeTiles ComputeTiling(int64_t numel, int64_t ranks, int64_t nprg, int64_t dtype_size,
                                 TableSpan tbl, bool per_rank) {
  int64_t message = numel * MsgBytesPerElem(ranks, dtype_size, per_rank);
  const Regime* rg = &tbl.regimes[tbl.count - 1];  // catch-all (floor 0)
  for (int i = 0; i < tbl.count; ++i) {
    if (message >= tbl.regimes[i].floor) {
      rg = &tbl.regimes[i];
      break;
    }
  }
  int64_t large = TileFor(rg->large, ranks, nprg, dtype_size, per_rank);
  int64_t medium = TileFor(rg->medium, ranks, nprg, dtype_size, per_rank);
  if (medium > large) medium = large;
  int64_t small = TileFor(rg->small, ranks, nprg, dtype_size, per_rank);
  if (small > medium) small = medium;
  return {large, medium, small};
}

// Mirrors Python _plan_dynamic. Ineligible when numel < small or ranks % nprg != 0.
// (platform, ranks, collective) selects the tiling table + byte accounting: all-gather counts
// gathered bytes (per_rank=false), all-reduce counts per-rank bytes (per_rank=true).
inline TilingPlan ComputePlan(int64_t numel, int64_t ranks, int64_t nprg, int64_t dtype_size,
                              const std::string& platform,
                              Collective collective = Collective::kAllGather) {
  TilingPlan plan;
  if (numel <= 0 || ranks <= 0 || nprg <= 0 || ranks % nprg != 0) return plan;

  RegimeTiles t = ComputeTiling(numel, ranks, nprg, dtype_size,
                                ResolveTable(platform, ranks, collective), PerRank(collective));
  plan.large = t.large;
  plan.medium = t.medium;
  plan.small = t.small;

  // trace_numel = largest tile that fits (large -> medium -> small).
  if (numel >= t.large) {
    plan.trace_numel = t.large;
    plan.eligible = true;
  } else if (numel >= t.medium) {
    plan.trace_numel = t.medium;
    plan.eligible = true;
  } else if (numel >= t.small) {
    plan.trace_numel = t.small;
    plan.eligible = true;
  }  // else: below floor -> ineligible
  if (plan.eligible) plan.unroll = PickUnroll(numel, plan.trace_numel);
  return plan;
}

}  // namespace collective_tiling
}  // namespace distributed
}  // namespace torch_neuronx
