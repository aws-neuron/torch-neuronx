# Dynamic NKI Kernel Optimization Tips

Accumulated patterns and techniques for optimizing dynamic-shape NKI kernels
(both aten ops and collectives).
Reference: `/workspace/private-nki-staging` for NKI API capabilities.

---

## 1. DMA Descriptor Generation (DGE) modes

**Finding (from dynamic all_gather optimization):** Explicitly setting
`dge_mode=nisa.dge_mode.hwdge` on `nisa.dma_copy` calls with indirect offsets
(`scalar_offset=..., indirect_dim=0`) reduces per-DMA setup time by 2-8%.

**When to apply:** Any `dma_copy` that uses a runtime-computed scalar offset (i.e.,
the offset isn't known at compile time). These are common in dynamic kernels where the
tile offset depends on the runtime element count.

**Modes available:**
- `nisa.dge_mode.none` — runtime pre-generates descriptors (default for static patterns)
- `nisa.dge_mode.swdge` — GpSimd engine generates descriptors at execution time
- `nisa.dge_mode.hwdge` — dedicated hardware block generates descriptors (fastest for
  indirect-offset DMAs)

**Example:**
```python
# Before (auto-selected DGE):
nisa.dma_copy(dst=out.ap(pattern, scalar_offset=off, indirect_dim=0), src=tile)

# After (explicit hwdge):
nisa.dma_copy(dst=out.ap(pattern, scalar_offset=off, indirect_dim=0), src=tile,
              dge_mode=nisa.dge_mode.hwdge)
```

---

## 2. Balanced flat tiling (from stride2_gather)

**Pattern:** Reshape the problem as a flat element stream and tile it as `[P_MAX, F_SIZE]`
blocks (P_MAX=128 partition dimension, F_SIZE=free dimension tile width), regardless of the
logical tensor shape. This ensures the partition dimension is always full (128) for the bulk
of any tensor, fixing shape-dependent under-utilization.

**When to apply:** Elementwise or reduction kernels where the logical tensor shape causes
partition-dimension under-fill (e.g., a `(1048576, 1)` tensor fills only 1 partition lane).

**Key technique:** Use `.ap()` access patterns to express flat tiling:
```python
# Contiguous burst-load [P_MAX, F_SIZE] block from flat buffer
tile = nl.ndarray((P_MAX, F_SIZE), dtype=..., buffer=nl.sbuf)
nisa.dma_copy(dst=tile, src=flat_buf.ap([[F_SIZE, P_MAX], [1, F_SIZE]], offset))
```

---

## 3. On-chip rearrangement via Vector Engine

**Pattern:** Do data rearrangement (transpose, lane select, scatter) **on-chip** in SBUF using
the Vector Engine (`nisa.tensor_copy` with `engine=nisa.engine.vector`), while DMA engines
handle straight burst loads/stores to HBM. The Vector Engine works in parallel with DMA.

**When to apply:** Kernels that need to reorder elements between input and output layouts
(e.g., stride-2 gather, transpose, channel reordering).

**Key advantage:** The Vector Engine's on-chip copy is essentially free (overlapped with DMA),
unlike HBM→HBM copies that consume DMA bandwidth.

```python
# On-chip strided read + contiguous write (parallel with DMA bursts)
nisa.tensor_copy(dst=contiguous_tile, src=strided_tile[:, ::2], engine=nisa.engine.vector)
```

---

## 4. Avoid unnecessary HBM round-trips (bounce buffers)

**Anti-pattern:** Load data from HBM → SBUF → process → write to shared_hbm → DMA copy
shared_hbm → final output. The last DMA copy is an extra HBM round-trip.

**Better:** Write the processed result directly to the final output location if possible.
Use output tensors allocated with `nl.shared_hbm` and write them with the correct access
pattern to land data in the right place.

**Dynamic kernel specific:** The dynamic kernels often stage through shared_hbm buffers
because the runtime offset isn't known. If the output AP pattern can encode the dynamic
offset directly, eliminate the staging buffer.

---

## 5. Minimize instruction count in dynamic loops — unrolling

**Finding (from tiling loop + exp optimization):** `nl.fori_loop` has per-iteration overhead
from the sequencer (register compare + branch + back-edge fence ~7µs). For tile counts of
1-2, the loop overhead can dominate. For elementwise ops at 2048x4096, loop overhead was 27%
of device time before unrolling.

**Techniques:**
- Unroll the loop body to process N tiles per iteration. Each slot `k` addresses
  `cursor + k*tile_size_m` via its own scalar offset. Split the trip count with
  `n >> log2(N)` for the bulk, `n & (N-1)` for the remainder (1 tile/iter).
- **Optimal unroll factor for elementwise ops:** 4–8x depending on SBUF budget.
  Going from 1→2→4 each gave 20-30% improvement; 4→8 gives another 10-20% on
  high-M shapes. Diminishing returns beyond 8.
- **SBUF budget constraint:** Each unrolled slot needs its own in_tile + out_tile in SBUF.
  Compute `unroll * 2 * h_segments * tile_size_h * dtype_bytes` and keep it under ~128 KB
  per partition (leaving headroom for other allocations). Adapt unroll at trace time.
- **Multi-stage decomposition (for unroll ≥ 8):** Instead of one bulk + one remainder loop,
  use a cascade: 8-wide bulk → 4-wide stage (1 trip for bit 2) → 1-wide remainder for
  bits 0-1. Each extra fori_loop entry costs a fence (~7µs) even at zero iterations, so a
  2-wide level does not pay off — stop stages at width 4.
- Minimize SBUF scalar operations inside the loop body (tensor_scalar, tensor_tensor for offset
  arithmetic) — these are sequencer instructions that can't overlap with DMA/compute.

---

## 6. Software pipelining — when it does NOT help

**Finding (from dynamic all_gather iteration 1):** Manually reordering operations to overlap
DMA load of tile N+1 with CC/scatter of tile N produced a 13% regression.

**Root cause:** The NeuronCore compiler's instruction scheduler already handles DMA/compute
overlap effectively within `nl.fori_loop`. Manual reordering:
- Adds extra instruction overhead (prefetch offset arithmetic, buffer swap copies).
- May create WAW dependencies that the hardware serializes conservatively.
- Restricts the compiler's optimization space.

**When it MIGHT help:** Cross-tier overlap (between two separate fori_loops) where the compiler
inserts a barrier between them. But this requires fusing the tiers into a single static scope,
which is complex.

**Rule of thumb:** Trust the compiler's scheduler for intra-loop overlap. Focus on reducing the
number of instructions and DMA operations, not on reordering them.

---

## 7. Tile size and partition dimension utilization

**Pattern:** For dynamic kernels, the trace floor (the tile size the NEFF is compiled at)
determines hardware utilization. If `tile_size < 128` elements per partition, some partition
lanes are idle.

**Check:** For elementwise ops, ensure the tile's partition dimension is P_MAX=128 for the
bulk. The free dimension can vary. Use the `[P_MAX, free_tile]` layout.

**For matmul kernels:** The M-dimension (dynamic) maps to partition. If M < 128, partition
utilization drops. Consider padding M to 128 for the traced NEFF (the kernel already handles
M < traced_M via masking).

---

## 8. NKI data types and casting

**Available dtypes:** bf16, fp16, fp32, int8, int16, int32, uint8, uint16, uint32.
No float64 or int64 on device.

**Casting costs:** dtype conversion (e.g., bf16→fp32 for accumulation) uses the Scalar Engine.
If the kernel does an unnecessary cast (e.g., loading bf16, casting to fp32, doing elementwise,
casting back), it adds latency. Check if the operation can stay in the native dtype.

---

## 9. Access patterns for non-contiguous data

**`nisa.dma_copy` with `.ap()` patterns:** Supports multi-dimensional strided access up to
4D. For non-contiguous output (e.g., rank-major scatter), the pattern encodes the stride.

**Key constraint:** Strides in `.ap()` must be compile-time constants (except for `scalar_offset`
which can be runtime/indirect). This means inter-rank strides of `numel` (runtime) cannot be
expressed as a single multi-rank DMA — each rank needs a separate DMA with its own offset.

---

## 10. Profile-guided optimization workflow

1. **Measure both modes** (static + dynamic) on the same shape.
2. **Compare device_time** (not e2e, which includes host overhead).
3. **If device_ratio > 1.2:** the dynamic kernel has significant optimization headroom.
4. **Parse profiles** to break down time by engine:
   - DMA time high → reduce DMA count or data movement
   - Compute time high → optimize the arithmetic (tiling, engine selection)
   - Stall/idle time high → instruction scheduling issue (rare — compiler handles well)
5. **Target the dominant cost first.** Don't micro-optimize a 5% contributor.

---

## 11. Common dynamic kernel overhead sources

| Overhead | Cause | Mitigation |
|----------|-------|------------|
| DMA descriptor setup | Per-DMA fixed cost with indirect offsets | Use hwdge mode |
| Loop sequencer | fori_loop compare+branch+fence per iteration | Increase unroll (4-8x) |
| Offset arithmetic | tensor_scalar/tensor_tensor for indices | Minimize per-iter ops |
| Over-gather (tail) | Last tile gathers more than needed | Tiling table tuning |
| SBUF staging | Load→SBUF→HBM→output double-write | Direct output write |
| Partition underutil | Tile < 128 elements per partition | Balanced flat tiling |
| HBM bandwidth waste | All cores read full tensor at LNC=2 | SPMD hidden-dim split |
| Small free-dim tiles | Many H-segments = many DMAs per iteration | Increase tile_size_h |
| Interleaved ordering | Load-compute-store per slot stalls pipeline | Phase-batch the body |

---

## 12. Testing and verification

- Always run `uv run ./tools/build` after kernel changes.
- Use `python -m pytest tests/vertical/test_dynamic_aten_ops.py -k "<op>" -v -s` to measure.
- The test checks **correctness** (output matches reference within tolerance) AND **performance**.
- The CSV output has both `e2e_time_us` (includes compilation on first run) and `device_time_us`
  (pure on-device execution). Use `device_time_us` for optimization comparison.
- Profile artifacts are saved to `profile_rel_path` in the CSV row.

---

## 13. SPMD hidden-dim splitting (LNC parallelism)

**Finding (from exp optimization comparison):** Splitting the free dimension (H) across
`nl.num_programs` (SPMD cores) halves HBM traffic per core at LNC=2, cutting device time
by 30-50% for large-H shapes. This is one of the highest-impact optimizations available.

**When to apply:** Elementwise and unary kernels where H is large (≥2048) and divisible by
`num_programs`. The M-dimension (dynamic) cannot be split because that would desymmetrize
the per-core basic-block counts (required by NCC_IXGM002).

**Pattern:**
```python
num_progs = nl.num_programs(0)
prog_id = nl.program_id(0)
if hidden % num_progs == 0:
    h_span = hidden // num_progs
    h_base = prog_id * h_span
else:
    h_span = hidden  # fallback: each core does full H
    h_base = 0

# Tile only within [h_base, h_base + h_span)
```

**Key constraints:**
- Both programs MUST emit the same basic-block structure (same instruction counts) for
  NCC symmetry. Splitting H evenly guarantees this. If H % num_progs != 0, fall back to
  each core processing the full H.
- The output tensor must still be `[m_static, hidden]` — each core writes its slice using
  the appropriate h_base offset in the AP pattern.
- This is a pure bandwidth optimization: each core touches H/2 data, so aggregate DMA time
  drops proportionally.

**Impact observed:** On H=12288, SPMD splitting brought device_ratio from 2.4x → 1.5x
(vs 2.4x → 2.2x from unroll-only optimization without SPMD). The larger H is, the more
DMA-bound the kernel becomes, and the more SPMD splitting helps.

---

## 14. Phase-batched loop bodies vs interleaved

**Finding (from exp optimization comparison):** Ordering the unrolled loop body by *phase*
(all loads first, then all activations, then all stores) outperforms *interleaved* ordering
(load→compute→store per slot, then next slot) by 10-20% on high-M shapes.

**Why it helps:**
- Phase batching allows the DMA engine to stay busy with back-to-back loads without
  waiting for compute to finish on each tile individually.
- The compiler can see all loads as a group and schedule them in parallel across DMA
  queues, filling the DMA pipeline.
- All activations are then grouped, keeping the Activation Engine busy.
- All stores are grouped at the end, allowing the DMA engine to burst-write.

**When it does NOT help (per tip #6):** When the compiler's scheduler already achieves good
overlap within a small body (1-2 tiles). Phase batching matters most at higher unroll factors
(4-8) where there are enough operations to pipeline.

**Caveat:** Phase batching keeps all input tiles live in SBUF simultaneously until the
activation phase completes. This increases peak SBUF pressure: `unroll * h_segments *
tile_size_h * dtype_bytes * 2` (inputs + outputs). Factor this into the SBUF budget when
choosing the unroll factor.

```python
# Phase-batched (preferred for unroll >= 4):
def body(_i):
    in_tiles = []
    for slot in range(unroll):
        for seg in range(n_segs):
            in_tiles[slot][seg] = load(...)

    out_tiles = []
    for slot in range(unroll):
        for seg in range(n_segs):
            out_tiles[slot][seg] = activate(in_tiles[slot][seg])

    for slot in range(unroll):
        for seg in range(n_segs):
            store(out_tiles[slot][seg])
```

---

## 15. Free-dim tile width selection

**Finding (from exp optimization):** The free-dim tile width (`tile_size_h`) is a key knob
that trades DMA count against per-DMA throughput. The original exp kernel used 512; optimal
is 2048-4096 for bf16 depending on whether SPMD splitting is used.

**Guidelines:**
- **Without SPMD split:** Use `tile_size_h = min(hidden, 4096)` for bf16/fp16. This gives
  8KB per partition per tile, well past the DMA saturation knee (~4KB) while keeping H-segments
  minimal. For H=4096, this means 1 segment; for H=12288, 3 segments.
- **With SPMD split:** Use `tile_size_h = min(h_span, 4096 // dtype_bytes)` = 2048 for bf16.
  Since each core only handles H/2, the effective hidden dim is smaller and 2048 gives 1-2
  segments.
- **The saturation knee:** DMA throughput saturates at ~4KB per partition per burst. Below
  this, per-DMA fixed costs dominate. Above it, returns diminish. The sweet spot is 2048-4096
  bf16 elements per partition (4-8 KB).
- **SBUF constraint:** Each tile is `128 * tile_size_h * dtype_bytes`. At tile_size_h=4096,
  that's 1MB per tile total (8KB per partition). With unroll=4 and 2 tiles per slot (in+out),
  that's 8 tiles × 8KB = 64KB per partition — well within the 224KB budget.

---

## 16. Inherent dynamic kernel overhead floor (~2x)

**Finding (from exp optimization — 4 iterations):** After applying all available optimizations
(hwdge, wider tiles, 4x unroll), the device_ratio floor is approximately 1.96x-2.84x for the
exp kernel. This ~2x overhead is structural:

| Source | Reason | Static equivalent |
|--------|--------|-------------------|
| fori_loop sequencer | Compare + branch + back-edge fence per trip | Zero (flat schedule) |
| Per-DMA descriptor setup | N separate DMAs with hwdge vs 1-2 large DMAs | 1-2 DMA commands |
| Offset arithmetic | tensor_scalar per slot per trip | None (offsets baked in) |
| Instruction encoding | Indirect-offset DMAs are longer instructions | Direct-address DMAs |

The static kernel has the shape baked in at compile time and emits a completely flat instruction
schedule with no loops. The dynamic kernel's flexibility inherently costs ~2x.

**When to stop optimizing:** If device_ratio ≤ 2.0 for most shapes, you've reached the
practical floor. Further gains require fundamentally different approaches (e.g., multi-NEFF
bucketing where different compiled NEFFs serve different shape ranges).

---

## 17. Dynamic collective tiling tables — structure and constraints

**Context:** Dynamic all_gather/all_reduce use multi-tier tiling tables to decompose a runtime
`numel` into (large, medium, small) tiles plus a tail. The tables live in BOTH Python
(`dynamic_collectives.py`) and C++ (`DynamicAllGatherTiling.h`) and must stay in lock-step.

**Table structure:**
```python
# Each regime: (name, floor_gathered_bytes, (large_bytes, medium_bytes, small_bytes))
_DEFAULT_LARGE = [
    ("A", 16*1024*1024, (4*1024*1024, 2*1024*1024, 1*1024*1024)),
    ("B",  4*1024*1024, (2*1024*1024, 1*1024*1024,   512*1024)),
    ...
    ("F",            0, (  128*1024,    64*1024,     32*1024)),  # catch-all
]
```

**Invariants (pre-check before every build):**
- Tiles: powers of two, non-increasing, nested (`large % medium == 0`, `medium % small == 0`)
- Floors: strictly descending; last == 0
- Never flip a dynamically-routed shape to native-fallback

**Key insight:** The element tile = `target_bytes / (dtype_size * world)`, floored to a power
of two, `>= n_prgs`. Larger tiles → fewer iterations → less per-DMA overhead, but each tile
carries more data through the bounce buffer.

**Regime restructuring is in scope — not just tuning tile sizes within existing regimes:**
- Add/remove regimes to better match the actual workload's message-size distribution.
- Move floor boundaries to align with natural breakpoints in the workload shapes.
- Split a regime that covers too wide a message-size range.
- The goal: each message size should fall in a regime where its tile decomposition yields
  few tiles with minimal over-gather. Parse the measured CSV to find shapes with poor
  decompositions and evaluate whether different boundaries would help.

---

## 18. Single-tile rows — bounce-bound overhead (no tile headroom)

**Finding (from dynamic all_gather tuning):** When `numel == large_tile` for a regime, the
kernel runs exactly 1 tile with no loop iterations. The entire dynamic/torch gap is from the
**HBM bounce**: load → collective → scatter ≈ 2x data movement + per-tile setup.

**Rule:** Never try to tune single-tile rows by shrinking tiles — that only ADDS tiles and
makes things slower. These rows are bounce-bound and require kernel-logic changes (e.g.,
eliminating the staging buffer, direct DGE to output) to improve.

**How to detect:** From the CSV, check `n_large + n_medium + n_small + has_tail <= 1`.
If true, the row is single-tile.

---

## 19. Multi-tier tile decomposition and unroll interaction

**Context:** The dynamic collective kernel decomposes runtime `numel` greedily:
`n_large` large tiles → `n_medium` medium tiles → `n_small` small tiles → 1 tail.
Each tier runs its own `fori_loop`. The unroll factor applies per-tier: `n >> log2(unroll)`
bulk iterations + remainder.

**Tuning interactions:**
- Enlarging a tier's tile reduces that tier's iteration count (fewer DMAs, less fence overhead).
- But if the enlarged tile overshoots `numel`, more data ends up in the "over-gather" tail.
- When adjusting one regime, check that neighboring shapes (one regime boundary away) don't
  regress — the regime switch point may flip them into a worse decomposition.
- The unroll factor is selected by `_pick_unroll(numel, trace_numel)` — if `numel/trace < 2`,
  unroll stays at 1. To enable unrolling, the trace floor must be small enough that most runtime
  numels produce ≥4 tiles.

---

## 20. Per-rank scatter overhead in dynamic collectives

**Finding (from all_gather profiling):** The dynamic kernel issues 56 DMA transfers vs torch's
4 for the same data volume. Root cause: after `ncc.all_gather`, the kernel scatters each rank's
tile to its output position with a separate DMA (one per rank per tile). With 16 ranks × ~4
tiles, that's 64 DMAs.

**Potential mitigations (require NKI FW grounding):**
- Coalesce adjacent rank outputs when they map to contiguous HBM regions.
- Use `.ap()` with a wider multi-tile pattern to scatter multiple tiles in one DMA.
- Write directly to the output tensor during the collective (eliminate the staging buffer).

These are kernel-logic changes and must be validated against:
- `/workspace/private-nki-staging/python/nki/collectives/_ops.py` — all_gather semantics
- The dynamic-loop contract (one NEFF, no per-shape recompile)

---

## 21. Platform-scoped tiling (trn2 vs trn3)

**Context:** Tiling tables are keyed by platform profile. Today trn2 and trn3 share the
`"default"` profile. Optimal tile sizes differ across platforms (different DMA bandwidth,
collective hardware, SBUF sizes).

**Rule:** Before editing shared tables, FORK a dedicated profile for the measured platform:
1. Python: add `_TRN2_SMALL`/`_TRN2_LARGE` = copies of `_DEFAULT_*`; register in
   `_TILING_TABLES`; change `_profile_for_platform("trn2") -> "trn2"`.
2. C++: add `TilingProfile::kTrn2`, `kTrn2Small`/`kTrn2Large`, `ProfileForPlatform` branch,
   and `ResolveTable` case.
3. Rebuild + run parity test (sweeps both platforms).

This freezes the other platform while allowing measurement-driven tuning on the target.


---

## 22. Weight-resident loop restructuring (loop-invariant operand hoisting)

**Finding (from dynamic_m_matmul optimization):** When a dynamic fori_loop body reads a
loop-invariant operand from HBM (e.g., rhs/weight matrix in matmul), it re-reads it every
iteration because the compiler cannot hoist loads out of fori_loop. For M=2048 with 16
m-tiles, this creates 16× read amplification of the weight (~256 MB for a 16 MB weight).

**The correct restructuring pattern:**

```python
# WRONG: one fori_loop wrapping everything — rhs reloaded per m-tile
def body(_i):
    for n_block:
        for k_block:
            load rhs[k_block, n_block]  # <-- reloaded every iteration!
            load lhs[m_tile, k_block]
            matmul
nl.fori_loop(0, num_m_tiles, body)

# WRONG: k_block outside fori_loop — coherence bug (cross-loop RAW dependency)
for n_block:
    for k_block:
        load rhs once
        fori_loop(m_tiles): ...write partial to HBM...
        # Next k_block reads same HBM location — no fence between fori_loops!

# CORRECT: n_block outside, k_block INSIDE body (no cross-loop dependency)
for n_block in range(num_block_n):
    # Load rhs for ALL k_blocks of this n_block ONCE, into SBUF
    for tk in range(num_tile_k):
        dma_copy(rhs_resident[tk], rhs[tk, n_block])  # outside fori_loop

    m_tile_start = ...  # per-n_block offset register
    def body(_i):
        result_accum = 0  # fp32 accumulation in SBUF
        for k_block:  # sequential, INSIDE body — fp32 stays in SBUF
            load lhs[m_tile, k_block]
            matmul(lhsT, rhs_resident[k_block], result_accum)
        store result_accum to res[m_tile, n_block]
        advance m_tile_start
    fori_loop(0, num_m_tiles_reg, body)
```

**Key rules:**
1. **Only hoist dimensions that write to non-overlapping output regions.** n_blocks write to
   disjoint column ranges of the output → safe to have separate fori_loops per n_block.
   k_blocks accumulate into the SAME output region → MUST stay inside the same fori_loop body.
2. **Each fori_loop needs its own m_tile_start register and trip-count register** — allocated
   at the top level, one per n_block.
3. **The invariant operand (rhs) lives in SBUF between the pre-load and the fori_loop.** Size
   the allocation to fit: `num_tile_k * block_width * dtype_bytes` per partition. If it doesn't
   fit, shrink `block_width` (see tip #23).
4. **Don't use scalar_offset/hwdge on the invariant load** — it's a static-offset DMA outside
   the loop; the compiler picks the optimal pre-generated descriptor mode automatically.

**Impact:** Eliminates M-proportional read amplification of the invariant operand. For a
2048×4096 weight with M=2048 (16 tiles): goes from 256 MB reads → 16 MB reads per core.
Combined with LNC sharding (tip #13), total rhs reads drop by ~8× (halved columns × no reloads).

---

## 23. Adaptive SBUF-fitted tiling (block_size_n as f(K))

**Finding (from dynamic matmul):** When the loop-invariant operand (rhs) is loaded once and
held resident in SBUF, the n-block width must be sized so ALL k-tiles of rhs fit per partition:
`num_tile_k * block_width * dtype_bytes ≤ SBUF_budget_per_partition`.

**Pattern:** Define a cascade of candidate widths and pick the widest that fits:
```python
_BUDGET = 192 * 1024 - 16 * 1024  # SBUF budget with headroom (per partition)
_CANDIDATES = (1024, 512, 256, 128)

def mm_block_size_n(num_tile_k: int) -> int:
    for cand in _CANDIDATES:
        if num_tile_k * cand * 2 <= _BUDGET:
            return cand
    return _CANDIDATES[-1]
```

**When to apply:** Any kernel where a loop-invariant operand is held resident. The key insight
is that K and block_width are inversely related under a fixed SBUF budget:
- K=2048 (16 tiles): block_width=1024 fits (16×1024×2 = 32 KB/partition)
- K=16384 (128 tiles): block_width=1024 = 256 KB (over); shrink to 512 = 128 KB (fits)
- K=131072 (1024 tiles): even width=128 = 256 KB (over); DECLINE the shape in can_handle.

**The adapter must gate:** If even the narrowest candidate exceeds the budget, `can_handle`
returns False and the shape falls back to the default (non-dynamic) lowering. This prevents
silent SBUF overflow / corruption.

---

## 24. Inter-fori-loop HBM coherence — no implicit fence

**Finding (from dynamic matmul restructuring failure):** When two separate `nl.fori_loop`
invocations access the same HBM address (one writes, the next reads), there is **no implicit
DMA fence** between them. The NKI compiler does not insert barriers between top-level code
blocks separated by fori_loop boundaries.

**Symptom:** Correctness failures (~1% of elements wrong, max_diff ~1.0 in bf16) that
disappear with uniform inputs (where all values are identical, masking the race).

**Rule:** Never design a kernel where fori_loop N+1 reads from an HBM location that
fori_loop N wrote to. Instead:
- Keep accumulation across dependent iterations **inside a single fori_loop body** (in SBUF)
- Or write each fori_loop to a **non-overlapping** HBM region (different output columns/rows)
- There is no user-accessible fence/barrier API in NKI to enforce ordering between fori_loops

**Safe patterns:**
- Multiple fori_loops writing to different output slices (n_block sharding): ✅
- Multiple fori_loops reading from shared read-only input: ✅ (no write hazard)
- fori_loop A writes to `res[0:1024]`, fori_loop B reads from `res[0:1024]`: ❌ UNSAFE

---

## 25. hwdge on rhs/static loads — DON'T force it

**Finding (from dynamic matmul experiment):** Adding `dge_mode=nisa.dge_mode.hwdge` to DMA
loads that use **static slice indexing** (compile-time known addresses, no `scalar_offset`)
causes 21-36% regression.

**Root cause:** For static-address DMAs, the compiler pre-generates optimal DMA descriptors
at compile time and stores them in the NEFF. Forcing hwdge overrides this with a runtime
hardware descriptor generator that has higher per-DMA overhead for addresses that are already
known at compile time.

**Rule:**
- `dge_mode=hwdge` ONLY on DMAs with `scalar_offset` or `vector_offset` (runtime-computed
  addresses where the compiler cannot pre-generate descriptors).
- NEVER force hwdge on static-indexed DMAs (`src=tensor[a:b, c:d]` with compile-time a,b,c,d).
- When in doubt, omit `dge_mode` — the compiler's auto-selection is correct for static cases.

---

## 26. Strided partition DMA patterns — catastrophic slowdown

**Finding (from dynamic matmul experiment):** Using an AP pattern with `stride_partition=1`
(e.g., `pattern=[[1, 128], [k_max, 128]]` to load a transposed tile directly) causes **18×
regression** despite eliminating nc_transpose operations.

**Root cause:** With partition stride = 1, adjacent hardware partitions access memory addresses
only `dtype_bytes` apart (e.g., 2 bytes for bf16). The DMA engine must scatter tiny 2-byte
fragments across 128 partitions, which is far below the DMA saturation knee (~4 KB per burst).
This creates thousands of tiny DMAs instead of a few large bursts.

**Rule:** The partition-dimension stride in AP patterns MUST be large (≥ 128 elements, ideally
≥ 512) for acceptable DMA throughput. The free-dimension stride can be 1 (contiguous burst).

**Correct pattern for matmul lhs load:**
```python
# GOOD: partition stride = k_max (rows apart), free stride = 1 (contiguous K elements)
# Reads [M_tile, K_tile] with M on partition — DMA bursts K_tile contiguous bytes/partition
pattern=[[k_max, tile_size_m], [1, tile_size_k]]  # then nc_transpose to [K, M]

# BAD: partition stride = 1 (adjacent K elements), free stride = k_max (rows)
# Reads [K_tile, M_tile] with K on partition — 2-byte scatter across 128 partitions
pattern=[[1, tile_size_k], [k_max, tile_size_m]]  # 18× SLOWER, avoid!
```

**When transpose is needed:** Accept the nc_transpose + tensor_copy overhead (3 ops per tile)
rather than trying to fuse the transpose into a strided DMA load. The DMA bandwidth loss from
non-contiguous partition access far exceeds the cost of an on-chip transpose.

---

## 27. Ragged tiling for non-aligned dimensions

**Finding (from PR #1554 matmul optimization):** Real model shapes often have dimensions that
are NOT multiples of the tile size (e.g., DeepSeek-V3: K=352, N=352). Requiring alignment
forces fallback to the default (non-dynamic) lowering, losing the dynamic-shape benefit.

**Pattern:** Use ceiling division for tile counts and track per-tile widths:
```python
num_tile_k = div_ceil(k_max, tile_size_k)  # final tile may be < tile_size_k
k_tile_widths = [min(tile_size_k, k_max - t * tile_size_k) for t in range(num_tile_k)]

# In matmul: use k_this = k_tile_widths[tk_global] for the contraction dimension
nisa.nc_matmul(
    dst=psum[0:M, 0:N_width],
    stationary=lhsT[0:k_this, 0:M],   # ragged K
    moving=rhs[0:k_this, n_start : n_start + N_width],  # ragged K and N
)
```

**Key constraints for nc_matmul with ragged tiles:**
- The contraction (K) dimension can be any size ≤ 128 (partition dim)
- The free (N) dimension of the moving operand can be any size ≤ 512
- Zero-padding the tile to 128×512 and masking the output is NOT needed — NKI handles
  sub-tile matmul natively via slice indexing

**Adapter changes:** Remove alignment checks (e.g., `K % 1024 != 0`) from `can_handle`.
Keep only: K > 0, N divisible by num_programs (for LNC shard), SBUF budget check.

---

## 28. PSUM bank budget constraint (tiles_in_block_k)

**Finding (from PR #1554):** The kernel reduced `tiles_in_block_k` from 8 to 4 with comment:
"transpose(<=4) + accumulation(<=tiles_in_block_n) PSUM banks <= 8". This reflects a hardware
constraint on NeuronCore.

**Context:** PSUM has 8 banks. Each live `nl.ndarray(..., nl.psum)` tile occupies a bank.
Within one k-block processing:
- `tiles_in_block_k` nc_transpose outputs (one per k-tile being transposed in the same scope)
- `tiles_in_block_n` accumulation psum_tiles (one per n-tile being accumulated)
- Total: tiles_in_block_k + tiles_in_block_n ≤ 8

**Rule:** When choosing tiles_in_block_k, ensure:
`tiles_in_block_k + tiles_in_block_n ≤ 8` (PSUM bank limit)

For the matmul kernel with tiles_in_block_n=2: tiles_in_block_k ≤ 6.
PR chose 4 for headroom. Going back to 8 requires either reducing tiles_in_block_n or
managing PSUM lifetime more carefully (allocating in-body so dead tiles are freed).
