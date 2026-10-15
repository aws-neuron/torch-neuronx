# Packed Int64 Design

## Overview

Neuron hardware does not natively support 64-bit tensor operations. The packed
int64 feature represents `int64` tensors using pairs of `int32` values in device
memory. This keeps the full 64 bits resident on device across transfers and
avoids the expensive host roundtrip that `cpu_bounce` mode performs for every
op. Computation still runs on the low 32 bits only (see Limitations).

## Modes

Controlled by `TORCH_NEURONX_INT64_MODE`:

| Mode | Env Value | Behavior |
|------|-----------|----------|
| Packed | `packed` (**default**) | Store int64 as two int32 lanes on device. Extract low words for compute on device. |
| CPU Bounce | `cpu_bounce` | Download to CPU, cast int64→int32, re-upload. Safe but slow. |
| Native | `native` | Pass int64 through to the compiler as S64 HLO. Requires compiler support. |

The mode is read
once and cached (Python `get_int64_mode()`, C++ `GetInt64Mode()`); both default
to `packed`.

## Packed Storage Layout

Each int64 element is stored as two adjacent int32 values in little-endian order
(`[low_word, high_word]`):

```
Logical int64 tensor: shape=(3, 4), stride=(4, 1), offset=0
Physical int32 view:  shape=(3, 4, 2), stride=(8, 2, 1), offset=0
                                   ^          ^       ^
                              trailing    doubled   lane
                              dim of 2    strides   stride
```

For a scalar (rank-0): `shape=() → physical shape=(1, 2), stride=(2, 1)`.

The layout math lives in two mirrored places:
- Python: `packed_storage.packed_physical_layout` / `packed_physical_view`
- C++: `packed_int64::physical_sizes` / `physical_strides` / `physical_view`

Storage offset is also scaled by 2 (each logical int64 element occupies two
int32 slots). int64 device tensors are allocated at full 8-bytes-per-element
width by the `empty_strided_neuron` factory, so the physical int32 view always
has backing storage for both lanes.

The low/high word aliases (`low_word_alias`/`high_word_alias`) drop the trailing
lane dimension to give an int32 view of one lane: shape = logical shape, strides
= doubled logical strides, offset = `storage_offset * 2 (+1 for the high lane)`.
For a rank-0 scalar the alias has empty shape **and** empty strides — the offset
alone selects the single word (PyTorch's `set_` requires size and stride lengths
to match, so a scalar cannot carry a length-1 stride).

This layout is byte-identical to the CPU's native little-endian int64
representation viewed as int32 pairs, so CPU↔Neuron transfers need no byte
swapping or arithmetic.

## Data Flow

### CPU → Neuron Transfer

```
CPU int64 tensor
    │  view(torch.int32)  ← zero-copy reinterpretation
    ▼
CPU int32 pairs [lo₀, hi₀, lo₁, hi₁, ...]
    │  bulk NRT DMA into the packed physical view of the device tensor
    ▼
Device int64 tensor (packed physical storage)
```

All 64 bits preserved, no casting. Implemented in
`cast_policy.copy_cpu_to_neuron` (packed branch). Non-contiguous sources are
made contiguous on the host first, then re-strided on device.

### Computation (e.g., `a + b`)

```
Device int64 tensor (packed)
    │  execution view: low_word_alias (int32, stride-2) → contiguous int32
    │  via neuron::nki_strided_copy_  (NKI burst-load + on-chip select)
    ▼
Device int32 tensor (contiguous, low words only)
    │  compile & execute StableHLO graph with int32 IO
    ▼
Device int32 result
    │  materialize: high = low >> 31 (sign extend);
    │  stack [low, high] into (*, 2); bulk D2D copy into packed physical view
    ▼
Device int64 tensor (packed)
```

Only the low 32 bits participate in computation; the high lane is reconstructed
by sign extension. Values outside `[-2³¹, 2³¹-1]` are truncated (see
Limitations).

### Neuron → CPU Transfer

```
Device int64 tensor (packed physical storage)
    │  bulk NRT DMA (download int32 pairs from the physical view)
    ▼
CPU int32 pairs
    │  view(torch.int64)  ← zero-copy reinterpretation
    ▼
CPU int64 tensor (all 64 bits intact)
```

Implemented in `cast_policy.copy_neuron_to_cpu` (packed branch).

## Dispatch Integration

In packed mode, `int64` is reported as an **unsupported** execution dtype
(`IsSupportedDtype(Long)` returns false unless native mode is on). This is
deliberate: it routes int64 operands through the downcast machinery
(`DowncastArgs` → `NeuronSafeCast`), which is where the packed-specific handling
hooks in:

- **Long → Int (operand downcast):** `NeuronSafeCast` calls
  `packed_int64::execution_view`, which extracts the low words on device via the
  NKI strided-copy op. No CPU bounce.
- **Int → Long (result upcast):** `NeuronSafeCast` calls
  `packed_int64::materialize_from_i32`, which sign-extends and writes the packed
  pairs.

`NeuronDispatcher::IsToCopyMlirDtypeSupported` also allows `Long` in packed (or
native) mode so `_to_copy` does not force a CPU bounce.

### `_to_copy` Int → Long must bypass the MLIR pipeline

Because `IsToCopyMlirDtypeSupported(Long)` is true in packed mode, an explicit
device-side cast like `int32_tensor.to(torch.int64)` would otherwise satisfy the
`mlir_can_handle` check in `DispatchToCopyImpl` and route through the MLIR
lowering pipeline. That pipeline cannot produce packed int64 storage — it
downcasts the result straight back to int32, so the cast becomes a **silent
no-op** (the tensor stays int32). `DispatchToCopyImpl` therefore has an explicit
packed Int→Long short-circuit, placed *before* the `mlir_can_handle` check, that
routes through `NeuronSafeCast` → `materialize_from_i32` (with stride
preservation). This keeps `_to_copy` consistent with the operand-cast path and
matters for any op whose Python implementation finishes with `result.to(int64)`
(e.g. the value-dependent ops like `masked_select`, which re-lower on every call
and rely on that final cast for the upcast).

### Recursion guard (important)

`materialize_from_i32` allocates its int64 output with `torch::empty`, which maps
directly to the `empty_strided_neuron` factory and does **not** redispatch.
Allocating via a composite op such as `torch::zeros` would re-enter the Neuron
dispatcher, get downcast Long→Int, execute, and then upcast Int→Long through
`UpcastExecutionOutputs → NeuronSafeCast → materialize_from_i32` again —
recursing infinitely until the stack overflows. The same constraint applies to
any new allocation added inside the cast path.

## NKI Strided-Copy Kernel

The lane extraction is the performance-critical step. A naive stride-2 gather
(a single `dma_copy` whose *source* is a strided HBM slice `src[:, parity::2]`)
reads 4-byte words with 4-byte gaps and cannot coalesce into HBM bursts;
measured at ~0.5 GB/s it dominated int64 op latency.

The kernel in `python_ops/nki_kernels/stride2_gather.py` instead:

1. burst-loads the contiguous packed tile `[rows, 2K]` HBM → SBUF (full bursts),
2. selects the parity lane on chip via an **SBUF → SBUF `nisa.dma_copy`**
   (pure data movement; bit-exact for int32), then
3. burst-stores the contiguous `[rows, K]` result SBUF → HBM.

`parity` selects the lane: 0 = low word (even columns), 1 = high word (odd
columns). It is a kernel constant argument, so each parity compiles to its own
cached NEFF. The caller (`run_stride2_gather`/`run_stride2_scatter`) rebases the
source/destination to the even boundary (`storage_offset - parity`) and passes
`parity` so column `parity` of the loaded tile is the requested lane. The high
lane (`parity=1`) is needed by the overflow check below; the low lane
(`parity=0`) is the normal execution-view path.

Two implementation constraints, learned empirically:

- The on-chip select must use `nisa.dma_copy`, **not** `nl.copy`. `nl.copy`
  routes through a compute engine that rounds int32 through an fp32 path and
  silently corrupts the low bits.
- The tile loop uses `nl.affine_range` since tiles are independent (no data
  dependencies between iterations), allowing NKI to pipeline DMA operations.

### Op registration

`neuron::nki_strided_copy_(Tensor src, Tensor(a!) dst) -> Tensor(a!)` is defined
and implemented in Python (`torch_neuronx/nki_strided_copy_op.py`, registered for
the `PrivateUse1` key at import). The C++ dispatch path invokes it through a
cached typed operator handle
(`c10::Dispatcher::singleton().findSchemaOrThrow(...).typed<...>()`), resolved
once per process. The Python implementation routes:

- stride-2 source, contiguous dst → `run_stride2_gather` (low-word extraction)
- contiguous source, stride-2 dst → `run_stride2_scatter` (writes target lanes,
  preserves the others)
- anything else → `_nrt_copy_strided_neuron_to_neuron` fallback

> Note: materialization (Int→Long) uses the `stack [low, high]` + contiguous D2D
> copy approach in both the C++ and Python paths, because it must write *both*
> lanes of a freshly allocated tensor. `run_stride2_scatter` writes only the
> selected lanes (preserving the others) and is reserved for in-place
> partial-lane updates; it is not currently on the materialization path.

## Overflow Check (opt-in)

Packed compute only uses the low 32 bits, so an int64 operand outside
`[-2³¹, 2³¹-1]` is silently truncated. Setting
`TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW=1` enables a guard that validates
operands at downcast time: it gathers the high lane (`parity=1`) via the NKI op
and checks `high == low >> 31` (the sign extension of an in-range int32). On
mismatch it raises (C++ `TORCH_CHECK` / Python `OverflowError`) directing the
user to `cpu_bounce` mode.

The check is implemented in both dispatch paths so behavior is uniform:
`packed_int64::execution_view` (C++ dispatch, covers arithmetic ops) and
`utils.convert_for_neuron` (Python op path). It is **off by default** — it
roughly doubles the gather work and forces a device sync, so it is a debug aid,
not a production safeguard. Note it validates *inputs*, not *results*: a value
that fits going in can still overflow during the op (matching TorchNeuron, which
does no range checking at all).

## Cache Key Isolation

Different int64 modes produce different compiled graphs. The dispatch cache key
includes a mode suffix:

- `packed` → `aten::add#p64--...`
- `native` → `aten::add#n64--...`
- `cpu_bounce` → `aten::add--...` (no suffix, backward compatible)

This prevents a compiled NEFF from one mode being incorrectly reused by another.
Applied in `CacheKeyGenerator::Generate` and `GenerateFromPython`.

## Performance

Measured on a single Neuron core with `torch.add`/`torch.mul` on int64 tensors
(50 iterations after warmup). "packed (naive gather)" is the original stride-2
HBM-read kernel; "packed (NKI)" is the current burst-load + on-chip-select
kernel.

| Shape | cpu_bounce | packed (naive gather) | packed (NKI) |
|-------|-----------|-----------------------|--------------|
| (1024,)      | 0.40 ms  | 0.83 ms  | 0.83 ms  |
| (1024, 1024) | 3.39 ms  | 25.75 ms | 2.40 ms  |
| (4096, 4096) | 144 ms   | 376 ms   | 29.8 ms  |

The current kernel makes packed mode faster than `cpu_bounce` at scale (≈4.8×
at 4096²) by keeping the lane extraction on device and bandwidth-bound. The
gather still moves ~2.5× the bytes of a plain int32 op (it reads the discarded
high lanes), so packed int64 does not match native int32 throughput. Small
shapes are dominated by fixed per-op launch overhead and show little difference
between modes.

The remaining known optimization is caching the execution view across ops so
repeated ops on an unchanged operand do not re-gather (see TorchNeuron's
`packed_int64_i32_execution_view_cached`).

## Architecture

### Python Layer

| File | Purpose |
|------|---------|
| `torch_neuronx/int64_policy.py` | `Int64Mode` enum, env var reading, `get_int64_mode()` and `is_*_int64_mode()` helpers |
| `torch_neuronx/packed_storage.py` | Layout math, physical/low/high views, `packed_int64_execution_view`, `materialize_int64_from_i32` |
| `torch_neuronx/python_ops/nki_kernels/stride2_gather.py` | NKI gather/scatter kernels and `run_stride2_gather`/`run_stride2_scatter` |
| `torch_neuronx/nki_strided_copy_op.py` | Registers `neuron::nki_strided_copy_` and routes to the kernels |
| `torch_neuronx/utils.py` | `convert_for_neuron()` routes int64 to the packed execution view; optional overflow check |
| `torch_neuronx/python_ops/cast_policy.py` | CPU↔Neuron transfers via byte reinterpretation in packed mode |

### C++ Layer

| File | Purpose |
|------|---------|
| `csrc/core/dispatch/DispatchUtils.h` | `Int64Mode` enum, `GetInt64Mode()`, `IsPackedInt64Enabled()`, `IsNativeInt64Enabled()` |
| `csrc/core/dispatch/DispatchUtils.cpp` | `IsSupportedDtype(Long)` returns false in packed mode (forces downcast routing) |
| `csrc/core/dispatch/DtypeUtils.cpp` | `NeuronSafeCast` → `execution_view` (down) / `materialize_from_i32` (up) in packed mode |
| `csrc/core/dispatch/NeuronDispatcher.cpp` | `IsToCopyMlirDtypeSupported` allows Long in packed/native mode; `DispatchToCopyImpl` short-circuits packed Int→Long to `NeuronSafeCast` (avoids the MLIR no-op) |
| `csrc/core/cache/CacheKeyGenerator.cpp` | `#p64` / `#n64` mode suffix prevents cross-mode cache collisions |
| `csrc/core/utils/PackedInt64Utils.h/.cpp` | Layout helpers, `physical_view`, `low_word_alias`/`high_word_alias`, `execution_view`, `materialize_from_i32`, gated overflow check |

## Limitations

1. **Precision loss during compute**: Only the low 32 bits are used for
   arithmetic. Int64 values outside `[-2³¹, 2³¹-1]` produce incorrect results
   after computation. Set `TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW=1` to raise
   on out-of-range operands (off by default for performance).
2. **Storage and transfers are lossless**: The full 64 bits are preserved in
   device memory and across CPU↔Neuron transfers. Only the compute step
   truncates.
3. **Gather reads the discarded high lanes**: Lane extraction moves ~2× the
   logical data, so packed int64 ops are bandwidth-bound relative to int32.
4. **Float64 / complex128**: Use the same packed-view mechanism
   (float64→float32, complex128→complex64; same shape, lossy truncation during
   compute). Note float64 still uses CPU bounce in the current cast policy.
