# All-to-All collective support (`all_to_all`, `all_to_all_single` / a2a-v)

Reference for the all-to-all family on the `"neuron"` backend. The C++
`NeuronBackend` owns dispatch; see `AI-distributed.md` for the single-backend
contract and `lower_collective_hlo` lowering path this shares.

> **Navigation → codegraph.** For source, ask:
> `codegraph_explore "NeuronBackend::alltoall alltoall_base nki_all_to_all_v_single _all_to_all_v_hbm UseNkiAllToAllV"`.

## Surface

Two `torch.distributed` entry points, both handled in
`csrc/distributed/NeuronBackend.cpp`:

- **`dist.all_to_all(out_list, in_list)`** → `NeuronBackend::alltoall`
  (`:768`). Equal-shaped per-rank tensors; lowered as an even `AllToAll` via
  the shared `lower_collective_hlo` path (op_type `ALLTOALL`).
- **`dist.all_to_all_single(out, in, output_split_sizes, input_split_sizes)`**
  → `NeuronBackend::alltoall_base` (`:839`). Supports **uneven / variable**
  per-rank splits (a2a-v).

## `alltoall_base` routing (three paths)

`alltoall_base` normalizes `None` splits to equal chunks, then branches:

1. **Even splits** (`even_in && even_out`) → delegate to the even
   `alltoall` (native `AllToAll` lowering). No a2a-v.
2. **Uneven splits, NKI a2a-v** (`UseNkiAllToAllV()`) — enabled by default.
   Calls
   the Python driver `nki_all_to_all_v_single`
   (`distributed/ops/all_to_all_v.py`) via pybind, which runs the
   `ncc.all_to_all_v` NKI kernel. Sends exactly the requested per-rank counts —
   no send-side padding.
3. **Uneven splits, padding fallback** (`UseNkiAllToAllV()` false) — selected
   with `NEURON_USE_NKI_ALLTOALLV=0`.
   Pad each sender chunk to `max_in_size`, do an even
   `alltoall`, then unpad. Correct only when all non-zero chunks are equal size
   (e.g. the permute/ring pattern).

**`NEURON_USE_NKI_ALLTOALLV`** (`UseNkiAllToAllV()`): defaults on. Set to `0`
to use the compatibility padding fallback. It is read per-call at dispatch time
(not cached), so tests can toggle it around individual calls.

## NKI a2a-v path (layout F)

`ncc.all_to_all_v` treats tensors as flat element buffers; per-rank
send/recv counts + displacements ride in a **`uint32` metadata tensor**, not in
the tensor shapes — so variable payloads move without dynamic shapes.

**Kernels** (`python_ops/nki_kernels/all_to_all_v.py`):
- `_all_to_all_v_hbm(input, metadata, replica_group, dst_rows,
  recv_counts_known, has_rdispls)` (`:40`) — dma_copy input+metadata into named
  `shared_hbm` buffers, `ncc.all_to_all_v(...)`, dma_copy out. (Named buffers
  required — else `NCC_IBIR440`; collectives can't touch IO tensors directly.)
- `_all_reduce_max_hbm` (`:25`) — reconciles per-rank capacities to a global max.

**Layout F** = `recv_counts_known=False, has_rdispls=False` (the only
runtime-supported combo). Send side is packed exactly; the runtime places
rank `r`'s received chunk equally-spaced at `(dst.total / rg_size) * r`. The
driver then gathers the real `output_split_sizes[r]` rows into the contiguous
output.

**Driver** `nki_all_to_all_v_single(output, input, output_split_sizes,
input_split_sizes, replica_groups)` (`distributed/ops/all_to_all_v.py:52`):
1. `H = prod(input.shape[1:])`; reshape input to `[rows, H]`.
2. **Globally-uniform NEFF:** an in-graph `all_reduce(MAX)` over
   `[send_rows, recv_rows]` yields `global_send_cap`/`global_recv_rows` so every
   rank compiles an **identical** a2a-v NEFF (the NKI NEFF key is over shapes +
   constant args, not tensor values — differing per-rank counts are safe, but
   differing shapes would diverge the barrier signature).
3. Pad `src` to `[global_send_cap, H]`; `dst_rows = ep * global_recv_rows`.
4. Build the `(3, EP)` uint32 metadata (`_build_layout_f_metadata`): row 0
   `send_counts*H`, row 1 `send_displs*H` (exclusive cumsum), row 2 zeros
   (runtime recv_counts writeback).
5. Run `_all_to_all_v_hbm`; gather `output_split_sizes[r]` rows from row
   `r*global_recv_rows` into `output`.

## Validation (`alltoall`, `alltoall_v`)

- **Hardware capability:** `alltoall` (path 1) requires the Mesh algorithm —
  group size in `{4, 8, 16}` or a multiple of 32; replica-group start ranks
  multiples of group size. The NKI branch re-checks these (it bypasses
  `alltoall`) plus an **LNC=2 guard on the NeuronSwitch (Trn3) fabric**.
- **Empty tensors:** ranks with explicit zero splits still enter a2a-v so peers can
  exchange data; implicit all-empty equal splits retain the no-op fast path.

## Runtime dependency

The a2a-v mesh subtype (`op=ALLTOALL_V`, `subtype=a2a_v`) requires
**NRT `aws-neuronx-runtime-lib >= 2.x.62579` and `aws-neuronx-collectives >=
2.x.63596`**. Older NRT lowers the collective but fails at load with
`ENC:__select_mesh_subtype ... not a supported operation for mesh (op_type 12)`.
Confirm with `NEURON_RT_LOG_LEVEL=INFO` — a healthy run logs `Composing Mesh:
op=ALLTOALL_V subtype[..]=a2a_v` on every rank.

## Tests

`tests/distributed/collective_ops/test_alltoall_v.py` — mirrors
`test_alltoall_base.py` with a **variable-offset verifier** (uneven splits can't
reuse the uniform-`chunk_size` verifier). Covers uneven splits, the permute/ring
pattern, dtypes, 2D (`H>1`), even-delegate (no regression), empty, async,
partial groups, and invalid-split errors.

## Notes / limits

- **Eager only.** `torch.compile` capture of variable a2a-v is not wired: real
  EP routing derives splits from the router as device tensors, and
  `all_to_all_single`'s `SymInt[]` splits force a host sync (`.tolist()`) that
  graph-breaks under Dynamo. Compiled MoE would need a custom-op dispatcher
  (device-tensor counts + fixed capacity + `register_fake`), analogous to
  `DeepEPExpertParallel`.
- Real EP (e.g. torchtitan `ExpertParallel`) calls `all_to_all_single` from an
  eager `input_fn`/`output_fn` hook → this eager a2a-v path.

## See also

- `AI-distributed.md` — backend contract, `lower_collective_hlo`, SPMD/mesh.
- `AI-compilation-kernels.md` — NKI kernel → NEFF pipeline, cache keying.
