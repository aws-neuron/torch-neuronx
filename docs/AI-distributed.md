# Distributed (`torch_neuronx/distributed/` + `torch_neuronx/csrc/distributed/`)

Multi-device training uses a custom `torch.distributed` backend named
`"neuron"`. Python half in `torch_neuronx/distributed/`; C++ `NeuronBackend`
/ `NeuronWork` / `NeuronWatchdog` in `torch_neuronx/csrc/distributed/`.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'NeuronBackend|lower_collective_hlo|register_neuron_collectives|register_sharding|NeuronWork|NeuronWatchdog' torch_neuronx/distributed torch_neuronx/csrc/distributed`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Registration & the single-backend contract

`torch_neuronx/distributed/backend.py` registers at import:
`dist.Backend.register_backend("neuron", _create_neuron_process_group,
extended_api=True, devices=["neuron"])` + rendezvous handler. User code then
calls `init_process_group(backend="neuron", ...)` as usual.

**`_create_neuron_process_group` unconditionally constructs the C++
`NeuronBackend`** - there is no Python alternative anymore. The legacy
pure-Python `ProcessGroupNeuron` and the `TORCH_NEURONX_CPP_PROCESS_GROUP`
selection were **removed**; collective lowering, bucketing, split-allgather,
and PREMUL_SUM all live in C++ (`torch_neuronx/csrc/distributed/NeuronBackend.cpp`). (`__init__.py` still
`setdefault`s `TORCH_NEURONX_CPP_PROCESS_GROUP=1`, but **nothing reads it** -
vestigial no-op.)

Because it registers via the extended-API path, PyTorch labels it
`BackendType.CUSTOM`. **`CUSTOM` carries no device awareness**, so generic
`ProcessGroup::barrier()` paths are xfailed
(`tests/pytorch_tests_native/skip_registry/distributed_xfails_registry.py`).

## Collective lowering (contract)

The C++ backend owns the collective surface (`allreduce`, `allgather`,
`reduce_scatter`, `all_to_all`, `broadcast`, `barrier`, plus coalesced and
`_*_base` variants). For the shared in-NEFF lowering path, it invokes the
GIL-held `lower_collective_hlo(op_type, inputs, replica_groups, reduce_op,
kwargs)` callback (`torch_neuronx/distributed/cpp_backend.py`) on a lowering
cache miss:
- Dispatches on `c10d.OpType` to one of four `_lower_*_torch_mlir` helpers,
  each tracing the matching `_c10d_functional.*` op through a kernel in
  `torch_neuronx/python_ops/torch_mlir/ops/collectives.py` into StableHLO, returning a
  `LoweringResult` with `has_collectives=True`.
- **`replica_groups` are encoded as `"[[r0,r1,...],...]"`** (via
  `_format_replica_groups`), parsed by `TorchToNeuron::createReplicaGroups`.
  **`group_size` comes from `len(replica_groups[0])`, not `WORLD_SIZE`** - so
  partial groups lower correctly. `_REDUCE_INT_TO_STR` includes `PREMUL_SUM`.

Uneven NKI a2a-v is a separate path. C++ calls
`torch_neuronx.distributed.ops.all_to_all_v.nki_all_to_all_v_single`, which
builds host-side metadata and invokes the NKI kernel.

**Functional collectives** (`torch_neuronx/distributed/ops/functional_collectives.py`):
`register_neuron_collectives()` registers `PrivateUse1` keys for
`all_gather_into_tensor[_out]`, `reduce_scatter_tensor`,
`reduce_scatter_tensor_coalesced`, `all_reduce`, and `all_reduce_coalesced`.
**All use `asyncOp=False`** to avoid the `CompositeExplicitAutograd` fallback
retaining tensor refs in `NeuronWork`'s Future.

**Runtime setup** (`_neuron_runtime_setup`, once on first PG creation): derive
`LOCAL_RANK`/`LOCAL_WORLD_SIZE` -> set `NEURON_RT_VISIBLE_CORES`; rank 0 picks
a free port -> all ranks set `NEURON_RT_ROOT_COMM_ID`; call `_C._set_world_size`
/ `_set_rank` / `_set_local_world_size` / `_set_local_device_start_index`;
`_lazy_init()`.

## SPMD / mesh registry (contract)

`torch_neuronx/distributed/mesh_registry.py` hooks
`DeviceMesh._init_process_groups` and `dist.new_group` to capture topology at
creation. **Without this, each rank only sees its own PG, breaking SPMD
compilation.** `get_all_replica_groups(pg_name)` returns precomputed groups and
respects `TORCH_NEURONX_SPMD_DISABLE=1`.

**DTensor sharding** (`torch_neuronx/distributed/tensor/register_sharding.py`): SDPA fwd has
4 strategies (replicate, TP on `num_heads` `Shard(1)`, batch `Shard(0)`,
context-parallel on seq `Shard(2)`). SDPA backward has the same four
placements. Its context-parallel strategy keeps gradients and attention inputs
on `Shard(2)` so the custom handler can run without redistribution. The handler
in `torch_neuronx/distributed/tensor/cp_ring_handler.py` uses the monolithic NKI
ring backward when eligible and otherwise calls PyTorch's templated ring
backward. `linear_out_strategy` is einsum-based; `linear_backward_strategy` covers
replicate/shard-batch/shard-n/shard-k.

## Environment variables

The collective-tuning vars (`NEURON_USE_SPLIT_ALLGATHER`,
`NEURON_USE_SPLIT_REDUCE_SCATTER`, and `COLLECTIVE_BUCKETSIZE_IN_MB`) and their
defaults live in `docs/environment_variables.md` (authoritative). `TORCH_NEURONX_SPMD_DISABLE=1`
(covered in the SPMD section above) bypasses the mesh registry. Standard
`torch.distributed` variables (`MASTER_ADDR`, `MASTER_PORT`,
`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, and `LOCAL_WORLD_SIZE`) apply as usual.

## See also

- `docs/AI-all-to-all.md` - the all-to-all family (`all_to_all`, `all_to_all_single`
  / NKI a2a-v) dispatch paths, packed explicit-displacement driver, and NRT
  requirement.
- `docs/AI-csrc.md` - `NeuronWork`/`NeuronWatchdog` internals + `NeuronBindings.cpp`.
- `docs/AI-dynamo-backend.md` - `collective_legalization` +
  `dtensor_redistribute_optimization` FX passes.
- `docs/AI-runtime-api.md` - the `_C` bindings `_neuron_runtime_setup` calls.
