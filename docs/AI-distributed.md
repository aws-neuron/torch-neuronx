# Distributed (`torch_neuronx/distributed/` + `csrc/distributed/`)

Multi-device training uses a custom `torch.distributed` backend named
`"neuron"`. Python half in `torch_neuronx/distributed/`; C++ `NeuronBackend`
/ `NeuronWork` / `NeuronWatchdog` in `torch_neuronx/csrc/distributed/`.

> **Navigation → codegraph.** For the registration factory, the collective
> lowering helpers, the DTensor sharding strategies, and the C++
> Work/Watchdog classes as source, ask codegraph:
> `codegraph_explore "NeuronBackend lower_collective_hlo register_neuron_collectives register_sharding NeuronWork NeuronWatchdog"`.
> This doc keeps the single-backend contract, the lowering/SPMD invariants,
> the env vars, and authority routing.

## Registration & the single-backend contract

`distributed/backend.py` registers at import:
`dist.Backend.register_backend("neuron", _create_neuron_process_group,
extended_api=True, devices=["neuron"])` + rendezvous handler. User code then
calls `init_process_group(backend="neuron", ...)` as usual.

**`_create_neuron_process_group` unconditionally constructs the C++
`NeuronBackend`** — there is no Python alternative anymore. The legacy
pure-Python `ProcessGroupNeuron` and the `TORCH_NEURONX_CPP_PROCESS_GROUP`
selection were **removed**; collective lowering, bucketing, split-allgather,
and PREMUL_SUM all live in C++ (`NeuronBackend.cpp`). (`__init__.py` still
`setdefault`s `TORCH_NEURONX_CPP_PROCESS_GROUP=1`, but **nothing reads it** —
vestigial no-op.)

Because it registers via the extended-API path, PyTorch labels it
`BackendType.CUSTOM`. **`CUSTOM` carries no device awareness**, so generic
`ProcessGroup::barrier()` paths are xfailed
(`tests/.../distributed_xfails_registry.py`).

## Collective lowering (contract)

The C++ backend owns the collective surface (`allreduce`, `allgather`,
`reduce_scatter`, `all_to_all`, `broadcast`, `barrier`, plus coalesced and
`_*_base` variants) and submits directly from C++. Python's only role at
collective time is the `lower_collective_hlo(op_type, inputs, replica_groups,
reduce_op, kwargs)` callback (`distributed/cpp_backend.py`), invoked **GIL-held
on a lowering cache miss**:
- Dispatches on `c10d.OpType` to one of four `_lower_*_torch_mlir` helpers,
  each tracing the matching `_c10d_functional.*` op through a kernel in
  `python_ops/torch_mlir/ops/collectives.py` into StableHLO, returning a
  `LoweringResult` with `has_collectives=True`.
- **`replica_groups` are encoded as `"[[r0,r1,...],...]"`** (via
  `_format_replica_groups`), parsed by `TorchToNeuron::createReplicaGroups`.
  **`group_size` comes from `len(replica_groups[0])`, not `WORLD_SIZE`** — so
  partial groups lower correctly. `_REDUCE_INT_TO_STR` includes `PREMUL_SUM`.

**Functional collectives** (`distributed/ops/functional_collectives.py`):
`register_neuron_collectives()` registers `PrivateUse1` keys for
`all_gather_into_tensor[_out]` and `reduce_scatter_tensor` used by compiled
graphs. **All use `asyncOp=False`** to avoid the `CompositeExplicitAutograd`
fallback retaining tensor refs in `NeuronWork`'s Future.

**Runtime setup** (`_neuron_runtime_setup`, once on first PG creation): derive
`LOCAL_RANK`/`LOCAL_WORLD_SIZE` → set `NEURON_RT_VISIBLE_CORES`; rank 0 picks
a free port → all ranks set `NEURON_RT_ROOT_COMM_ID`; call `_C._set_world_size`
/ `_set_rank` / `_set_local_world_size` / `_set_local_device_start_index`;
`_lazy_init()`.

## SPMD / mesh registry (contract)

`mesh_registry.py` hooks `DeviceMesh._init_process_groups` to capture mesh
topology at creation — **without this, each rank only sees its own PG, breaking
SPMD compilation**. `get_all_replica_groups(pg_name)` returns precomputed
groups; respects `TORCH_NEURONX_SPMD_DISABLE=1`.

**DTensor sharding** (`distributed/tensor/register_sharding.py`): SDPA fwd has
4 strategies (replicate, TP on `num_heads` `Shard(1)`, batch `Shard(0)`,
context-parallel on seq `Shard(2)`); SDPA bwd mirrors (replicate/TP/batch);
`linear_out_strategy` is einsum-based; `linear_backward_strategy` covers
replicate/shard-batch/shard-n/shard-k.

## Environment variables

The collective-tuning vars (`NEURON_USE_SPLIT_ALLGATHER`,
`NEURON_USE_SPLIT_REDUCE_SCATTER`, `COLLECTIVE_BUCKETSIZE_IN_MB`,
`COLLECTIVE_MIN_BUCKETS`) and their defaults live in
`docs/environment_variables.md` (authoritative). `TORCH_NEURONX_SPMD_DISABLE=1`
(covered in the SPMD section above) bypasses the mesh registry. Standard
`torch.distributed` vars (`MASTER_ADDR`/`PORT`, `WORLD_SIZE`, `RANK`,
`LOCAL_RANK`) apply as usual.

## See also

- `AI-all-to-all.md` — the all-to-all family (`all_to_all`, `all_to_all_single`
  / NKI a2a-v) dispatch paths, layout-F driver, and NRT requirement.
- `AI-csrc.md` — `NeuronWork`/`NeuronWatchdog` internals + `NeuronBindings.cpp`.
- `AI-dynamo-backend.md` — `collective_legalization` +
  `dtensor_redistribute_optimization` FX passes.
- `AI-runtime-api.md` — the `_C` bindings `_neuron_runtime_setup` calls.
