# All-to-All collective support (`all_to_all`, `all_to_all_single` / a2a-v)

Reference for the all-to-all family on the `"neuron"` backend. The C++
`NeuronBackend` owns dispatch; see `docs/AI-distributed.md` for the backend
contract and shared collective lowering path.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'NeuronBackend::alltoall|alltoall_base|nki_all_to_all_v_single|_all_to_all_v_hbm|UseNkiAllToAllV' torch_neuronx/csrc/distributed torch_neuronx/distributed tests/distributed`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Surface and routing

`torch_neuronx/csrc/distributed/NeuronBackend.cpp` handles both list-form and
single-tensor APIs:

- `dist.all_to_all(out_list, in_list)` calls `NeuronBackend::alltoall`. Equal
  chunk shapes use the even `ALLTOALL` lowering. If chunk sizes differ only in
  dim 0, the backend concatenates the input list and routes the packed buffer
  and per-rank row counts to `nki_all_to_all_v_single`. All trailing dimensions
  must match. Variable list-form routing requires the NKI path.
- `dist.all_to_all_single(out, in, output_split_sizes, input_split_sizes)` calls
  `NeuronBackend::alltoall_base`. It normalizes omitted splits to equal chunks.
  Equal input and output splits delegate to `alltoall`. Uneven splits use NKI
  a2a-v when enabled and supported. Otherwise, they use the one-dimensional
  padding fallback.

`UseNkiAllToAllV()` reads `NEURON_USE_NKI_ALLTOALLV` at dispatch time. The NKI
path is enabled when the variable is unset or set to `1`. Any other value
disables NKI. Uneven `alltoall_base` then uses the padding fallback; unequal
list-form `all_to_all` raises an error.

### Uneven `alltoall_base` fallback

The padding fallback supports only one-dimensional tensors. Keep NKI a2a-v
enabled for tensors with trailing dimensions. The current fallback allocates
one-dimensional padded buffers, so its copy fails for those tensors.

For one-dimensional tensors, `alltoall_base` pads each sender chunk to the
local maximum. It runs an even `alltoall` and unpads the result. This fallback
is correct only when every rank has the same largest send chunk, such as the
permute pattern.

## NKI a2a-v buffers and metadata

`torch_neuronx/python_ops/nki_kernels/all_to_all_v.py` defines the kernels.
Collective source, destination, and metadata tensors must be named
`nl.shared_hbm` buffers. The kernels stage input and metadata into those
buffers, run the collective, and stage the result back to the kernel output.
Do not pass kernel I/O tensors directly to the collective.

`torch_neuronx/distributed/ops/all_to_all_v.py` owns the host-side layout:

1. Flatten each row to width `H` and keep the payload packed by destination.
2. Build `uint32` metadata with shape `[4, EP]`. Its rows are send counts,
   send displacements, receive counts, and receive displacements. Counts and
   displacements are in elements, so row counts are multiplied by `H`.
3. All-reduce `[send_rows, recv_rows]` with MAX. This gives every rank the same
   source and destination capacities and therefore the same kernel shapes.
4. Round both capacities up to a multiple of `EP`. Pad only the buffers; the
   metadata still addresses the real packed prefixes.
5. Call `_all_to_all_v_hbm` with `recv_counts_known=True` and
   `has_rdispls=True`. Explicit receive displacements make the received blocks
   a packed source-major prefix.
6. Copy the packed prefix into the output tensor, or copy each source block
   into its corresponding output-list tensor.

`tests/distributed/collective_ops/test_alltoall_v.py` checks the `[4, EP]`
metadata values, global capacities, packed list output, uneven single-tensor
routing, uneven list-form routing, empty ranks, even delegation, and invalid
splits.

## Group-size support

The even `alltoall` lowering accepts group sizes 2, 4, 8, 16, or a multiple of
32. It also validates replica-group alignment.

The NKI a2a-v path accepts group sizes 4, 8, 16, or a multiple of 32. It also
validates replica-group alignment and the platform-specific logical-core
requirement in `NeuronBackend::ValidateNkiAllToAllV`.

For uneven `alltoall_base`, an unsupported NKI group size uses the padded even
fallback. Unequal list-form `all_to_all` does not have that fallback and
requires a group size supported by NKI a2a-v.

## Runtime dependency

The a2a-v mesh subtype requires runtime and collectives packages that support
`ALLTOALL_V`. Upgrade both packages if the runtime rejects the operation at
load time.

## Notes and limits

- Ranks with explicit zero splits still enter a2a-v so peers can exchange data.
  An implicit globally empty equal exchange keeps the no-op fast path.
- Variable a2a-v is an eager path. `all_to_all_single` split lists are host
  values, so device-derived routing counts are not directly captured by
  `torch.compile`.

## See also

- `docs/AI-distributed.md` - backend, collective lowering, SPMD, and mesh.
- `docs/AI-compilation-kernels.md` - NKI and NEFF compilation paths.
