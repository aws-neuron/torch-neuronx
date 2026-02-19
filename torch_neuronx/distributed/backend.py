import contextlib
import os
import socket
import threading
import time

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    ProcessGroup,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
)
from torch.distributed.constants import default_pg_timeout

import torch_neuronx

# Import C++ NeuronWork and NeuronWatchdog from bindings
from torch_neuronx._C import NeuronWatchdog, NeuronWork
from torch_neuronx.python_ops.base import _create_and_raise_detailed_error
from torch_neuronx.python_ops.xla_ops.all_gather_xla import AllGatherXlaOp
from torch_neuronx.python_ops.xla_ops.reduce_scatter_xla import ReduceScatterXLAOp
from torch_neuronx.utils import flatten_tensors as _flatten_tensors
from torch_neuronx.utils import get_device_from_tensors as _get_device_from_tensors

from .ops import all_gather_op, all_reduce_op, all_to_all_op, reduce_scatter_op
from .ops.functional_collectives import register_neuron_collectives
from .utils import (
    _calculate_allgather_splits,
    _execute_with_xla_op_check,
    _gather_splits_direct_to_output,
    _gather_splits_with_rank_views,
    _should_use_split_allgather,
    _validate_no_zero_dim0,
    create_global_replica_group,
    get_free_port,
    get_reduce_scatter_inputs_outputs,
    get_reduce_type,
    parse_and_update_inputs,
)

_WORLD_SIZE = None


def _create_neuron_process_group(prefix_store, rank, size, timeout):
    return ProcessGroupNeuron(prefix_store, rank, size, timeout)


def _register_neuron_backend():
    dist.Backend.register_backend("neuron", _create_neuron_process_group, devices=["neuron"])


def rendezvous_handler():
    pass


_register_neuron_backend()

dist.register_rendezvous_handler("neuron", rendezvous_handler)

register_neuron_collectives()
# Feature flag: Set NEURON_USE_SPLIT_ALLGATHER=1 to enable new split-based allgather
_USE_SPLIT_ALLGATHER: bool = os.environ.get("NEURON_USE_SPLIT_ALLGATHER", "1") == "1"

# Feature flag: Set NEURON_USE_SPLIT_REDUCE_SCATTER=1 to enable new split-based reduce_scatter,
# enabled by default
_USE_SPLIT_REDUCE_SCATTER: bool = os.environ.get("NEURON_USE_SPLIT_REDUCE_SCATTER", "1") == "1"
# Bucket size limit for collective operations (in MB)
_COLLECTIVE_BUCKETSIZE_BYTES: int = (
    int(os.environ.get("COLLECTIVE_BUCKETSIZE_IN_MB", "512")) * 1024 * 1024
)


def _create_work(pg, outputs, op_type, device, work_stream):
    """Helper to create NeuronWork with common parameters.

    - outputs_ is ALWAYS set in work (for result() method)
    - Stashing is NOT done here - caller decides based on asyncOp
    - record_end_event and register_with_tensors are NOT called here
      because they must happen AFTER stashing (order: stash -> record -> enqueue)

    Args:
        outputs: Output tensors - returned by result()
    """
    # Create work with outputs only - these are returned by result()
    work = NeuronWork(
        pg._pg_uid,
        getattr(pg, "_group_name", "default"),
        device.index if hasattr(device, "index") else device,
        pg.rank(),
        op_type,
        pg._get_next_seq_num(),
        _flatten_tensors(outputs) if outputs is not None else [],
        pg._enable_timing,
        pg._timeout_ms,
        work_stream,
    )

    # NOTE: No stashing, record_end_event, or register_with_tensors here.
    return work


def _ret_work(
    pg: "ProcessGroupNeuron",
    tensors,
    op_type: str,
    opts,
    collective_fn=None,
    device=None,
    inputs=None,
):
    """
    Unified work creation with optional collective execution.

    This function handles two scenarios:
    1. Edge cases (collective_fn=None): Create Work for operations that don't
       execute collectives (e.g., empty tensors). No stream management needed.
    2. Main path (collective_fn provided): Execute collective with NCCL-style
       stream selection and synchronization.

    Args:
        pg: The process group (for sequence number tracking)
        tensors: Input/output tensors for the collective
        op_type: Type of operation (for Work tracking)
        opts: Options object (must have asyncOp attribute for main path)
        collective_fn: Optional lambda that performs the collective operation.
                      If None, this is an edge case with no collective work.
        device: Device to run on (inferred from tensors if None)

    Returns:
        NeuronWork object tracking the operation
    """
    # Get device from tensors, with fallback to current_device() for operations
    if device is None:
        device = _get_device_from_tensors(tensors)

    # EDGE CASE: No collective work (empty tensors, etc.)
    if collective_fn is None:
        work_stream = torch_neuronx.current_stream(device)
        work = _create_work(pg, tensors, op_type, device, work_stream)
        work.record_end_event()
        work.register_with_tensors()
        return work

    async_op = getattr(opts, "asyncOp", False)

    # Choose stream based on asyncOp
    if async_op:
        neuron_stream = pg._get_neuron_stream(device)
        neuron_stream.wait_stream(torch_neuronx.current_stream())
        work_stream = neuron_stream
    else:
        work_stream = torch_neuronx.current_stream(device)

    # 1. Create work
    # 2. Stash tensors (BEFORE collective execution)
    # 3. Execute collective fn()
    # 4. Record end event
    # 5. Enqueue work
    #
    # This order is critical: stashing BEFORE execution ensures the caching
    # allocator knows the tensors are in-use while the collective is in-flight.

    # Step 1: Create work (outputs passed to constructor but NOT auto-stashed)
    work = _create_work(pg, tensors, op_type, device, work_stream)

    # Step 2: Stash tensors BEFORE executing collective
    # For Neuron, we ALWAYS stash inputs and outputs regardless of async_op.
    # Unlike CUDA where sync ops are truly synchronous, Neuron's NRT may queue
    # operations even for "sync" ops.
    if async_op:
        if inputs is not None:
            work.stash(_flatten_tensors(inputs))
        if tensors is not None:
            work.stash(_flatten_tensors(tensors))

    # Step 3: Execute collective on the chosen stream
    if async_op:
        with torch_neuronx.stream(neuron_stream):
            collective_fn()
    else:
        try:
            collective_fn()
        except Exception:
            raise

    # Step 4: Record end event and register with tensors
    work.record_end_event()
    work.register_with_tensors()

    # Enqueue ALL work for watchdog monitoring
    pg._enqueue_work(work)

    return work


def _set_rt_visible_cores(rank, world_size, store):
    torch_neuronx._C._reset_vnc_count()

    # Check if LOCAL_RANK and LOCAL_WORLD_SIZE are already available
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")

    if local_rank_env is not None and local_world_size_env is not None:
        local_rank = int(local_rank_env)
        local_world_size = int(local_world_size_env)

        # Only set NEURON_RT_VISIBLE_CORES if user hasn't set it
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)
        return local_rank, local_world_size

    # Get local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # Store this worker's info: rank -> "ip:rank"
    store.set(f"worker_{rank}", f"{local_ip}:{rank}")

    # Poll until all workers have registered
    while True:
        try:
            # Check if all workers have registered
            registered_count = 0
            for i in range(world_size):
                try:
                    store.get(f"worker_{i}")
                    registered_count += 1
                except Exception:
                    break

            if registered_count == world_size:
                break
        except Exception:
            pass

        time.sleep(0.1)  # Small delay before next poll

    # Collect all worker info and determine local rank/world_size
    workers_on_node = []
    my_ip = local_ip

    for i in range(world_size):
        worker_info = store.get(f"worker_{i}").decode()
        worker_ip, worker_rank = worker_info.split(":")

        if worker_ip == my_ip:
            workers_on_node.append(int(worker_rank))

    # Sort to get consistent local rank assignment
    workers_on_node.sort()
    local_rank = workers_on_node.index(rank)
    local_world_size = len(workers_on_node)

    # Only set NEURON_RT_VISIBLE_CORES if user hasn't set it
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)

    return local_rank, local_world_size


def _set_root_comm_id(rank, local_rank, world_size, store):
    """Set root communication address/port. Here after setting the
    root-comm-id, we need to run the barrier to lock the port. We
    also set the visible cores to LOCAL_RANK and restrict the number
    of cores that each process can see.

    Note: We also don't allow initializing the runtime before
    init_process_group right now. To allow that case, we need to infer
    the visible device from device index and need setting of visible
    cores in the backend.
    """
    if os.environ.get("NEURON_RT_ROOT_COMM_ID") is None:
        if rank == 0:
            root_port = get_free_port()
            store.set("NEURON_RT_PORT", root_port)
        else:
            root_port = store.get("NEURON_RT_PORT").decode("UTF-8")

        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"


def _neuron_runtime_setup(rank, size, store):
    assert (
        not torch_neuronx.is_neuron_runtime_initialized()
    ), "Neuron runtime should not be initialized before the init_process_group."
    # Always determine local rank and world size using TCP store method
    local_rank, local_world_size = _set_rt_visible_cores(rank, size, store)
    _set_root_comm_id(rank, local_rank, size, store)

    torch_neuronx._C._set_world_size(size)
    torch_neuronx._C._set_rank(rank)
    torch_neuronx._C._set_local_world_size(local_world_size)
    torch_neuronx._C._set_local_device_start_index(local_rank)

    # Do nrt_init and check if devices per core is 1. We only allow one device per
    # process in the distributed setting.
    torch_neuronx._lazy_init()

    # Set up vnc_id mapping
    assert (
        torch_neuronx._C._vnc_count() == 1
    ), "Attempted to have more than one device per process in distributed setup"

    # We do the barrier immediately so that we reserve the port
    torch_neuronx._C._nrt_barrier(0, rank, size)


def _validate_coalesced_shapes(tensors):
    """Validate all tensors have same shape except dimension 0."""
    if len(tensors) <= 1:
        return

    first_shape = list(tensors[0].shape)[1:]  # Everything except dim 0
    for tensor in tensors[1:]:
        if list(tensor.shape)[1:] != first_shape:
            raise ValueError(
                f"reduce_scatter_tensor_coalesced requires all input tensors "
                f"to have the same shape except dimension 0. "
                f"Got shapes: {[t.shape for t in tensors]}"
            )


def _interleave_chunks_for_reduce_scatter(tensors, chunk_sizes, world_size, dtype, device):
    """Interleaving using pre-allocated buffer and in-place copies.

    Performance optimization: Uses narrow() and copy_() instead of split/cat
    to reduce memory allocations and improve performance (0.19s -> 0.0019s).

    Args:
        tensors: List of input tensors to interleave
        chunk_sizes: Pre-computed chunk sizes for each tensor
        world_size: Number of ranks
        dtype: Output tensor dtype
        device: Output tensor device

    Returns:
        Concatenated tensor with interleaved chunks
    """
    # Calculate total size
    total_chunk_per_rank = sum(chunk_sizes)
    total_size = total_chunk_per_rank * world_size

    remaining_dims = list(tensors[0].shape[1:])

    # Pre-allocate output buffer
    result_shape = [total_size, *remaining_dims]
    result = torch.empty(result_shape, dtype=dtype, device=device)

    # Copy chunks in-place using views (narrow + copy_ avoids temp tensors)
    offset = 0
    for rank_idx in range(world_size):
        for tensor, chunk_size in zip(tensors, chunk_sizes, strict=False):
            src_start = rank_idx * chunk_size
            src_chunk = tensor.narrow(0, src_start, chunk_size)
            result.narrow(0, offset, chunk_size).copy_(src_chunk)
            offset += chunk_size

    return result


def _split_scattered_result(scattered_output, input_tensors, world_size):
    """Split scattered result back into per-tensor chunks."""
    chunk_sizes = [tensor.shape[0] // world_size for tensor in input_tensors]
    return torch.split(scattered_output, chunk_sizes, dim=0)


def _copy_chunks_to_outputs(scattered_output, outputs, chunk_sizes):
    """Copy scattered result chunks to output tensors using pre-computed offsets.

    Performance optimization: Uses narrow() instead of split() to avoid
    creating intermediate tensors.

    Args:
        scattered_output: The scattered result tensor
        outputs: List of output tensors to copy to
        chunk_sizes: Pre-computed chunk sizes for each output
    """
    offset = 0
    for out, chunk_size in zip(outputs, chunk_sizes, strict=False):
        out.copy_(scattered_output.narrow(0, offset, chunk_size))
        offset += chunk_size


def _filter_empty_tensors(inputs, outputs):
    """Filter out pairs where input tensor is empty."""
    non_empty_pairs = [
        (inp, out) for inp, out in zip(inputs, outputs, strict=False) if inp.numel() > 0
    ]

    if not non_empty_pairs:
        return [], []

    non_empty_inputs, non_empty_outputs = zip(*non_empty_pairs, strict=False)
    return list(non_empty_inputs), list(non_empty_outputs)


class ProcessGroupNeuron(ProcessGroup):
    """ProcessGroup for Neuron device. See ProcessGroup for doc.

    Here we are implementing only a Python subclass. For implementing a
    C++/Python extension, see
    https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html.
    """

    def __init__(self, store, rank, size, timeout):
        super().__init__(rank, size)
        # The first process group creation happens when we do init_process_group. That
        # is when we create the default process group. In that case, we need to
        # initialize the runtime and set the root-comm-id. We need to do init runtime
        # for two reasons: 1) the core id should always be equal to local_rank, hence we
        # need to set that here 2) To lock in the root-comm-id, we need to run the barrier
        # which requires nrt_init.
        self._seq_num = 0
        self._seq_num_lock = threading.Lock()

        self._pg_uid = str(id(self))

        self._enable_timing = False
        self._neuron_streams = {}
        # TODO: look for way to set via environment variable
        # Also look in kernel.py to add for all collectives
        # Use default_pg_timeout for non-NCCL backends per PyTorch convention
        self._timeout_ms = (
            timeout.total_seconds() * 1000 if timeout else default_pg_timeout.total_seconds() * 1000
        )

        global _WORLD_SIZE
        if _WORLD_SIZE is None:
            _WORLD_SIZE = size
            _neuron_runtime_setup(rank, size, store)

        # Initialize NaN checking (disabled by default)
        self._enable_nan_check = False

        self._watchdog = NeuronWatchdog()
        self._watchdog.start()

    def _get_neuron_stream(self, device):
        """Get or create dedicated neuron stream for async collectives."""
        key = str(device.index)
        if key not in self._neuron_streams:
            # Create high-priority stream for collectives
            self._neuron_streams[key] = torch_neuronx.Stream(device)
        return self._neuron_streams[key]

    def _get_next_seq_num(self) -> int:
        "Get the next sequence number for work tracking."
        with self._seq_num_lock:
            seq = self._seq_num
            self._seq_num += 1
            return seq

    def _enqueue_work(self, work):
        """
        Add work to watchdog monitoring queue.

        The watchdog thread will monitor this work for completion/timeout
        and automatically handle tensor cleanup.

        Args:
            work: NeuronWork instance to monitor
        """
        self._watchdog.enqueue_work(work)

    def _set_enable_nan_check(self, enable_nan_check: bool) -> None:
        """Set whether to enable NaN checking for collective operations.

        Args:
            enable_nan_check: Whether to enable NaN checking
        """
        self._enable_nan_check = enable_nan_check

    def getBackendName(self):  # noqa N802
        return "neuron"

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    def _get_backend(self, device):
        return self

    @property
    def group_name(self):
        return self._group_name

    @property
    def _device_types(self):
        return [torch.device("neuron")]

    def _validate_root_rank(self, opts):
        if opts.rootRank < 0 or opts.rootRank >= self.size():
            raise ValueError(
                f"Invalid rootRank {opts.rootRank} for broadcast with world size {self.size()}"
            )

    def allreduce(
        self,
        tensors: list[torch.Tensor] | torch.Tensor,
        all_reduce_options: AllreduceOptions | None = None,
    ):
        """Reduce tensors across all ranks and distribute result to all ranks.

        Args:
            tensors: List of tensors to reduce
            all_reduce_options: Options containing reduction operation (SUM, PRODUCT, etc.)

        Returns:
            Work object for async tracking
        """
        # Default arguments are evaluated once at definition time, not per-call
        # Shared mutable state across function calls, hence going this route
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]

        if all_reduce_options is None:
            all_reduce_options = AllreduceOptions()
        replica_groups = [dist.get_process_group_ranks(self)]
        processed_tensors, reduce_type = parse_and_update_inputs(tensors, all_reduce_options)

        def collective_fn():
            all_reduce_op(
                processed_tensors,
                replica_groups,
                reduce_type,
                out=tuple(tensors),
            )

        return _ret_work(
            self,
            tensors,
            "allreduce",
            all_reduce_options,
            collective_fn=collective_fn,
        )

    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions | None = None,
    ):
        """Low-level implementation of allgather operation without slicing.

        This method gathers input tensors from all ranks and concatenates them
        into a single output tensor along dimension 0.

        Args:
            output_tensor: Destination tensor to receive gathered results.
                    Shape should be [world_size * input_tensor.shape[0], *input_tensor.shape[1:]]
            input_tensor: Source tensor to be gathered from all ranks.
            opts: AllgatherOptions containing operation parameters.

        Returns:
            Work handle for asynchronous completion tracking.

        Note:
            Output shape: [world_size * input.shape[0], *input.shape[1:]]
            Example: If input_tensor is [2, 3] with world_size=4,
                    output_tensor will be [8, 3] containing all gathered values.
        """
        if opts is None:
            opts = AllgatherOptions()
        replica_groups = [dist.get_process_group_ranks(self)]
        world_size = len(replica_groups[0])

        if len(output_tensor.shape) != len(input_tensor.shape):
            output_tensor = output_tensor.squeeze()
            input_tensor = input_tensor.squeeze()

        if _USE_SPLIT_ALLGATHER and _should_use_split_allgather(
            input_tensor, world_size, _COLLECTIVE_BUCKETSIZE_BYTES
        ):

            def collective_fn():
                def core_fn():
                    splits = _calculate_allgather_splits(input_tensor, world_size)
                    _gather_splits_direct_to_output(
                        splits, output_tensor, input_tensor.numel(), replica_groups, world_size
                    )

                ag_op = AllGatherXlaOp()
                args = (input_tensor, replica_groups)
                kwargs = {"out": output_tensor}
                _execute_with_xla_op_check(ag_op, "_allgather_base", core_fn, None, args, kwargs)
        else:

            def collective_fn():
                all_gather_op(input_tensor, replica_groups, out=output_tensor)

        return _ret_work(
            self,
            [output_tensor],
            "allgather_base",
            opts,
            collective_fn=collective_fn,
            inputs=[input_tensor],
        )

    def allgather(self, output_tensors_list, input_tensors, opts: AllgatherOptions | None = None):
        """Gather tensors from all ranks and distribute sliced results to each rank.

        Each process contributes input_tensors and receives slices for each rank's
        contribution in the output_tensors_list.

        Args:
            output_tensors_list: List of tensors to store results. Each tensor will receive
                                data from one rank's contribution.
            input_tensors: Tensor to be gathered across all ranks.
            opts: AllgatherOptions containing operation parameters.

        Returns:
            Work handle for asynchronous completion tracking.

        Note:
            Output structure: List[Tensor], where each tensor has the same shape as input_tensors.
            Example: If input_tensor is [2, 3] with world_size=4,
                output_tensors_list will be [tensor[2,3], tensor[2,3], tensor[2,3], tensor[2,3]],
                each containing data from a different rank.
        """
        if opts is None:
            opts = AllgatherOptions()
        replica_groups = [dist.get_process_group_ranks(self)]
        world_size = len(replica_groups[0])

        all_tensors = input_tensors + list(output_tensors_list)
        _validate_no_zero_dim0(all_tensors)

        if _USE_SPLIT_ALLGATHER and _should_use_split_allgather(
            input_tensors[0], world_size, _COLLECTIVE_BUCKETSIZE_BYTES
        ):

            def collective_fn():
                def core_fn():
                    splits = _calculate_allgather_splits(input_tensors[0], world_size)
                    _gather_splits_with_rank_views(
                        splits, output_tensors_list[0], replica_groups, world_size
                    )

                ag_op = AllGatherXlaOp()
                args = (input_tensors, replica_groups)
                kwargs = {"opts": opts, "out": tuple(output_tensors_list[0]), "slice_output": True}
                _execute_with_xla_op_check(ag_op, "allgather", core_fn, None, args, kwargs)
        else:

            def collective_fn():
                all_gather_op(
                    input_tensors,
                    replica_groups,
                    opts=opts,
                    out=tuple(output_tensors_list[0]),
                    slice_output=True,
                )

        return _ret_work(
            self,
            list(output_tensors_list[0]),
            "allgather",
            opts,
            collective_fn=collective_fn,
            inputs=input_tensors,
        )

    def allgather_coalesced(
        self, output_tensors_list, input_tensors, opts: AllgatherOptions | None = None
    ):
        """Coalesced version of allgather that handles multiple input tensors efficiently.

        Gathers multiple input tensors from all ranks and distributes the results
        with a more efficient communication pattern than separate allgather calls.

        Args:
            output_tensors_list: Nested list structure to store results:
                    [
                    [tensor_0_from_rank_0, tensor_0_from_rank_1, ...],  # For first input tensor
                    [tensor_1_from_rank_0, tensor_1_from_rank_1, ...],  # For second input tensor
                    ...
                    ]
                    Each inner list corresponds to one input tensor and contains
                    contributions from all ranks.
            input_tensors: List of tensors to be gathered from all ranks.
            opts: AllgatherOptions containing operation parameters.

        Returns:
            Work handle for asynchronous completion tracking.

        Note:
            Output structure: List[List[Tensor]] where:
            - Outer list has one entry per input tensor
            - Inner lists have one entry per rank
            - Each tensor has the same shape as its corresponding input tensor

            Example: With 2 input tensors [tensor[2,3], tensor[4,5]] and world_size=2:
                    Output will be:
                    [
                    [tensor[2,3], tensor[2,3]],    # First tensor from ranks 0,1
                    [tensor[4,5], tensor[4,5]]     # Second tensor from ranks 0,1
                    ]
        """
        if opts is None:
            opts = AllgatherOptions()
        replica_groups = [dist.get_process_group_ranks(self)]
        world_size = len(replica_groups[0])
        total_output_bytes = sum(t.numel() * t.element_size() * world_size for t in input_tensors)

        if _USE_SPLIT_ALLGATHER and total_output_bytes > _COLLECTIVE_BUCKETSIZE_BYTES:
            # For split path, call sub-collectives synchronously within collective_fn
            # The async behavior is handled by the outer _ret_work call
            def collective_fn():
                # Create synchronous opts for sub-collectives
                sync_opts = AllgatherOptions()
                sync_opts.asyncOp = False
                for input_tensor, output_list in zip(
                    input_tensors, output_tensors_list, strict=False
                ):
                    self.allgather([output_list], [input_tensor], sync_opts)
        else:
            # flattened_outputs is just a list of references to user-provided tensors.
            # Safe outside collective_fn since no device allocation happens here.
            flattened_outputs = [
                tensor for output_list in output_tensors_list for tensor in output_list
            ]

            def collective_fn():
                all_gather_op(
                    input_tensors,
                    replica_groups,
                    opts=opts,
                    out=tuple(flattened_outputs),
                    slice_output=True,
                )

        # Flatten output_tensors_list for stashing (list of lists -> flat list)
        all_outputs = [t for sublist in output_tensors_list for t in sublist]
        return _ret_work(
            self,
            all_outputs,
            "allgather_coalesced",
            opts,
            collective_fn=collective_fn,
            inputs=input_tensors,
        )

    def allgather_into_tensor_coalesced(
        self, outputs, inputs, opts: AllgatherOptions | None = None
    ):
        """Coalesced all-gather operation gathers multiple tensors into pre-allocated output tensors

        This is similar to _allgather_base but handles multiple input/output tensor pairs
        efficiently in a single collective operation.

        Args:
            outputs: List of pre-allocated output tensors. Each output tensor should have shape
                    [world_size * input.shape[0], *input.shape[1:]] where input is the
                    corresponding input tensor.
            inputs: List of input tensors to be gathered from all ranks.
            opts: AllgatherOptions containing operation parameters.

        Returns:
            Work object for asynchronous operation tracking.

        Note:
            Output shape: [world_size * input.shape[0], *input.shape[1:]] for each tensor pair.
            Example: With 2 input tensors [tensor[2, 3], tensor[4, 5]] and world_size=4,
                    outputs will be [tensor[8, 3], tensor[16, 5]], each containing
                    concatenated gathered values from all ranks.
        """
        if opts is None:
            opts = AllgatherOptions()
        replica_groups = [dist.get_process_group_ranks(self)]

        non_empty_indices = [i for i, inp in enumerate(inputs) if inp.numel() > 0]
        if len(non_empty_indices) == 0:
            return _ret_work(
                self, outputs, "allgather_into_tensor_coalesced", opts, collective_fn=None
            )

        non_empty_inputs = [inputs[i] for i in non_empty_indices]
        non_empty_outputs = [outputs[i] for i in non_empty_indices]

        all_tensors = non_empty_inputs + non_empty_outputs
        _validate_no_zero_dim0(all_tensors)
        total_output_bytes = sum(o.numel() * o.element_size() for o in non_empty_outputs)
        if _USE_SPLIT_ALLGATHER and total_output_bytes > _COLLECTIVE_BUCKETSIZE_BYTES:

            def collective_fn():
                sync_opts = AllgatherOptions()
                sync_opts.asyncOp = False
                for input_tensor, output_tensor in zip(
                    non_empty_inputs, non_empty_outputs, strict=False
                ):
                    self._allgather_base(output_tensor, input_tensor, sync_opts)
        else:

            def collective_fn():
                all_gather_op(
                    non_empty_inputs,
                    replica_groups,
                    out=tuple(non_empty_outputs),
                    slice_output=False,
                )

        assert len(outputs) == len(
            inputs
        ), "Internal error: output list length changed during operation"
        return _ret_work(
            self,
            outputs,
            "allgather_into_tensor_coalesced",
            opts,
            collective_fn=collective_fn,
            inputs=inputs,
        )

    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions | None = None
    ) -> torch.distributed.Work:
        """Broadcast tensors from root rank to all other ranks in the process group.

        This function implements the broadcast collective operation for Neuron devices.
        It takes a list of tensors and BroadcastOptions, and broadcasts the tensor
        from the root rank to all other ranks in the process group.

        Args:
            tensors: A list containing a single tensor to be broadcasted.
                     The length of the list must be 1.
            opts: Broadcast options containing the root rank and other settings.

        Returns:
            A Work object for asynchronous operation tracking.
        """
        if opts is None:
            opts = BroadcastOptions()
        # Get the ranks of the processes in the current process group.
        replica_groups: list[list[int]] = [dist.get_process_group_ranks(self)]
        # Extract the input tensor from the list.
        input_tensor: torch.Tensor = tensors[0]

        if input_tensor.numel() == 0:
            return _ret_work(self, tensors, "broadcast", opts, collective_fn=None)

        self._validate_root_rank(opts)

        # If the current rank is not the root rank, zero out the input tensor.
        # This is a common pattern in broadcast implementations where non-root ranks
        # receive data from the root.
        if self.rank() != opts.rootRank:
            with torch.no_grad():
                input_tensor.zero_()

        def collective_fn():
            # Perform an all-reduce operation with SUM. In a broadcast, after the root
            # sends its data, an all-reduce can be used to ensure all participants
            # have the correct data (effectively a broadcast if only the root had data initially).
            all_reduce_op(tensors, replica_groups, "SUM", out=tuple(tensors))

        return _ret_work(self, tensors, "broadcast", opts, collective_fn=collective_fn)

    def reduce_scatter(
        self,
        output_tensors,
        input_tensors_list,
        reduce_scatter_options: ReduceScatterOptions | None = None,
    ):
        """Reduce tensors across ranks and scatter results.

        Args:
            output_tensors: List containing output tensor
            input_tensors_list: List containing list of input tensors (one per rank)
            reduce_scatter_options: Options containing reduction operation

        Returns:
            Work object for async tracking
        """
        if reduce_scatter_options is None:
            reduce_scatter_options = ReduceScatterOptions()

        replica_groups = [dist.get_process_group_ranks(self)]
        input_tensors_list, reduce_type = parse_and_update_inputs(
            input_tensors_list, reduce_scatter_options
        )

        # Calculate input bytes for bucket size check
        total_numel = sum(tensor.numel() for tensor in input_tensors_list[0])
        input_bytes = total_numel * input_tensors_list[0][0].element_size()

        if _USE_SPLIT_REDUCE_SCATTER and input_bytes > _COLLECTIVE_BUCKETSIZE_BYTES:
            # Check can_handle with original input list before concatenation
            rs_op = ReduceScatterXLAOp()
            args = (input_tensors_list[0], replica_groups, reduce_type)
            kwargs = {"out": output_tensors[0]}
            can_handle_result = rs_op.can_handle(*args, **kwargs)
            if isinstance(can_handle_result, tuple):
                can_handle_impl, can_handle_error_msg = can_handle_result
            else:
                can_handle_impl, can_handle_error_msg = can_handle_result, None

            if can_handle_impl:

                def collective_fn():
                    if len(input_tensors_list[0]) == 1:
                        input_tensor = input_tensors_list[0][0]
                    else:
                        input_tensor = torch.cat(input_tensors_list[0], dim=0)
                    sync_opts = ReduceScatterOptions()
                    sync_opts.asyncOp = False
                    sync_opts.reduceOp = reduce_scatter_options.reduceOp
                    self._reduce_scatter_base(output_tensors[0], input_tensor, sync_opts)
            else:
                base_debug_msg = "No implementation could handle operation reduce_scatter. "
                _create_and_raise_detailed_error(
                    RuntimeError,
                    "reduce_scatter",
                    base_debug_msg + can_handle_error_msg,
                    args,
                    kwargs,
                    None,
                )
        else:

            def collective_fn():
                reduce_scatter_op(
                    input_tensors_list[0],  # passing input as a list of tensors to PG backend
                    replica_groups,
                    reduce_type,
                    out=output_tensors[0],  # passing output as a tensor to PG backend
                )

        # Flatten input_tensors_list for stashing
        all_inputs = [t for sublist in input_tensors_list for t in sublist]
        return _ret_work(
            self,
            output_tensors,
            "reduce_scatter",
            reduce_scatter_options,
            collective_fn=collective_fn,
            inputs=all_inputs,
        )

    def reduce_scatter_coalesced(self, output_tensors, input_tensors_list, opts):
        raise NotImplementedError(__class__.reduce_scatter_coalesced)

    def _reduce_scatter_base(
        self,
        output_tensor,
        input_tensor,
        reduce_scatter_options: ReduceScatterOptions | None = None,
    ):
        """Base reduce-scatter operation on a single tensor.

        Args:
            output_tensor: Output tensor receiving scattered result
            input_tensor: Input tensor to be reduced and scattered
            reduce_scatter_options: Options containing reduction operation

        Returns:
            Work object for async tracking
        """
        if reduce_scatter_options is None:
            reduce_scatter_options = ReduceScatterOptions()
        replica_groups = [dist.get_process_group_ranks(self)]
        input_tensor, reduce_type = parse_and_update_inputs(input_tensor, reduce_scatter_options)
        world_size = len(replica_groups[0])
        input_bytes = input_tensor.numel() * input_tensor.element_size()

        if _USE_SPLIT_REDUCE_SCATTER and input_bytes > _COLLECTIVE_BUCKETSIZE_BYTES:

            def collective_fn():
                def core_fn():
                    input_gen, output_gen = get_reduce_scatter_inputs_outputs(
                        input_tensor, output_tensor, world_size
                    )

                    # Iterate over buckets with explicit cleanup
                    for input_tensor_bucketed, output_tensor_bucketed in zip(
                        input_gen, output_gen, strict=False
                    ):
                        reduce_scatter_op(
                            [input_tensor_bucketed],
                            replica_groups,
                            reduce_type,
                            out=output_tensor_bucketed,
                        )

                rs_op = ReduceScatterXLAOp()
                args = ([input_tensor], replica_groups, reduce_type)
                kwargs = {"out": output_tensor}
                _execute_with_xla_op_check(
                    rs_op, "_reduce_scatter_base", core_fn, None, args, kwargs
                )
        else:

            def collective_fn():
                reduce_scatter_op(
                    [input_tensor],
                    replica_groups,
                    reduce_type,
                    out=output_tensor,
                )

        return _ret_work(
            self,
            [output_tensor],
            "reduce_scatter_base",
            reduce_scatter_options,
            collective_fn=collective_fn,
            inputs=[input_tensor],
        )

    def reduce_scatter_tensor_coalesced(
        self, outputs, inputs, opts: ReduceScatterOptions | None = None
    ):
        """Coalesced reduce-scatter operation using chunk interleaving.

        Performs a single reduce-scatter collective by interleaving tensor chunks
        to ensure each rank receives its portion of all tensors after reduction.

        Args:
            outputs: List of pre-allocated output tensors. Each output tensor should have shape
                    [input.shape[0] // world_size, *input.shape[1:]]
            inputs: List of input tensors to be reduced and scattered. All tensors must have
                the same shape except dimension 0, which must be divisible by world_size.
            opts: ReduceScatterOptions containing the reduction operation.

        Returns:
            Work object for asynchronous operation tracking.

        Raises:
            ValueError: If input tensors have incompatible shapes (except dimension 0)

        Note:
            This coalesced implementation optimizes performance by performing a single
            collective operation instead of multiple sequential operations.

            Equivalent naive approach (multiple collectives):
                for inp, out in zip(inputs, outputs):
                     reduce_scatter_op([inp], replica_groups, reduce_type, out=out)

        Example:
            With world_size=2, reducing 2 tensors with SUM operation:

            # Rank 0 inputs
            input1_r0 = torch.tensor([[1.], [2.], [3.], [4.]])      # [4, 1]
            input2_r0 = torch.tensor([[5.], [6.], [7.], [8.], [9.], [10.]])  # [6, 1]
            # Rank 1 inputs
            input1_r1 = torch.tensor([[10.], [20.], [30.], [40.]])  # [4, 1]
            input2_r1 = torch.tensor([[50.], [60.], [70.], [80.], [90.], [100.]])  # [6, 1]
            # After reduce_scatter with SUM:
            # Rank 0 receives:
            output1_r0  # [[11.], [22.]] - first half of reduced input1
            output2_r0  # [[55.], [66.], [77.]] - first half of reduced input2
            # Rank 1 receives:
            output1_r1  # [[33.], [44.]] - second half of reduced input1
            output2_r1  # [[88.], [99.], [110.]] - second half of reduced input2
        """
        if opts is None:
            opts = ReduceScatterOptions()

        replica_groups = [dist.get_process_group_ranks(self)]
        world_size = len(replica_groups[0])
        reduce_type = get_reduce_type(opts.reduceOp)
        non_empty_inputs, non_empty_outputs = _filter_empty_tensors(inputs, outputs)
        if len(non_empty_inputs) == 0:
            return _ret_work(
                self, outputs, "reduce_scatter_tensor_coalesced", opts, collective_fn=None
            )

        # Single tensor optimization: skip interleaving, call _reduce_scatter_base directly
        if len(non_empty_inputs) == 1:
            return self._reduce_scatter_base(non_empty_outputs[0], non_empty_inputs[0], opts)

        # Validate shape compatibility (only needed for multiple tensors)
        _validate_coalesced_shapes(non_empty_inputs)

        # Calculate input bytes for bucket size check
        input_bytes = sum(inp.numel() * inp.element_size() for inp in non_empty_inputs)

        # Pre-calculate values for the optimized helper function
        first_input = non_empty_inputs[0]
        dtype = first_input.dtype
        device = first_input.device
        remaining_dims = list(first_input.shape[1:])
        chunk_sizes = [tensor.shape[0] // world_size for tensor in non_empty_inputs]
        total_chunk_per_rank = sum(chunk_sizes)

        if _USE_SPLIT_REDUCE_SCATTER and input_bytes > _COLLECTIVE_BUCKETSIZE_BYTES:
            # For split path, we need to check can_handle first
            # Create temporary tensors for the check (not used in async)
            temp_concatenated = _interleave_chunks_for_reduce_scatter(
                non_empty_inputs, chunk_sizes, world_size, dtype, device
            )
            temp_output_shape = [total_chunk_per_rank, *remaining_dims]
            temp_scattered = torch.empty(temp_output_shape, dtype=dtype, device=device)

            rs_op = ReduceScatterXLAOp()
            args = ([temp_concatenated], replica_groups, reduce_type)
            kwargs = {"out": temp_scattered}
            can_handle_result = rs_op.can_handle(*args, **kwargs)
            if isinstance(can_handle_result, tuple):
                can_handle_impl, can_handle_error_msg = can_handle_result
            else:
                can_handle_impl, can_handle_error_msg = can_handle_result, None

            if can_handle_impl:

                def collective_fn():
                    # Allocate intermediate tensors INSIDE collective_fn() so they stay
                    # alive on the async stream until the collective completes
                    concatenated = _interleave_chunks_for_reduce_scatter(
                        non_empty_inputs, chunk_sizes, world_size, dtype, device
                    )
                    output_shape = [total_chunk_per_rank, *remaining_dims]
                    scattered_output = torch.empty(output_shape, dtype=dtype, device=device)

                    sync_opts = ReduceScatterOptions()
                    sync_opts.asyncOp = False
                    sync_opts.reduceOp = opts.reduceOp
                    self._reduce_scatter_base(scattered_output, concatenated, sync_opts)

                    # Copy results using optimized helper
                    _copy_chunks_to_outputs(scattered_output, non_empty_outputs, chunk_sizes)

                return _ret_work(
                    self,
                    outputs,
                    "reduce_scatter_tensor_coalesced",
                    opts,
                    collective_fn=collective_fn,
                    inputs=inputs,
                )
            else:
                base_debug_msg = (
                    "No implementation could handle operation reduce_scatter_tensor_coalesced. "
                )
                _create_and_raise_detailed_error(
                    RuntimeError,
                    "reduce_scatter_tensor_coalesced",
                    base_debug_msg + can_handle_error_msg,
                    args,
                    kwargs,
                    None,
                )
        else:

            def collective_fn():
                # Allocate intermediate tensors INSIDE collective_fn() so they stay
                # alive on the async stream until the collective completes
                concatenated = _interleave_chunks_for_reduce_scatter(
                    non_empty_inputs, chunk_sizes, world_size, dtype, device
                )
                output_shape = [total_chunk_per_rank, *remaining_dims]
                scattered_output = torch.empty(output_shape, dtype=dtype, device=device)

                # Perform single reduce-scatter operation
                reduce_scatter_op([concatenated], replica_groups, reduce_type, out=scattered_output)

                # Copy results using optimized helper
                _copy_chunks_to_outputs(scattered_output, non_empty_outputs, chunk_sizes)

            return _ret_work(
                self,
                outputs,
                "reduce_scatter_tensor_coalesced",
                opts,
                collective_fn=collective_fn,
                inputs=inputs,
            )

    def barrier(self, opts: BarrierOptions | None = None):
        """Synchronize all ranks in the process group.

        Args:
            opts: Barrier options

        Returns:
            Work object for async tracking

        Note:
            Only implemented for default process group.
        """
        if opts is None:
            opts = BarrierOptions()
        if dist.get_world_size() != self.size():
            raise NotImplementedError(
                f"Barrier is implemented only for default group, called barrier on {self}"
            )

        def collective_fn():
            # Call nrt_barrier here
            torch_neuronx._C._nrt_barrier(0, self.rank(), self.size())

        return _ret_work(self, None, "barrier", opts, collective_fn=collective_fn)

    def reduce(self, tensors: list[torch.Tensor], opts: ReduceOptions | None = None):
        """Reduce tensors from all ranks to root rank in the process group.

        This function implements the reduce collective operation for Neuron devices.
        Reusing the implementation of all_reduce.

        Args:
            tensors: A list of tensors to be reduced. All tensors must have the same
                    shape and dtype across all ranks.
            opts: Reduce options containing the root rank and reduction operation.

        Returns:
            A Work object for asynchronous operation tracking.

        Note:
            This implementation uses temporary tensor copies for the all_reduce
            operation to preserve original tensors on non-root ranks.
        """
        if opts is None:
            opts = ReduceOptions()

        self._validate_root_rank(opts)
        # only change the tensors on root rank
        output_tensors = (
            tensors
            if self.rank() == opts.rootRank
            else [torch.empty_like(tensor) for tensor in tensors]
        )
        # Get the ranks of the processes in the current process group
        replica_groups = [dist.get_process_group_ranks(self)]

        def collective_fn():
            # Perform all-reduce operation on temporary tensors
            all_reduce_op(
                tensors,
                replica_groups,
                get_reduce_type(opts.reduceOp),
                out=tuple(output_tensors),
            )

        return _ret_work(
            self,
            tensors if self.rank() == opts.rootRank else output_tensors,
            "reduce",
            opts,
            collective_fn=collective_fn,
        )

    def allreduce_coalesced(
        self, tensors, all_reduce_coalesced_options: AllreduceCoalescedOptions | None = None
    ):
        """Coalesced all-reduce operation for multiple tensors.

        Args:
            tensors: List of tensors to reduce
            all_reduce_coalesced_options: Options containing reduction operation

        Returns:
            Work object for async tracking

        Note:
            Performs single collective for multiple tensors. Empty tensors are filtered.
        """
        if all_reduce_coalesced_options is None:
            all_reduce_coalesced_options = AllreduceCoalescedOptions()
        replica_groups = [dist.get_process_group_ranks(self)]

        non_empty_indices = [i for i, tensor in enumerate(tensors) if tensor.numel() > 0]

        if len(non_empty_indices) == 0:
            return _ret_work(
                self,
                tensors,
                "allreduce_coalesced",
                all_reduce_coalesced_options,
                collective_fn=None,
            )

        non_empty_tensors = [tensors[i] for i in non_empty_indices]

        def collective_fn():
            all_reduce_op(
                non_empty_tensors,
                replica_groups,
                get_reduce_type(all_reduce_coalesced_options.reduceOp),
                out=tuple(non_empty_tensors),
            )

        return _ret_work(
            self,
            tensors,
            "allreduce_coalesced",
            all_reduce_coalesced_options,
            collective_fn=collective_fn,
        )

    def alltoall(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensor_list: list[torch.Tensor],
        opts: AllToAllOptions | None = None,
    ):
        """All-to-all collective operation for Neuron devices.

        This function implements the all-to-all collective operation where each rank
        sends different data to every other rank in the process group.

        Args:
            output_tensor_list: List of output tensors, one for each rank in the group.
                               Each tensor will receive data from the corresponding rank.
            input_tensor_list: List of input tensors, one for each rank in the group.
                              Each tensor will be sent to the corresponding rank.
            opts: All-to-all options (currently unused but kept for interface compatibility).

        Returns:
            A Work object for asynchronous operation tracking.
        """
        if opts is None:
            opts = AllToAllOptions()
        # Early return for empty tensors, not moving this to all_to_all_xla.py
        # as XLA AllToAll doesn't support empty tensors on Neuron
        if all(input_tensor.numel() == 0 for input_tensor in input_tensor_list):
            return _ret_work(self, output_tensor_list, "alltoall", opts, collective_fn=None)

        # Get the current process group ranks
        current_group_ranks = dist.get_process_group_ranks(self)
        group_size = len(current_group_ranks)

        # For XLA AllToAll, we need to construct replica groups properly
        # XLA expects all replica groups in the world to be represented
        world_size = dist.get_world_size()
        if world_size % group_size != 0:
            raise ValueError(
                f"Cannot create global_replica_group when world_size:{world_size} "
                f"is not divisible by group_size: {group_size}."
            )

        # Construct replica groups for XLA: partition all ranks into consecutive groups
        # This creates [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        # for world_size=16, group_size=4
        global_replica_groups = create_global_replica_group(world_size, current_group_ranks)

        input_shape = input_tensor_list[0].shape

        def collective_fn():
            # Move temp allocation inside collective_fn so it happens on the correct stream
            # when async_op=True (collective runs on neuron_stream, not default stream)
            concat_shape = (group_size * input_shape[0], *input_shape[1:])
            temp_result = torch.empty(
                concat_shape, dtype=input_tensor_list[0].dtype, device=input_tensor_list[0].device
            )

            # Execute the collective operation (returns concatenated result)
            # The XLA implementation now returns a single concatenated tensor
            all_to_all_op(input_tensor_list, global_replica_groups, out=(temp_result,))

            # Step 3: Split the result back into individual tensors for each rank
            # This logic was moved from the XLA HLO to the backend level
            for i in range(group_size):
                # Calculate slice indices for this rank's data
                start_idx = i * input_shape[0]
                end_idx = (i + 1) * input_shape[0]

                # Slice the concatenated result and copy to the corresponding output tensor
                output_tensor_list[i].copy_(temp_result[start_idx:end_idx])

        return _ret_work(
            self,
            list(output_tensor_list),
            "alltoall",
            opts,
            collective_fn=collective_fn,
            inputs=list(input_tensor_list),
        )

    def alltoall_base(
        self,
        output,
        input,
        output_split_sizes,
        input_split_sizes,
        opts: AllToAllOptions | None = None,
    ):
        """Base all-to-all collective operation for Neuron devices.

        This function implements the base all-to-all collective operation with support
        for custom split sizes. Due to XLA AllToAll constraints, only specific split
        size patterns are supported.

        Args:
            output: Output tensor to receive the gathered data.
            input: Input tensor to be scattered across ranks.
            output_split_sizes: List specifying how to split the output tensor.
                               Can be None (equal splits) or list of all 1s.
            input_split_sizes: List specifying how to split the input tensor.
                              Can be None (equal splits) or list of all 1s.
            opts: All-to-all options (currently unused but kept for interface compatibility).

        Returns:
            A Work object for asynchronous operation tracking.

        Note:
            Due to XLA AllToAll constraints, only the following split size patterns are supported:
            1. None (defaults to equal splits where each chunk has size tensor_size // world_size)
            2. Lists of all 1s (e.g., [1, 1, 1, 1] for world_size=4)

            Any other split size configuration will raise NotImplementedError.
        """
        if opts is None:
            opts = AllToAllOptions()
        group_size = len(dist.get_process_group_ranks(self))

        # Validate split sizes - only None or lists of all 1s are supported
        def _validate_split_sizes(split_sizes, tensor_name):
            if (
                split_sizes is None
                or len(split_sizes) == 0
                or all(size == 1 for size in split_sizes)
            ):
                return  # None is valid (equal splits)

            if not isinstance(split_sizes, list | tuple):
                raise NotImplementedError(
                    f"alltoall_base: {tensor_name}_split_sizes must be None or a list/tuple, "
                    f"got {type(split_sizes)}"
                )

            if len(split_sizes) != group_size:
                raise NotImplementedError(
                    f"alltoall_base: {tensor_name}_split_sizes length ({len(split_sizes)}) "
                    f"must match group_size ({group_size})"
                )

            # Check if all elements are 1 (the only supported non-None pattern)
            if not all(size == 1 for size in split_sizes):
                raise NotImplementedError(
                    f"alltoall_base: Only even split sizes "
                    f"are supported. Got {tensor_name}_split_sizes: {split_sizes}"
                )

            raise NotImplementedError(f"Unsupported split_sizes found: {split_sizes} ")

        _validate_split_sizes(output_split_sizes, "output")
        _validate_split_sizes(input_split_sizes, "input")

        # For supported cases (None or all 1s), we can use the regular alltoall implementation
        # Split input tensor into chunks for each rank
        if input_split_sizes is None or len(input_split_sizes) == 0:
            # Equal splits
            input_split_sizes = input.shape[0] // group_size
        input_tensor_list = torch.split(input, input_split_sizes, dim=0)

        # Create output tensor list to receive results
        if output_split_sizes is None or len(output_split_sizes) == 0:
            # Equal splits
            output_split_sizes = output.shape[0] // group_size
        output_tensor_list = torch.split(output, output_split_sizes, dim=0)

        # Call the existing alltoall implementation
        return self.alltoall(output_tensor_list, input_tensor_list, opts)

    def gather(
        self,
        output_tensors_list: list[list[torch.Tensor]],
        input_tensor_list: list[torch.Tensor],
        opts: GatherOptions | None = None,
    ):
        """Gather tensors from all ranks to root rank.

        Args:
            output_tensors_list: Nested list to store gathered tensors (only on root)
            input_tensor_list: List of tensors to gather
            opts: Gather options containing root rank

        Returns:
            Work object for async tracking (or None for non-root ranks)
        """
        if opts is None:
            opts = GatherOptions()
        self._validate_root_rank(opts)

        replica_groups = [dist.get_process_group_ranks(self)]

        # Create dummy output list for non root ranks
        if self.rank() != opts.rootRank:
            output_tensors_list = [
                torch.empty(
                    input_tensor_list[0].shape[0],
                    *input_tensor_list[0].shape[1:],
                    dtype=input_tensor_list[0].dtype,
                    device=input_tensor_list[0].device,
                )
            ] * dist.get_world_size(self)
            output_tensors_list = [output_tensors_list]

        def collective_fn():
            # TODO: For perf optimization we can possibly not slice on non root ranks
            all_gather_op(
                input_tensor_list,
                replica_groups,
                slice_output=True,
                opts=opts,
                out=tuple(output_tensors_list[0]),
            )

        # Flatten output_tensors_list for stashing
        all_outputs = [t for sublist in output_tensors_list for t in sublist]
        return _ret_work(
            self,
            all_outputs,
            "gather",
            opts,
            collective_fn=collective_fn,
            inputs=input_tensor_list,
        )

    def scatter(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensors_list: list[list[torch.Tensor]],
        opts: ScatterOptions | None = None,
    ):
        """Scatter tensors from root rank to all ranks.

        Args:
            output_tensor_list: List of output tensors (one per rank)
            input_tensors_list: Nested list of input tensors (only used on root)
            opts: Scatter options containing root rank

        Returns:
            Work object for async tracking

        Note:
            Implemented using reduce_scatter with SUM operation.
        """
        if opts is None:
            opts = ScatterOptions()
        if self.rank() == opts.rootRank:
            assert (
                len(input_tensors_list[0]) == self.size()
            ), "Incorrect number of inputs for scatter"
            inputs = input_tensors_list
        else:
            inputs = [
                [torch.zeros_like(output_tensor)] * self.size()
                for output_tensor in output_tensor_list
            ]

        rs_opts = ReduceScatterOptions()
        rs_opts.reduceOp = dist.ReduceOp.SUM
        return self.reduce_scatter(output_tensor_list, inputs, rs_opts)

    def recv_anysource(self, *args):
        raise NotImplementedError

    def monitored_barrier(self, *args):
        raise NotImplementedError

    def Options(self, *args):  # noqa N802
        raise NotImplementedError

    def __del__(self):
        """Clean up watchdog on process group destruction.

        Note: This may be called during Python GC, but the atexit handler
        should have already stopped the watchdog. This is a backup for
        explicit del calls during normal execution.
        """
        if hasattr(self, "_watchdog"):
            # Best effort - watchdog may already be stopped by atexit handler
            with contextlib.suppress(Exception):
                self._watchdog.stop()
