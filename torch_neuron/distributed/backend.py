import os
import socket
import time

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllToAllOptions,
    BroadcastOptions,
    GatherOptions,
    ProcessGroup,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
)

import torch_neuron

from .ops import all_gather_op, all_reduce_op, all_to_all_op, reduce_scatter_op
from .utils import create_global_replica_group, get_free_port, get_reduce_type

_WORLD_SIZE = None


def _create_neuron_process_group(prefix_store, rank, size, timeout):
    return ProcessGroupNeuron(prefix_store, rank, size, timeout)


def _register_neuron_backend():
    dist.Backend.register_backend("neuron", _create_neuron_process_group, devices=["neuron"])


def rendezvous_handler():
    pass


_register_neuron_backend()

dist.register_rendezvous_handler("neuron", rendezvous_handler)


def _ret_work(ret):
    # TODO: Need to implement better version of Work. This basic version works
    # when there is single stream. When we have multi-stream execution,
    # this needs to be implemented.
    fut = torch.futures.Future()
    fut.set_result(ret)
    return torch._C._distributed_c10d._create_work_from_future(fut)


def _set_rt_visible_cores(rank, world_size, store):
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
        not torch_neuron.is_neuron_runtime_initialized()
    ), "Neuron runtime should not be initialized before the init_process_group."
    # Always determine local rank and world size using TCP store method
    local_rank, local_world_size = _set_rt_visible_cores(rank, size, store)
    _set_root_comm_id(rank, local_rank, size, store)

    # Do nrt_init and check if devices per core is 1. We only allow one device per
    # process in the distributed setting.
    torch_neuron._lazy_init()

    # Set up vnc_id mapping
    assert (
        torch_neuron._C._vnc_count() == 1
    ), "Attempted to have more than one device per process in distributed setup"
    torch_neuron._C._set_world_size(size)
    torch_neuron._C._set_rank(rank)
    torch_neuron._C._set_local_world_size(local_world_size)
    torch_neuron._C._set_local_device_start_index(local_rank)

    # We do the barrier immediately so that we reserve the port
    torch_neuron._C._nrt_barrier(0, rank, size)


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
        global _WORLD_SIZE
        if _WORLD_SIZE is None:
            _WORLD_SIZE = size
            _neuron_runtime_setup(rank, size, store)

    def getBackendName(self):  # noqa N802
        return "neuron"

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

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

    def allreduce(self, tensors, all_reduce_options):
        replica_groups = [dist.get_process_group_ranks(self)]
        all_reduce_op(
            tensors,
            replica_groups,
            get_reduce_type(all_reduce_options.reduceOp),
            out=tuple(tensors),
        )
        return _ret_work(tensors)

    def _allgather_base(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, opts: AllgatherOptions
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
        replica_groups = [dist.get_process_group_ranks(self)]

        # Handle dimension mismatch by squeezing the tensor with more dimensions
        if len(output_tensor.shape) != len(input_tensor.shape):
            output_tensor = output_tensor.squeeze()
            input_tensor = input_tensor.squeeze()

        all_gather_op(input_tensor, replica_groups, out=output_tensor)
        return _ret_work(output_tensor)

    def allgather(self, output_tensors_list, input_tensors, opts: AllgatherOptions = None):
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
        replica_groups = [dist.get_process_group_ranks(self)]
        all_gather_op(
            input_tensors,
            replica_groups,
            opts=opts,
            out=tuple(output_tensors_list[0]),
            slice_output=True,
        )
        return _ret_work(output_tensors_list)

    def allgather_coalesced(
        self, output_tensors_list, input_tensors, opts: AllgatherOptions = None
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
        replica_groups = [dist.get_process_group_ranks(self)]
        world_size = len(replica_groups[0])
        # Flatten the nested output structure for XLA backend compatibility
        # XLA expects a flat list of tensors rather than nested structure
        temp_outputs = []
        for input_tensor in input_tensors:
            concat_shape = (world_size * input_tensor.shape[0], *input_tensor.shape[1:])
            temp_output = torch.empty(
                concat_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            temp_outputs.append(temp_output)

        all_gather_op(
            input_tensors,
            replica_groups,
            opts=opts,
            out=tuple(temp_outputs),
            slice_output=False,
        )

        for i, (temp_output, output_list) in enumerate(
            zip(temp_outputs, output_tensors_list, strict=False)
        ):
            input_shape = input_tensors[i].shape
            chunks = torch.split(temp_output, input_shape[0], dim=0)
            # Copy each chunk to the corresponding output tensor
            for j, chunk in enumerate(chunks):
                output_list[j].copy_(chunk)

        return _ret_work(output_tensors_list)

    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions
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
        # Get the ranks of the processes in the current process group.
        replica_groups: list[list[int]] = [dist.get_process_group_ranks(self)]
        # Extract the input tensor from the list.
        input_tensor: torch.Tensor = tensors[0]

        if input_tensor.numel() == 0:
            return _ret_work(tensors)

        self._validate_root_rank(opts)

        # If the current rank is not the root rank, zero out the input tensor.
        # This is a common pattern in broadcast implementations where non-root ranks
        # receive data from the root.
        if self.rank() != opts.rootRank:
            input_tensor.zero_()

        # Perform an all-reduce operation with SUM. In a broadcast, after the root
        # sends its data, an all-reduce can be used to ensure all participants
        # have the correct data (effectively a broadcast if only the root had data initially).
        all_reduce_op(tensors, replica_groups, "SUM", out=tuple(tensors))

        # Return a Work object to track the asynchronous completion of the operation.
        return _ret_work(tensors)

    def reduce_scatter(self, output_tensors, input_tensors_list, reduce_scatter_options):
        # Passing [0] to both input and output as the c10d inside torch wraps both
        # customer inputs and outputs with one more level of list
        replica_groups = [dist.get_process_group_ranks(self)]
        reduce_scatter_op(
            input_tensors_list[0],  # passing input as a list of tensors to PG backend
            replica_groups,
            get_reduce_type(reduce_scatter_options.reduceOp),
            out=output_tensors[0],  # passing output as a tensor to PG backend
        )
        return _ret_work(output_tensors)

    def reduce_scatter_coalesced(self, output_tensors, input_tensors_list, opts):
        raise NotImplementedError(__class__.reduce_scatter_coalesced)

    def _reduce_scatter_base(self, output_tensor, input_tensor, reduce_scatter_options):
        replica_groups = [dist.get_process_group_ranks(self)]
        reduce_scatter_op(
            [input_tensor],
            replica_groups,
            get_reduce_type(reduce_scatter_options.reduceOp),
            out=output_tensor,
        )
        return _ret_work(output_tensor)

    def barrier(self, opts):
        if dist.get_world_size() != self.size():
            raise NotImplementedError(
                f"Barrier is implemented only for default group," f"called barrier on {self}"
            )
        # Call nrt_barrier here
        torch_neuron._C._nrt_barrier(0, self.rank(), self.size())
        return _ret_work(None)

    def reduce(self, tensors: list[torch.Tensor], opts: ReduceOptions):
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

        self._validate_root_rank(opts)
        # only change the tensors on root rank
        output_tensors = (
            tensors
            if self.rank() == opts.rootRank
            else [torch.empty_like(tensor) for tensor in tensors]
        )
        # Get the ranks of the processes in the current process group
        replica_groups = [dist.get_process_group_ranks(self)]

        # Perform all-reduce operation on temporary tensors
        all_reduce_op(
            tensors,
            replica_groups,
            get_reduce_type(opts.reduceOp),
            out=tuple(output_tensors),
        )

        return _ret_work(tensors) if self.rank == opts.rootRank else _ret_work(None)

    def allreduce_coalesced(self, tensors, all_reduce_coalesced_options):
        replica_groups = [dist.get_process_group_ranks(self)]
        all_reduce_op(
            tensors,
            replica_groups,
            get_reduce_type(all_reduce_coalesced_options.reduceOp),
            out=tuple(tensors),
        )
        return _ret_work(tensors)

    def alltoall(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensor_list: list[torch.Tensor],
        opts: AllToAllOptions,
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
        # Early return for empty tensors, not moving this to all_to_all_xla.py
        # as XLA AllToAll doesn't support empty tensors on Neuron
        if all(input_tensor.numel() == 0 for input_tensor in input_tensor_list):
            return _ret_work(output_tensor_list)

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

        # Create a temporary tensor to hold the concatenated result from the collective
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

        return _ret_work(output_tensor_list)

    def alltoall_base(self, output, input, output_split_sizes, input_split_sizes, opts):
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
        opts: GatherOptions,
    ):
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

        # TODO: For perf optimization we can possibly not slice on non root ranks
        all_gather_op(
            input_tensor_list,
            replica_groups,
            slice_output=True,
            opts=opts,
            out=tuple(output_tensors_list[0]),
        )

        return _ret_work(output_tensors_list) if self.rank() == opts.rootRank else _ret_work(None)

    def scatter(
        self,
        output_tensor_list: list[torch.Tensor],
        input_tensors_list: list[list[torch.Tensor]],
        opts: ScatterOptions,
    ):
        if torch.distributed.get_rank() == opts.rootRank:
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
