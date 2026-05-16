"""C++ NeuronBackend integration — Python HLO callback and PG creation."""

import torch
from torch._C._distributed_c10d import OpType

from torch_neuronx._C import LoweringResult, NeuronBackend
from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type
from torch_neuronx.python_ops.xla_builder.compilation import XLABuilderCompiler
from torch_neuronx.python_ops.xla_ops.all_gather_xla import AllGatherXlaOp
from torch_neuronx.python_ops.xla_ops.all_reduce_xla import AllReduceXLAOp
from torch_neuronx.python_ops.xla_ops.all_to_all_xla import AllToAllXlaOp
from torch_neuronx.python_ops.xla_ops.reduce_scatter_xla import ReduceScatterXLAOp
from torch_neuronx.utils import global_version_hash

# Lazily-initialized XLA ops and compiler
_xla_ops = {}
_compiler = None

_OP_TYPE_TO_XLA = {
    OpType.ALLREDUCE: ("allreduce", AllReduceXLAOp),
    OpType.ALLGATHER: ("allgather", AllGatherXlaOp),
    OpType.REDUCE_SCATTER: ("reduce_scatter", ReduceScatterXLAOp),
    OpType.ALLTOALL: ("alltoall", AllToAllXlaOp),
}


def _get_compiler():
    global _compiler
    if _compiler is None:
        _compiler = XLABuilderCompiler()
    return _compiler


def _get_xla_op(op_type):
    if op_type not in _xla_ops:
        entry = _OP_TYPE_TO_XLA.get(op_type)
        if entry is None:
            raise ValueError(f"Unknown collective op_type: {op_type}")
        _xla_ops[op_type] = entry[1]()
    return _xla_ops[op_type]


_REDUCE_INT_TO_STR = {
    0: "SUM",
    1: "AVG",
    2: "PRODUCT",
    3: "MIN",
    4: "MAX",
    5: "BAND",
    6: "BOR",
    7: "BXOR",
    8: "PREMUL_SUM",
    9: "UNUSED",
}


def lower_collective_hlo(op_type, inputs, replica_groups, reduce_op, kwargs):
    """Python lowering callback for C++ NeuronBackend on cache miss.

    Lowers a collective operation to HLO via XLA builder. Called from C++
    with GIL held. Returns a LoweringResult.

    Args:
        op_type: c10d.OpType enum value
        inputs: list[Tensor] — input tensors
        replica_groups: list[list[int]] — replica group configuration
        reduce_op: int — c10d::ReduceOp enum value
        kwargs: dict — additional kwargs from C++

    Returns:
        LoweringResult with stablehlo_bytes, cache_key, has_collectives=True
    """
    xla_op = _get_xla_op(op_type)
    op_name = _OP_TYPE_TO_XLA[op_type][0]
    if xla_op.kernel is None:
        xla_op.kernel = xla_op._create_kernel(xla_op.hlo_fn)
    kernel = xla_op.kernel

    reduce_str = _REDUCE_INT_TO_STR.get(reduce_op, "SUM")

    # Build kwargs for the HLO function
    hlo_kwargs = {"replica_groups": replica_groups, "reduce": reduce_str}
    hlo_kwargs.update(kwargs)

    lowering_inputs = list(inputs)

    # C++ always passes out as a tuple, but reduce_scatter expects a single tensor.
    out = hlo_kwargs.get("out")
    if (
        out is not None
        and op_type == OpType.REDUCE_SCATTER
        and isinstance(out, list | tuple)
        and len(out) == 1
    ):
        hlo_kwargs["out"] = out[0]

    # Generate cache key
    cache_key = kernel._generate_cache_key(kernel.op_name, tuple(lowering_inputs), hlo_kwargs)
    cache_key = f"{cache_key}:{global_version_hash()}"

    # Lower to HLO
    compiler = _get_compiler()
    hlo = compiler.get_or_compile_hlo(cache_key, kernel.hlo_fn, tuple(lowering_inputs), hlo_kwargs)
    hlo_bytes = hlo.module_proto.SerializeToString()

    # Build output specs from the output tensors in kwargs
    out = hlo_kwargs.get("out")
    output_specs = []
    original_output_specs = []
    if out is not None:
        out_tensors = out if isinstance(out, list | tuple) else (out,)
        for t in out_tensors:
            if isinstance(t, torch.Tensor):
                output_specs.append((tuple(t.shape), _dtype_to_scalar_type(t.dtype)))
                original_output_specs.append((tuple(t.shape), _dtype_to_scalar_type(t.dtype)))

    # All inputs are kept for collectives (no dead-code elimination)
    kept_input_indices = list(range(len(lowering_inputs)))

    return LoweringResult(
        stablehlo_bytes=hlo_bytes,
        cache_key=cache_key,
        kept_input_indices=kept_input_indices,
        kept_output_indices=list(range(len(output_specs))),
        output_specs=output_specs,
        original_output_specs=original_output_specs,
        passthrough_mappings={},
        input_specs=[],
        kw_input_specs=[],
        op_name=op_name,
        is_inplace=False,
        has_collectives=True,
        is_empty_tensor=False,
        input_output_aliases=[],
    )


def _dtype_to_scalar_type(dtype):
    """Convert torch.dtype to c10::ScalarType int."""
    scalar_type = torch_dtype_to_scalar_type(dtype)
    if scalar_type is None:
        raise ValueError(f"Unsupported dtype for scalar_type mapping: {dtype}")
    return scalar_type


def create_neuron_cpp_backend(options, rank, size, global_world_size):
    """Create a C++ NeuronBackend process group."""
    return NeuronBackend(options, rank, size, global_world_size, lower_collective_hlo)
