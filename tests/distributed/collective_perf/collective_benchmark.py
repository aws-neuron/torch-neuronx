"""
Collective operations performance benchmark.

Measures bandwidth utilization and latency for distributed collective operations.

Usage:
    torchrun --nproc_per_node=32 collective_benchmark.py \
        --op all_reduce --sizes 1M,10M,100M
    torchrun --nproc_per_node=32 collective_benchmark.py \
        --op all --sizes 1M
        # --op all: pipeline perf only, scales data internally
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass
class BenchmarkResult:
    """Result of a single collective benchmark."""

    op_name: str
    data_bytes: int
    world_size: int
    latency_ms: float
    latency_std_ms: float
    algo_bandwidth_gbs: float


def setup_distributed():
    """Initialize distributed process group with neuron backend."""
    import torch_neuronx
    from torch_neuronx.distributed.backend import _register_neuron_backend

    if "neuron" not in dist.Backend.backend_type_map:
        _register_neuron_backend()

    dist.init_process_group(backend="neuron")

    return dist.get_rank(), dist.get_world_size(), "neuron"


BYTES_PER_GB = 1e9


def calculate_bandwidth(per_rank_bytes, time_sec, world_size, op_type):
    """
    Calculate achieved bandwidth based on collective type.

    Uses ring-algorithm data movement as theoretical minimum baseline:
    - AllReduce: 2 * (n-1)/n * per_rank_bytes (reduce-scatter + all-gather)
    - AllGather/ReduceScatter: (n-1) * per_rank_bytes

    Ring represents the minimum data that must be moved. Actual Neuron
    algorithms (Mesh, RDH, Bandwidth-Optimal) may move more data to
    achieve better latency or link utilization.

    """
    if op_type == "all_reduce":
        algo_bytes = 2 * (world_size - 1) / world_size * per_rank_bytes
    elif op_type in ["all_gather_into_tensor", "reduce_scatter_tensor"]:
        algo_bytes = (world_size - 1) * per_rank_bytes
    else:
        algo_bytes = per_rank_bytes

    algo_bandwidth = algo_bytes / time_sec / BYTES_PER_GB  # GB/s
    return algo_bandwidth


def benchmark_collective(op_fn, tensor, world_size, op_name, warmup=5, iterations=10):
    """
    Benchmark a collective operation.

    Args:
        op_fn: Function that performs the collective (takes tensor, returns None)
        tensor: Input tensor
        world_size: Number of ranks
        op_name: Name of the operation for bandwidth calculation
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        BenchmarkResult
    """
    data_bytes = tensor.numel() * tensor.element_size()

    # Warmup
    for _ in range(warmup):
        dist.barrier()
        op_fn(tensor)
        torch.neuron.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        op_fn(tensor)
        torch.neuron.synchronize()
        latencies.append(time.perf_counter() - start)

    avg_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    algo_bw = calculate_bandwidth(data_bytes, avg_latency, world_size, op_name)

    return BenchmarkResult(
        op_name=op_name,
        data_bytes=data_bytes,
        world_size=world_size,
        latency_ms=avg_latency * 1000,
        latency_std_ms=std_latency * 1000,
        algo_bandwidth_gbs=algo_bw,
    )


def create_collective_ops(device, world_size, elements_per_rank, dtype, scale_all_reduce=False):
    """Create collective operations and their input tensors.

    Args:
        device: Device to create tensors on
        world_size: Number of ranks
        elements_per_rank: Base number of elements per rank
        dtype: Data type for tensors
        scale_all_reduce: If True, scale all_reduce tensor by world_size to match
            data movement of all_gather_into_tensor/reduce_scatter_into_tensor. Used with
            --op all to ensure comparable data movement across all collective types.
    """
    ops = {}

    # AllReduce - optionally scale to match data movement of other ops
    ar_elements = elements_per_rank * world_size if scale_all_reduce else elements_per_rank
    tensor = torch.ones(ar_elements, device=device, dtype=dtype)
    ops["all_reduce"] = (lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM), tensor)

    # AllGatherIntoTensor
    tensor = torch.ones(elements_per_rank, device=device, dtype=dtype)
    output_tensor = torch.zeros(elements_per_rank * world_size, device=device, dtype=dtype)
    ops["all_gather_into_tensor"] = (
        lambda t, ot=output_tensor: dist.all_gather_into_tensor(ot, t),
        tensor,
    )

    # ReduceScatterTensor
    input_tensor = torch.ones(elements_per_rank * world_size, device=device, dtype=dtype)
    output_tensor = torch.zeros(elements_per_rank, device=device, dtype=dtype)
    ops["reduce_scatter_tensor"] = (
        lambda t, it=input_tensor: dist.reduce_scatter_tensor(t, it, op=dist.ReduceOp.SUM),
        output_tensor,
    )

    return ops


def parse_size(size_str):
    """Parse size string like '1M', '100K', '1G' to number of elements."""
    size_str = size_str.upper().strip()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}

    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * mult)

    return int(size_str)


def format_bytes(num_bytes):
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description="Collective operations benchmark")
    parser.add_argument(
        "--op",
        type=str,
        default="all",
        help="Operation (all_reduce, all_gather_into_tensor, reduce_scatter_tensor, all)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1M",
        help="Comma-separated tensor sizes (e.g., '1K,1M,10M,100M')",
    )
    parser.add_argument("--warmup_runs", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--benchmark_runs", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="collective_benchmark_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    results = []

    for elements_per_rank in sizes:
        # When running all ops, scale all_reduce to match data movement of others
        scale_ar = args.op == "all"
        ops = create_collective_ops(device, world_size, elements_per_rank, dtype, scale_ar)

        ops_to_run = ops if args.op == "all" else {args.op: ops[args.op]} if args.op in ops else {}

        for op_name, (op_fn, tensor) in ops_to_run.items():
            result = benchmark_collective(
                op_fn,
                tensor,
                world_size,
                op_name,
                warmup=args.warmup_runs,
                iterations=args.benchmark_runs,
            )
            results.append(result)

    # Save results to JSON if requested (rank 0 only)
    if args.save_results and rank == 0:
        output_data = {
            "benchmark_type": "collective",
            "config": {
                "op": args.op,
                "sizes": args.sizes,
                "dtype": args.dtype,
                "warmup_runs": args.warmup_runs,
                "benchmark_runs": args.benchmark_runs,
                "world_size": world_size,
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
