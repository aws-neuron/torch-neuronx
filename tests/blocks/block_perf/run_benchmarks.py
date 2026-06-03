"""Generic benchmark runner for transformer blocks."""

import argparse
import contextlib

# Block modules will be imported dynamically based on --block argument
import importlib
import json
import os
import sys
import time
from collections.abc import Generator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.profiler import ProfilerActivity, profile
from utils import printf

from torch_neuronx.profiling import NeuronConfig, NeuronProfiler


def find_project_root(start_path):
    """Find project root by looking for setup.py or .git directory."""
    current = os.path.abspath(start_path)
    while current != "/":
        # Check for typical project root indicators
        if any(
            os.path.exists(os.path.join(current, marker))
            for marker in ["setup.py", "pyproject.toml", ".git"]
        ):
            return current
        current = os.path.dirname(current)
    return None


# Find and add project root to path
project_root = find_project_root(os.path.dirname(__file__))
if not project_root:
    raise RuntimeError(
        "Could not find project root. Expected to find setup.py, pyproject.toml, or .git directory."
    )

sys.path.insert(0, project_root)

# Add the parent of block_perf (which contains both block_perf and block_def)
# This allows importing from sibling directories
blocks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if blocks_dir not in sys.path:
    sys.path.insert(0, blocks_dir)


def maybe_enable_autocast(
    enable: bool, dtype: torch.dtype = torch.bfloat16, device_type: str = "neuron"
) -> Generator[None, None, None]:
    """Context manager to conditionally enable autocast."""
    if enable:
        return torch.autocast(device_type, dtype=dtype)
    else:
        return contextlib.nullcontext()


class StackedBlocks(nn.Module):
    """Generic wrapper to stack multiple identical blocks."""

    def __init__(self, block_fn, n_layers, *block_args, **block_kwargs):
        """
        Args:
            block_fn: Function to create a single block
            n_layers: Number of layers to stack
            *block_args: Arguments to pass to block_fn
            **block_kwargs: Keyword arguments to pass to block_fn
        """
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [block_fn(*block_args, **block_kwargs) for _ in range(n_layers)]
        )

    def forward(self, x, *args, **kwargs):
        """Forward pass through all stacked layers."""
        for layer in self.layers:
            result = layer(x, *args, **kwargs)
            x = result[0] if isinstance(result, tuple) else result
        return x


def setup_device(device_name):
    """
    Setup and validate device.

    Args:
        device_name: Device name (cpu, cuda, neuron)

    Returns:
        torch.device object
    """
    if device_name == "neuron":
        try:
            import torch_neuronx  # Register neuron device
        except ImportError:
            print("Warning: torch_neuronx not found, falling back to CPU")
            device_name = "cpu"

    return torch.device(device_name)


def setup_distributed(device_type):
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        print("Single chip benchmark")
        return
    world_size = int(world_size)
    rank = int(os.environ.get("RANK", "0"))
    if device_type == "neuron":
        import torch_neuronx

        backend = "neuron"
    elif device_type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


class ModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs

        class Output:
            def __init__(self, tensor):
                self.last_hidden_state = tensor

        return Output(outputs)


def run_block(block, input_tensor, block_module, **kwargs):
    """
    Generic block runner that delegates to block-specific logic.

    Args:
        block: The block to run
        input_tensor: Input tensor or tuple of tensors
        block_module: The block's module (contains run_block function)
        **kwargs: Additional arguments (e.g., is_causal)

    Returns:
        Block output
    """
    outputs = block_module.run_block(block, input_tensor, **kwargs)

    if hasattr(outputs, "last_hidden_state"):
        return outputs

    class Output:
        def __init__(self, tensor):
            self.last_hidden_state = tensor

    return Output(outputs)


def warmup_block(
    block,
    input_tensor,
    warmup_runs,
    is_causal=True,
    forward_only=True,
    block_module=None,
    enable_autocast=False,
    autocast_dtype=None,
    device_name="cpu",
    optimizer=None,
):
    """
    Perform warmup runs to handle compilation/initialization.

    Args:
        block: Block to warmup
        input_tensor: Input tensor for forward pass
        warmup_runs: Number of warmup iterations
        is_causal: Whether to use causal masking
        forward_only: Whether to run only forward or forward+loss+backward
        block_module: Block module containing run_block function
        optimizer: Optional optimizer for optimizer.step() benchmarking

    Returns:
        List of warmup timings in milliseconds
    """
    if forward_only:
        mode_str = "FORWARD"
    else:
        mode_str = "FORWARD + LOSS + BACKWARD"
        if optimizer is not None:
            mode_str += " + OPTIMIZER"
    printf("\n" + "=" * 40)
    printf(f"WARMUP PHASE ({mode_str})")
    printf("=" * 40)

    warmup_timings = []
    for i in range(warmup_runs):
        total_ms, _ = run_block_step(
            block,
            input_tensor,
            is_causal=is_causal,
            forward_only=forward_only,
            block_module=block_module,
            enable_autocast=enable_autocast,
            autocast_dtype=autocast_dtype,
            device_name=device_name,
            optimizer=optimizer,
        )
        warmup_timings.append(total_ms)
        printf(f"  Warmup run {i + 1}: {total_ms:.2f} ms")

    return warmup_timings


def run_block_step(
    block,
    input_tensor,
    is_causal=True,
    forward_only=True,
    block_module=None,
    enable_autocast=False,
    autocast_dtype=None,
    device_name="cpu",
    trace_json=None,
    optimizer=None,
):
    """
    Run a single block step, measuring forward and optionally backward times.

    When forward_only is True, use torch.no_grad() and measure only forward.

    Returns (total_ms, prof)
    """
    prof = None
    total_ms = None

    # Enable/disable grad based on mode
    context_manager = torch.no_grad() if forward_only else torch.set_grad_enabled(True)

    with context_manager:
        if not forward_only:
            block.zero_grad(set_to_none=True)

        with maybe_enable_autocast(enable_autocast, autocast_dtype, device_name):

            def execute_block():
                start_time = time.perf_counter()
                if forward_only:
                    _ = run_block(block, input_tensor, block_module, is_causal=is_causal)
                else:
                    outputs = run_block(block, input_tensor, block_module, is_causal=is_causal)
                    loss = compute_loss(outputs)
                    loss.backward()
                    # Apply optimizer step if provided
                    if optimizer is not None:
                        optimizer.step()
                torch.neuron.synchronize()
                return (time.perf_counter() - start_time) * 1000

            total_ms = execute_block()
    return total_ms, prof


def benchmark_passes(
    block,
    input_tensor,
    benchmark_runs,
    is_causal=True,
    forward_only=True,
    block_module=None,
    enable_autocast=False,
    autocast_dtype=None,
    device_name="cpu",
    optimizer=None,
):
    """Benchmark pass timings."""
    if forward_only:
        mode_str = "FORWARD"
    else:
        mode_str = "FORWARD + LOSS + BACKWARD"
        if optimizer is not None:
            mode_str += " + OPTIMIZER"
    printf("\n" + "=" * 40)
    printf(f"BENCHMARK PHASE ({mode_str})")
    printf("=" * 40)

    timings = []
    for i in range(benchmark_runs):
        total_ms, _ = run_block_step(
            block,
            input_tensor,
            is_causal=is_causal,
            forward_only=forward_only,
            block_module=block_module,
            enable_autocast=enable_autocast,
            autocast_dtype=autocast_dtype,
            device_name=device_name,
            optimizer=optimizer,
        )
        timings.append(total_ms)
        printf(f"  Run {i + 1}: {total_ms:.2f} ms")

    return timings


def compute_loss(outputs) -> torch.Tensor:
    """Squash the scalar loss to a small magnitude to avoid large gradients.

    Uses tanh to bound the loss in [-1, 1].
    """
    return torch.tanh(outputs.last_hidden_state.sum())


def profile_block_step(
    block,
    input_tensor,
    is_causal=True,
    forward_only=False,
    block_module=None,
    enable_autocast=False,
    autocast_dtype=None,
    device_name="cpu",
    trace_json=None,
    trace_steps=3,
    optimizer=None,
    neuron_profile=False,
    with_stack=False,
):
    """Profile either forward-only or fwd+loss+bwd as one region."""
    if forward_only:
        title = "PROFILER ANALYSIS (FORWARD)"
    else:
        title = "PROFILER ANALYSIS (FWD+LOSS+BWD"
        if optimizer is not None:
            title += "+OPT"
        title += ")"
    printf("\n" + "=" * 40)
    printf(title)
    printf("=" * 40)

    for i in range(trace_steps):
        if neuron_profile:
            neuron_config = NeuronConfig(
                profile_output_dir="./output",
            )
            neuron_profiler = NeuronProfiler(neuron_config)
            profiler_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                record_shapes=True,
                experimental_config=neuron_config,
                with_stack=with_stack,
            )
        else:
            neuron_profiler = None
            profiler_ctx = profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
            )
        with profiler_ctx as prof:
            total_ms, _ = run_block_step(
                block,
                input_tensor,
                is_causal=is_causal,
                forward_only=forward_only,
                block_module=block_module,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                device_name=device_name,
                trace_json=trace_json,
                optimizer=optimizer,
            )
        prefix = "\n" if i == 0 else ""
        printf(f"{prefix}Run {i} total ms for run_block_step: {total_ms}.")

    if neuron_profile:
        printf("\nDetailed Time Breakdown (sorted by self CPU time):")
        printf(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        if trace_json:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                neuron_profiler.export_trace(prof, filename="trace.json")
                printf(f"\nTrace exported to: " f"{neuron_config.get_session_dir()}/trace.json")
            if dist.is_initialized():
                dist.barrier()
    else:
        printf("\nDetailed Time Breakdown (sorted by self CPU time):")
        printf(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        if trace_json:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                prof.export_chrome_trace(trace_json)
                printf(f"\nTrace exported to: {trace_json}")
            if dist.is_initialized():
                dist.barrier()
    return prof


def calculate_mfu_trn2(tflops):
    """
    Calculate Block FLOPs Utilization for AWS TRN2.

    Args:
        tflops: Achieved TFLOPS

    Returns:
        Dictionary with MFU metrics
    """
    from torch_neuronx.utils import get_logical_neuron_cores

    # TRN2 single neuron core specifications:
    # - 128x128 systolic array
    # - 2 FLOPs per cycle per processing unit (BF16)
    # - 2.4 GHz clock rate
    systolic_array_size = 128 * 128  # 16,384 processing units
    flops_per_cycle = 2  # BF16 operations
    clock_rate_ghz = 2.4
    num_devices = dist.get_world_size() if dist.is_initialized() else 1
    lnc = int(get_logical_neuron_cores())

    # Calculate theoretical peak in FLOPS
    theoretical_peak_flops = (
        systolic_array_size * flops_per_cycle * clock_rate_ghz * 1e9 * num_devices * lnc
    )
    theoretical_peak_tflops = theoretical_peak_flops / 1e12

    mfu = (tflops / theoretical_peak_tflops) * 100

    return {
        "systolic_array_size": systolic_array_size,
        "flops_per_cycle": flops_per_cycle,
        "clock_rate_ghz": clock_rate_ghz,
        "theoretical_peak_tflops": theoretical_peak_tflops,
        "achieved_tflops": tflops,
        "mfu_percent": mfu,
        "total_devices": num_devices,
        "lnc": lnc,
    }


def collect_memory_stats(device_name):
    """Collect and print memory stats for neuron devices."""
    if device_name != "neuron":
        return None
    import torch_neuronx

    stats = torch_neuronx.memory_stats()
    memory_stats = {
        "allocated_bytes_current": stats["allocated_bytes.all.current"],
        "allocated_bytes_peak": stats["allocated_bytes.all.peak"],
        "num_alloc_retries": stats["num_alloc_retries"],
        "num_ooms": stats["num_ooms"],
    }
    printf("\nMemory Stats:")
    printf(f"  Current allocated: {memory_stats['allocated_bytes_current'] / 1e9:.2f} GB")
    printf(f"  Peak allocated: {memory_stats['allocated_bytes_peak'] / 1e9:.2f} GB")
    printf(f"  Alloc retries (OOM): {memory_stats['num_alloc_retries']}")
    printf(f"  OOMs: {memory_stats['num_ooms']}")
    return memory_stats


def print_performance_metrics(timings, total_flops, device_name, forward_only=True):
    """
    Calculate and print performance metrics.

    Args:
        timings: List of benchmark timings in milliseconds
        total_flops: Total FLOPs for the operation (forward or forward+backward)
        device_name: Device name for MFU calculation
        forward_only: Whether metrics are for forward-only or forward+backward
    """
    printf("\n" + "=" * 40)
    printf("PERFORMANCE METRICS")
    printf("=" * 40)

    # Calculate statistics
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    printf("\nExecution Time Statistics:")
    printf(f"  Average: {avg_time:.2f} ms")
    printf(f"  Std Dev: {std_time:.2f} ms")
    printf(f"  Min: {min_time:.2f} ms")
    printf(f"  Max: {max_time:.2f} ms")

    # Calculate throughput
    flops_per_sec = total_flops / (avg_time / 1000)
    tflops = flops_per_sec / 1e12

    operation_type = "forward pass" if forward_only else "forward+loss+backward"
    printf("\nFLOPs Analysis:")
    printf(f"  Total FLOPs per {operation_type}: {total_flops:,}")
    printf(f"  Throughput: {tflops:.2f} TFLOPS")

    # Calculate MFU for Neuron devices
    mfu_metrics = None
    if device_name == "neuron":
        mfu_metrics = calculate_mfu_trn2(tflops)
        printf("\nBlock FLOPs Utilization (BFU):")
        printf("  TRN2 Single Core Configuration:")
        printf(f"    - Systolic array: 128x128 ({mfu_metrics['systolic_array_size']:,} units)")
        printf(f"    - FLOPs per cycle: {mfu_metrics['flops_per_cycle']} (BF16)")
        printf(f"    - Clock rate: {mfu_metrics['clock_rate_ghz']} GHz")
        printf(f"  Stats (Devices={mfu_metrics['total_devices']}, LNC={mfu_metrics['lnc']}):")
        printf(f"    - Theoretical Peak: {mfu_metrics['theoretical_peak_tflops']:.2f} TFLOPS")
        printf(f"    - Achieved: {mfu_metrics['achieved_tflops']:.2f} TFLOPS")
        printf(f"    - MFU: {mfu_metrics['mfu_percent']:.2f}%")

    return {
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "tflops": tflops,
        "total_flops": total_flops,
        "mfu_metrics": mfu_metrics,
    }


def benchmark_block(
    block_module,
    config,
    device_name="cpu",
    warmup_runs=3,
    benchmark_runs=5,
    is_causal=True,
    preset_name=None,
    block_type=None,
    forward_only=False,
    n_layers=1,
    dtype=torch.float32,
    tp_size=1,
    autocast_dtype=None,
    trace_json=None,
    trace_steps=3,
    with_optimizer=False,
    neuron_profile=False,
    with_stack=False,
    exec_mode="eager",
):
    """
    Generic benchmark function for transformer blocks.

    Args:
        block_module: Block module with create_block and count_flops functions
        config: Block configuration dictionary
        device_name: Device to run on
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        is_causal: Whether to use causal masking
        preset_name: Name of the preset configuration used
        block_type: Type of block being benchmarked

    Returns:
        Dictionary with benchmark results
    """
    printf("=" * 80)
    printf("BLOCK PERFORMANCE BENCHMARK")
    printf("=" * 80)
    printf("\nConfiguration:")

    if block_type:
        printf(f"  Block type: {block_type}")
    if preset_name:
        printf(f"  Preset: {preset_name}")

    for key, value in config.items():
        printf(f"  {key}: {value}")
    printf(f"  Number of layers: {n_layers}")
    printf(f"  Device: {device_name}")
    printf(f"  Data type: {dtype}")
    enable_autocast = autocast_dtype is not None and dtype == torch.float32
    if enable_autocast:
        printf(f"  Autocast: enabled with dtype {autocast_dtype}")
    else:
        printf("  Autocast: disabled")
    # Build mode string
    if forward_only:
        mode_str = "Forward only"
    else:
        mode_str = "Forward + Loss + Backward"
        if with_optimizer:
            mode_str += " + Optimizer"
    printf(f"  Mode: {mode_str}")
    printf(f"  Execution: {exec_mode}")
    printf(f"  Warmup runs: {warmup_runs}")
    printf(f"  Benchmark runs: {benchmark_runs}")

    # Setup device
    device = setup_device(device_name)

    tp_mesh = None
    dp_mesh = None
    if torch.distributed.is_initialized():
        dp_size = torch.distributed.get_world_size() // tp_size
        device_mesh = init_device_mesh(device.type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]

    # Create block(s) using module's create_block function
    if n_layers == 1:
        # Single block (original behavior)
        block = block_module.create_block(config, device, dtype, tp_mesh, dp_mesh)
    else:
        # Stack multiple blocks
        block = StackedBlocks(
            block_module.create_block, n_layers, config, device, dtype, tp_mesh, dp_mesh
        )

    if exec_mode == "compile":
        block = torch.compile(block, backend="neuron")

    # Create input tensor
    if block_type == "qwen3":
        # Qwen3Model expects token indices (input_ids)
        input_tensor = torch.randint(
            0,
            config["vocab_size"],
            (config["batch_size"], config["seq_len"]),
            device=device,
            dtype=torch.long,
        )
    elif block_type == "rope":
        # RoPE expects query and key tensors: [batch, seq_len, num_heads, head_dim]
        xq = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["num_attention_heads"],
            config["head_dim"],
            device=device,
            dtype=dtype,
        )
        xk = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["num_key_value_heads"],
            config["head_dim"],
            device=device,
            dtype=dtype,
        )
        input_tensor = (xq, xk)
    else:
        # llama decoder layers and rmsnorm expect continuous tensors
        input_tensor = torch.randn(
            config["batch_size"],
            config["seq_len"] // tp_size,
            config["hidden_size"],
            device=device,
            dtype=dtype,
        )
    # Convert to DTensor for TP
    if tp_mesh and block_type == "rms_norm":
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            input_tensor = DTensor.from_local(input_tensor, tp_mesh, [Replicate()])
        else:
            input_tensor = DTensor.from_local(input_tensor, tp_mesh, [Shard(1)])

    # Create optimizer if requested
    optimizer = None
    if with_optimizer and not forward_only:
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)
        num_params = sum(p.numel() for p in block.parameters())
        printf(f"\nCreated AdamW optimizer (lr=1e-4) for {num_params} parameters")

    # Warmup + Benchmark + Profile
    warmup_timings = warmup_block(
        block,
        input_tensor,
        warmup_runs,
        is_causal,
        forward_only,
        block_module,
        enable_autocast,
        autocast_dtype,
        device_name,
        optimizer,
    )
    benchmark_timings = benchmark_passes(
        block,
        input_tensor,
        benchmark_runs,
        is_causal,
        forward_only,
        block_module,
        enable_autocast,
        autocast_dtype,
        device_name,
        optimizer,
    )

    # Profile
    if trace_json:
        profile_block_step(
            block,
            input_tensor,
            is_causal,
            forward_only,
            block_module,
            enable_autocast,
            autocast_dtype,
            device_name,
            trace_json=trace_json,
            trace_steps=trace_steps,
            optimizer=optimizer,
            neuron_profile=neuron_profile,
            with_stack=with_stack,
        )

    # Calculate FLOPs
    # Forward pass FLOPs from the module (for a single block)
    single_block_forward_flops = block_module.count_flops(config)
    # Multiply by number of layers
    forward_flops = single_block_forward_flops * n_layers
    # For backward pass, we use 2x forward FLOPs as standard approximation
    # Total for forward+loss+backward = forward + 2*forward = 3*forward
    total_flops = forward_flops if forward_only else forward_flops * 3

    # For FSDP: all ranks compute the same FLOPs, scale by dp_size
    if dist.is_initialized():
        dp_size = dist.get_world_size() // tp_size
        total_flops = total_flops * dp_size  ## Total system FLOPS

    # Print metrics
    metrics = print_performance_metrics(benchmark_timings, total_flops, device_name, forward_only)

    # Collect memory stats
    memory_stats = collect_memory_stats(device_name)

    printf("\n" + "=" * 80)

    return {
        "config": config,
        "mode": "forward_only" if forward_only else "forward_backward",
        "warmup_timings": warmup_timings,
        "benchmark_timings": benchmark_timings,
        "metrics": metrics,
        "memory_stats": memory_stats,
    }


def main():
    # First, create a parser just to get the block type
    parser_temp = argparse.ArgumentParser(add_help=False)
    parser_temp.add_argument(
        "--block",
        type=str,
        default="llama",
        choices=[
            "llama",
            "gpt_oss",
            "gpt_oss_transformer",
            "qwen3",
            "qwen3_torchtitan",
            "rms_norm",
            "rope",
            "mixtral_moe",
        ],
    )
    args_temp, _ = parser_temp.parse_known_args()

    # Dynamically import the block module
    try:
        block_module = importlib.import_module(f"block_perf.{args_temp.block}")
    except ImportError as e:
        raise ImportError(f"Could not import block module 'block_perf.{args_temp.block}'") from e

    # Now create the full parser with all arguments
    parser = argparse.ArgumentParser(description="Benchmark Transformer Block Performance")

    # Block configuration
    parser.add_argument(
        "--block",
        type=str,
        default="llama",
        choices=[
            "llama",
            "qwen3",
            "qwen3_torchtitan",
            "rms_norm",
            "gpt_oss",
            "gpt_oss_transformer",
            "rope",
            "mixtral_moe",
        ],
        help="Block type to benchmark (default: llama)",
    )

    # Add preset argument with dynamic choices from the loaded module
    parser.add_argument(
        "--preset",
        type=str,
        default="llama-7b",
        choices=block_module.get_presets(),
        help="Use preset configuration for specific model sizes",
    )

    # Input configuration
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (default: 2048)")

    # Benchmark configuration
    parser.add_argument(
        "--warmup_runs", type=int, default=1, help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--benchmark_runs", type=int, default=3, help="Number of benchmark runs K (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="neuron",
        choices=["neuron", "cpu", "cuda"],
        help="Device to run on (default: neuron)",
    )

    # Output configuration
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    # Profiling options
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Profile only the forward pass (default profiles fwd+loss+bwd)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of identical blocks to stack (default: 1)",
    )

    # Dtype configuration
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model and inputs (default: bfloat16)",
    )

    parser.add_argument(
        "--autocast_dtype",
        type=str,
        choices=["bfloat16", "float16"],
        help="Autocast dtype (only enabled when --dtype is float32)",
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel degree (default: 1)",
    )

    parser.add_argument(
        "--trace_json",
        type=str,
        help="Export Chrome trace JSON to specified file",
    )

    parser.add_argument(
        "--trace_steps",
        type=int,
        default=3,
        help="How many warmup steps for the profiler before it captures.",
    )

    parser.add_argument(
        "--with-optimizer",
        action="store_true",
        help="Include optimizer.step() in backward pass (creates AdamW optimizer)",
    )

    parser.add_argument(
        "--neuron-profile",
        action="store_true",
        help="Enable Neuron/NRT profiling (default: PyTorch profiler only)",
    )
    parser.add_argument(
        "--mfu-threshold",
        type=float,
        default=None,
        help="Minimum MFU %% threshold - fails if below (perf regression gate)",
    )
    parser.add_argument(
        "--peak-memory-gb",
        type=float,
        default=None,
        help="Max peak memory in GB - fails if exceeded (memory regression gate)",
    )

    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Capture Python stack traces with line numbers (default: False)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable NeuronMetrics collection and print full metrics/op-tracking report",
    )

    parser.add_argument(
        "--exec_mode",
        type=str,
        default="eager",
        choices=["eager", "compile"],
        help="Execution mode: eager (default) or compile",
    )

    # Parse all arguments
    args = parser.parse_args()
    setup_distributed(args.device)

    # Enable metrics collection if requested
    if args.metrics:
        import torch_neuronx.metrics as met

        met.set_enabled(True)
        met.clear_all()
        import torch_neuronx

        torch_neuronx.clear_op_tracking()
        # Enable detailed per-type concatenation profiling via centralized metrics toggle
        met.set_concatenation_profiling_enabled(True)

    # Re-import the module (already imported above but we do it again for clarity)
    block_module = importlib.import_module(f"block_perf.{args.block}")

    printf(f"\nUsing preset configuration: {args.preset}")

    # Create configuration using the block module's method
    config = block_module.create_config(args.preset, args.batch_size, args.seq_len)

    # Map string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_mapping[args.dtype]

    # Handle autocast dtype
    autocast_dtype = None
    if args.autocast_dtype:
        autocast_dtype = dtype_mapping[args.autocast_dtype]

    # Run benchmark
    results = benchmark_block(
        block_module=block_module,
        config=config,
        device_name=args.device,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        preset_name=args.preset,
        block_type=args.block,
        forward_only=args.forward_only,
        n_layers=args.n_layers,
        dtype=dtype,
        tp_size=args.tp_size,
        autocast_dtype=autocast_dtype,
        trace_json=args.trace_json,
        trace_steps=args.trace_steps,
        with_optimizer=args.with_optimizer,
        neuron_profile=args.neuron_profile,
        with_stack=args.with_stack,
        exec_mode=args.exec_mode,
    )

    # Save results if requested
    if args.save_results and (
        (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()
    ):
        output_data = {
            "block_type": args.block,
            "device": args.device,
            "exec_mode": args.exec_mode,
            "results": {
                "config": results["config"],
                "mode": results.get("mode"),
                "warmup_timings": results["warmup_timings"],
                "benchmark_timings": results["benchmark_timings"],
                "metrics": results["metrics"],
                "memory_stats": results.get("memory_stats"),
            },
        }

        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        printf(f"\nResults saved to {args.output_file}")

    # Perf regression check (only on rank 0)
    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        if args.mfu_threshold is not None:
            mfu_metrics = results["metrics"].get("mfu_metrics")
            if mfu_metrics:
                actual_mfu = mfu_metrics["mfu_percent"]
                printf(f"\nPerf gate: MFU {actual_mfu:.2f}% >= {args.mfu_threshold}%")
                assert (
                    actual_mfu >= args.mfu_threshold
                ), f"FAILED: MFU regression: {actual_mfu:.2f}% < {args.mfu_threshold}% minimum"
                printf("PASSED")

        if args.peak_memory_gb is not None:
            mem_stats = results.get("memory_stats")
            if mem_stats:
                peak_gb = mem_stats["allocated_bytes_peak"] / 1e9
                printf(f"\nMemory gate: Peak {peak_gb:.2f} GB <= {args.peak_memory_gb} GB")
                assert (
                    peak_gb <= args.peak_memory_gb
                ), f"FAILED: Memory regression: {peak_gb:.2f} GB > {args.peak_memory_gb} GB max"
                printf("PASSED")

    # Print metrics report if enabled
    if args.metrics:
        import torch_neuronx
        import torch_neuronx.metrics as met

        printf("\n" + "=" * 40)
        printf("NEURON METRICS REPORT")
        printf("=" * 40)

        # Concatenation status diagnostic
        concat_status = torch_neuronx.get_concatenation_status()
        printf("\nConcatenation Status:")
        printf(f"  Enabled: {concat_status['enabled']}")
        printf(f"  Active: {concat_status['is_active']}")
        printf(
            f"  Concatenated ops in executed: {len(concat_status['concatenated_ops_in_executed'])}"
        )
        printf(f"  Individual ops in executed: {len(concat_status['individual_ops_in_executed'])}")
        printf(f"  Groups tracked: {concat_status['concatenated_groups_tracked']}")
        printf(f"  Diagnosis: {concat_status['diagnosis']}")

        # All collected metrics (compilation + concatenation + future)
        all_metrics = met.get_metrics_dict()
        if all_metrics:
            printf("\nCollected Metrics:")
            for name, value in sorted(all_metrics.items()):
                if isinstance(value, dict):
                    printf(f"  {name}: {value.get('formatted', value)}")
                else:
                    printf(f"  {name}: {value}")
        else:
            printf("\nNo metrics collected (enable with TORCH_NEURONX_METRICS_ENABLED=1)")

        # Op tracking
        executed_ops = torch_neuronx.get_executed_ops()
        printf(f"\nExecuted ops ({len(executed_ops)} unique):")
        for op in sorted(executed_ops):
            marker = " [CONCATENATED]" if "|" in op else ""
            printf(f"  {op}{marker}")

        concatenated_ops = torch_neuronx.get_concatenated_ops()
        if concatenated_ops:
            execution_counts = torch_neuronx.get_concatenated_op_execution_counts()
            printf(f"\nConcatenated op groups ({len(concatenated_ops)}):")
            for concat_name, original_ops in concatenated_ops.items():
                count = execution_counts.get(concat_name, 0)
                cache_hits = max(0, count - 1)
                printf(f"  {concat_name}  (executions={count}, cache_hits={cache_hits})")
                for orig in original_ops:
                    printf(f"    <- {orig}")

        # Per-type concatenation profiles
        profiles = torch_neuronx.get_concatenation_profiles()
        if profiles:
            printf(f"\nPer-type concatenation profiles ({len(profiles)} types):")
            for op_name, profile in sorted(profiles.items()):
                printf(f"  {op_name}:")
                printf(f"    Invocations: {profile['invocation_count']}")
                printf(f"    Avg merge time: {profile['avg_merge_time_ms']:.3f} ms")
                printf(f"    Avg group size: {profile['avg_group_size']:.1f} ops")
                if len(profile["merge_times_ns"]) <= 10:
                    times_ms = [t / 1e6 for t in profile["merge_times_ns"]]
                    printf(f"    Merge times (ms): {[f'{t:.3f}' for t in times_ms]}")

        # Full report
        full_report = met.report()
        if full_report:
            printf("\nFull Metrics Report:")
            printf(full_report)

        printf("=" * 40)

    printf("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
