"""Generic benchmark runner for transformer blocks."""

import os

os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "1"
import argparse

# Block modules will be imported dynamically based on --block argument
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


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

import torch.nn as nn  # noqa: E402


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
            x = layer(x, *args, **kwargs)
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
            import torch_neuron  # Register neuron device
        except ImportError:
            print("Warning: torch_neuron not found, falling back to CPU")
            device_name = "cpu"

    return torch.device(device_name)


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


def run_block(block, input_tensor, block_type=None, **kwargs):
    """
    Generic block runner that handles different block types.

    Args:
        block: The block to run
        input_tensor: Input tensor
        block_type: Type of block (determines which kwargs to pass)
        **kwargs: Additional arguments (e.g., is_causal)

    Returns:
        Block output
    """
    outputs = block(input_tensor) if block_type == "rms_norm" else block(input_tensor, **kwargs)

    # Ensure output has last_hidden_state
    if hasattr(outputs, "last_hidden_state"):
        return outputs

    class Output:
        def __init__(self, tensor):
            self.last_hidden_state = tensor

    return Output(outputs)


def warmup_block(
    block, input_tensor, warmup_runs, is_causal=True, forward_only=True, block_type=None
):
    """
    Perform warmup runs to handle compilation/initialization.

    Args:
        block: Block to warmup
        input_tensor: Input tensor for forward pass
        warmup_runs: Number of warmup iterations
        is_causal: Whether to use causal masking
        forward_only: Whether to run only forward or forward+loss+backward
        block_type: Type of block being benchmarked

    Returns:
        List of warmup timings in milliseconds
    """
    mode_str = "FORWARD" if forward_only else "FORWARD + LOSS + BACKWARD"
    print("\n" + "=" * 40)
    print(f"WARMUP PHASE ({mode_str})")
    print("=" * 40)

    warmup_timings = []
    for i in range(warmup_runs):
        total_ms, _ = run_block_step(
            block,
            input_tensor,
            is_causal=is_causal,
            forward_only=forward_only,
            block_type=block_type,
        )
        warmup_timings.append(total_ms)
        print(f"  Warmup run {i+1}: {total_ms:.2f} ms")

    return warmup_timings


def run_block_step(
    block, input_tensor, is_causal=True, forward_only=True, with_profile=False, block_type=None
):
    """
    Run a single block step, measuring forward and optionally backward times.

    When forward_only is True, use torch.no_grad() and measure only forward.
    When with_profile is True, wrap in a profiler context and return the profile.

    Returns (total_ms, prof)
    """
    prof = None
    total_ms = None

    # Enable/disable grad based on mode
    context_manager = torch.no_grad() if forward_only else torch.set_grad_enabled(True)

    with context_manager:
        if not forward_only:
            block.zero_grad(set_to_none=True)

        start_time = time.perf_counter()
        if with_profile:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=False,
            ) as prof:
                if forward_only:
                    _ = run_block(block, input_tensor, block_type, is_causal=is_causal)
                else:
                    outputs = run_block(block, input_tensor, block_type, is_causal=is_causal)
                    loss = compute_loss(outputs)
                    loss.backward()
        else:
            if forward_only:
                _ = run_block(block, input_tensor, block_type, is_causal=is_causal)
            else:
                outputs = run_block(block, input_tensor, block_type, is_causal=is_causal)
                loss = compute_loss(outputs)
                loss.backward()
    torch.neuron.synchronize()
    total_ms = (time.perf_counter() - start_time) * 1000
    return total_ms, prof


def benchmark_passes(
    block, input_tensor, benchmark_runs, is_causal=True, forward_only=True, block_type=None
):
    """Benchmark pass timings."""
    mode_str = "FORWARD" if forward_only else "FORWARD + LOSS + BACKWARD"
    print("\n" + "=" * 40)
    print(f"BENCHMARK PHASE ({mode_str})")
    print("=" * 40)

    timings = []
    for i in range(benchmark_runs):
        total_ms, _ = run_block_step(
            block,
            input_tensor,
            is_causal=is_causal,
            forward_only=forward_only,
            block_type=block_type,
        )
        timings.append(total_ms)
        print(f"  Run {i+1}: {total_ms:.2f} ms")

    return timings


def compute_loss(outputs) -> torch.Tensor:
    """Squash the scalar loss to a small magnitude to avoid large gradients.

    Uses tanh to bound the loss in [-1, 1].
    """
    return torch.tanh(outputs.last_hidden_state.sum())


def profile_block_step(block, input_tensor, is_causal=True, forward_only=False, block_type=None):
    """Profile either forward-only or fwd+loss+bwd as one region."""
    title = "PROFILER ANALYSIS (FORWARD)" if forward_only else "PROFILER ANALYSIS (FWD+LOSS+BWD)"
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    _, prof = run_block_step(
        block,
        input_tensor,
        is_causal=is_causal,
        forward_only=forward_only,
        with_profile=True,
        block_type=block_type,
    )

    print("\nDetailed Time Breakdown (sorted by self CPU time):")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    return prof


def calculate_mfu_trn2(tflops):
    """
    Calculate Block FLOPs Utilization for AWS TRN2.

    Args:
        tflops: Achieved TFLOPS

    Returns:
        Dictionary with MFU metrics
    """
    # TRN2 single neuron core specifications:
    # - 128x128 systolic array
    # - 2 FLOPs per cycle per processing unit (BF16)
    # - 2.4 GHz clock rate
    systolic_array_size = 128 * 128  # 16,384 processing units
    flops_per_cycle = 2  # BF16 operations
    clock_rate_ghz = 2.4

    # Calculate theoretical peak in FLOPS
    theoretical_peak_flops = systolic_array_size * flops_per_cycle * clock_rate_ghz * 1e9
    theoretical_peak_tflops = theoretical_peak_flops / 1e12

    mfu = (tflops / theoretical_peak_tflops) * 100

    return {
        "systolic_array_size": systolic_array_size,
        "flops_per_cycle": flops_per_cycle,
        "clock_rate_ghz": clock_rate_ghz,
        "theoretical_peak_tflops": theoretical_peak_tflops,
        "achieved_tflops": tflops,
        "mfu_percent": mfu,
    }


def print_performance_metrics(timings, total_flops, device_name, forward_only=True):
    """
    Calculate and print performance metrics.

    Args:
        timings: List of benchmark timings in milliseconds
        total_flops: Total FLOPs for the operation (forward or forward+backward)
        device_name: Device name for MFU calculation
        forward_only: Whether metrics are for forward-only or forward+backward
    """
    print("\n" + "=" * 40)
    print("PERFORMANCE METRICS")
    print("=" * 40)

    # Calculate statistics
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    print("\nExecution Time Statistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    # Calculate throughput
    flops_per_sec = total_flops / (avg_time / 1000)
    tflops = flops_per_sec / 1e12

    operation_type = "forward pass" if forward_only else "forward+loss+backward"
    print("\nFLOPs Analysis:")
    print(f"  Total FLOPs per {operation_type}: {total_flops:,}")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    # Calculate MFU for Neuron devices
    mfu_metrics = None
    if device_name == "neuron":
        mfu_metrics = calculate_mfu_trn2(tflops)
        print("\nBlock FLOPs Utilization (BFU):")
        print("  TRN2 Single Core Configuration:")
        print(f"    - Systolic array: 128x128 ({mfu_metrics['systolic_array_size']:,} units)")
        print(f"    - FLOPs per cycle: {mfu_metrics['flops_per_cycle']} (BF16)")
        print(f"    - Clock rate: {mfu_metrics['clock_rate_ghz']} GHz")
        print(f"  Theoretical Peak: {mfu_metrics['theoretical_peak_tflops']:.2f} TFLOPS")
        print(f"  Achieved: {mfu_metrics['achieved_tflops']:.2f} TFLOPS")
        print(f"  MFU: {mfu_metrics['mfu_percent']:.2f}%")

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
    print("=" * 80)
    print("BLOCK PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("\nConfiguration:")
    if block_type:
        print(f"  Block type: {block_type}")
    if preset_name:
        print(f"  Preset: {preset_name}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Device: {device_name}")
    print(f"  Data type: {dtype}")
    print(f"  Mode: {'Forward only' if forward_only else 'Forward + Loss + Backward'}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Benchmark runs: {benchmark_runs}")

    # Setup device
    device = setup_device(device_name)

    # Create block(s) using module's create_block function
    if n_layers == 1:
        # Single block (original behavior)
        block = block_module.create_block(config, device, dtype)
    else:
        # Stack multiple blocks
        block = StackedBlocks(block_module.create_block, n_layers, config, device, dtype)

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
    else:
        # llama decoder layers and rmsnorm expect continuous tensors
        input_tensor = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["hidden_size"],
            device=device,
            dtype=dtype,
        )

    # Warmup + Benchmark + Profile
    warmup_timings = warmup_block(
        block, input_tensor, warmup_runs, is_causal, forward_only, block_type
    )
    benchmark_timings = benchmark_passes(
        block, input_tensor, benchmark_runs, is_causal, forward_only, block_type
    )

    # Profile
    profile_block_step(block, input_tensor, is_causal, forward_only, block_type)

    # Calculate FLOPs
    # Forward pass FLOPs from the module (for a single block)
    single_block_forward_flops = block_module.count_flops(config)
    # Multiply by number of layers
    forward_flops = single_block_forward_flops * n_layers
    # For backward pass, we use 2x forward FLOPs as standard approximation
    # Total for forward+loss+backward = forward + 2*forward = 3*forward
    total_flops = forward_flops if forward_only else forward_flops * 3

    # Print metrics
    metrics = print_performance_metrics(benchmark_timings, total_flops, device_name, forward_only)

    print("\n" + "=" * 80)

    return {
        "config": config,
        "mode": "forward_only" if forward_only else "forward_backward",
        "warmup_timings": warmup_timings,
        "benchmark_timings": benchmark_timings,
        "metrics": metrics,
    }


def main():
    # First, create a parser just to get the block type
    parser_temp = argparse.ArgumentParser(add_help=False)
    parser_temp.add_argument(
        "--block", type=str, default="llama", choices=["llama", "gpt_oss", "qwen3", "rms_norm"]
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
        choices=["llama", "qwen3", "rms_norm", "gpt_oss"],
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

    # Parse all arguments
    args = parser.parse_args()

    # Re-import the module (already imported above but we do it again for clarity)
    block_module = importlib.import_module(f"block_perf.{args.block}")

    print(f"\nUsing preset configuration: {args.preset}")

    # Create configuration using the block module's method
    config = block_module.create_config(args.preset, args.batch_size, args.seq_len)

    # Map string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_mapping[args.dtype]

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
    )

    # Save results if requested
    if args.save_results:
        output_data = {
            "block_type": args.block,
            "device": args.device,
            "results": {
                "config": results["config"],
                "mode": results.get("mode"),
                "warmup_timings": results["warmup_timings"],
                "benchmark_timings": results["benchmark_timings"],
                "metrics": results["metrics"],
            },
        }

        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to {args.output_file}")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
