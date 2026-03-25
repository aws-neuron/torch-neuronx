"""Detailed profiling runner for transformer blocks.

Variant of run_benchmarks.py focused on per-operator profiling output:
individual event traces, stack exports, and optional Chrome trace files.
"""

import argparse
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
        if any(
            os.path.exists(os.path.join(current, marker))
            for marker in ["setup.py", "pyproject.toml", ".git"]
        ):
            return current
        current = os.path.dirname(current)
    return None


project_root = find_project_root(os.path.dirname(__file__))
if not project_root:
    raise RuntimeError(
        "Could not find project root. Expected to find setup.py, pyproject.toml, or .git directory."
    )

sys.path.insert(0, project_root)

blocks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if blocks_dir not in sys.path:
    sys.path.insert(0, blocks_dir)

import importlib  # noqa: E402

import torch.nn as nn  # noqa: E402


class StackedBlocks(nn.Module):
    """Generic wrapper to stack multiple identical blocks."""

    def __init__(self, block_fn, n_layers, *block_args, **block_kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [block_fn(*block_args, **block_kwargs) for _ in range(n_layers)]
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


def setup_device(device_name):
    if device_name == "neuron":
        try:
            import torch_neuronx  # noqa: F401
        except ImportError:
            print("Warning: torch_neuronx not found, falling back to CPU")
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


def run_block(block, input_tensor, block_module, **kwargs):
    """Generic block runner that delegates to the block module's run_block."""
    outputs = block_module.run_block(block, input_tensor, **kwargs)

    if hasattr(outputs, "last_hidden_state"):
        return outputs

    class Output:
        def __init__(self, tensor):
            self.last_hidden_state = tensor

    return Output(outputs)


def warmup_block(
    block, input_tensor, warmup_runs, is_causal=True, forward_only=True, block_module=None
):
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
            block_module=block_module,
        )
        warmup_timings.append(total_ms)
        print(f"  Warmup run {i + 1}: {total_ms:.2f} ms")

    return warmup_timings


def run_block_step(
    block,
    input_tensor,
    is_causal=True,
    forward_only=True,
    with_profile=False,
    block_module=None,
):
    """Run a single block step, measuring forward and optionally backward times.

    Returns (total_ms, prof)
    """
    prof = None
    total_ms = None

    context_manager = torch.no_grad() if forward_only else torch.set_grad_enabled(True)

    with context_manager:
        if not forward_only:
            block.zero_grad(set_to_none=True)

        if with_profile:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                if forward_only:
                    start_time = time.perf_counter()
                    _ = run_block(block, input_tensor, block_module, is_causal=is_causal)
                    total_ms = (time.perf_counter() - start_time) * 1000
                else:
                    start_time = time.perf_counter()
                    outputs = run_block(block, input_tensor, block_module, is_causal=is_causal)
                    loss = compute_loss(outputs)
                    loss.backward()
                    total_ms = (time.perf_counter() - start_time) * 1000
        else:
            if forward_only:
                start_time = time.perf_counter()
                _ = run_block(block, input_tensor, block_module, is_causal=is_causal)
                total_ms = (time.perf_counter() - start_time) * 1000
            else:
                start_time = time.perf_counter()
                outputs = run_block(block, input_tensor, block_module, is_causal=is_causal)
                loss = compute_loss(outputs)
                loss.backward()
                total_ms = (time.perf_counter() - start_time) * 1000

    return total_ms, prof


def benchmark_passes(
    block, input_tensor, benchmark_runs, is_causal=True, forward_only=True, block_module=None
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
            block_module=block_module,
        )
        timings.append(total_ms)
        print(f"  Run {i + 1}: {total_ms:.2f} ms")

    return timings


def compute_loss(outputs) -> torch.Tensor:
    """Squash the scalar loss to a small magnitude to avoid large gradients."""
    return torch.tanh(outputs.last_hidden_state.sum())


def profile_block_step(
    block,
    input_tensor,
    is_causal=True,
    forward_only=False,
    block_module=None,
    export_trace=False,
    trace_path="trace.json",
):
    """Profile either forward-only or fwd+loss+bwd and show detailed operator breakdown."""
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
        block_module=block_module,
    )

    print("\nAll Operations (sorted by self CPU time):")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

    print("\nAll Operations (sorted by total CPU time):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))

    print("\nMemory-Intensive Operations:")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

    print("\nIndividual Operation Events:")
    events_list = list(prof.events())
    print(f"Total events captured: {len(events_list)}")
    print("\nFirst 100 events (showing details):")
    for i, evt in enumerate(events_list):
        if i >= 100:
            print(f"\n... and {len(events_list) - 100} more events")
            print("\nTo see all events, check the exported Chrome trace or stack files.")
            break
        cpu_time_us = evt.cpu_time_total
        print(f"  {i + 1}. {evt.name}")
        print(f"      CPU time: {cpu_time_us:.2f} us")

        if hasattr(evt, "input_shapes") and evt.input_shapes:
            print(f"      Input shapes: {evt.input_shapes}")

        if hasattr(evt, "cpu_memory_usage") and evt.cpu_memory_usage != 0:
            print(f"      Memory: {evt.cpu_memory_usage / 1024:.2f} KB")

    if export_trace:
        try:
            prof.export_chrome_trace(trace_path)
            print(f"\nChrome trace exported to: {trace_path}")
            print("   View in Chrome: Open chrome://tracing and load the file")
            print("   View in Perfetto: Upload to https://ui.perfetto.dev/")
        except Exception as e:
            print(f"\nError exporting Chrome trace: {e}")
            print("   Attempting fallback export...")
            try:
                events = []
                for event in prof.events():
                    event_dict = {
                        "name": event.name,
                        "cat": "Op",
                        "ph": "X",
                        "ts": event.cpu_time_total / 1000.0,
                        "dur": event.cpu_time_total / 1000.0,
                        "pid": 0,
                        "tid": 0,
                    }
                    if hasattr(event, "input_shapes") and event.input_shapes:
                        event_dict["args"] = {"shapes": str(event.input_shapes)}
                    events.append(event_dict)

                trace_data = {"traceEvents": events}
                with open(trace_path, "w") as f:
                    json.dump(trace_data, f)
                print(f"Trace exported to: {trace_path} (fallback format)")
            except Exception as e2:
                print(f"Failed to export trace: {e2}")

    stack_path = trace_path.replace(".json", "_stacks.txt")
    try:
        prof.export_stacks(stack_path, "self_cpu_time_total")
        print(f"Stack traces exported to: {stack_path}")
    except Exception as e:
        print(f"Could not export stack traces: {e}")

    return prof


def calculate_mfu_trn2(tflops):
    """Calculate Block FLOPs Utilization for AWS TRN2."""
    systolic_array_size = 128 * 128
    flops_per_cycle = 2
    clock_rate_ghz = 2.4

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
    print("\n" + "=" * 40)
    print("PERFORMANCE METRICS")
    print("=" * 40)

    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    print("\nExecution Time Statistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    flops_per_sec = total_flops / (avg_time / 1000)
    tflops = flops_per_sec / 1e12

    operation_type = "forward pass" if forward_only else "forward+loss+backward"
    print("\nFLOPs Analysis:")
    print(f"  Total FLOPs per {operation_type}: {total_flops:,}")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

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
    export_trace=False,
    trace_path="trace.json",
):
    print("=" * 80)
    print("BLOCK PERFORMANCE BENCHMARK (PROFILE MODE)")
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

    if config.get("lora_rank", 0) > 0:
        print("\n  LoRA Configuration:")
        print(f"    Rank: {config['lora_rank']}")
        print(f"    Alpha: {config.get('lora_alpha', 8.0)}")
        print(f"    Dropout: {config.get('lora_dropout', 0.0)}")
        print(f"    Targets: {config.get('lora_targets', [])}")
        print(f"    Use BMM: {config.get('use_bmm', False)}")

    device = setup_device(device_name)

    if n_layers == 1:
        block = block_module.create_block(config, device, dtype)
    else:
        block = StackedBlocks(block_module.create_block, n_layers, config, device, dtype)

    if block_type == "qwen3":
        input_tensor = torch.randint(
            0,
            config["vocab_size"],
            (config["batch_size"], config["seq_len"]),
            device=device,
            dtype=torch.long,
        )
    elif block_type == "grpo":
        input_tensor = None
    else:
        input_tensor = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["hidden_size"],
            device=device,
            dtype=dtype,
        )

    warmup_timings = warmup_block(
        block, input_tensor, warmup_runs, is_causal, forward_only, block_module
    )
    benchmark_timings = benchmark_passes(
        block, input_tensor, benchmark_runs, is_causal, forward_only, block_module
    )

    profile_block_step(
        block,
        input_tensor,
        is_causal,
        forward_only,
        block_module,
        export_trace=export_trace,
        trace_path=trace_path,
    )

    single_block_forward_flops = block_module.count_flops(config)
    forward_flops = single_block_forward_flops * n_layers
    total_flops = forward_flops if forward_only else forward_flops * 3

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
    parser_temp = argparse.ArgumentParser(add_help=False)
    parser_temp.add_argument(
        "--block",
        type=str,
        default="llama",
        choices=[
            "llama",
            "llama_lora",
            "gpt_oss",
            "qwen3",
            "rms_norm",
            "grpo",
        ],
    )
    args_temp, _ = parser_temp.parse_known_args()

    try:
        block_module = importlib.import_module(f"block_perf.{args_temp.block}")
    except ImportError as e:
        raise ImportError(f"Could not import block module 'block_perf.{args_temp.block}'") from e

    parser = argparse.ArgumentParser(
        description="Detailed profiling runner for transformer blocks"
    )

    parser.add_argument(
        "--block",
        type=str,
        default="llama",
        choices=[
            "llama",
            "llama_lora",
            "qwen3",
            "rms_norm",
            "gpt_oss",
            "grpo",
        ],
        help="Block type to benchmark (default: llama)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="llama-7b",
        choices=block_module.get_presets(),
        help="Use preset configuration for specific model sizes",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (default: 2048)")

    parser.add_argument(
        "--warmup_runs", type=int, default=1, help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--benchmark_runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="neuron",
        choices=["neuron", "cpu", "cuda"],
        help="Device to run on (default: neuron)",
    )

    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )

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
    parser.add_argument(
        "--export_trace",
        action="store_true",
        help="Export Chrome trace for visualization (default: False)",
    )
    parser.add_argument(
        "--trace_path",
        type=str,
        default="trace.json",
        help="Path for exported trace file (default: trace.json)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model and inputs (default: bfloat16)",
    )

    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank r (0 disables LoRA)")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help=(
            "Comma-separated target module names "
            "(e.g. q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj)"
        ),
    )
    parser.add_argument(
        "--use_bmm",
        action="store_true",
        help="Use batch matrix multiplication for parallel LoRA computation",
    )

    args = parser.parse_args()

    block_module = importlib.import_module(f"block_perf.{args.block}")

    print(f"\nUsing preset configuration: {args.preset}")

    config = block_module.create_config(args.preset, args.batch_size, args.seq_len)

    config["lora_rank"] = args.lora_rank
    config["lora_alpha"] = args.lora_alpha
    config["lora_dropout"] = args.lora_dropout
    config["lora_targets"] = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    config["use_bmm"] = args.use_bmm

    dtype_mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_mapping[args.dtype]

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
        export_trace=args.export_trace,
        trace_path=args.trace_path,
    )

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
