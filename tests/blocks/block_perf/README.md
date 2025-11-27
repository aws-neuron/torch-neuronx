# Block Performance Benchmarks

Performance benchmarks for transformer blocks (currently supporting Llama).

## Usage

```bash
# Default configuration (llama-7b preset)
python run_benchmarks.py

# Use specific model preset
python run_benchmarks.py --preset llama-13b

# Custom batch size and sequence length with preset
python run_benchmarks.py --preset llama-70b --batch_size 4 --seq_len 1024

# Run on different devices
python run_benchmarks.py --device neuron  # default
python run_benchmarks.py --device cpu
python run_benchmarks.py --device cuda

# Save results to JSON
python run_benchmarks.py --save_results --output_file results.json
```

### Profiling Options

```bash
# Profile only the forward pass (profiles just the block forward)
python run_benchmarks.py --forward-only

# Default (no flag): profiles forward + loss + backward as one profile
python run_benchmarks.py
```

### Run with TP

```bash
# Currently supported only for llama
torchrun --nproc-per-node=8 run_benchmarks.py --tp_size=8

## Parameters

- `--block`: Block type to benchmark [llama] (default: llama)
- `--preset`: Preset configuration [llama-7b, llama-13b, llama-30b, llama-70b] (default: llama-7b)
- `--batch_size`: Batch size (default: 1)
- `--seq_len`: Sequence length (default: 2048)
- `--warmup_runs`: Number of warmup runs (default: 1)
- `--benchmark_runs`: Number of benchmark runs (default: 3)
- `--device`: Device to run on [neuron, cpu, cuda] (default: neuron)
- `--forward-only`: Profile only the forward pass (default profiles forward + loss + backward)
- `--save_results`: Save results to JSON file (flag)
- `--output_file`: Output file for results (default: benchmark_results.json)
- `--tp_size`: Tensor parallel degree to use for the block test

## Preset Configurations

| Preset     | d_model | n_heads | d_head | d_ff   |
|------------|---------|---------|--------|--------|
| llama-7b   | 4096    | 32      | 128    | 16384  |
| llama-13b  | 5120    | 40      | 128    | 20480  |
| llama-30b  | 6656    | 52      | 128    | 26624  |
| llama-70b  | 8192    | 64      | 128    | 32768  |

## Metrics

The benchmark provides comprehensive performance metrics:

- **Execution Time Statistics**
  - Average, standard deviation, min, and max latency (ms)

- **FLOPs Analysis**
  - Total FLOPs per forward pass
  - Throughput in TFLOPS

- **Block FLOPs Utilization (BFU)** - For Neuron devices only
  - Theoretical peak performance based on TRN2 specifications
  - Achieved TFLOPS
  - MFU percentage (Model FLOPs Utilization)
