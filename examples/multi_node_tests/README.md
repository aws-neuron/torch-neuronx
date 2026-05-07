# Multi-Node Training Example

This example demonstrates how to run multi-node data parallel training using PyTorch's DistributedDataParallel (DDP) with AWS Neuron devices.

## Files

- `data_parallel_multi_node_example.py` - Main training script with a simple neural network
- `run_parallel_training.sh` - Unified script for launching parallel training (data parallel or tensor parallel)
- `run_multi_node.sh` - SLURM job submission script

## Quick Start

### Single Node Training
```bash
# Run on single node with default parameters
./run_parallel_training.sh
```

### Multi-Node Training with SLURM
```bash
# Submit multi-node job via SLURM
sbatch run_multi_node.sh
```

## Script Arguments

The training script supports the following arguments:

```bash
python data_parallel_multi_node_example.py [OPTIONS]

Options:
  --num-samples INT     Number of samples in dataset (default: 512)
  --input-dim INT       Input dimension (default: 10)
  --batch-size INT      Batch size per process (default: 1)
  --steps INT           Number of training steps (default: 1)
  --lr FLOAT            Learning rate (default: 0.001)
  --log-interval INT    Steps interval for logging loss (default: 10)
  --compare-params      Compare gradients and weights across ranks (default: True)
  --no-compare-params   Disable parameter comparison
  --cleanup             Cleanup parameter files after comparison
```

## Environment Variables

The shell script uses these environment variables:

- `PARALLELISM_TYPE` - Type of parallelism: "data_parallel" or "tensor_parallel" (default: "data_parallel")
- `PROCESSES_PER_NODE` - Number of processes per node (default: 64)
- `NNODES` - Number of nodes (default: 2)
- `MASTER_ADDR` - Master node address (auto-detected in SLURM)
- `MASTER_PORT` - Master port (default: 29503)

## Manual Launch

You can also launch training manually using torchrun:

```bash
# On master node (replace <master_ip> with actual IP)
torchrun --nproc_per_node=64 --nnodes=2 --node_rank=0 \
         --master_addr=<master_ip> --master_port=29503 \
         data_parallel_multi_node_example.py

# On worker node
torchrun --nproc_per_node=64 --nnodes=2 --node_rank=1 \
         --master_addr=<master_ip> --master_port=29503 \
         data_parallel_multi_node_example.py
```

## Customization for Tensor Parallelism

This example can be extended for tensor parallelism by:

1. Creating a `tensor_parallel_multi_node_example.py` script
2. Using the generic launcher:
   ```bash
   PARALLELISM_TYPE=tensor_parallel ./run_parallel_training.sh
   ```
3. The shell script structure is designed to be reusable for different parallelism approaches

### Future Tensor Parallel Usage
```bash
# When tensor parallel implementation is available
PARALLELISM_TYPE=tensor_parallel PROCESSES_PER_NODE=32 ./run_parallel_training.sh
```

## What the Example Does

1. **Model**: Creates a simple 3-layer neural network (Linear -> ReLU -> Linear)
2. **Data**: Generates synthetic random data for training
3. **Training**: Performs distributed training across multiple Neuron devices
4. **Logging**: Prints loss values at configurable intervals during training
5. **Verification**: Optionally compares gradients and weights across all ranks to ensure synchronization (enabled by default)

## Expected Output

The script will output:
- Initialization messages from each rank
- Training progress with loss values at specified intervals
- Gradient and weight synchronization verification (if enabled)
- Completion status

## Performance Options

```bash
# Default behavior (with parameter comparison enabled)
python data_parallel_multi_node_example.py

# Explicitly enable comparison (same as default)
python data_parallel_multi_node_example.py --compare-params

# Disable comparison for faster training
python data_parallel_multi_node_example.py --no-compare-params

# Adjust logging frequency
python data_parallel_multi_node_example.py --log-interval 50

# Long training with comparison disabled
python data_parallel_multi_node_example.py --steps 1000 --no-compare-params --log-interval 100
```

## Troubleshooting

- Ensure all nodes have access to the same file system
- Verify Neuron devices are available on all nodes
- Check network connectivity between nodes
- Review SLURM logs if using cluster submission

## Requirements

- PyTorch with Neuron support
- AWS Neuron SDK
- EFA drivers (for multi-instance training)
- SLURM (for cluster deployment)
