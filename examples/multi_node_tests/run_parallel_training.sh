#!/bin/bash

# Generic script for running parallel training (data parallel or tensor parallel)
# This script can be extended for different parallelism strategies

# Default configuration
PARALLELISM_TYPE=${PARALLELISM_TYPE:-"data_parallel"}
PROCESSES_PER_NODE=${PROCESSES_PER_NODE:-64}
NNODES=${NNODES:-2}
TRAINING_SCRIPT=""

# Set script name based on parallelism type
case $PARALLELISM_TYPE in
    "data_parallel")
        TRAINING_SCRIPT="./data_parallel_multi_node_example.py"
        ;;
    "tensor_parallel")
        TRAINING_SCRIPT="./tensor_parallel_multi_node_example.py"  # Future implementation
        ;;
    *)
        echo "Unknown parallelism type: $PARALLELISM_TYPE"
        echo "Supported types: data_parallel, tensor_parallel"
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script $TRAINING_SCRIPT not found"
    exit 1
fi

echo "Running $PARALLELISM_TYPE training with $TRAINING_SCRIPT"
echo "Processes per node: $PROCESSES_PER_NODE"
echo "Number of nodes: $NNODES"

# Install EFA Driver (only required for multi-instance training)
ORIG_DIR=$(pwd)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
cat aws-efa-installer.key | gpg --fingerprint
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
cd
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer
cd $ORIG_DIR

echo "$(date '+%Y-%m-%d %H:%M:%S') === SLURM Environment ==="
echo "$(date '+%Y-%m-%d %H:%M:%S') SLURM_NODEID: $SLURM_NODEID"
echo "$(date '+%Y-%m-%d %H:%M:%S') SLURM_NNODES: $SLURM_NNODES"
echo "$(date '+%Y-%m-%d %H:%M:%S') SLURM_NTASKS: $SLURM_NTASKS"
echo "$(date '+%Y-%m-%d %H:%M:%S') SLURM_NODELIST: $SLURM_NODELIST"
echo "$(date '+%Y-%m-%d %H:%M:%S') SLURM_PROCID: $SLURM_PROCID"
echo "$(date '+%Y-%m-%d %H:%M:%S') ========================="

# Default values for non-SLURM environments
NODEID=${NODEID:-0}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
LOG_PATH=${LOG_PATH:-logs/default/$NODEID/}

if [ -v SLURM_NNODES ]
then
    # SLURM runs
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    export FI_EFA_FORK_SAFE=1
    # Reserve local ports if RESERVE_PORTS is true (default)
    if [ "${RESERVE_PORTS:-true}" = "true" ]; then
        sudo sysctl -w net.ipv4.ip_local_reserved_ports=29503
    fi
    if which lctl >/dev/null 2>&1; then
        sudo lctl set_param 'osc.*.max_dirty_mb=64' # Cap max space each connection to FSx reserves so we avoid OODs
    fi
    # Get the first node as master
    MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
    NODEID=$SLURM_NODEID
    NNODES=$SLURM_NNODES

    export EXPLICIT_LOGDIR=null
    : ${SLURM_RESTART_COUNT:=0}
    LOG_PATH=logs/$SLURM_JOB_ID/$SLURM_RESTART_COUNT/$NODEID/
    mkdir -p $LOG_PATH

    # Redirect output to per-node log files
    exec > >(tee -a $LOG_PATH/stdout.log)
    exec 2> >(tee -a $LOG_PATH/stderr.log >&2)
else
    # Non-SLURM runs, ensure log directory exists
    mkdir -p $LOG_PATH
    # Redirect output to default log files
    exec > >(tee -a $LOG_PATH/stdout.log)
    exec 2> >(tee -a $LOG_PATH/stderr.log >&2)
fi

export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29503}

export DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NNODES --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo "=== Distributed Training Config ==="
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Node $NODEID of $NNODES starting..."
echo "==================================="

# Launch with torchrun
echo "Starting torchrun on node $NODEID..."
echo "Launching training script: $TRAINING_SCRIPT"
torchrun $DISTRIBUTED_ARGS $TRAINING_SCRIPT
TORCHRUN_EXIT_CODE=$?

echo "All DDP processes finished with exit code: $TORCHRUN_EXIT_CODE"

# Ensure clean exit
if [ $TORCHRUN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully on node $NODEID"
    exit 0
else
    echo "Training failed on node $NODEID with exit code $TORCHRUN_EXIT_CODE"
    exit $TORCHRUN_EXIT_CODE
fi
