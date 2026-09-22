# Training Llama3-8B with torch-neuronx

This tutorial demonstrates how to train the Llama3-8B model using
TorchTitan on AWS Trainium instances.

## Setup TorchTitan

```shell
cd /workspace
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 8d4f65a3
pip install -r requirements.txt
pip install -e .
pip install --no-deps triton
```

## Download Tokenizer

Download the tokenizer from Hugging Face. A Hugging Face token is required for
Llama models:

```shell
export HF_TOKEN=...
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer
```

Tokenizers are saved to `./assets/hf/<model-name>/` by default.

## Train Llama3-8B (Eager Mode)

The command below uses 64 NeuronCores
on a single `trn2.48xlarge` node with TP4 + FSDP16:

```shell
export NEURON_RT_NUM_CORES=64
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:0" \
-m torchtitan.train --module llama3 --config llama3_8b \
--parallelism.tensor-parallel-degree 4 \
--parallelism.data-parallel-shard-degree -1 \
--training.local-batch-size 1 \
--training.global-batch-size 16 \
--profiling.no-enable-profiling
```

## Train Llama3-8B with torch.compile

Enable `torch.compile` with the Neuron backend by adding `--compile.enable
--compile.backend neuron`:

```shell
export NEURON_RT_NUM_CORES=64
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:0" \
-m torchtitan.train --module llama3 --config llama3_8b \
--parallelism.tensor-parallel-degree 4 \
--parallelism.data-parallel-shard-degree -1 \
--training.local-batch-size 1 \
--training.global-batch-size 16 \
--profiling.no-enable-profiling \
--compile.enable --compile.backend neuron
```

## Train Llama3-8B with Simple FSDP + torch.compile

Use the `simple_fsdp` module for simplified FSDP with torch.compile:

```shell
export NEURON_RT_NUM_CORES=64
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:0" \
-m torchtitan.train --module simple_fsdp.llama3 --config simple_fsdp_llama3_8b \
--parallelism.tensor-parallel-degree 4 \
--parallelism.data-parallel-shard-degree -1 \
--training.global-batch-size 16 \
--training.local-batch-size 1 \
--profiling.no-enable-profiling \
--compile.enable --compile.backend neuron
```
