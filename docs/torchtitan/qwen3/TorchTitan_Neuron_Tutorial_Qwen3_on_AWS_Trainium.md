# Training Qwen3-32B with torch-neuronx

This tutorial demonstrates how to train the Qwen3-32B dense model using
TorchTitan on AWS Trainium (Trn2) instances.

## Setup TorchTitan

```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 8d4f65a3167861dcc98c59343cdf00d8fe055c4d
git apply ~/torch-neuronx/docs/torchtitan/neuron_attention_patch.diff
uv pip install --system -r requirements.txt
uv pip install -e .
```

The patch adds a `force_local_contiguous` workaround in the SDPA attention path for the PyTorch issue [#173041](https://github.com/pytorch/pytorch/issues/173041) so DTensor local shards are made contiguous under `torch.compile`. See
[neuron_attention_patch.diff](../neuron_attention_patch.diff) for the full diff.

## Download Tokenizer (for c4 dataset)

```shell
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-32B --assets tokenizer
```

## Train Qwen3-32B (Eager Mode)

```shell
export NEURON_RT_NUM_CORES=64
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:0" \
-m torchtitan.train --module qwen3 --config qwen3_32b \
--parallelism.tensor-parallel-degree 4 \
--parallelism.data-parallel-shard-degree -1 \
--training.local-batch-size 1 \
--training.global-batch-size 16 \
--activation-checkpoint.mode full
```

## Train Qwen3-32B with torch.compile

Enable `torch.compile` with the Neuron backend by adding `--compile.enable
--compile.backend neuron`.

```shell
export NEURON_RT_NUM_CORES=64
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:0" \
-m torchtitan.train --module qwen3 --config qwen3_32b \
--parallelism.tensor-parallel-degree 4 \
--parallelism.data-parallel-shard-degree -1 \
--training.local-batch-size 1 \
--training.global-batch-size 16 \
--activation-checkpoint.mode full \
--compile.enable --compile.backend neuron
```

The first training step will take a few minutes, but after that, expect consistent throughput.
