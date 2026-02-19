# Training Qwen3-8B with torch-neuronx

## Setup TorchTitan
```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 0a2107f984639e23a0e5b07fc278785345f03b73
git apply ~/torch-neuronx/docs/torchtitan/qwen3/TorchTitan.diff
uv pip install --system -r requirements.txt
```


## Train with TorchTitan

Download tokenizer (for c4 dataset)
```shell
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B --assets tokenizer
```

Test with a single core to ensure environment setup succeeded:
```shell
export NEURON_RT_NUM_CORES=1
export CONFIG_FILE=~/torch-neuronx/docs/torchtitan/qwen3/qwen3_8b_tp4_fsdp.toml
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES --rdzv_id 101 \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:29500" \
-m torchtitan.train --job.config_file ${CONFIG_FILE} \
--model.flavor debugmodel \
--model.hf_assets_path ./tests/assets/tokenizer \
--training.global_batch_size 2 \
--parallelism.tensor_parallel_degree 1
```

Train the full scale model with single Trn2.48xlarge node:
```shell
export NEURON_RT_NUM_CORES=64 # num_trn_cores
export CONFIG_FILE=~/torch-neuronx/docs/torchtitan/qwen3/qwen3_8b_tp4_fsdp.toml
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES --rdzv_id 101 \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:29500" \
-m torchtitan.train --job.config_file ${CONFIG_FILE}
```
The first training step will take a few minutes, but after that, expect consistent throughput.
