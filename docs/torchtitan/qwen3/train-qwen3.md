# Training Qwen3-8B with torch-neuronx

## Build torch-neuronx with torch nightly build
Starting with the torch-neuronx DLC

```shell
# enable torch nightly build
cd ~
git clone git@github.com:aws-neuron/torch-neuronx.git
cd torch-neuronx
git apply docs/torchtitan/qwen3/TorchNeuronx.diff

# install bazel
bash install_bazelisk.sh

# install astral-uv
pip install uv

# install updated requirements
uv pip install --system -e . --force-reinstall

# build torch-neuronx
./tools/build
```

## Setup TorchTitan
```shell
cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git apply ~/torch-neuronx/docs/torchtitan/qwen3/TorchTitan.diff
pip install -r requirements.txt
pip install triton
pip install -U tensorboard
```


## Train with TorchTitan

Download tokenizer (for c4 dataset)
```shell
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B --assets tokenizer
```

Train model
```shell
export NEURON_RT_NUM_CORES=64 # num_trn_cores
export CONFIG_FILE=~/torch-neuronx/docs/torchtitan/qwen3/qwen3_8b_tp4_fsdp.toml
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES --rdzv_id 101 \
--local-ranks-filter 0 --role rank --tee 3 \
--rdzv_backend c10d --rdzv_endpoint "localhost:29500" \
-m torchtitan.train --job.config_file ${CONFIG_FILE}
```
The first training step will take a few minutes, but after that, expect consistent throughput.
