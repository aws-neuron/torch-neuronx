# Check small loss gradients

This example implements opt-in loss formulas from arXiv:2610.31642v1.
It also compares stock and opt-in gradients at the same represented inputs.
It does not change the backend dispatcher.

The public TorchNeuron backend has an Apache 2.0 license.
This example is AI-assisted preparation for human review.
Neuron hardware execution remains unverified.

## Use the supported contract

Use ordinary dense tensors with a complete, replicated, class-last vocabulary.
The vocabulary must contain at least two classes.
Logits and class-wise differences must be finite.

Hard-label loss supports unweighted `int64` targets.
It does not support class weights, ignored labels, label smoothing, or probability targets.
Targets have shape `logits.shape[:-1]`.

Distillation requires fixed teacher logits.
Pass logits scaled by the chosen temperature.
Apply any temperature-squared multiplier outside the function.
Centered student-minus-teacher gaps must have magnitude at most 20.
Only first-order reverse-mode student gradients are supported.
Second derivatives and trainable teachers are unsupported.
Forward-mode automatic differentiation is unsupported.
FP16 distillation requires explicit FP32 or FP64 computation.
Set `compute_dtype=torch.float32` or `compute_dtype=torch.float64`.
The bound is a reference contract, not a new backend limit.

Run eager validation before compiled execution.
Use `validate=False` only after the actual inputs pass preflight.
Continue enforcing the input contract for every compiled call.
The reference uses temporary full-vocabulary tensors.
It does not preserve the memory contract of a fused or sharded loss.

## Run CPU checks

Use an isolated CPU environment.
The prepared package used PyTorch 2.14.1 and pytest 9.1.1.
Use your SDK's supported packages on a Neuron host.
Run these commands from the `torch-neuronx` repository root.
The pytest options isolate the CPU example from the SDK test configuration and hardware fixtures.

```sh
PYTHONPATH=examples/numerical_stability \
python -m pytest examples/numerical_stability/test_stable_softmax.py -q \
  -o addopts='' --confcutdir=examples/numerical_stability

python examples/numerical_stability/check_loss_gradients.py \
  --device cpu --mode both --dtype float32 --output cpu-fp32.json
```

CPU `aot_eager` checks do not test Neuron lowering or hardware execution.

## Run Neuron checks

Run these commands on a supported Neuron host.
Set the flags before Python imports the backend.

```sh
TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1 \
TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1 \
python examples/numerical_stability/check_loss_gradients.py \
  --device neuron --mode both --dtype float32 --output neuron-fp32.json

TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1 \
TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1 \
python examples/numerical_stability/check_loss_gradients.py \
  --device neuron --mode both --dtype bfloat16 \
  --autocast --loss-fp32 --output neuron-bf16-amp.json
```

Also test BF16 with AMP disabled, with and without `--loss-fp32`.
Keep compiler flags and device-generation details with each receipt.
The runner records package versions, devices, dtypes, and gradient errors.
It records native operation tracking and CPU fallback lists.
CPU fallback causes the Neuron gate to fail.
An empty execution list alone does not prove native graph execution.

`--loss-fp32` casts both loss implementations before evaluation.
The input gradient still returns to the input dtype.
AMP does not guarantee FP32 arithmetic for every custom composition.
Neuron FP64 is not an independent high-precision oracle.
The runner uses CPU FP64 references at the same represented inputs.

## Interpret the receipt

The exit status is zero when the candidate gradient and input-preservation checks pass.
For Neuron, the fallback check must also pass.
Stock-loss failures remain recorded diagnostics.
They do not fail the candidate gate.

The runner checks target-margin sweeps, tied maxima, wrong targets, and common shifts.
It checks fixed-teacher stationarity and representable near-equality differences.
It reports component errors and gradient row sums.
A zero row sum alone does not prove correct components.
The pass gate does not certify forward-loss parity or complete API coverage.

Before changing a default primitive, test the full public loss contract.
Cover class weights, ignored labels, smoothing, reductions, shapes, sharding, and derivative behavior.
Test native and compiled paths separately.
Measure memory and train-step performance after correctness checks.
