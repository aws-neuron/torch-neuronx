# Neuron PyTorch MLIR Backend

This document describes the MLIR-based backend for lowering PyTorch ATen operations to run on AWS
Neuron devices. The backend provides a comprehensive system for registering, compiling, and
executing PyTorch operations through MLIR lowering, neuronx-cc compilation, and Neuron Runtime
execution.

> **Note**: Throughout this document, "HLO" refers to StableHLO, the stable version of Google's High
> Level Operations intermediate representation used in the MLIR compilation pipeline.

## Table of Contents

- [Operation Registration](#operation-registration)
- [Operation Execution](#operation-execution)
  - [Data Type Handling](#data-type-handling)
  - [Lowering](#lowering)
  - [Graph Module Generation](#graph-module-generation)
  - [Dynamo Optimization](#dynamo-optimization)
  - [Output Construction](#output-construction)
- [Preprocessing](#preprocessing)
- [Operations](#operations)
- [Custom Decompositions](#custom-decompositions)
- [Testing](#testing)

## Operation Registration

ATen operators are supported through MLIR lowering using the
[`register_aten`](operation_registry.py#L8) decorator. This decorator simplifies the registration
process by automatically handling metadata extraction and registration details.

### Basic Registration Example

Using [`aten::linear`](ops/linear_algebra.py#L35) as an example:

```python
@register_aten(["aten::linear"])
def torch_linear(input, weight, bias=None):
    return torch.nn.functional.linear(input, weight, bias)
```

### Registration Process

Under the hood, the registration process involves several steps:

1. **Import-time Discovery**: When `torch_neuronx` is imported (auto-loaded when `torch` is
   imported):
   - `auto_register_neuron_ops` discovers all neuron-registered ATen operations
   - [`register_aten`](operation_registry.py#L10) adds the neuron-defined function `torch_linear`
     into [\_PENDING_OPERATIONS](operation_registry.py#65) along with metadata such as
     `static_argnums` and `static_argnames`

2. **Finalization**: `finalize_registrations` is called to register ATen operations using
   `aten_lib.impl`

3. **Wrapper Creation**: An [`ImplementationWrapper(TorchMlirOpImpl)`](operation_registry.py#L120)
   object is created with the `torch_linear` function set as its [`torch_fn`](op_impl.py#L25)

4. **Kernel Creation**: Depending on whether preprocessing is used, a `TorchMlirKernel` object is
   created for the `TorchMlirOpImpl` object

With these steps, `torch.linear` operations with inputs on the `neuron` device will be dispatched to
the neuron-defined `torch_linear` function and enter the [`execute`](op_impl.py#L58) function.

### Registration Tips

When registering new operations, consider these guidelines:

1. **Check MLIR Support**: Verify the operation is supported in torch-mlir by checking the
   [supported operations list](https://github.com/aws-neuron/torch-mlir/blob/main/projects/pt1/python/torch_mlir/jit_ir_importer/build_tools/torch_ods_gen.py).
   If not supported, implement custom lowering and consider adding it to `decompositions.py` for
   reuse by `torch.compile`

2. **Avoid CompositeImplicitAutograd**: Don't register operations marked as
   `CompositeImplicitAutograd` in
   [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml),
   as this can break PyTorch's autograd system

3. **Skip Metadata-Only Operations**: Avoid registering operations that only manipulate metadata
   without actual computation (e.g., view operations, shape queries)

## Operation Execution

### Data Type Handling

Since Neuron has better support for float32, the system casts float64 inputs to float32 and saves
the original inputs in an [`ExecutionContext`](op_impl.py#L87) before executing `TorchMlirKernel`.
The results are later cast back to the correct dtype based on the inputs and operation requirements.

### Lowering

The lowering process involves several key steps:

#### Input Preprocessing

- **Scalar to Tensor Conversion**: By default, when Dynamo first encounters a scalar value
  (int/float), it bakes the exact value into the graph and final StableHLO (HLO). This continues
  until Dynamo detects value changes for these arguments, then converts them to symbolic values
  (SymInt/SymFloat). Since neuronx-cc doesn't handle symbolic values well and we want to cache HLO
  when input arguments don't change the computation graph, scalar values are converted to scalar
  tensors. This ensures they're represented as tensors in the HLO, eliminating re-compilation needs
  for the same computation. Static arguments like `dim` are not converted since their actual values
  change the computation graph.

- **Empty Tensor Handling**: The system bypasses compilation and computation if input tensors are
  empty (`numel() == 0`). In such cases, no computation is needed, and outputs are directly returned
  using [`_handle_empty_tensors`](kernel.py#L515) with correct shape and value depending on the
  operation type.

- **HLO Generation**: The [`_generate_hlo`](kernel.py#L204) function performs the heavy lifting to
  generate the FX graph and lower it to HLO.

### Graph Module Generation

The graph generation process follows these steps:

1. **Meta Tensor Pass**: First, a pass is run on the defined `torch_fn` with original inputs using
   meta tensors to determine the correct expected number of outputs and tensor dtypes. This step
   ensures correctness of actual execution outputs, as Dynamo optimization may remove unused
   tensors, empty tensors, or pass-through tensors from the graph.

2. **FX Graph Creation**: [`make_fx`](kernel.py#L200) is called with a custom decomposition table to
   create a decomposed graph module. This allows early application of decompositions to ensure
   correct inputs and outputs are generated for the graph.

At this stage, you may see a graph like:

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[5, 4]"):
        # No stacktrace found for following nodes
        t: "f32[4, 5]" = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
        mm: "f32[3, 5]" = torch.ops.aten.mm.default(arg0_1, t);  arg0_1 = t = None
        return (mm,)
```

### Dynamo Optimization

To enable capturing the generated HLO for operation concatenation, the system runs a custom Dynamo
backend [`capture_backend`](kernel.py#L256) explicitly instead of calling `torch.compile`.

The system uses [`aot_autograd`](kernel.py#L327) for graph transformation: it applies custom
decompositions to break down complex operations into MLIR-compatible primitives and captures the
decomposed graph without actual compilation, providing more pipeline control than `torch.compile`.

During this backend process, key information is captured:

1. **`captured_gm`**: The actual computation graph used for the operation after optimization
2. **`kept_input_indices`**: Dynamo removes unused input arguments from the graph, so we track which
   inputs need to be passed to Neuron during runtime for correct execution
3. **`kept_output_indices`**: Sometimes outputs are optimized away by Dynamo (e.g., `None` outputs
   and empty output tensors)

Once the graph is generated,
[`convert_fx_to_stablehlo`](../../neuron_dynamo_backend/fx/fx_transform.py#L147) is called to run
through different pipelines and gradually lower the FX graph into StableHLO using MLIR. This process
includes [custom pipelines](../../neuron_dynamo_backend/fx/pipelines) and
[FX passes](../../neuron_dynamo_backend/fx/passes) that perform graph transformations such as
[dtype conversion](../../neuron_dynamo_backend/fx/passes/dtype_conversion.py) and
[random operation legalization](../../neuron_dynamo_backend/fx/passes/random_op_legalization.py).

At this stage, you may see outputs from different pipeline stages:

```mlir
// torch-raw-to-neuron-backend-pipeline
builtin.module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %0 = torch.aten.t %arg1 : !torch.vtensor<[5,4],f32> -> !torch.vtensor<[4,5],f32>
    %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    return %1 : !torch.vtensor<[3,5],f32>
  }
}

// torchdynamo-export-to-torch-backend-pipeline
builtin.module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[5,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,5],f32>
    %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    return %1 : !torch.vtensor<[3,5],f32>
  }
}

// torch-backend-to-stablehlo-backend-pipeline
builtin.module {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<3x5xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }
}
```

The final HLO output:

```mlir
module {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<3x5xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }
}
```

Once the HLO is generated, `_execute_async` or `_execute_sync` calls neuronx-cc to compile the HLO
into NEFF and execute the NEFF with provided input and output tensors.

### Output Construction

After NEFF execution, results are written into provided `execution_tensors`, which may have
different dtypes and number of outputs than PyTorch expected due to optimization and casting. The
system runs:

1. **`self._upcast_execution_results`**: Upcast `torch.float32` to `torch.float64` if needed
2. **`self._reconstruct_outputs`**: Construct correct outputs by adding tensors optimized away by
   Dynamo

## Preprocessing

[`register_aten`](operation_registry.py#L8) supports preprocessing for operations that involve
dynamic shape outputs or require CPU-side processing. For example, `torch.repeat_interleave` with a
tensor `repeats` has output tensor shape that depends on the sum of repeats, making it
data-dependent and dynamic.

### Preprocessing Example

```python
@register_aten(
    [
        "aten::repeat_interleave.self_Tensor",
    ],
    static_argnums=(2,),
    static_argnames=("output_size",),
    uses_preprocessing=True,
)
def torch_repeat_interleave_self_tensor(input, repeats, dim=None, output_size=None):
    output_size = get_repeat_interleave_output_size(repeats, input, dim, output_size)
    return compute_repeat_interleave, (input, repeats, dim), {"output_size": output_size}
```

The preprocessing function calculates the `output_size`, then returns:

- The actual computation function (`compute_repeat_interleave`)
- Preprocessed inputs
- Additional keyword arguments

This allows the kernel to call the actual function with preprocessed inputs.

## Operations

All neuron ATen operations are registered under [`torch_neuronx/python_ops/torch_mlir/ops`](ops/),
organized by category:

- **[Linear Algebra](ops/linear_algebra.py)**: Matrix operations (mm, bmm, addmm, etc.)
- **[Comparison](ops/comparison.py)**: Comparison and clamping operations
- **[Tensor Operations](ops/tensor_ops.py)**: Basic tensor manipulations
- **[Indexing](ops/indexing.py)**: Indexing and slicing operations
- **[Convolution](ops/convolution.py)**: Convolution operations
- **[Normalization](ops/normalization.py)**: Normalization layers
- **[Activation](ops/activation.py)**: Activation functions
- **[And more...](ops/)**

Please refer to the operations [README](ops/README.md) for more details.

## Custom Decompositions

For operations where PyTorch and Torch MLIR don't have existing decompositions, custom
decompositions are implemented. For example, [`aten::nonzero_static`](ops/indexing.py#L230) doesn't
have an existing decomposition, so it's implemented using basic operations in
`torch_neuronx/neuron_dynamo_backend/decompositions.py`. This allows the decomposition to be shared
between ATen operations and `torch.compile`.

When running [`make_fx`](kernel.py#L200), these custom decompositions are utilized to decompose
operations during graph generation, ensuring compatibility with the MLIR lowering pipeline.

## Testing

Tests for ATen operations are located in `tests/python_ops/test_<operation_name>.py`. Example:
[`test_linear.py`](../../tests/python_ops/test_linear.py).

### Test Template

```python
import pytest
import torch
import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestYourOperation:
    def test_operation_basic(self):
        with track_neuron_ops():
            input_cpu = torch.rand(shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.your_operation(input_cpu)
            output_neuron = torch.your_operation(input_neuron)

            assert_op_runs_on_neuron("aten::your_operation")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
```

### Key Requirements

- **Device check**: Skip tests when no Neuron devices available
- **Operation tracking**: Use `track_neuron_ops()` to monitor Neuron dispatch
- **Verification**: Confirm operation runs on Neuron and produces correct results
- **Parametrization**: Test multiple shapes, dtypes, and edge cases

### Running Tests

```bash
# Run specific operation tests
./tools/run-test-parallel tests/python_ops/test_linear.py

# Run all python_ops tests
./tools/run-test-parallel
```
