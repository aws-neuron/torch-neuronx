# C++ Dispatch — Onboarding Guide for ops

## Onboarding NKI Kernels to C++ Dispatch

All NKI kernels use C++ dispatch with thread-safe caching. Performance-critical
ops (SDPA) have dedicated C++ handlers; other NKI ops use the unified
`neuron::nki_call` path.

### Writing a Custom NKI Kernel (for customers)

Custom NKI kernels automatically get C++ dispatch with zero C++ changes.
There are two ways to register a kernel:

#### Method 1: `@nki_op`

Creates a proper PyTorch custom op with fake/meta kernel support,
autograd integration, and `torch.compile` compatibility.

```python
import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_neuronx import nki_op

# Step 1: Write the NKI kernel
@nki.jit
def vector_add_kernel(a, b):
    """Element-wise addition on Neuron hardware."""
    sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sa, src=a)

    sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb, src=b)

    sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)

    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=sc)
    return out

# Step 2: Wrap with @nki_op to create a PyTorch custom op
@nki_op("mylib::vector_add")
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return vector_add_kernel(a, b)

# Step 3: Use it — C++ dispatch happens automatically
a = torch.randn(128, 128, device="neuron")
b = torch.randn(128, 128, device="neuron")
result = vector_add(a, b)  # 1st call: compile + cache, 2nd+: cache hitso the

# Also works with torch.compile
compiled_fn = torch.compile(vector_add, backend="neuron")
result = compiled_fn(a, b)
```

#### Method 2: `wrap_nki`

Wraps an NKI kernel directly without creating a named custom op.
Simpler but no `torch.compile` graph capture support.

```python
from torch_neuronx import wrap_nki

wrapped_add = wrap_nki(vector_add_kernel)
result = wrapped_add(a, b)  # C++ dispatch, same caching
```

#### Dispatch flow (both methods)

```
vector_add(a, b)
  → nki_kernel_wrapper HOP
  → torch.ops.neuron.nki_call(tensor_args, kernel_idx, ...)
  → C++ DispatchNeuronNkiCall
  → dispatch_cache_ lookup
    hit:  pure C++ execution (no Python/GIL)
    miss: GIL → compile to StableHLO → NEFF → cache → execute
```

### Best Practices for C++ Dispatch Performance

**Define `@nki.jit` kernels at module level, not inside functions.**

The C++ dispatch cache is keyed by `kernel_idx`, which is assigned once per
unique kernel object. If a kernel is defined inside a function (e.g., inside
`@nki_op`), a new kernel object is created on every call, resulting in a new
`kernel_idx` and always a cache miss — no C++ dispatch benefit.

```python
# GOOD: kernel defined at module level — same kernel_idx on every call
@nki.jit
def my_kernel(a, b):
    ...

@nki_op("mylib::my_op")
def my_op(a, b):
    return my_kernel(a, b)  # cache hit on 2nd+ call

# BAD: kernel defined inside function — new kernel_idx every call
@nki_op("mylib::my_op")
def my_op(a, b):
    @nki.jit
    def my_kernel(a, b):  # re-created on every call
        ...
    return my_kernel(a, b)  # always cache miss
```

Same applies to `wrap_nki` — define at module level:

```python
# GOOD
wrapped_kernel = wrap_nki(my_kernel)  # module level

# BAD
def my_function(a, b):
    wrapped = wrap_nki(my_kernel)  # re-created on every call
    return wrapped(a, b)
```

### Onboarding ATen NKI Overrides

ATen NKI overrides register Python wrappers via `aten_lib.impl()` that internally
call `torch.ops.neuron.nki_call()`. Op-specific post-processing (e.g., SDPA stride
fixup, 9-tuple output) is handled in the Python wrapper.

**Steps:**

1. **Write a traceable wrapper function** (module-level, traceable by make_fx).

2. **Add get_lower_func() to the NKI impl class** — creates a TorchMlirKernel
   with the traceable function and returns a lower_func closure. This is how
   make_composite_lower() discovers the lowering function.

3. **Register** via register_aten_nki_lower_func() to get kernel_idx, create
   a Python wrapper (can_handle -> nki_call -> post-process), register with
   aten_lib.impl() in _register_common_ops().

4. **For performance-critical ops** (like SDPA), add a dedicated C++ handler
   instead of a Python wrapper — see DispatchSdpaFwdImpl as an example.

No C++ changes needed for simple ops — the C++ dispatch
(DispatchNeuronNkiCall) handles all NKI kernels automatically.


---
