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

## Onboarding Preprocessing Ops

Preprocessing ops use a factory-function protocol: instead of computing the result
directly, the `torch_fn` returns `(compute_fn, args, kwargs[, postprocess_fn])`.

### Step 1: Choose the Tier

Add `preprocessing=` to your `@register_aten` decorator:

- **`Preprocessing.SHAPE_ONLY`** — preprocessing only reshapes/validates based on shapes.
  Does NOT read tensor values. Gets full C++ caching (2-5x faster).
- **`Preprocessing.VALUE_DEPENDENT`** — preprocessing reads tensor values (`.item()`,
  `torch.nonzero()`, etc.). Uses optimized Python dispatch (1.03-1.32x faster).

**When in doubt, use `Preprocessing.VALUE_DEPENDENT`** — it's the safe default. A wrong
`Preprocessing.SHAPE_ONLY` classification can cause silent correctness bugs (stale cache).

#### `relower_every_call=True`

The C++ dispatcher normally caches a `DispatchCacheEntry` and replays it, so a hit skips
Python entirely. That key covers op name, shapes, dtypes, strides, and static args. It never
covers tensor values.

That is a problem for any op whose graph depends on a value:

```
one_hot(tensor([0,1,2]), -1)  ->  num_classes 3   ->  output [3, 3]
one_hot(tensor([0,1,9]), -1)  ->  num_classes 10  ->  output [3, 10]
```

Both calls build the same key, so the second replays the first's NEFF and returns `[3, 3]`.
No error, just the wrong shape.

`relower_every_call=True` gives up that entry. Every call re-enters Python, re-runs the
preprocessing factory, and re-lowers. Execution stays in C++.

**It does not disable caching.** The HLO/NEFF cache inside `kernel.lower` still applies, keyed
on the *preprocessed* args, where the resolved value is a static arg. In a training loop where
`num_classes` is always the vocab size, that cache hits and nothing recompiles. Only a changed
resolved value costs a compile. The per-call cost is the Python round-trip, not a compile.

When to use which:

- Preprocessing only resolves values into the graph → `relower_every_call=True`.
- Preprocessing replaces the op's runtime input tensors (`_index_put_impl_` turns a bool mask
  into `nonzero` indices) → `python_dispatch=True`, because C++ derives execution inputs from
  the raw args.

Mutually exclusive; both require `preprocessing=`.

#### How to Choose

Ask: **would my preprocessing produce the same `(compute_fn, args, kwargs)` for
any two inputs with identical shapes, dtypes, and static params?**

- If YES → `Preprocessing.SHAPE_ONLY`. The preprocessing only inspects `.shape`, `.dtype`,
  `.dim()`, `.device`, and static arg values.
- If NO → `Preprocessing.VALUE_DEPENDENT`. The preprocessing reads actual tensor data.

**Quick test**: could your preprocessing run on **meta tensors** (tensors with
shapes but no data) without error? If yes → `Preprocessing.SHAPE_ONLY`. If it would crash
(e.g., `.item()` on meta tensor) → `Preprocessing.VALUE_DEPENDENT`.

#### Common Patterns

| Pattern | Classification | Why |
|---------|---------------|-----|
| Reshape args based on `.dim()` / `.shape` | `shape_only` | Shape metadata only |
| Select conv1d/conv2d/conv3d based on ndim | `shape_only` | Shape metadata only |
| Bounds check via `torch.any(idx >= size)` | `shape_only` | Validation-only side effect — raises or passes, doesn't change the compiled graph |
| Replace empty tensors with dummies (`_replace_empty`) | `shape_only` | `numel() == 0` is shape-based |
| Scalar→tensor conversion | `shape_only` | Type conversion, not value read |
| `mask.sum().item()` for output size | `value_dependent` | Reads tensor values, output shape changes |
| `torch.nonzero(bool_index)` for bool→int conversion | `value_dependent` | Output shape depends on number of True values |
| `scalars.tolist()` to unpack tensor values | `value_dependent` | Reads tensor values |
| `lr.item()` to extract learning rate | `value_dependent` | Reads tensor value |
| `torch.max(x).item()` for num_classes | `value_dependent` | Reads tensor value |
| Filter empty tensors from list (inplace ops) | `value_dependent` | Changes which tensors the NEFF operates on |
| Int→float dtype promotion for div | `value_dependent` | Changes tensor dtypes in a way the C++ cache key doesn't capture |
| Bool/int + scalar dtype promotion (`result_type`) | `value_dependent` | Promotes non-float tensors to int64; C++ cache key uses original dtype |

### Step 2: Register

```python
from torch_neuronx.python_ops.torch_mlir.operation_registry import Preprocessing, register_aten

@register_aten(
    ["aten::my_op", "aten::my_op.out"],
    static_argnums=(1,),           # compile-time constant positions
    output_params=("out",),        # for .out variants (required!)
    preprocessing=Preprocessing.SHAPE_ONLY,    # or `Preprocessing.VALUE_DEPENDENT`
)
def my_op(x, dim, value, out=None):
    # Preprocessing: transform args
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    def compute_fn(x, dim, value):
        return torch.scatter(x, dim, index, value)

    return compute_fn, (x, dim, value), {}
```

**Important**: For `.out` / `.grad_input` variants, always set `output_params`
to the names of the output tensor kwargs. Without this, `kernel.lower()` will
crash when passing out-tensor kwargs to the torch function.

### Step 3: Preprocessing Return Protocol

```python
# 3-tuple (no postprocess):
return (compute_fn, new_args, new_kwargs)

# 4-tuple (with postprocess):
return (compute_fn, new_args, new_kwargs, postprocess_fn)
```

Supported postprocess patterns:
- `None` — no postprocess (most ops)
- `lambda result: None` — inplace ops (mutation via `out=` kwarg)
- `lambda result: result.to(original_dtype)` — dtype cast after execution

### Step 4: How It Works

**Shape-only** → `_make_preprocessing_lower` wraps preprocessing into `lower_func`.
C++ `DispatchImpl` calls it on cache miss. Cache hits are pure C++, zero Python.

**Value-dependent** → `_make_preprocessing_dispatch` creates an optimized Python
function registered via `aten_lib.impl`. Runs preprocessing + cached kernel
execution directly, skipping `Operation.__call__` overhead.

**Value-dependent + `relower_every_call=True`** → `_make_preprocessing_lower` as above, but the
dispatch entry is never cached: every call re-lowers in Python and executes in C++.

### Step 5: Testing Checklist

- Test with int64 and float64 inputs (Neuron downcasts to 32-bit)
- Test with empty tensors
- Test error cases (wrong dtype, mixed CPU/neuron devices)
- For inplace ops: verify original tensor is mutated
- For `.out` variants: verify `output_params` is set
