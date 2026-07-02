# Backward-compatibility guidelines for torch-neuronx CRs

This document covers BC considerations when reviewing CRs against
`torch-neuronx`. The top-level principle is the same as upstream
PyTorch: **anything that changes any user-visible behavior is
potentially BC-breaking**. It must then be classified as (a) behavior
not exercised in practice, (b) a bug fix, or (c) a voluntary
BC-breaking change with a migration path.

As a reviewer, **be paranoid**. A key part of your job is to identify
user-visible change and ensure the author thought through the
implications.

## What constitutes a BC break in this repo

The public surface of `torch-neuronx` has five layers. Each has its
own BC rules.

### 1. The `torch_neuronx` Python package

Anything re-exported from `torch_neuronx/__init__.py` or a documented
submodule (e.g., `torch_neuronx.nki_op`, `torch_neuronx.Stream`,
`torch_neuronx.NeuronProfiler`). Rules mirror upstream PyTorch's
[Public API definition](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation):

| Change type | BC impact | Action required |
|---|---|---|
| Remove a public symbol | Breaking | Deprecation period required |
| Rename a public symbol | Breaking | Deprecation period required |
| Remove / reorder positional args | Breaking | Deprecation period required |
| Add required arg (no default) | Breaking | Add default instead |
| Change default value | Potentially breaking | Document in release notes |
| Change return type or shape | Breaking | Deprecation period required |
| Remove private symbol (`_foo`) used outside the package | Potentially breaking | Validate via code-search |

**"Looks public, is public"** — a non-underscored symbol at a
non-underscored path is public even if undocumented.

### 2. The `torch.neuron.*` device module

`torch._register_device_module("neuron", neuron)` exposes
`torch.neuron.is_available()`, `torch.neuron.current_device()`,
`torch.neuron.device_count()`, `torch.neuron.memory_stats()`, etc.
These are called by user code directly and by tools (profilers,
DDP / FSDP harnesses). Removing, renaming, or changing the return
type of any of them is a break.

### 3. The `'neuron'` device semantics

The device string `"neuron"` itself, plus the tensor behavior it
implies:

- Which dtypes are accepted (int32/float32 native; int64/float64
  bounced on entry).
- The contiguity contract (contiguous required on-device; everything
  else goes through `ensure_contiguous_on_device`).
- The result of mixing `"neuron"` and `"cpu"` tensors in the same op
  (the implicit bounce behavior).

A CR that changes any of these needs a release-notes entry and a
description of user-visible impact.

### 4. `torch.compile(backend="neuron")`

The README states the supported parameters:

| Parameter | Supported | Notes |
|---|---|---|
| `model` | Yes | |
| `fullgraph` | Yes | |
| `dynamic` | `False` only | |
| `mode` | `"default"` only | |
| `options` | No | |
| `disable` | Yes | |

Any change to this table is BC-visible. Silently starting to support
a new mode is BC-additive (generally fine, but document it). Silently
rejecting a combination that previously worked is a break.

### 5. Environment variables

`docs/environment_variables.md` is the canonical list. Renaming or
removing a `TORCH_NEURONX_*` / `NEURON_*` env var is a BC break for
anyone with it in a shell rc, Dockerfile, or Apollo config. If
unavoidable:

1. Support both names for one release.
2. Emit a `FutureWarning` when the old name is read.
3. Document the migration in the CR description + release notes.

### 6. On-disk artifacts (NEFF cache)

Users' `TORCH_NEURONX_DEBUG_DIR` accumulates NEFFs keyed by a hash of
compile inputs. Changing the hash schema silently invalidates users'
caches — not strictly wrong, but surprising.

Worse: if the hash still *matches* after a semantics change, users
get wrong results from a stale NEFF. That is a silent-correctness
regression.

**Bump the cache-key schema version any time the hash function or
its inputs change.** See `docs/AI-compilation-kernels.md` for the
current schema.

### 7. The `_C` extension ABI

Function signatures in `torch_neuronx._C` are an ABI for the Python
wheel. Changing a signature breaks any pinned older wheel:

- Don't change existing binding signatures in place.
- Add new bindings alongside.
- Delete only after a deprecation release.

## Python version support

`torch-neuronx` pins a Python version in `.python-version`. The
supported window is whatever the file says, typically the last 2–3
CPython versions. All Python code must be compatible with the oldest
supported version. C++ code compiled against CPython must work with
free-threaded builds if the pinned versions include 3.13t+.

## When BC breaks are acceptable

### With proper deprecation

BC breaks are acceptable when:

1. A `FutureWarning` has been emitted for at least one release.
2. A migration path is documented.
3. The change is in release notes.
4. The benefit is substantial enough to justify the churn.

### Deprecation pattern (Python)

```python
import warnings

def old_function(x, old_arg=None, new_arg=None):
    if old_arg is not None:
        warnings.warn(
            "old_arg is deprecated and will be removed in "
            "torch_neuronx X.Y. Use new_arg instead.",
            FutureWarning,
            stacklevel=2,
        )
        new_arg = old_arg
    # ... rest of implementation
```

Use `FutureWarning` (user-facing), not `DeprecationWarning` (dev-
facing, silenced by default).

### Deprecation pattern (env var)

```python
import os, warnings

def _read_env():
    if "TORCH_NEURONX_OLD_NAME" in os.environ and \
       "TORCH_NEURONX_NEW_NAME" not in os.environ:
        warnings.warn(
            "TORCH_NEURONX_OLD_NAME is deprecated; "
            "use TORCH_NEURONX_NEW_NAME.",
            FutureWarning,
        )
        return os.environ["TORCH_NEURONX_OLD_NAME"]
    return os.environ.get("TORCH_NEURONX_NEW_NAME")
```

### Without deprecation (rare)

Immediate BC breaks may be acceptable for:

- Security vulnerabilities.
- Serious correctness bugs that make the API unusable.
- APIs explicitly marked experimental / internal / beta in the CR
  that introduced them.

Document the justification in the CR description.

## Common BC pitfalls

### 1. Changing function signatures

**Bad:**
```python
# Before
def compile_neuron(model, fullgraph): ...

# After — breaks any caller passing fullgraph positionally in a
# different slot or by name from a different position
def compile_neuron(model, dynamic, fullgraph): ...
```

**Good:**
```python
# After — new args go to the end with defaults
def compile_neuron(model, fullgraph, dynamic=False): ...
```

### 2. Removing public attributes

**Bad:**
```python
class NeuronProfiler:
    # self.last_trace removed
    ...
```

**Good:**
```python
class NeuronProfiler:
    @property
    def last_trace(self):
        warnings.warn(
            "NeuronProfiler.last_trace is deprecated",
            FutureWarning,
        )
        return self._new_trace_api()
```

### 3. Changing default behavior

**Bad:**
```python
# Silently changing default dtype promotion rule
def some_op(x, promote_int64=True):  # was False
    ...
```

**Good:**
```python
def some_op(x, promote_int64=None):
    if promote_int64 is None:
        warnings.warn(
            "promote_int64 default is changing from False to True "
            "in torch_neuronx X.Y.",
            FutureWarning,
        )
        promote_int64 = False   # keep old default during deprecation
    ...
```

### 4. Changing exception types

**Bad:**
```python
# Users catching ValueError will miss the new exception
raise TypeError("invalid dtype for neuron op")   # was ValueError
```

**Good:**
```python
class NeuronDtypeError(ValueError):   # inherits from old type
    pass

raise NeuronDtypeError("invalid dtype for neuron op")
```

### 5. Renaming an env var

**Bad:**
```python
# Before
os.environ.get("TORCH_NEURONX_FOO")
# After
os.environ.get("TORCH_NEURONX_BAR")   # silently stops reading FOO
```

**Good:** see the env var deprecation pattern above.

### 6. Silently invalidating the NEFF cache

**Bad:**
```python
# Change how cache key is computed without bumping schema version
key = hash((shapes, dtypes, new_thing_you_added))
```

**Good:**
```python
CACHE_SCHEMA_VERSION = 3   # was 2
key = hash((CACHE_SCHEMA_VERSION, shapes, dtypes, new_thing))
# → old keys (schema 2) are simply misses, not mis-hits
```

## Review checklist for BC

When reviewing, explicitly check:

- [ ] **No removed public APIs** — Or a deprecation path exists.
- [ ] **No changed positional signatures** — New args have defaults
      and come last.
- [ ] **No changed defaults** — Or a `FutureWarning` is emitted and
      the change is scheduled for a named release.
- [ ] **No changed return types / shapes** — Or a migration path is
      documented.
- [ ] **No changed exception types** — Or the new type inherits from
      the old.
- [ ] **Deprecations use `FutureWarning`** — Not `DeprecationWarning`.
- [ ] **Deprecations set `stacklevel=2`** — Warning points at user
      code, not library internals.
- [ ] **Env var changes handled** — Both names supported for one
      release, old name warns.
- [ ] **NEFF cache schema versioned** — Any change to cache-key
      inputs bumps the schema.
- [ ] **`_C` ABI stable** — New bindings added alongside; old ones
      untouched.
- [ ] **README / environment_variables.md updated** — User-visible
      changes are reflected in the docs.
- [ ] **Other user-visible behavior changes** — Listed explicitly for
      the author.

## Questions to ask when unsure

When uncertain about BC impact:

1. Would existing user code break silently (worst case)?
2. Would existing user code raise an exception (recoverable, but
   still broken)?
3. Is there a migration path that doesn't require users to change
   code immediately?
4. Is this change in the release notes / CR description?
5. Can I find callers of this API via code-search? If yes, have they
   been audited or notified?

If the answer to (5) is uncertain, raise it in the review as a point
to investigate — do not approve on faith.

## When to escalate

Some BC questions are bigger than one reviewer. Escalate to the
package owners if:

- The CR changes a fundamental `torch.compile(backend="neuron")`
  contract (e.g., a newly-supported mode, a newly-rejected input).
- The CR changes the on-device dtype contract (e.g., introduces
  native int64 support, changes bounce behavior).
- The CR changes the `'neuron'` device-module API surface.
- The CR deletes or renames an env var older than one release.

In these cases, request **Needs Discussion** in the review
recommendation rather than **Approve** or **Request Changes**.
