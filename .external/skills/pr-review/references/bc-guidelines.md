# Backward-Compatibility Guidelines for torch-neuronx Reviews

Treat each user-visible behavior change as a possible compatibility break.
Classify it as a compatible addition, a bug fix, or an intentional break with a migration path.

Use the upstream [public API definition](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation)
for general Python policy.
Use this guide for torch-neuronx surfaces.

## Public compatibility surfaces

### 1. The `torch_neuronx` Python package

Treat a non-underscored symbol at a documented, non-underscored path as public.
Check symbols re-exported from `torch_neuronx/__init__.py` and documented submodules.

| Change | Impact | Required action |
|---|---|---|
| Remove or rename a public symbol | Breaking | Keep a deprecated alias for the documented migration period. |
| Remove or reorder a positional parameter | Breaking | Preserve the old call shape during migration. |
| Add a required parameter | Breaking | Add it at the end with a default. |
| Change a default | Potentially breaking | Preserve the old default during migration. |
| Change a return type or shape | Breaking | Document and stage the migration. |
| Change an exception family | Breaking | Preserve the old exception family when possible. |
| Remove a used underscored symbol | Potentially breaking | Search public callers before removal. |

Documentation does not define the complete public surface.
Check whether users can import a symbol from a documented path.

### 2. The `torch.neuron.*` device module

`torch._register_device_module("neuron", neuron)` exposes a user-visible API.
Check availability, device, memory, stream, event, and profiler functions.
Preserve names, signatures, return types, and error behavior.

### 3. The `"neuron"` device contract

The device contract includes:

- Accepted dtypes and transfer behavior.
- On-device contiguity behavior.
- CPU and Neuron tensor interaction.
- Stream and synchronization semantics.
- Error behavior for unsupported layouts or dtypes.

A change to these behaviors needs a user-impact statement and a migration plan.

### 4. `torch.compile(backend="neuron")`

Treat argument, shape, dtype, graph-break, and return-value behavior as public.
Read the current support table in `README.md`.
Do not rely on a copied list because the contract can change.

A new supported case is compatible when it does not change an existing case.
A newly rejected case or changed default is a compatibility break.

### 5. Environment variables

`docs/environment_variables.md` is the canonical list.
Renaming or removing a listed variable breaks existing user configurations.

For a rename:

1. Read both names during the migration period.
2. Prefer the new name when both names exist.
3. Emit a `FutureWarning` for the old name.
4. Set `stacklevel=2` when Python emits the warning.
5. Document when support for the old name ends.

### 6. On-disk compiled artifacts

A cache-key change can cause a safe cache miss or an unsafe cache hit.
Read `docs/AI-compilation-kernels.md` and identify the touched cache first.
Do not apply one cache's identity rules to another cache.

The HLO precompile cache stores tracing and StableHLO results.
Its implementation is `torch_neuronx/python_ops/compilation/hlo_cache.py`.
It owns a manifest, key logic, and `_SCHEMA_VERSION`.

The default persistent NEFF key is content-addressed.
`CompilationCacheKey` hashes serialized HLO or MLIR bytes with compiler configuration.
The compiler configuration includes platform, logical core count, optimization level, compiler arguments, compiler version, and enabled layout-optimization state.

- Change the HLO cache key or local schema when its serialized payload or interpretation changes.
- Preserve the persistent NEFF content identity.
- Add an explicit persistent-key input only for output-changing behavior that serialized IR and compiler configuration do not encode.
- Prove an artifact created under old semantics becomes a cache miss.
- Never serve an old artifact under new semantics.
- Document expected cache invalidation in the change description.

### 7. Documented public bindings

The private `_C` extension ships in the same wheel as its Python callers.
Do not infer a cross-version package ABI promise from an internal `_C` call.
A binding can still be a compatibility surface when public documentation or supported user code calls it directly.

- Search public documentation, examples, and external callers before assigning compatibility impact.
- Preserve documented public binding names, signatures, defaults, returns, and exceptions.
- Add a compatible path or migration when a documented public binding changes.
- Treat private Python-to-`_C` changes as same-wheel implementation changes unless public evidence says otherwise.
- Test the public wrapper and any documented direct binding call that the change affects.

## Python version support

Read `.python-version` and package metadata for supported versions.
Use syntax and standard-library APIs that work on the oldest supported version.
Check each supported CPython ABI that the change affects.

## Compatible additions

A change is usually compatible when it:

- Adds a new public symbol without changing an old symbol.
- Adds an optional keyword-only parameter with a stable default.
- Supports a new dtype, shape, or compile mode without changing old cases.
- Adds a new environment variable without changing an old variable.
- Extends an HLO precompile key or manifest for changed precompile behavior.
- Preserves the persistent NEFF content key, adding identity only for behavior absent from serialized IR and compiler configuration.

Require tests and user documentation for new behavior.

## Bug fixes and compatibility

A bug fix can change observable behavior.
Document the old behavior, intended behavior, and affected inputs.

An immediate behavior change can be acceptable for:

- A security vulnerability.
- A serious correctness defect.
- An API with a clear experimental stability statement.

Prefer a migration period for other user-visible changes.
Classify an ambiguous user-impact decision as `NEEDS_DISCUSSION`.

## Deprecation patterns

### Rename a Python parameter

Keep the old parameter during the migration period.
Reject a call that passes conflicting values through both names.

```python
import warnings


def compile_neuron(model, fullgraph=False, *, old_option=None, new_option=None):
    if old_option is not None:
        if new_option is not None and new_option != old_option:
            raise ValueError("old_option and new_option conflict. Pass only new_option.")
        warnings.warn(
            "old_option is deprecated. Use new_option.",
            FutureWarning,
            stacklevel=2,
        )
        new_option = old_option
    return run_compile(model, fullgraph=fullgraph, option=new_option)
```

### Rename an environment variable

Read the new name first.
Warn only when code uses the old name.

```python
import os
import warnings


def read_setting():
    if "TORCH_NEURONX_NEW_NAME" in os.environ:
        return os.environ["TORCH_NEURONX_NEW_NAME"]
    if "TORCH_NEURONX_OLD_NAME" in os.environ:
        warnings.warn(
            "TORCH_NEURONX_OLD_NAME is deprecated. Use TORCH_NEURONX_NEW_NAME.",
            FutureWarning,
            stacklevel=2,
        )
        return os.environ["TORCH_NEURONX_OLD_NAME"]
    return None
```

### Change an exception type

Preserve the old exception family when callers can catch it.

```python
class NeuronDtypeError(ValueError):
    pass
```

### Change compiled-artifact cache semantics

Identify the cache before changing its identity.
For the HLO precompile cache, update its real key, manifest, or local schema as required.
For persistent NEFF caching, preserve HLO-or-MLIR-bytes plus compiler-configuration identity.
Add an explicit input only for behavior absent from those serialized values.
Add a test that creates an entry with old semantics and proves the new reader misses it.

## Common compatibility defects

### Insert a positional parameter

Do not insert a new parameter before an existing parameter.
Append an optional parameter or make it keyword-only.

### Change a default immediately

Keep the old behavior during the documented migration period.
Warn about the future default and state when it changes.

### Remove a public attribute

Keep a deprecated property or method that forwards to the replacement.
Remove it only after the documented migration period.

### Rename an environment variable silently

Read both names during migration.
A silent rename disables existing user configurations.

### Change cache semantics without invalidating old artifacts

This can serve an artifact with stale semantics.
Identify the HLO precompile or persistent NEFF cache first.
Update only identity not already encoded by serialized IR and compiler configuration.
Test that artifacts from old semantics become misses.

### Change a documented public binding

This can break supported user code even when the binding is implemented in `_C`.
Search public callers first.
Preserve the documented contract or provide a migration path.
Do not assign public compatibility impact only because private same-wheel Python code calls `_C`.

### Change an exception family

Existing callers can depend on a specific base exception.
Make a new exception inherit from the prior exception when possible.

## Evidence to collect

For each possible break, collect:

1. The old signature or behavior.
2. The new signature or behavior.
3. Public callers or documented examples.
4. The warning and migration period, if required.
5. Tests for old and new call patterns.
6. The user documentation update.
7. Cache compatibility or documented-public-binding evidence when applicable.

Do not approve a possible break based only on the author's intent.
Verify old behavior in source, documentation, or a focused test.

## Review checklist

- [ ] No public symbol is removed or renamed without a migration path.
- [ ] New parameters are optional, ordered safely, and have stable defaults.
- [ ] Defaults, return types, shapes, and exception families remain compatible.
- [ ] Deprecations use `FutureWarning` and `stacklevel=2`.
- [ ] Environment-variable changes support both names during migration.
- [ ] The device contract remains compatible or has an approved migration.
- [ ] The backend contract remains compatible or has an approved migration.
- [ ] HLO precompile changes update that cache's key, manifest, or schema as needed.
- [ ] Persistent NEFF changes preserve content identity and add only behavior absent from serialized IR and compiler configuration.
- [ ] Old artifacts miss after every semantics change.
- [ ] Documented public bindings stay compatible or have a migration path supported by public-caller evidence.
- [ ] `README.md` and `docs/environment_variables.md` reflect user-visible changes.
- [ ] Tests prove the current contract and migration path.

## Questions to ask when unsure

1. Can existing user code produce a silent incorrect result?
2. Can existing user code raise a new exception?
3. Can users migrate without an immediate code change?
4. Does public documentation describe the change?
5. Are public callers known and tested?

Return `NEEDS_DISCUSSION` when evidence cannot settle an intentional compatibility decision.
State the decision, affected users, evidence, and migration choices.
