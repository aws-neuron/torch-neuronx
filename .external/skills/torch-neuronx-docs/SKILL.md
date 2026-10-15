---
name: torch-neuronx-docs
description: Routes torch-neuronx architecture and source-location questions to the correct AI guide and current source. Use when a user asks how a subsystem works, where behavior is implemented, what calls a symbol, or which repository file owns a contract.
---

# torch-neuronx Source Navigation

Use this skill to explain the repository. Keep the task report-only.
Read the relevant guide first. Verify each answer in the current source.

## Prerequisites

- Run commands from the torch-neuronx repository root.
- Confirm the root with `git rev-parse --show-toplevel`.
- Confirm the guides with `git ls-files docs | grep -E 'AI-.*[.]md$'`.
- Use `rg` for source search when it is available.
- Use `git grep -n -E` when `rg` is unavailable.
- Treat the source as authoritative when a guide differs from the source.
- Do not require Neuron hardware for source navigation.

### Safety

- This skill is report-only. Do not edit files.
- Do not create a commit unless the user asks for a separate implementation task.
- Before a GitHub write, show the exact content and wait for approval.
- Open a pull request only as a draft after approval.
- Never merge, mark ready, force-push, or push to `main`.
- Disclose AI assistance in a pull request description.

## Workflow

1. Restate the question as one behavior, symbol, or subsystem.
2. Select one primary guide from the routing table.
3. Read the primary guide before searching the source.
4. Read a linked guide only when the question crosses a subsystem boundary.
5. Extract exact symbols, environment variables, and source paths from the guide.
6. Search only the key source paths in the routing table first.
7. Expand the search only when the first search has no result.
8. Read the definition and its direct callers.
9. Check tests for examples and expected behavior.
10. Report the result with repository-relative `file:line` citations.
11. State each conflict between a guide and the current source.
12. Stop without changing the repository.

## Subsystem routing

| Question type | Read first | Search these source paths |
|---|---|---|
| Package purpose, import sequence, or repository layout | `docs/AI-package-overview.md` | `README.md`, `pyproject.toml`, `torch_neuronx/__init__.py` |
| Public device, stream, memory, profiler, or NKI API | `docs/AI-runtime-api.md` | `torch_neuronx/`, `torch_neuronx/csrc/` |
| `torch.compile` backend, FX passes, graph conversion, or executor behavior | `docs/AI-dynamo-backend.md` | `torch_neuronx/neuron_dynamo_backend/` |
| Eager ATen operation, dtype policy, fallback, or operation registration | `docs/AI-python-ops.md` | `torch_neuronx/python_ops/`, `torch_neuronx/csrc/core/dispatch/` |
| C++ extension, dispatcher, allocator, cache, stream, or execution engine | `docs/AI-csrc.md` | `torch_neuronx/csrc/` |
| Process group, collective lowering, DTensor, mesh, or distributed work | `docs/AI-distributed.md` | `torch_neuronx/distributed/`, `torch_neuronx/csrc/distributed/` |
| `all_to_all` or variable split all-to-all | `docs/AI-all-to-all.md` | `torch_neuronx/distributed/`, `torch_neuronx/csrc/distributed/`, `tests/distributed/collective_ops/` |
| NEFF compilation, cache key, compiler process, or stored artifact | `docs/AI-compilation-kernels.md` | `torch_neuronx/kernels/`, `torch_neuronx/csrc/core/compilation/`, `torch_neuronx/csrc/core/cache/` |
| Build, wheel, Bazel, CMake, or vendored torch-mlir | `docs/AI-build-infra.md` | `setup.py`, `BUILD.bazel`, `MODULE.bazel`, `CMakeLists.txt`, `tests/csrc/CMakeLists.txt`, `bazel/`, `third_party/torch-mlir/` |
| Test runner, fixture, marker, retry, or upstream test registry | `docs/AI-testing.md` | `conftest.py`, `tests/`, `tools/` |
| Usage example or demonstration | `docs/AI-examples.md` | `examples/` |

### Authoritative companion documents

Use these documents when the selected guide points to them:

| Topic | Document |
|---|---|
| Build and test entry points | `README.md`, `CONTRIBUTING.md`, `BAZEL_BUILD.md` |
| Environment variables | `docs/environment_variables.md` |
| Decomposition authoring | `docs/decompositions.md` |
| Current int64 default and routing policy | `torch_neuronx/int64_policy.py` |
| Packed int64 design background before the `native` default | `docs/packed_int64_design.md` |
| C++ dispatch onboarding | `docs/cpp_dispatch_onboarding_guide.md` |
| NUMA behavior | `docs/numa_optimization.md` |
| Memory and statistics APIs | `docs/statistics/torch_neuronx_stats_readme.md` |
| Vendored torch-mlir | `third_party/torch-mlir/README.md` |

## Search procedure

Start with the narrowest exact symbol search:

```sh
rg -n 'ExactSymbol' torch_neuronx tests
```

Search a behavior or a group of related symbols:

```sh
rg -n 'first_term|second_term|ENVIRONMENT_VARIABLE' torch_neuronx tests tools
```

Use tracked files when `rg` is unavailable:

```sh
git grep -n -E 'ExactSymbol|ENVIRONMENT_VARIABLE' -- torch_neuronx tests tools
```

Find a file when the question gives only a partial name:

```sh
if command -v rg >/dev/null 2>&1; then
    git ls-files | rg 'partial_name|subsystem_name'
else
    git ls-files | grep -E 'partial_name|subsystem_name'
fi
```

Search in this order:

1. Find the symbol definition.
2. Find registration or export sites.
3. Find direct call sites.
4. Find tests that name the symbol or behavior.
5. Find configuration reads for the related environment variable.
6. Read adjacent code before you infer control flow.

Use the selected guide to constrain each search. Do not search the full tree first.
Use a second guide only when the first guide links to that subsystem.

### Source checks

- Verify that each cited path exists with `git ls-files <path>`.
- Verify that each cited line still contains the relevant symbol.
- Check both Python and C++ registration when a boundary crosses `_C`.
- Check tests for observable behavior. Do not use tests as the only design source.
- Separate a source fact from an inference.
- State that an inference is an inference.

## Citation format

Cite one line as `FILE:LINE`, using the repository-relative file name.
Cite a range as `FILE:START-END`, using the repository-relative file name.
Cite the guide when it supplies a contract or subsystem map.
Cite the source when it supplies current implementation behavior.
Cite both when the answer depends on a guide claim that the source confirms.

Use this answer structure:

```text
Result: <direct answer>
How it works: <short control-flow explanation>
Evidence:
- docs/AI-<guide>.md:<line> - <documented contract>
- torch_neuronx/<path>:<line-range> - <current implementation>
Tests:
- tests/<path>:<line-range> - <observed behavior>
Drift: none found
```

Replace `Drift: none found` with a precise conflict when the guide is stale.
Do not claim that a call path is complete unless the search confirms it.

## Hand-offs

Stay in this skill for explanation, navigation, and source citation.
Hand off an implementation of a new ATen lowering to `aten-op-lowering`.
Hand off an upstream PyTorch test failure to `pytorch-upstream-test-debugger`.
Hand off a pull request or branch review to `pr-review`.
Hand off a compiler failure reproduction package to `compiler-artifact-repro`.
Return to this skill when a hand-off task needs repository orientation.

## Verification

Before the report, run these checks:

```sh
git status --short
git ls-files 'docs/AI-*.md'
if command -v rg >/dev/null 2>&1; then
    rg -n 'ExactSymbol' <selected-source-paths>
else
    git grep -n -E 'ExactSymbol' -- <selected-source-paths>
fi
```

Confirm that `git status --short` did not change during the task.
Confirm every evidence line from the current checkout.
Confirm that the report contains no proposed edit unless the user asks for one.
No build or hardware test is required for a source-navigation report.

## Failure handling

Return `BLOCKED` when the repository root or selected guide is missing.
Include the missing path and the command that failed.
Include the exact command that the user can run after restoring the checkout.

Return `BLOCKED` when the question requires runtime proof and Neuron hardware is unavailable.
List the source evidence that you could verify.
List each runtime command that you could not run.
Give the exact `tools/run-test <test-path>` command for the selected test.

Return `BLOCKED` when a required build or credential is unavailable.
Do not replace missing evidence with an assumption.
Report the exact build or access command that failed.
Report the exact command to retry after the prerequisite is available.

## Output

Return the direct answer first.
Then give the short control flow and `file:line` evidence.
List relevant tests after the implementation evidence.
List guide drift, search gaps, and unverified runtime behavior.
End with `BLOCKED` only when a required fact cannot be verified.
Never edit files, create commits, or perform GitHub writes in this workflow.
