# Build Infrastructure

The build compiles `_C.so` (the pybind11 extension), **builds the vendored
torch-mlir native libs** (`_mlir.so`, `_torchMlir.so`, `_stablehlo.so`),
generates protobuf stubs, and packs everything - including the bundled
`torch_mlir` Python package - into a wheel. **Bazel is primary; CMake is an
IDE fallback that builds only `_C.so`, not the torch-mlir libs.** `setup.py`
orchestrates both.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'BuildBazelExtension|_fix_rpath|generate_build_config|torch_extension' setup.py BUILD.bazel MODULE.bazel bazel tools third_party/torch-mlir`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Vendored torch-mlir - the link-time gap

torch-neuronx **vendors a subset of `aws-neuron/torch-mlir` in-tree** at
`third_party/torch-mlir/`. It is **not** a PyPI dependency - `import torch_mlir`
resolves to the vendored package because the wheel ships it (`setup.py`
`package_dir = {"torch_mlir": "third_party/torch-mlir/python/torch_mlir"}`).
The vendored commit and update procedure, including the LLVM + StableHLO
re-pin in `WORKSPACE`, are documented in
**`third_party/torch-mlir/README.md`**. Inspect the tree for the current
vendored file set.

The native-library dependency graph is defined by the Bazel targets.
`BUILD.bazel` links `mlir_capi_deps` into `_C.so`. In
`third_party/torch-mlir/BUILD.bazel`, `_mlir.so` and `_torchMlir.so` depend on
the `_C_import` `cc_import`, while `_stablehlo.so` depends on `_mlir_import`.
`_torchMlir.so` is nanobind, not pybind11. The MLIR Python targets use
`MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.` where required. After Bazel produces
the three `.so`s, `BuildBazelExtension` copies them into
`third_party/torch-mlir/python/torch_mlir/_mlir_libs/` (not `torch_neuronx/`),
then `setup.py::_fix_rpath` rewrites RUNPATH for the installed wheel layout.
Read that function for the current search paths instead of copying them here.
The LLVM + StableHLO pins that must stay in lockstep with the vendored rev live
in the **top-level `WORKSPACE`** (`LLVM_COMMIT`/`LLVM_SHA256`,
`STABLEHLO_COMMIT`).

## Bump recipes (procedural gap)

For **vendored torch-mlir**, follow `third_party/torch-mlir/README.md`
(authoritative). Two dependency update recipes:

**Bump PyTorch:** keep the exact `torch==X.Y.Z` build pin in
`pyproject.toml` synchronized with `.bazelrc` `PYTORCH_VERSION`. Update the
matching wheel hashes in `bazel/python_config.bzl`, and update
`bazel/torch.BUILD` if library or header layout changed. `setup.py` derives the
selected or installed PyTorch version and emits an exact `torch==X.Y.Z` wheel
requirement. Then run `./tools/build && ./tools/build-and-test`.

**Bump NRT:** `prerequisites.sh` installs or upgrades the latest available
Neuron runtime package instead of pinning a version. The repository defaults
`NRT_LOCAL_PATH` to `/opt/aws/neuron`. Bazel uses that value as the NRT local
repository root, and CMake reads NRT headers and libraries from its `include/`
and `lib/` directories. `bazel/nrt.BUILD` expects headers under `include/` and
`lib/libnrt.so`; its linker and RPATH entries currently name
`/opt/aws/neuron/lib`. If the selected runtime changes the install root or
layout, update the path configuration and build definitions. Then run
`./tools/build && ./tools/run-cpp-test`.

## Bazel configuration specifics (not in BAZEL_BUILD.md)

`.bazelrc` sets C++20, `-O2 -DNDEBUG`, local spawn strategy, and the PyTorch and
NRT path passthroughs. `BUILD.bazel` adds package-scoped `PACKAGE_COPTS`,
including `-Werror`, so dependencies do not inherit it.

## `setup.py` orchestration

Backend selection uses `USE_BAZEL=1` for `BuildBazelExtension`,
`USE_CMAKE=1` for `BuildCMakeExtension`, and torch `CppExtension` otherwise.
All paths regenerate protobuf stubs, generate `_build_config.py`, and compile
`_C.so`. The Bazel path also calls `setup.py::_fix_rpath`; that function is the
authority for RUNPATH entries.

`setup.py::_get_version` builds the package version from the selected PyTorch
major/minor, `torch_neuronx/_version.py` Neuron major/minor, a build counter,
and a git-derived suffix. Release branches omit the `.dev` suffix. Read
`pyproject.toml` for current Python and build dependency pins.
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` prevents self-import during the build.

**CMake limitation:** does **not** build the torch-mlir libs - the MLIR /
StableHLO lowering paths require the Bazel build. CMake is only for IDE work
on `_C.so` (`USE_CMAKE=1 python setup.py build_ext`).

## Wheel packaging

`./tools/build --wheel [release]`. Bazel builds `_C.so` + the three torch-mlir
`.so`s; `_C.so` -> `torch_neuronx/`, torch-mlir libs ->
`third_party/torch-mlir/python/torch_mlir/_mlir_libs/`; patchelf fixes every
RPATH; `bdist_wheel` packs **two** Python packages (`torch_neuronx/` +
`torch_mlir/`). After install, the venv has both import names with **no
separate `pip install torch-mlir`** - only `torch-neuronx` shows in `pip list`
though `from torch_mlir import fx` works.

## Vendored-tree tests

- `tools/run-torch-mlir-lit-tests` - Bazel LIT FileCheck under
  `third_party/torch-mlir/test/`; known failures XFAILed in-file.
- `tools/run-torch-mlir-e2e-tests` - upstream E2E via `fx_importer_stablehlo`;
  requires a prior `./tools/build` so `_mlir.so` exists.

## Continuous integration

The active CI workflows define repository automation and required checks.
`CONTRIBUTING.md` gives developer commands and high-level expectations, but
its CI job list can lag the workflows.

## Authoritative human docs (cite, don't restate)

- Bazel reference (targets, Bazelisk, `--override_repository`) -> `BAZEL_BUILD.md`.
- Dev build/test commands, lint, op-coverage HTML, and high-level CI expectations
  -> `CONTRIBUTING.md`.
- Prerequisites / `--docs` output -> `README.md`.
- Vendored commit + update procedure -> `third_party/torch-mlir/README.md`.

## See also

- `docs/AI-testing.md` - test runners + fixtures.
- `docs/AI-csrc.md` - the C++ extension compiled here.
- `docs/AI-compilation-kernels.md` - NEFF cache + `CompilerSubprocess`.
