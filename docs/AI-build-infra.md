# Build Infrastructure

The build compiles `_C.so` (the pybind11 extension), **builds the vendored
torch-mlir native libs** (`_mlir.so`, `_torchMlir.so`, `_stablehlo.so`),
generates protobuf stubs, and packs everything — including the bundled
`torch_mlir` Python package — into a wheel. **Bazel is primary; CMake is an
IDE fallback that builds only `_C.so`, not the torch-mlir libs.** `setup.py`
orchestrates both.

> **Navigation → codegraph.** For the file map (`setup.py`, `BUILD.bazel`,
> `MODULE.bazel`, `bazel/*.bzl`, `tools/*`, `.github/workflows/*`) and the
> torch-mlir import sites, ask codegraph:
> `codegraph_explore "setup.py BuildBazelExtension _fix_rpath generate_build_config torch_extension"`.
> This doc keeps the link-time symbol-ownership story, the PyTorch/NRT bump
> recipes, and the CMake-can't-do-MLIR limitation — the gap no source file
> or human doc states.

## Vendored torch-mlir — the link-time gap

torch-neuronx **vendors a subset of `aws-neuron/torch-mlir` in-tree** at
`third_party/torch-mlir/`. It is **not** a PyPI dependency — `import torch_mlir`
resolves to the vendored package because the wheel ships it (`setup.py`
`package_dir = {"torch_mlir": "third_party/torch-mlir/python/torch_mlir"}`).
What's vendored vs excluded, and the bump procedure (including the LLVM +
StableHLO re-pin in `WORKSPACE`), are owned by
**`third_party/torch-mlir/README.md`** (authoritative) — don't restate them.

The agent-facing gap that README doesn't cover is **symbol ownership — why
`_mlir.so` is tiny**: `BUILD.bazel`'s `mlir_capi_deps` links MLIR + torch-mlir
CAPI symbols **into `_C.so`**, making `_C.so` the canonical owner at runtime.
`_mlir.so` / `_torchMlir.so` link against it via the `_C_import` `cc_import`
rule (`_torchMlir.so` is nanobind, not pybind11 — must match `_mlir.so`'s type
casters). `_mlir.so` is built with `MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.` so
it imports under `torch_mlir.*`. After Bazel produces the three `.so`s,
`BuildBazelExtension` copies them into
`third_party/torch-mlir/python/torch_mlir/_mlir_libs/` (not `torch_neuronx/`),
then `_fix_rpath` (patchelf) rewrites RUNPATH so they find `_C.so`, torch libs,
and `/opt/aws/neuron/lib`. The LLVM + StableHLO pins that must stay in lockstep
with the vendored rev live in the **top-level `WORKSPACE`**
(`LLVM_COMMIT`/`LLVM_SHA256`, `STABLEHLO_COMMIT`).

## Bump recipes (procedural gap)

For **vendored torch-mlir**, follow `third_party/torch-mlir/README.md`
(authoritative). The two recipes no human doc owns:

**Bump PyTorch:** `pyproject.toml` `torch~=X.Y.Z` → `.bazelrc`
`PYTORCH_VERSION` → `bazel/python_config.bzl` SHA-256s → `bazel/torch.BUILD`
if libs/headers changed → `./tools/build && ./tools/build-and-test`.

**Bump NRT:** `prerequisites.sh` pin → `bazel/nrt.BUILD` / `CMakeLists.txt` if
layout changed → default path `/opt/aws/neuron` (override `NRT_LOCAL_PATH`) →
`./tools/build && ./tools/run-cpp-test`.

## `.bazelrc` specifics (not in BAZEL_BUILD.md)

C++20, `-O2 -DNDEBUG`, `-Werror` scoped to package targets (not deps),
`--spawn_strategy=local` (sandbox disabled so the build reaches system libs),
`PYTORCH_VERSION` / `NRT_LOCAL_PATH` (default `/opt/aws/neuron`) passthroughs.

## `setup.py` orchestration

Backend via env: `USE_BAZEL=1` → `BuildBazelExtension`, `USE_CMAKE=1` →
`BuildCMakeExtension`, neither → torch `CppExtension`. All paths: regenerate
proto stubs (`BuildPyCommand`) → `generate_build_config(is_release)` sets
`BUILD_TYPE` in `_build_config.py` → compile `_C.so` → (Bazel only)
`_fix_rpath` → `$ORIGIN/../torch/lib:/opt/aws/neuron/lib`. Version
`0.1.0+{git_sha}`; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` during build to prevent
self-import. `pyproject.toml`: Python `>=3.10,<3.14`, `torch~=2.11.0`.

**CMake limitation:** does **not** build the torch-mlir libs — the MLIR /
StableHLO lowering paths require the Bazel build. CMake is only for IDE work
on `_C.so` (`USE_CMAKE=1 python setup.py build_ext`).

## Wheel packaging

`./tools/build --wheel [release]`. Bazel builds `_C.so` + the three torch-mlir
`.so`s; `_C.so` → `torch_neuronx/`, torch-mlir libs →
`third_party/torch-mlir/python/torch_mlir/_mlir_libs/`; patchelf fixes every
RPATH; `bdist_wheel` packs **two** Python packages (`torch_neuronx/` +
`torch_mlir/`). After install, the venv has both import names with **no
separate `pip install torch-mlir`** — only `torch-neuronx` shows in `pip list`
though `from torch_mlir import fx` works.

## Vendored-tree tests

- `tools/run-torch-mlir-lit-tests` — Bazel LIT FileCheck under
  `//third_party/torch-mlir/test/...`; known failures XFAILed in-file.
- `tools/run-torch-mlir-e2e-tests` — upstream E2E via `fx_importer_stablehlo`;
  requires a prior `./tools/build` so `_mlir.so` exists.

## CI internals (gap content; job list in CONTRIBUTING.md)

- `nightly-dispatcher.yml` runs 2 AM UTC daily; `nightly.yml` builds base image
  → torch layer → wheel → promote to stable.
- `_block-perf.yml`: reusable block + collective perf; **MFU golden thresholds
  enforced on PRs (baseline × 0.9), not nightly.**
- Adding a CI job: add to `pull.yml`, use the
  `./.github/actions/run-test-suite` composite action; runner labels include
  `neuron.internal_compute:trn2.48xlarge`.

## Authoritative human docs (cite, don't restate)

- Bazel reference (targets, Bazelisk, `--override_repository`) → `BAZEL_BUILD.md`.
- Dev build/test commands, lint, op-coverage HTML, CI job list → `CONTRIBUTING.md`.
- Prerequisites / `--docs` output → `README.md`.
- What's vendored vs excluded + the vendored-torch-mlir bump recipe → `third_party/torch-mlir/README.md`.

## See also

- `AI-testing.md` — test runners + fixtures.
- `AI-csrc.md` — the C++ extension compiled here.
- `AI-compilation-kernels.md` — NEFF cache + `CompilerSubprocess`.
