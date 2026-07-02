---
name: torch-neuronx-docs
description: "Knowledge base for the torch-neuronx package — a PyTorch backend for AWS Neuron devices. Use when the user asks any question about this repo: what it is, how it's structured, how the eager path works, how torch.compile(backend=\"neuron\") works, the C++ _C extension, ProcessGroupNeuron / collectives / DTensor, the NEFF compilation cache, how to build / test / extend the code, or where to find a specific file. Navigation (where is X / what calls it / show me the source) is delegated to the codegraph MCP server — these docs carry the subsystem contracts, invariants, and authority routing codegraph can't index. Two modes: read-only (default) and verify (audits the docs' concrete claims against source for a high-stakes task)."
---

# torch-neuronx Knowledge Base

> **Freshness.** Navigation content (signatures, call graphs, file trees) is
> served live by **codegraph** — see the Bootstrap section below — so it
> can't go stale. These docs hold only stable subsystem *contracts*; if a
> contract here disagrees with source, source wins and that's a bug to fix
> (run **verify** mode, bottom of this file).

## Navigation is delegated to codegraph (highly recommended)

These docs are **deliberately thin**: they carry subsystem *contracts*
(defaults, gated behaviors, invariants), *authority routing* (which
human doc owns what), and the *right codegraph query* per subsystem. They
do **not** carry the code tree, signatures, call graphs, or per-symbol
writeups — that is **codegraph's** job, and codegraph does it better
because it returns current verbatim source plus a call/blast-radius graph
that can never go stale.

So: **"where is X / what calls it / what breaks if I change it / show me
the source"** → ask codegraph. **"what's the contract / default / which
doc is authoritative"** → these docs. Each subsystem doc opens with the
exact `codegraph_explore` query for its area.

### Bootstrap codegraph (step 0)

If the `mcp__codegraph__*` tools are **not** available in this session,
codegraph isn't wired into the agent yet. Print these steps for the user
(install + agent registration need a session restart before the tools
appear — a skill can't make them load mid-turn):

```sh
# 1. Install the CLI (one-time, per machine)
#    macOS / Linux:
curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh
#    Windows (PowerShell):
#    irm https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.ps1 | iex

# 2. Register the MCP server into the agent, then RESTART the session
codegraph install

# 3. Build the initial index for this repo
codegraph init .
```

If the tools **are** available, refresh the index before relying on it —
the DB is local per-machine and git-ignored, so it can lag the working
tree:

```sh
codegraph sync .    # sync changes since last index
```

Without codegraph these docs still function as a contract reference, but
the navigation layer is gone — code lookups fall back to Read/grep.

## What this is

`torch-neuronx` is a **PyTorch backend for AWS Neuron devices** (Trainium /
Inferentia). It adds a first-class `'neuron'` device by hooking into
PyTorch's [`PrivateUse1`](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
mechanism and exposing two execution paths:

1. **Eager mode** — `tensor.to('neuron')` runs individual ATen ops on
   Neuron by routing them through a C++ dispatcher and per-op implementations.
2. **`torch.compile` mode** — `torch.compile(model, backend="neuron")`
   compiles whole FX graphs to NEFF via a StableHLO pipeline and executes
   them through the Neuron Runtime (NRT).

The subsystem docs live in `docs/` under the `AI-` prefix (beside the
human-authored docs); this `SKILL.md` is their index. Start with
`docs/AI-package-overview.md`, then drill into the matching subsystem doc.

> **Resolving doc paths (read this if a doc "isn't found").** Every
> `docs/AI-*.md` path below is **relative to the torch-neuronx repo root**,
> not your current working directory. If a `Read docs/AI-*.md` fails, your
> CWD is elsewhere (common when this skill is loaded by a dispatched
> sub-agent) — the docs are **not** missing. Resolve the repo root and read
> the absolute path:
> ```sh
> ROOT=$(git -C "<dir-of-this-SKILL.md>" rev-parse --show-toplevel)
> #   then read "$ROOT/docs/AI-*.md"
> ```
> This `SKILL.md` lives at `<repo-root>/.kiro/skills/torch-neuronx-docs/SKILL.md`,
> so the repo root is four directories up from it — use that if `git` is
> unavailable.

## Subsystem map

| If the user asks about... | Read |
|---------------------------|------|
| "What is torch-neuronx?", "how is this repo organized?" | `docs/AI-package-overview.md` |
| `torch.neuron.*`, `torch_neuronx.Stream`, `memory_allocated`, `NeuronProfiler`, `@nki_op`, `wrap_nki`, any top-level `torch_neuronx.*` API | `docs/AI-runtime-api.md` |
| `torch.compile(backend="neuron")`, FX passes, aliasing analysis, flex attention, vocab sharding, MegaCache, `neuron_dynamo_backend` | `docs/AI-dynamo-backend.md` |
| How an ATen op is implemented in eager mode, `@neuron_op`, MLIR / NKI backends, cast policy, contiguous variants, CPU fallback, adding a new op | `docs/AI-python-ops.md` |
| The `_C` extension, `NeuronDispatcher`, allocators, concatenation engine, compilation cache, lazy materialization, streams C++ impl, profiler, `NeuronWork` | `docs/AI-csrc.md` |
| Multi-device training, the `"neuron"` c10d backend, `all_reduce`/`all_gather`/`reduce_scatter`/`all_to_all`/`broadcast`/`barrier`, DTensor sharding, bucketing, premul-sum | `docs/AI-distributed.md` |
| How NEFF caching works, how `neuronx-cc` is invoked, `CompilerSubprocess`, `NEFFCacheKey`, `CompilationJob`, artifact layout | `docs/AI-compilation-kernels.md` |
| Running / writing tests, `./tools/run-test[-parallel]`, `./tools/run-cpp-test`, markers, upstream-test harness | `docs/AI-testing.md` |
| "How do I use X?", "show me an example of Y" | `docs/AI-examples.md` |
| Building the extension, compiler flags, dev env, bumping PyTorch / NRT / **vendored torch-mlir**, the `third_party/torch-mlir/` tree, CI, wheel packaging | `docs/AI-build-infra.md` |

## Human-authored docs (the AI- docs must not duplicate these)

The `AI-*.md` docs sit **in `docs/` beside the repo's human-written docs**.
Those human files are **authoritative**: when a question is about a
documented contract, the human doc (and source) wins over an AI- summary.
The AI- docs are a **navigation/contract layer** that points to the human
docs, never restates them.

**No-duplication invariant.** An AI- doc covers only the agent-facing gap
codegraph and the human docs both leave: cross-cutting contracts,
invariants, gated behaviors, and the index into the doc set. Anything a
human doc already owns (env-var tables, command recipes, op-compatibility
lists, authoring guides) is cited by inline path, not reproduced.

| Human doc | Owns |
|----------|--------|
| `README.md` | Elevator pitch, the two execution paths, 64-bit/contiguous limitations, build/test quickstart, examples table |
| `CONTRIBUTING.md` | Build/test commands, reports & coverage, HuggingFace tests, CI expectations, op-coverage expansion |
| `BAZEL_BUILD.md` | Bazel build + test invocation, targets, build-architecture file roles |
| `docs/torch_api_compatibility.md` / `docs/ops.html` | ATen + `torch.distributed` op compatibility |
| `docs/environment_variables.md` | Complete env-var reference + artifact directory layout |
| `docs/decompositions.md` | Decomposition authoring guide |
| `docs/packed_int64_design.md` | Packed-int64 representation, `TORCH_NEURONX_INT64_MODE`, storage layout, compute limits |
| `docs/cpp_dispatch_onboarding_guide.md` | C++ dispatch onboarding (preprocessing protocol, NKI/ATen override onboarding) |
| `docs/numa_optimization.md` | NUMA pinning / `neuron-numa-launch` |
| `docs/statistics/torch_neuronx_stats_readme.md` | Metrics + memory API tour |
| `docs/torchtitan/` | TorchTitan training tutorials |
| `examples/*/README.md`, `examples/compile/QUICKSTART.md` | Per-example setup + launch |

## Quick answers

### How is the backend wired into PyTorch?

Two `pyproject.toml` entry points register it automatically:
`torch.backends` → `torch_neuronx:_autoload` (registers the `'neuron'`
device) and `torch_dynamo_backends` → `neuron_backend`. On import,
`torch_neuronx/__init__.py` loads `_C`, renames the `PrivateUse1` backend
to `"neuron"`, registers `torch.neuron.*`, registers Python op impls, and
sets env defaults. See `docs/AI-package-overview.md` + `docs/AI-runtime-api.md`
(or codegraph the boot symbols).

### Two important invariants

- **Neuron computes on 32-bit types.** `float64`/`uint64` are silently
  down-cast on entry, re-widened on exit. `int64` is governed by
  `TORCH_NEURONX_INT64_MODE` (default `packed`): packed keeps 64 bits on
  device as `int32` lane-pairs (lossless storage + transfer) while
  *arithmetic* runs on the low 32 bits; `cpu_bounce` does the host-roundtrip
  down-cast; `native` passes `S64` to the compiler. The dynamo `Executor`
  path still down-casts `int64`→`int32` via `view()`. Authoritative:
  `docs/packed_int64_design.md`.
- **Only contiguous tensors execute on-device.** Non-contiguous inputs are
  materialized via the `contiguous_*` op family, or for eager copies via
  `csrc/utils/CopyUtils.cpp`.

### `torch-mlir` is vendored in-tree

`import torch_mlir` resolves to `third_party/torch-mlir/python/torch_mlir/`,
**not** a PyPI dependency. The wheel ships both `torch_neuronx` and
`torch_mlir`, plus three native libs (`_mlir.so`, `_torchMlir.so`,
`_stablehlo.so`). The vendored tree is stripped to `TorchToNeuron` +
`TorchToStablehlo`. Pin: `third_party/torch-mlir/TORCH_MLIR_UPSTREAM_SHA`;
LLVM + StableHLO pinned in top-level `WORKSPACE`. See `docs/AI-build-infra.md`.

### Build + test quickstart

The `prerequisites.sh` → `uv` env → `neuronx-cc` install → `./tools/build`
→ `./tools/run-test` / `run-cpp-test` flow is in `README.md` (authoritative).
For what the build/test machinery *does* (Bazel vs CMake, vendored torch-mlir,
test orchestration), see `docs/AI-build-infra.md` and `docs/AI-testing.md`.

### Key environment variables

Complete list in `docs/environment_variables.md`. Most common:
`TORCH_NEURONX_DEBUG_DIR`, `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`,
`NEURON_LAUNCH_BLOCKING=1`, `TORCH_NEURONX_DISABLE_NEFF_CACHE=1`,
`NEURON_CC_FLAGS`, `TORCH_NEURONX_LOG_LEVEL=3`.

### Conventions

- Paths are **repo-relative** (`torch_neuronx/...`, `tests/...`, `tools/...`).
- Each doc opens with its codegraph query and cross-links via `See also`.

---

## Maintaining this skill

The division of labor is what keeps this skill cheap to own:

- **Navigation freshness is codegraph's job.** Signatures, call graphs, and
  file trees are never written down here — they're served live by
  `codegraph_explore`. `codegraph sync .` keeps that current; there is **no
  doc-refresh workflow to run** for navigation drift.
- **Contracts change rarely.** What these docs hold — defaults, invariants,
  gated env-var behaviors, registration order, the no-duplication routing —
  moves only when a subsystem's *contract* moves (a new public API, a flipped
  default, a removed selection path). When that happens, edit the one
  affected doc inline at the same altitude; describe the contract, not the
  patch. No PR/issue numbers in doc bodies.
- **What does NOT belong here:** bug fixes, perf tweaks, refactors, new ops
  following an existing pattern, test edits — the source (and codegraph) is
  authoritative for those. If an edit would duplicate a human doc from the
  table above, write a pointer instead.

### Verify mode (audit a doc against source)

Run when a high-stakes task is about to depend on the docs (a PR review of a
cross-cutting change, a debug session on a subtle dispatcher/cache-key bug),
or when you suspect a contract has drifted. Trigger phrases: `verify the
knowledge base`, `audit the docs skill`, `verify torch-neuronx-docs`, or
`/skill torch-neuronx-docs verify` (optionally scope: `verify AI-python-ops.md`).

The docs are now small enough that verify is a light pass, not a fan-out:

1. **Scope** — all `docs/AI-*.md` (skip `AI-examples.md` unless asked), or the
   one doc named.
2. **Per doc, one agent** (fan out per-section only if a doc still exceeds one
   context). Each agent: read the doc, extract every **verifiable** claim
   (path / symbol / signature / registration order / env var / public-API
   export — *not* soft motivation claims), locate each in source via
   codegraph/Read (never memory), and return per-claim verdicts:

   ```
   [VERIFIED]   <doc:anchor> — <claim>   evidence: <file:line>
   [WRONG]      <doc:anchor> — <claim>   reality: <file:line>   fix: <one sentence>
   [MISSING]    <doc:anchor> — <claim>   searched: <pattern>; symbol/path not found
   [UNVERIFIABLE] <doc:anchor> — <claim>   reason: <not checkable from source>
   ```

   Pass each agent `model: "opus"` — these do contract-synthesis, not just
   grep-and-cite, and Haiku has fabricated claims before. Verify is
   **read-only**: it produces a report, it does not edit or commit.
3. **Aggregate** into one report grouped by doc then verdict (`WRONG` first),
   with a per-doc pass rate and a short list of high-confidence fixes the user
   can approve as a batch.
4. **Hand off.** The user picks which findings to fix; apply them as small
   per-doc commits. If a doc returns >5 `WRONG`, it has drifted enough that
   the whole doc needs a rewrite, not spot fixes.

Most `[UNVERIFIABLE]` verdicts are fine (design rationale, mental-model
prose). Only flag the ones that are unverifiable **and** specific (e.g. "called
1000× per step") — those should be measured-and-cited or cut.
