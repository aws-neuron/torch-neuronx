---
name: aten-op-lowering
description: End-to-end runbook for adding a native Neuron lowering for an ATen op (e.g. aten::linspace, the upsample family) — ordered steps, exact commands, the platform gotchas that cause silent wrong results, how to triage each upstream failure into fix-vs-xfail, the registry contract, and how to open the PR. Use when the user says "implement aten::<op>", "lower <op> to Neuron", "add op coverage", "this op falls back to CPU", or is triaging upstream test failures for a newly lowered op. Companion to `pr-review` (reviews the finished PR) and `pytorch-upstream-test-debugger` (fixes pre-existing failures). Cross-references `torch-neuronx-docs` for subsystem contracts.
---

# Adding a native ATen op lowering

An ordered runbook. Do the steps in order — the ordering exists to keep the
device busy while you write code, and to avoid re-running 20-minute suites.

| Sibling skill | Job |
|---|---|
| `pr-review` | Review the finished PR (run it in branch mode before asking a human) |
| `pytorch-upstream-test-debugger` | Fix an *existing* failing test unrelated to a new op |
| `torch-neuronx-docs` | Subsystem contracts (`04-python-ops.md`, `05-csrc.md`, `08-testing.md`) |

**Navigation:** this repo has codegraph. `codegraph_explore "<symbols>"` before
grep/Read — it returns current verbatim source plus the call graph.

**Memory:** read `.kiro/skills/aten-lowering-orchestrator/references/lessons.md`
at the start of every run — it is this skill's memory. **Do not** write anything
to your own personal/local memory; a generalizable lesson goes into `lessons.md`
(via the orchestrator's retro), an op-specific quirk into that op's PR body.

**Speed / long-running commands:** the device and the build are the bottleneck,
not you. Any command that can exceed a foreground timeout — `./tools/build` cold,
NEFF compiles, an upstream suite — must be launched in the **background** and
polled incrementally (short `sleep` + `tail` on its log), never blocked on in the
foreground. Bazel is seconds when warm but minutes cold; start it, then keep
working. Don't burn a 2-minute foreground timeout waiting on a 15-minute run.

---

## Read this table first — 6 platform gotchas that cause silent wrong results

Every one of these produced a wrong-but-passing state in practice. Check them
against your design *before* writing code.

| # | Gotcha | What to do |
|---|---|---|
| 1 | **A float→int cast inside a graph does not truncate.** neuronx-cc elides convert pairs, so `(arange * scale).to(int64)` picks the wrong element. Invisible when the int tensor is a graph *output*, so a standalone cast test passes. | Emit an explicit `torch.floor` / `torch.trunc` **before** every cast. Grep candidate decompositions for `.to(torch.int` on a float value. |
| 2 | **`index_add` into the MINOR axis is pathological.** (8,32,64,256)→(8,32,64,64): `index_add(dim=3)` **2534 ms** vs **1.14 ms** with the axis moved. | `movedim(dim, 0).contiguous()` → `index_add(0, …)` → `movedim(0, dim)`. Applies to any scatter-shaped gradient. |
| 3 | **Narrow dtypes arrive WIDENED inside the kernel** — uint8/int8/int16/int64 → int32, float64 → float32 (`IsSupportedDtype`/`CastTarget`, `csrc/core/dispatch/DispatchUtils.cpp:136-147`), for **every** op. So `torch.iinfo(x.dtype)` gives ±2³¹ and a saturating clamp becomes a silent no-op; the framework's narrowing then *wraps* (−5 → 251). | Take bounds from the dtype the *op* accepts, never from the traced tensor. Same trap for float64. |
| 4 | **Map constants cannot come from Python lists.** A memoised CPU tensor *or* inline `torch.tensor` is lifted as `detach_(lift_fresh_copy(…))`; AOTAutograd rejects it ("expected to have an entirely functional graph"). Building it inside the body instead hits `Cannot copy out of meta tensor` in the fx importer. | Build maps from `torch.arange` + arithmetic on the op's device. Functional, works under eager tracing and AOTAutograd. |
| 5 | **A `neuron_decompositions` entry does NOT mean your decomposition runs.** PyTorch also publishes them as `py_impl(Autograd)` / `py_impl(CompositeImplicitAutograd)`, which resolve *above* `__torch_dispatch__` — the only place a `decompositions=` table is read. Your registration is dead code, silently. | `python -c "import torch; p=torch.ops.aten.<op>; [print(o, [str(k) for k in getattr(p,o).py_kernels]) for o in p.overloads()]"` — anything besides `Meta` is shadowing you. Pop those keys and clear `_dispatch_cache` (see `_release_upsample_builtin_python_decompositions`). **Pop, don't repoint**: the global table stays intact for other backends. |
| 6 | **Eager kernels return contiguous outputs unless you ask.** A channels_last input comes back contiguous, failing every `is_contiguous(memory_format=…)` assertion. | `register_aten(output_layout=fn)`; `fn` returns target strides, taken from **aten's own meta kernel** (not your traced output, which is contiguous; not the input's layout, which is wrong where aten itself returns contiguous). Doesn't survive a dtype bounce — float64/uint8 still land contiguous. |

Also: `make_fx` will not reveal #5 (no python dispatcher), and a NEFF cache hit
will not reveal any code change — **the cache key is (op, shapes, dtypes, static
args), not the graph.** When iterating, vary the shape or set
`TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache_<task>`.

---

## Step 0 — Start the baselines, then work while they run (do this first)

The device is single-tenant and it is the bottleneck; authoring is not. Start
the pristine-main baselines in the background **before** writing any code, so
every later failure is attributable without a second 20-minute run.

```bash
cd <repo-root>                          # your torch-neuronx checkout
git fetch origin main
git worktree add /tmp/main-baseline origin/main     # keeps your branch editable
cd /tmp/main-baseline
# the exact -k selection you will use later, else the baseline is not comparable
python -m pytest tests/pytorch_tests_native/test/test_nn.py -k "<op>" \
    -q -n 64 -p no:randomly --tb=no -rf > /tmp/<op>_main_baseline.txt 2>&1 &
```

Save the result into the repo dir (`.kiro/<suite>_main_baseline.txt`) so it
survives `/tmp` cleanup, and **delete it before committing**.

Then confirm the op isn't already handled:

```bash
./tools/list-implemented-ops | grep -i <op>
grep -rn "aten::<op>" torch_neuronx/python_ops/
git log --all --oneline -- '*<op>*'          # main may have a partial impl
grep -rn "<op>" third_party/torch-mlir/lib/Conversion/   # or a StableHLO pattern
```

If you inherited a branch/PR, sync it **now** — a stale base invalidates every
later result:

```bash
gh pr view <N> --json headRefName,baseRefName,state
git checkout <headRefName> && git merge origin/main -m "Merge origin/main into <branch>"
for f in <files you'll edit>; do echo "$(git rev-list --count HEAD..origin/main -- $f)  $f"; done
```

Base branch is **`main`** (a legacy `mainline` exists and is not the target).

## Step 1 — Extract the reference operator on CPU (no hardware, catches the most)

**First, probe autograd** (CPU-only, no device) so you know on day one whether
this op needs a backward — don't let it surface at review:

```bash
python .kiro/skills/aten-lowering-orchestrator/scripts/probe_op_autograd.py \
    --op aten::<op> --out agent_artifacts/<slug>/autograd_probe.json
```

Read the result: `backward_kind == "dedicated_op"` means eager backward
dispatches a **separate** ATen op (in `backward_ops`) that you must lower **in
this same PR** and prove `runs_on_neuron` — plan the lowering (Step 2) and its
`target_ops.py` row now, not later. `"derived"` means backward flows through
ordinary primitives + AOTAutograd (nothing extra to lower). `"none"` means the op
isn't differentiable. `"unknown"` (no OpInfo sample / exotic signature) means the
probe couldn't resolve it — resolve by hand from `tools/autograd/derivatives.yaml`
and record the answer, because the verifier treats unknown as a blocking gap.

Then extract the reference. Do not read aten's source and trust it. Any op that is *linear* in its input for
a fixed config — resampling, pooling, padding, and every one of their backwards —
is fully described by a matrix you can recover exactly:

```python
basis = torch.eye(n_in, dtype=torch.float32).reshape(n_in, 1, *in_spatial)
operator = fn(basis).reshape(n_in, -1).T          # (out, in), exact
```

Sweep `itertools.product` over sizes × scales × flags and diff your formula
against it. This answers, cheaply, three things that are otherwise guesswork:

- is the op **separable**? (compare against `torch.kron` of per-dim factors)
- is the backward the **adjoint** of the forward? (compare against `operator.T` —
  where it isn't, you have found an asymmetry you must reproduce)
- *which* configs disagree (sweep and print, don't bisect one shape at a time)

Use the input's **opmath dtype** for the basis (float32 for a float32 op) —
aten resolves indices in opmath, so a float64 basis makes it disagree with a
device that has no float64.

Commit the sweep as a CPU-only test that imports your real helpers. It needs no
hardware, so it runs everywhere, and it is where reference quirks stay pinned.
~3.6k cases run in 12 s and caught four aten asymmetries before any device run.

## Step 2 — Write the lowering

Lives in `torch_neuronx/python_ops/torch_mlir/ops/<family>.py` — the **existing**
file for the family (see that dir's `README.md`); a new file only for a genuinely
new family. Read the neighbouring functions and match them.

```python
@register_aten(
    ["aten::<op>", "aten::<op>.out"],           # functional + out share ONE decorator (see below)
    validate_static_inputs=_validate,           # dtype/shape guards, aten's wording
    # static_argnums=(...),                     # ONLY args that change the graph — see below
    # output_params=(...),                      # ONLY multi-output / @neuron_op ops — see below
    # output_layout=_layout,                    # only if the op must preserve layout
)
@register_aten(["aten::<op>_"], validate_static_inputs=_validate_inplace)  # in-place goes here
def torch_<op>(self, arg, out=None):
    return ...
```

- **Register every overload in the schema** — but measure which ones actually
  reach the backend first. For a structured op, `.default`, `.vec`, `.vec_out`
  and the `F.*` wrappers **all funnel into `.out`** (`.grad_input` for backwards),
  and `.vec` is CompositeImplicitAutograd so it decomposes above the backend key.
  Confirm with op tracking on unmodified main:
  ```python
  with track_neuron_ops():
      op(x.to("neuron"), ...)
      torch.neuron.synchronize()       # tracking is async; without this you get []
      print(get_executed_op_list())    # names the overload the backend saw
  ```
  Missing a genuinely-dispatched `.out` is costly: the fallback aliases `self` to
  `out`, so `out=` **silently corrupts `self`**.
- **`static_argnums` / `static_argnames` — mark an arg static ONLY if it changes
  the compute graph** (output shape, or a value the lowering bakes into structure /
  uses to specialize). It cuts *both* ways: too few → a reused NEFF returns wrong
  results; **too many → a separate NEFF per distinct value, i.e. runtime
  recompilation** every time a workload sweeps that arg. A plain numeric scalar
  that only feeds arithmetic is **not** static — pass it as a 0-d runtime tensor
  (like `mul.Scalar`, which does *not* mark its scalar static). The sharp contrast
  is `pow`: the **exponent** is static (`static_argnums=(1,)` — a constant exponent
  lets the compiler specialize the power), but `pow.Scalar`'s **base** was marked
  static by mistake and had to be removed to stop recompiling per base value
  (#1670). Ask "does this value change the graph, or just the numbers?" before
  adding it. `add`/`sub`/`div` mark their *mode* kwargs (`alpha`, `rounding_mode`)
  static for the same reason — those change the op, a scalar operand does not.
- **`out=` and `output_params` — a single `out` needs NEITHER on the C++
  `register_aten` path.** The dispatcher already aliases the caller's `out=` buffer
  and preserves its dtype natively, so `output_params=("out",)` is a no-op and a
  separate `_out` function is dead weight. **Fold functional + `.out` into one
  decorator** (see `addcmul`/`addcdiv`/`div`/`lerp`, and `masked_fill` after #1579
  review). `output_params` is load-bearing only for **multi-output** ops whose
  outputs aren't named `out` (`out0`/`out1`/`grad_input`/`save_mean`) and for ops
  on the `@neuron_op` / `python_dispatch` path. Take output dtype from `out` when
  absent; don't hardcode.
- **`output_params` / `static_argnums` / `static_argnames` are read from the FIRST
  registration in a function group** and applied to *every* overload under it
  (`operation_registry.py`, `finalize_registrations{,_cpp}` read `ops_list[0]`).
  Everything else — validator, `translate_args_fn`, preprocessing,
  `python_dispatch` — is per-overload, so stacking two decorators on one function
  is fine and is the norm. But two overloads that need *different* values for those
  three fields (e.g. a scalar at arg 0 vs arg 1, `pow.Scalar` vs `pow.Tensor_Scalar`)
  **must be separate functions**. Don't blindly copy a sibling's `static_argnums`.
- **Dtype guards must match aten exactly** — `test_ops.py::test_dtypes` compares
  against the OpInfo table. Probe the truth on CPU:
  ```python
  for dt in dtypes: try: aten.<op>(torch.ones(shape, dtype=dt), ...)
  ```
- **Don't call `.contiguous()` on a Neuron tensor from an op impl** — use
  `ensure_contiguous_on_device` (`cast_policy.py`). Inside a traced body it is
  fine (it becomes a graph op).
- **Shared helpers**: `grep -rn "<helper>" torch_neuronx/` before changing dtype
  or shape behaviour. `run_bitwise_shift_op` is also on the packed-int64 path.

Also lower the **compile path** if the op has a decomposition
(`neuron_dynamo_backend/decompositions.py`) — put the math in a leaf module both
paths import, so they cannot drift, and check gotcha #5.

Build: `./tools/build` (Bazel; seconds when warm).

**Lint gate — every commit must pass it.** The repo's `.pre-commit-config.yaml`
runs `ruff` + `ruff-format` (line-length 100, `select = E,F,I,N,W,UP,B,C4,SIM,RUF`),
`clang-format`, and whitespace/EOF/JSON fixers, and the same config runs in CI. A
git hook is NOT installed in a fresh clone/worktree, so **do it yourself**:
`pre-commit install` once, then before every commit
`pre-commit run --files <changed files>` (or `pre-commit run -a`) and fix until
green. If `pre-commit` isn't on PATH, `pip install --user pre-commit` (plus
`ruff==0.8.0` to match the pinned hook). Don't rely on `./tools/pre-commit` — it
can fail with `No module named pre_commit`. Expect the fixers to reformat and
fail once, then pass on a second `git add`.

## Step 3 — Unit tests (make these good enough that upstream passes first try)

`tests/python_ops/test_<op>.py`, plus
`tests/neuron_dynamo_backend/e2e/test_<family>_decomposition_e2e.py` if the op
has a decomposition.

Mandatory, from the acceptance criteria:

- **`assert_op_runs_on_neuron` in EVERY device test** — without it a test passes
  identically on CPU fallback, i.e. it tests nothing. It substring-matches, so
  the packet name covers the `.out` kernel. Derive it from the op object in
  parametrized tests. **For a differentiable op, this must also assert the
  *backward* op** (see the backward bullet below): the forward assert says
  nothing about where backward ran.
- **CPU as the reference** (`torch.testing.assert_close(got.cpu(), expected)`),
  never hand-written constants, so the test tracks upstream.
- **Bit-exact where the op only moves data** (gather/scatter/index): use
  `torch.equal`. Where a scatter accumulates, feed integer-valued payloads
  (`torch.randint(-8, 9, shape).float()`) so exact comparison stays legitimate.
- **`pytest.raises(..., match=...)`** — always `match=`, else an unrelated
  exception passes the test.
- **Cover this matrix** (each dimension is a parametrization axis; skip a cell
  only where the op genuinely can't take it, and say why):

  | Dimension | Values to cover |
  |---|---|
  | Shapes | scalar (0-d), 1D, 2D, 3D, 4D+; small (1–128), medium (1K–64K), large (1M+); non-square; **shapes from `SHAPE_REGISTRY`** |
  | Dtypes | float32, bfloat16, float16, int32, int64 (where applicable), bool |
  | Edge shapes | empty (numel=0), single-element, size-1 dims, odd shapes e.g. `(7,19)`, large sizes |
  | Numeric edges | inf, -inf, nan, 1e38, 1e-38, zeros, negative zeros |
  | Dtype casting | int64 (downcast→int32→upcast), float64 (downcast→float32→upcast) |
  | Mixed dtypes | e.g. float32 + bfloat16 where PyTorch promotes |
  | Broadcasting | different-but-broadcastable input shapes |
  | Variants | every registered overload: base, `.out`, `.grad_input`, `.Tensor`, `.Scalar`, in-place (`_`) — `out=` asserts the value written **and** `self` untouched, incl. non-contiguous `out=` |

  Also: `assert_op_runs_on_neuron` in every device test; autograd end-to-end; and
  channels_last if the op should preserve it. Pull registry shapes with
  `parametrize` from `SHAPE_REGISTRY`, don't hand-pick.
- **Differentiable op ⇒ lower and test the backward, on device.** Eager backward
  dispatches a **separate** ATen op (e.g. `aten::_weight_norm_interface_backward`)
  that CPU-falls-back silently unless it too is lowered — a forward-only
  `assert_op_runs_on_neuron` passes anyway. So: lower the backward op(s) in this
  same PR, add each to `target_ops.py` (its coverage row must be `runs_on_neuron`),
  and in the autograd test run `.backward()` and assert the **backward op's**
  device (`assert_op_runs_on_neuron(bwd_op, ...)`), then compare gradients to the
  CPU oracle. The `torch.compile` backward is auto-derived by AOTAutograd from the
  forward decomposition — it works once the forward compile path is `supported`
  and its decomposition uses lowered differentiable primitives; verify it doesn't
  graph-break. A backward that genuinely can't run on Neuron is an **allowlisted
  xfail** on its gradient test with a root-cause reason, never a silent fallback.
- **Read the upstream test you must satisfy before writing yours.** Mine used
  float32 while upstream used `torch.double` and also asserted the *gradient's*
  layout — so my tests passed and upstream failed. Copy upstream's dtypes,
  shapes and assertions.
- For a decomposition, **assert the routing**, not just numbers:
  ```python
  assert DispatchKey.Autograd not in op.py_kernels
  assert DispatchKey.CompositeImplicitAutograd not in op.py_kernels
  ```
  Numbers alone pass just as happily against PyTorch's shadowing built-ins.
- Distinct function objects per compile config + `torch._dynamo.reset()`, else
  `FailOnRecompileLimitHit`.

Prove a new guard isn't vacuous: reintroduce the bug and watch it fail.

## Step 4 — Run the upstream suites (the real gate)

**First discover which files actually contain your op's tests — by code, not from a
static list.** OpInfo names are generated at collection, so a static grep misses
them and a fixed file list runs suites that don't touch your op. Collect-only over
the whole tree (cheap: no execution), then boundary-match the op name and bucket by
file. The boundary regex is the same one the verifier uses (`_op_keyword_matches`),
so `pow` won't drag in `float_power` / `linalg_matrix_power`.

```bash
# ~ a few min for --collect-only; do it once, reuse the file set for every run below.
python -m pytest tests/pytorch_tests_native/test/ -k "<op>" \
    --collect-only -q --continue-on-collection-errors -p no:randomly 2>/dev/null \
  | grep -iE '(^|[^a-z0-9])<op>([^a-z0-9]|$)' \
  | awk -F'::' '{print $1}' | sort -u > /tmp/<op>_files.txt
cat /tmp/<op>_files.txt        # the exact files to run — usually 1-4, not 7
```

Then run **only those files**, filtered to your op:

```bash
CACHE=/tmp/cache_<op>; mkdir -p $CACHE/{hlo,neff,tmp}
NUM_WORKERS=64 PYTEST_ADDOPTS="-k '<op>' -p no:randomly" \
TORCH_NEURONX_HLO_CACHE_DIR=$CACHE/hlo TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE/neff TMPDIR=$CACHE/tmp \
  .claude/skills/agent-isolation/scripts/with-hw-lock -- \
  uv run python tests/pytorch_tests_native/runner/run_upstream_tests.py \
      --output-dir /tmp/reports/<op> --total-timeout 7200 \
      --pytorch-fork-branch release/pt_212 \   # MUST match installed torch — see note below
      $(sed 's#tests/pytorch_tests_native/test/##' /tmp/<op>_files.txt)   # add --runxfail to PYTEST_ADDOPTS for the runxfail leg
```

The runner clones the upstream tests on first use, and its `--pytorch-fork-branch`
**defaults to `main`, which is wrong for torch 2.12** (the repo default is intentionally
unchanged — this is handled here in the runbook, not in repo code). **Always pass the
branch matching the installed torch**: derive it as `release/pt_{major}{minor}` — get the
version without importing torch via
`python -c "from importlib.metadata import version; print(version('torch'))"` (e.g.
`2.12.1+cpu` → `release/pt_212`). The `main` fork's `torch/testing/_internal` overlay
references symbols absent from torch 2.12 (e.g. `torch.bcomplex32`), and conftest's
overlay keeps **no backup**, so cloning the wrong branch breaks `import torch`
*host-wide* until hand-restored. If `test/` already exists (baked into the Docker/CI
image) the clone is skipped entirely and the flag is moot — this bites only fresh
dev-pod clones.

`--continue-on-collection-errors` absorbs the ~5 pre-existing `import file mismatch`
collection errors (see below); they are not yours. If `/tmp/<op>_files.txt` is empty,
the op is untested by the source of truth — widen `-k` or confirm coverage before
proceeding (the verifier FAILs on "no upstream tests matched this op").

Reference — what each file *typically* covers (informational; the discovery step
above is authoritative for which ones actually apply to your op):

| File | Covers |
|---|---|
| `test_ops.py` | OpInfo: dtypes, `out=`, variant consistency, noncontiguous |
| `test_ops_gradients.py` | gradcheck / gradgradcheck (an acceptance criterion) |
| `test_ops_fwd_gradients.py` | forward-mode AD |
| `test_nn.py` | the behavioural matrix — memory formats, dtype consistency |
| `test_decomp.py` | decomposition vs op |
| `test_meta.py` | meta / fake-tensor shapes |
| `tests/vertical/test_target_ops_{,compile_}coverage.py` | on-device coverage + perf |

Use `runner/run_upstream_tests.py` (what CI runs), not `tools/run-test-parallel`
(it `--ignore`s this tree) or bare `pytest -n 64` (missing the `env_lock.so` shim →
"Failed to initialize Neuron Runtime" at scale). The runner writes JSON to
`--output-dir` and retries failures at 4 — don't pass `--json-report-file`,
`--download-neff-cache`, or `--refresh-neff-cache` (the cache flags replay baseline
NEFFs and mask a lowering change). A failure list that differs between runs *in both
directions* is flake, not regression.

**Which command / worker count for each tier** — `-n 64` (`NUM_WORKERS=64`) is the
upstream runner *only*, because only that path preloads the `env_lock.so` shim:

| Tier | Command | Workers |
|---|---|---|
| Unit (`tests/python_ops/…`, e2e) | bare `pytest … -n <N>` under `with-hw-lock` | `-n 16` (bare pytest has no shim; don't push to 64) |
| Vertical (`tests/vertical/…`) | `pytest` **directly** (not `run-test-parallel` — it `--ignore`s the tree) under `with-hw-lock` | `-n 16` |
| Upstream (`tests/pytorch_tests_native/…`) | `run_upstream_tests.py` (above) | `NUM_WORKERS=64`, retry-at-4 |

Efficiency rules that avoid re-runs:

- **One shared cache dir for every run.** Per-run dirs mean every suite
  recompiles: same selection was 20:17 cold vs 10:50 warm.
- **Quote `-k` and keep it on ONE line.** A wrapped `-k` silently drops the tail;
  the header must say `N/12429 tests collected (M deselected)`.
- **Don't run the full `test_indexing.py` locally** — a ~8.6 GB compile that
  `--timeout` can't interrupt.
- **float64/double variants compile pathologically slowly** (no native fp64;
  emulation) — a `*_float64` / `*_double` test (especially a *backward*, or the
  dtype-sweeping `test_dtypes`) hanging for tens of minutes is the **known
  platform limit** (`float64|double …(precision|not)` on the allowlist), not a
  flake. Recognize it immediately: **one re-run max, never escalate `--timeout`**;
  xfail the fp64 variants with the allowlisted reason and move on. When you remove
  a *blanket* "unimplemented" skip to enable an op, re-add a narrow fp64
  platform-limit xfail for its float64 variants in the same change or they unmask
  and hang.
- Collecting the whole `tests/pytorch_tests_native/test/` tree reports ~5
  pre-existing `import file mismatch` collection errors. Not yours.
- `-rX` lists xpasses; `-rs` lists skip reasons. You need both.
- `tools/run-test-parallel` **cannot run `tests/vertical/`** (it passes
  `--ignore=tests/vertical` and rewrites `-k`); call pytest directly there, and
  delete the `target_ops_*.csv` it writes into the repo root.

## Step 5 — Triage every failure

**Default to fixing.** Categorize only for a genuine platform limit:

| Genuinely categorize | Fix instead |
|---|---|
| complex dtypes, sparse | wrong values / wrong layout |
| float64 precision (silently downcast) | crashes, arity errors, dispatch failures |
| FMA / ~1 ULP vs an exact-endpoint reference | anything reproducing for **integer** dtypes |

**Strongest signal: does it reproduce for pure integer dtypes?** If yes it is a
real bug, not precision.

Before believing a failure is yours, **check the baseline from Step 0**. Then:

- **"Unexpected success" is not a failure** — it means an *upstream*
  `@unittest.expectedFailure` / OpInfo `xfail` no longer holds because your
  lowering is better. You cannot drop an upstream decorator; add the nodeid to
  `ATEN_UPSTREAM_STRICT_XFAIL_PASSES_ON_NEURON`.
- **Passes alone but fails under `-n`** (or vice versa) is order-dependent flake.
  Confirm 3× in isolation before calling it pre-existing.
- **A "compiler bug" needs more than reproducibility.** A single-shape wrong
  result survived a full rewrite of my graph and still turned out to be fixed by
  changing the op's rounding. Try changing your own emitted pattern before
  blaming neuronx-cc.
- **Dump the StableHLO before theorising** about miscompiles — there is no env
  var; wrap the converter (patch the *source* module, `kernel.py` imports it
  inside the function) and use a fresh shape or the cache hides it:
  ```python
  import torch_neuronx.neuron_dynamo_backend.fx.fx_transform as fx
  orig = fx.convert_fx_to_stablehlo
  fx.convert_fx_to_stablehlo = lambda *a, **k: (lambda r: (print(str(r[0])), r)[1])(orig(*a, **k))
  ```
  One dump showed `stablehlo.constant dense<2.14748365E+9>` — instantly revealing
  gotcha #3, after five wrong guesses.

### Registry contract

Single source of truth: `tests/pytorch_tests_native/skip_registry/`. Never flip
an inline `@pytest.mark.xfail`; never edit files under
`tests/pytorch_tests_native/test/`.

- **skip** = crashes/hangs/missing feature. **xfail** = runs, wrong output.
- Group by **root cause**, not by op name, with a reason naming the real cause.
- **Adding a skip/xfail for the op you just implemented is not acceptable**
  unless it's a true platform limit, spelled out.
- **Delete entries your work fixed** — verify 3× first, and attribute against
  pristine main (a stale entry may have been passing before you started).
- Verify the marks catch what you think: `-q` should xfail N, `--runxfail` should
  fail the same N. Equal counts means nothing is masking a pass.
- **The `--runxfail` leg is conditional — scope it, and skip it when nothing can
  be masked.** It exists only to catch a candidate that xfailed a genuinely-failing
  test, so run it only over *your op's* nodeids (`--runxfail -k "<op>"` on the
  discovered files), never the whole suite. **Skip even that** when all three hold:
  (a) the registry diff for this op is empty (you added no skip/xfail), (b) you
  touched no file under `tests/pytorch_tests_native/test/` (you never should), and
  (c) every op-relevant test passed in the plain run above. With no new xfail there
  is nothing for `--runxfail` to unmask — the plain green run already proves it. The
  full-suite `--runxfail` regression stays a CI backstop, not a local per-op gate.
- **Check whether the patterns you removed matched anything.** Dead patterns are
  common (`test_ops` names OpInfo entries `nn_functional_*`, so aten-spelled
  wildcards match nothing) — say so, or a reviewer hunts for tests that never
  existed.

## Step 6 — Perf and coverage

- Add every new overload to `tests/vertical/internal/target_ops.py` **and** a
  shape to the registry. Without both, the op silently gets no row — which is how
  a 2-second backward regression went unnoticed.
- Sanity-check steady-state time and cold compile per op. Forwards of a
  20 MiB gather-shaped op land at 0.3–1.0 ms; a backward in the seconds means
  gotcha #2.
- The vertical `roofline_ratio` column is meaningless without a handler for your
  family in `internal/roofline.py`.

### Optimization pass (advisory — catch a slow lowering here, not in a later rewrite)

A lowering that only passes correctness can still be slow; the point of this pass
is to surface that **now** rather than after a profile forces a rewrite. Reuse
`dynamic-kernel-optimize`'s profiling method and tips (not a second loop, and not
its NKI-kernel authoring — see step 2):

1. **Profile + triage.** From the vertical profile, read the *bottleneck engine's
   occupancy* and *achieved-vs-ideal traffic* (bytes actually moved / (N_in+N_out)·
   bytes). If the busy engine saturates near-ideal traffic, the lowering is at its
   ceiling — stop, record the numbers, move on. Engage the loop only on a headroom
   signal: bottleneck engine idle, achieved traffic ≫ ideal (redundant passes/
   copies), or steady-state well outside the sane band for the shape. (Occupancy
   alone is not enough — a lowering can be 100% DMA-busy moving 4× the necessary
   bytes.) **Prioritize the medium (1K–64K) and large (1M+) shapes** — those are
   compute/bandwidth-bound and where a lowering change actually shows up. Small
   shapes with `N_out < 64K` are exempt — launch/DMA overhead dominates and no
   lowering change touches it.
2. **Try lowering-level fixes only — no NKI authoring.** Any optimization stays at
   the torch-mlir lowering level: eliminate redundant copies/casts, fix output
   layout/stride, pick a cheaper decomposition, or adjust broadcast/tiling
   expressible in the lowering — one change → `uv run ./tools/build` → re-measure →
   **accept on a real speedup over the current best, revert on regression** → log.
   Drive accept/revert on *relative* improvement (this iteration vs the last), not
   an absolute target. **Cap at 3 cycles.** Reject any change that isn't a real
   speedup. Gate each accepted change on the *fast* correctness signal — the op's
   unit tests + vertical correctness — **not** the full upstream suite (too slow to
   run per cycle); then run the full upstream suite **once** after the last accepted
   change to confirm it's still green. Use `dynamic-kernel-optimize`'s **profiling
   method** and the
   lowering-applicable subset of its `optimization_tips.md` as reference only — **do
   not write or edit NKI kernels**, including for `dynamic_kernels/` ops. If the
   profile shows the only remaining win needs a custom kernel, stop and flag it for a
   human in the PR body (advisory); the agent never authors one.
3. **Record, don't gate.** Perf never blocks the verdict (see the roofline note
   above). Put before/after steady-state + the profiler signal (bottleneck engine,
   occupancy, achieved-vs-ideal traffic) in the PR body for the reviewer.

## Step 7 — Docs, review, PR

```bash
git fetch origin main && git rev-list --left-right --count origin/main...HEAD
```

Merge and **re-verify on the merged base** if `origin/main` touched your files.

- `docs/torch_api_compatibility.md` — flip `CPU_FALLBACK` → `Y` with a
  one-phrase limitation for what still falls back. Measure it with
  `track_neuron_ops`, don't guess.
- `docs/environment_variables.md` if you added an env var.
- Run **`/pr-review` in branch mode** and act on it before asking a human.
- PR from `.github/PULL_REQUEST_TEMPLATE.md`, base **`main`**, draft first, and
  **label it `torch-neuronx-agent`** (`gh pr create --label torch-neuronx-agent …`,
  or `gh pr edit <N> --add-label torch-neuronx-agent` if it opened without one).
  Report results as a table with counts; name pre-existing failures with the
  evidence they reproduce on a clean base; explain a counter-intuitive registry
  diff (adds and removes net to zero in the GitHub diff); recount test totals
  after the last edit.
- **Commit messages follow conventional-commits**, matching this repo's history:
  `feat:` / `fix:` / `docs:` / `chore:` / `test:` / `refactor:`, optional scope
  (`feat(dispatch): …`). One logical change per commit.
- **No `Co-Authored-By` trailer. Never reply to PR review comments without
  asking a human first.**
- Keep `.kiro/` and `torch_neuronx/dependency_commits.json` uncommitted; never
  `git add -A` (it picks them up). Delete `target_ops_*.csv` and any baseline
  file you parked in the repo.
- Push the branch early and often — a lost session means redoing everything.

### CI expectations

Upstream tests run **only in the merge queue** (`merge_group`); the
`Merge Queue Gate Status` on the PR is a deliberate no-op. Add
`ciflow:full-suite` when you touch shared dispatch code (verify with
`gh label list --search ciflow`). A red job showing `Scan timed out after 1800s`
with an empty `test-results/` is the ECR image-scan gate, not your code.

---

## Acceptance checklist

When this op is lowered in a batch, the **objective** items below (1–3, 5, 7,
and 4 for a differentiable op) are graded by
`.kiro/skills/aten-lowering-orchestrator/scripts/verify_op_acceptance.py`
from artifacts you produce — a script-computed verdict, not a self-assessment.
Produce the artifacts it expects (upstream pytest-json with **and** without
`--runxfail`, a registry snapshot, the eager **and** compile vertical coverage
CSVs, the unit-test rc) so the verifier can confirm you didn't silence a test.
Item 6 (both execution paths) is gated too: the verifier grades the compile-path
CSV with its own status vocabulary (`supported`, not `runs_on_neuron`), and if
you added a decomposition it must be run under `--require-compile`. Items 4,
8–12 are your own discipline; the reviewer checks them.

1. Unit tests pass **and** assert the op runs on Neuron in every device test.
2. All upstream tests for the op pass, or are registry-categorized with a correct
   root cause — verified under `--runxfail`.
3. No new skip/xfail for the op you implemented (true platform limits excepted,
   reason spelled out). Entries your work fixed are deleted.
4. **Differentiable op: backward is lowered and runs on Neuron.** Gradcheck
   passes (or the failure is shown to be within fp32 noise), the separate backward
   ATen op(s) dispatch on Neuron (asserted in the autograd test, `runs_on_neuron`
   in their coverage rows), and the compile backward doesn't graph-break. Gated by
   the verifier's `backward` check when the orchestrator runs `--differentiable
   --backward-op aten::<bwd>`; a backward that can't run on Neuron is an
   allowlisted xfail, never a silent CPU fallback.
5. Every overload in the schema is registered; `out=` tested for the written
   value **and** `self` left untouched.
6. Both execution paths tested — eager (`@register_aten`) and `torch.compile`
   (decompositions + StableHLO) lower through different machinery, and for a
   decomposition the *routing* is asserted, not just the numbers.
7. Op present in `target_ops.py` **with** a shape, so vertical reports it.
8. Perf sane: steady-state and cold compile measured; no seconds-long kernel.
9. `./tools/run-cpp-test` green if you touched `csrc/`.
10. Shared-code changes are **bounded and exercised**: enumerate which registered
    ops change behaviour, and delete any branch a test doesn't reach.
11. Every regression test was proven to fail without the fix.
12. Verification re-run on the **post-merge** base; PR counts match the final run.
