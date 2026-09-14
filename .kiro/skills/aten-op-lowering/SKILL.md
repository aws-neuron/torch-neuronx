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

**Under the orchestrator — durable handoff, not live memory.** When the
orchestrator spawns you, your context must live in **artifacts**, because a fix
round respawns a fresh Implementer onto your worktree, not the live you. So: work
in the worktree the orchestrator gives you and never destroy it; keep
`agent_artifacts/<slug>/handoff.json` current (chosen decomposition, overloads
registered, what you tried and rejected, current failing `checks[]`, next-step
hypothesis); and treat `completion.json` as a **contract** — write `state`,
`round`, `ready_for_verify` (false until every artifact is final), and refresh the
`heartbeat` timestamp every step. Writing `completion.json` with `ready_for_verify:
true` is your final action of a round and the ONLY signal that you are ready to be
graded. Do not open the PR; the orchestrator gates it on the verifier.

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

## Step 0 — Confirm the op isn't already handled (do this first)

Before you write code, confirm the op is not already lowered on `main`:

```bash
./tools/list-implemented-ops | grep -i <op>
grep -rn "aten::<op>" torch_neuronx/python_ops/
git log --all --oneline -- '*<op>*'          # main may have a partial impl
grep -rn "<op>" third_party/torch-mlir/lib/Conversion/   # or a StableHLO pattern
```

If you inherited a branch/PR, sync it **now** — a stale base invalidates every
later result. The merge also refreshes the policy files (`AGENTS.md`, `.kiro/`,
`lessons.md`); a branch can hold a stale copy. Read the policy from the merged
base, not from memory. Do not overwrite an intentional `lessons.md` edit; only
confirm the branch is not behind `origin/main` on these files:

```bash
gh pr view <N> --json headRefName,baseRefName,state
git checkout <headRefName> && git merge origin/main -m "Merge origin/main into <branch>"
git diff --stat origin/main -- AGENTS.md .kiro   # expect only your intended edits, nothing stale
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
    return torch.ops.aten.<op>(self, arg)   # call the aten op, do NOT inline the ref
```

- **In the body, call the aten op (`torch.ops.aten.<op>`), not an inline ref.**
  `make_fx` in `kernel.py` then reuses the registered decomposition, so one ref
  serves the functional op, the `.out` overload, and the compile path. Inlining
  the ref (e.g. `return <op>_ref(a, b)`) duplicates the math and drifts. See
  `ldexp`/`rsub` in `binary.py`. In a `register_decomposition([aten.<op>.out])`,
  do **not** write `out.copy_(<op>_ref(...))` — the dispatcher handles `out=`
  outside the lowering, so the functional and `.out` variants share ONE decomposition.
- **Register on the C++ dispatch path. Do NOT set `python_dispatch=True`.** A plain
  `@register_aten` uses the cached C++ dispatch — keep it. `python_dispatch=True`
  runs preprocessing AND execution in Python and holds the GIL for both. It hurts
  perf. The flag is deprecated and goes away. Do not add a new op to it. A reviewer
  rejects any new op that uses `python_dispatch=True`. If the op needs
  preprocessing that reads a tensor value (an output size or a static arg),
  set `relower_every_call=True` with `preprocessing=` (PR #1065): the preprocessing
  and the lowering run in Python, execution stays in C++, and the HLO/NEFF cache
  still applies (keyed on the preprocessed args). See `one_hot` / `histc`.
- **A CompositeImplicitAutograd op needs no backend registration** — it decomposes
  to lowered primitives and runs on Neuron with no CPU fallback. Register it only
  to attach a custom backward, and then wrap it in ONE `torch.autograd.Function`
  like `_CrossEntropyAutograd` in `loss.py`. Do not stack a `@register_aten` AND a
  separate `autograd.Function` for the same op (the `selu` / `selu_` double-register
  was a reviewer rejection).
- **Do not say "falls back to CPU" for fp64.** Neuron has no native fp64, so a
  double input is **downcast to fp32, computed, then cast back** — it does not fall
  back to CPU. Use that wording in docstrings, comments, and reviewer replies.
- **Comment a corner-case workaround so it is removable.** An extra op added only
  to fix `inf`/`nan` (e.g. a `>=` mask) needs a one-line "why" comment, so a future
  reader can delete it when the primitive is fixed. Drop dead casts: `.float()` on
  an already-fp32 tensor is a no-op, so remove a `x.float() if up else x` guard when
  both operands are always fp32.

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

Add the tests to the **existing family file**, not a new per-op file. A unary or
activation op goes in `tests/python_ops/test_unary_elementwise_ops.py` (add a row
to `POINTWISE_OPS` / `UNARY_OPS_LIST`); a binary op goes in its family file
(`test_sub`, `test_ldexp`, ...). `grep -rn "aten::<sibling>" tests/python_ops/` to
find the family file first. Make a new `test_<op>.py` ONLY for a genuinely new
family. A per-op file for an op that fits a family is a repeat reviewer rejection.
Also add `tests/neuron_dynamo_backend/e2e/test_<family>_decomposition_e2e.py` if the
op has a decomposition. This file goes in the dynamo backend tree, not
`tests/python_ops/`, because it tests the compile path, not the eager handler.
Assert the routing (both overloads in `neuron_decompositions`), then run the
decomposition under `torch.compile` for the forward and the backward. Run it with
your unit tests (bare `pytest` under `agent-run`); do not put an eager op-unit test
in this file.

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

> **Who runs this.** Standalone (a human invoked this skill), you run this step.
> **Under the orchestrator you do NOT run the upstream suite.** It is
> single-tenant, so the orchestrator runs one central lane after every
> Implementer releases its cores. You produce only the collection artifacts
> (`op_files.txt`, `op_nodeids.txt`) so the lane knows what to run. See the
> orchestrator SKILL "Central upstream lane".

**Set up the upstream tests once, at the branch that matches the installed torch.**
The runner clones the tests on first use, but its `--pytorch-fork-branch` defaults to
`main`. The `main` overlay of `torch/testing/_internal` uses symbols that a release
torch does not have (for example `torch.bcomplex32`), and conftest keeps **no backup**
of the overlay. A clone at the wrong branch breaks `import torch` *host-wide* until you
restore it by hand. To avoid this, clone the fork yourself at the matching branch and
populate `test/` before any run. Then the runner finds `test/` and skips its own clone,
so the branch flag is moot and the trap cannot fire.

```bash
# Derive the branch from the installed torch: release/pt_{major}{minor}.
# importlib.metadata reads the version WITHOUT importing torch.
FORK_BRANCH="release/pt_$(python -c 'from importlib.metadata import version; v=version("torch").split("+")[0].split("."); print(v[0]+v[1])')"
echo "$FORK_BRANCH"      # e.g. torch 2.13.0+cpu -> release/pt_213
git clone https://github.com/amazon-contributing/upstream-to-pytorch.git /workspace/upstream-to-pytorch
git -C /workspace/upstream-to-pytorch checkout "$FORK_BRANCH"
python tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo /workspace/upstream-to-pytorch
```

If `test/` is already present (baked into the image, or from a prior setup) the runner
skips the clone and you can go straight to discovery. On a fresh dev pod without the
one-time setup, pass `--pytorch-fork-branch "$FORK_BRANCH"` to the runner so it clones
the correct branch.

**Next, discover which files actually contain your op's tests.** Never collect the
whole `test/` tree. A whole-tree `--collect-only` is NOT cheap: pytest collects all
~183 files before `-k` filters, and the OpInfo files (`test_ops`, `test_decomp`,
`test_meta`) plus the ufunc files parametrize across the entire OpInfo database x
dtypes at collection time. One file alone (`test_unary_ufuncs.py`) takes ~77s and
builds 22k items to keep a few hundred. The whole tree is 20+ minutes and often does
not finish. Measured on the pilot host.

Scope the candidate files first, then collect only those:

1. The OpInfo-driven files always apply (the op comes from `@ops(op_db)`, not a
   literal name, so grep cannot find it there): `test_ops.py test_decomp.py
   test_meta.py`.
2. The ufunc file carries the op name in source, so a filesystem grep finds it
   fast (no import): `grep -rlwE '<op>' tests/pytorch_tests_native/test/*.py`.
   Keep `test_unary_ufuncs.py` (unary) or `test_binary_ufuncs.py` (binary). Do
   NOT add `test_torch.py`: it is huge, slow to collect, and its op tests are
   generic, not the OpInfo gate.

Collect the nodeids per file, in SEQUENCE, each under a per-file `timeout`. Do
NOT run a pytest process per file in parallel over the same venv: the
`conftest.py` overlay (`_overlay_torch_internal`) does a destructive swap of
`site-packages/torch/testing/_internal`, so two masters race and fail with
`ImportError` / `INTERNALERROR`. A per-file cap keeps the fast files even if one
file is slow. Separate ops have separate venvs, so the ops MAY run in parallel.
Filter to real nodeids (`^<test-dir>/...::`) — pytest prints warnings on stdout
that otherwise pollute `op_files.txt` with non-test paths. The combined 4-file
collect is ~10 min; measured on the pilot host, `test_ops.py` is the long pole.
Each per-file collect (`test_ops` / `test_meta` ~3.5-4 min) exceeds a 2-min
foreground timeout, so run this loop in the background (the long-running rule).

```bash
T=tests/pytorch_tests_native/test
: > /tmp/<op>_nodeids.txt
for f in test_ops.py test_decomp.py test_meta.py test_unary_ufuncs.py; do    # unary example
  timeout 420 python -m pytest "$T/$f" -k "<op>" --collect-only -q \
      -p no:randomly --continue-on-collection-errors 2>/dev/null \
    | grep -E "^$T/.*::" | grep -iE '(^|[^a-z0-9])<op>([^a-z0-9]|$)' \
    >> /tmp/<op>_nodeids.txt
done
sort -u -o /tmp/<op>_nodeids.txt /tmp/<op>_nodeids.txt
awk -F'::' '{print $1}' /tmp/<op>_nodeids.txt | sort -u > /tmp/<op>_files.txt
```

The older single combined invocation also works, but it loses everything if it
hits the cap:

```bash
FILES="$T/test_ops.py $T/test_decomp.py $T/test_meta.py $T/test_unary_ufuncs.py"
timeout 600 python -m pytest $FILES -k "<op>" --collect-only -q \
    -p no:randomly --continue-on-collection-errors 2>/tmp/<op>_collect.err \
  | grep -E "^$T/.*::" | grep -iE '(^|[^a-z0-9])<op>([^a-z0-9]|$)' > /tmp/<op>_nodeids.txt
awk -F'::' '{print $1}' /tmp/<op>_nodeids.txt | sort -u > /tmp/<op>_files.txt
```

The boundary regex is the same one the verifier uses (`_op_keyword_matches`), so
`pow` does not drag in `float_power` / `linalg_matrix_power`.

`/tmp/<op>_nodeids.txt` is the `op_nodeids.txt` artifact. Collect ONCE and reuse it
for every run below (Step 4 recipe) — do not re-collect per run.

Then run **only those files** (usually 1-4, e.g. `test_ops` /
`test_binary_ufuncs` / `test_unary_ufuncs` for a pointwise op), driven by the
**explicit collected nodeids** — not `-k` re-scanning all ~9 files (that collects
2000+ irrelevant items every run). The nodeids are already the `op_nodeids.txt`
artifact from the collect-only step; feed that list so pytest runs exactly your
op's tests:

The upstream runner is **single-tenant**: it self-parallelizes with xdist, and
its conftest overwrites `NEURON_RT_VISIBLE_CORES` from each worker global index,
so it always claims cores `0..N-1` and **ignores any `agent-run` slice**. It owns
the whole host. So run it only when the host is idle, gate on a device-free
check, and never run two instances at once. `with-hw-lock` serializes runner
instances, but a *sliced* unit test that holds a core still collides — the
device-free gate is the real guard.

```bash
CACHE=/tmp/cache_<op>; mkdir -p $CACHE/{hlo,neff,tmp}
# Gate: confirm no process holds a Neuron device. The runner needs the whole host.
python3 .kiro/skills/aten-lowering-orchestrator/scripts/neuron_devices_free.py || {
  echo "devices busy — wait, then retry"; exit 1; }
# The runner launches a BARE `pytest` resolved via PATH. Under the orchestrator
# (a worktree) put the overlaid venv first (the shared /opt venv is not overlaid:
# cannot import get_unsupported_dtypes_for_device). In a worktree: WT=$PWD.
# Standalone: skip this export -- the active /opt venv already resolves
# torch_neuronx to this checkout with get_unsupported_dtypes_for_device importable.
export VIRTUAL_ENV=$WT/.venv PATH=$WT/.venv/bin:$PATH
# Reuse the collected nodeids — one line each; run_upstream_tests.py forwards them.
NUM_WORKERS=64 PYTEST_ADDOPTS="-p no:randomly" \
TORCH_NEURONX_HLO_CACHE_DIR=$CACHE/hlo TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE/neff TMPDIR=$CACHE/tmp \
  .kiro/skills/agent-isolation/scripts/with-hw-lock -- \
  python tests/pytorch_tests_native/runner/run_upstream_tests.py \
      --output-dir /tmp/reports/<op> --total-timeout 7200 \
      $(sed 's#tests/pytorch_tests_native/test/##' /tmp/<op>_nodeids.txt)   # add --runxfail to PYTEST_ADDOPTS for the runxfail leg
      # test/ is already populated (setup above), so no --pytorch-fork-branch is
      # needed. On a fresh pod without setup, add: --pytorch-fork-branch "$FORK_BRANCH"
```

If the runner can't take raw nodeids, fall back to the discovered file list
(`/tmp/<op>_files.txt`) with `-k '<op>'` in `PYTEST_ADDOPTS` — still only those
files, never the whole tree.

**Global-invariant leg.** If your change registers a decomposition or edits
dispatch state (`py_kernels` pop/repoint, gotcha #5), also run the whole-registry
snapshot tests unfiltered: `python -m pytest
tests/pytorch_tests_native/test/test_decomp.py::HasDecompTest`. A `-k <op>` filter
deselects them, but the merge queue runs them. They are CPU-only (no NEFF).

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
| Unit (`tests/python_ops/…`, e2e) | bare `pytest … -n <N>` under `agent-run` | `-n` = slice size at most (see below); bare pytest has no shim, so don't push to 64 |
| Vertical (`tests/vertical/…`) | `pytest` **directly** (not `run-test-parallel` — it `--ignore`s the tree) under `agent-run` | `-n` = slice size at most |
| Upstream (`tests/pytorch_tests_native/…`) | `run_upstream_tests.py` (above) | `NUM_WORKERS=64`, retry-at-4. Single-tenant: whole host, device-gated, never two at once, no `agent-run` slice |

**Set `-n` to the core-slice size, not higher.** On the bare-pytest lanes,
`agent-run` loads the `agent_core_isolation` plugin, which pins each xdist worker
to ONE distinct core in the slice. So a slice of size S (from `agent-run BASE S`)
runs `-n S` cleanly. `-n` above S gives the extra workers no free core, and they
fail runtime init with "cores busy". `NEURON_RT_VISIBLE_CORES` indexes LOGICAL
cores, so on an LNC=2 host a slice `8-15` is 8 logical cores (not 4).

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

**A legitimate but unlisted xfail reason: cite a precedent, do NOT stop to ask.**
If the reason is a genuine limitation whose pattern is not on the allowlist, do
not block and do not widen `platform-limitations.md`. Add the xfail and end its
reason with `precedent: <an existing group reason or nodeid>` that already xfails
the same class (grep the registry for the sibling). The verifier confirms the
precedent is real and returns a **REVIEW** verdict, which still opens the draft
PR; a human accepts or rejects the reason at review. A reason with no real
precedent, or one matching a deny pattern (wrong fp32/int value, crash, layout,
CPU fallback), stays a FAIL — fix the op.

Before you decide a failure is yours, confirm it is not pre-existing. Re-run only
the failing nodeids on a pristine `origin/main` checkout, or check the
orchestrator's list of known pre-existing failures. Then:

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
- **Do not add ops to mask a compiler `inf`/`nan` bug.** If Neuron gives a wrong
  `inf`/`nan` where the CPU oracle is right and your math is sound, do NOT hide it
  with a `torch.where`/mask. xfail the affected case with a compiler-bug reason,
  note the compiler-ticket follow-up in the PR description, and package a standalone
  repro with the `compiler-artifact-repro` skill (see the hardtanh/hardshrink/relu6
  ticket follow-ups in #1810). Invoke `/compiler-artifact-repro`; it gates the ticket
  and produces the three required deliverables: the `ticket.md` description, a minimal
  standalone `repro.py`, and the offending HLO dump. Confirm the miscompile with an
  on-device divergence probe against the CPU oracle first; a wrong-looking torch-mlir
  decomp is not a bug if a native lowering shadows it (leaky_relu_backward). Then drop
  the op from the PR. But FIRST rule out reference-matching: if CPU `torch.<op>`
  itself special-cases the value (e.g. `logaddexp2` returns `inf` for `inf`+`inf`,
  which the log-sum-exp formula gives as `nan`), then a small correcting op matches
  the reference and is correct. Keep it, and comment it as reference-matching, not a
  removable workaround.
- **Complex is a platform limit: raise on it, and keep a raise-only reference call.**
  Do not half-support complex. An upstream test can assert the op's EXACT error for a
  complex input with a real dtype (`assertRaisesRegex(..., r"torch.linspace\(\): inferred
  dtype")`, #789). A `torch.<op>(...)` call that only raises (no return) reproduces that
  exact message by reusing torch's own dtype inference; it is required, not dead code.
  Skip only the test that wants a complex OUTPUT; never skip a whole real test for it.
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
- **Run both vertical suites for your op and read the rows.** Adding the op to
  `target_ops.py` does not run it — run both suites and confirm a green row for
  the eager path AND, when the op has a decomposition, the compile path:

  ```
  # Under agent-run; -n = slice size. Filter with -k, NOT --ops.
  pytest tests/vertical/test_target_ops_coverage.py \
      tests/vertical/test_target_ops_compile_coverage.py \
      -k "<op>" -n <slice> -q
  ```

  Read the CSV rows the run writes (`target_ops_coverage_<ts>.csv`,
  `target_ops_compile_coverage_<ts>.csv`): the eager row needs
  `status == runs_on_neuron` and `correctness_test == pass`; the compile row
  needs `status == supported` and `correctness_test == pass`. A missing compile
  row means the decomposition did not register — go back to Step 3.
  - **`--ops` filters only the EAGER suite.** The `--ops` option sets
    `NEURON_OPS_FILTER`, which only `test_target_ops_coverage.py` reads;
    `test_target_ops_compile_coverage.py` ignores it and sweeps every op (~680,
    slow). Use `-k "<op>"` to trim BOTH suites. Match the op name, not a
    superstring — for `logaddexp` use `-k "logaddexp and not logaddexp2"`.
  - A `Logical Neuron Core(s) not available ... (cores busy, ret=-16)` failure at
    Neuron init is a transient worker core-availability race, not a defect.
    Re-run the failed items with a smaller `-n` (for example `-n 2`).
  - Delete any `target_ops_*.csv` a stray run writes into the repo root.
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

**Before you commit, push, or open the PR: re-read
`.kiro/skills/aten-lowering-orchestrator/references/lessons.md` AND `AGENTS.md`,
then confirm you follow every applicable rule and lesson.** The lessons are prior
reviewer findings; `AGENTS.md` is the always-on repo policy (STE, no autonomous
GitHub action, draft PR only, disclose AI authorship). A clean pass here stops a
known comment from coming back on this PR.

- `docs/torch_api_compatibility.md`: flip `CPU_FALLBACK` to `Y`. For the note,
  match the repo convention: write `Native MLIR impl` for a native
  `@register_aten` lowering (the dominant phrasing, 200+ rows), or
  `Decomposes to <op>` when the op only decomposes to supported ops. Do not
  invent free-form text, for example "lowers to" or "passthrough registration".
  Add a one-phrase limitation for what still falls back. Measure it with
  `track_neuron_ops`, don't guess.
- `docs/environment_variables.md` if you added an env var.
- Run **`/pr-review` in branch mode** and act on it before asking a human.
- PR body follows
  `.kiro/skills/aten-lowering-orchestrator/references/pr-body-format.md` exactly
  (the canonical section order, the one-line-per-paragraph rule, and the
  tight-list rule that stop the two common GitHub render defects), base **`main`**,
  draft first, and **label it `torch-neuronx-agent`**
  (`gh pr create --label torch-neuronx-agent …`, or
  `gh pr edit <N> --add-label torch-neuronx-agent` if it opened without one).
  Name pre-existing failures with the evidence they reproduce on a clean base;
  explain a counter-intuitive registry diff (adds and removes net to zero in the
  GitHub diff); recount test totals after the last edit.
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

When this op is lowered in a batch, the **objective** items below (1–3, 5, 6, 7,
and 4 for a differentiable op) are graded by
`.kiro/skills/aten-lowering-orchestrator/scripts/verify_op_acceptance.py`
from artifacts you produce — a script-computed verdict, not a self-assessment.
Produce the artifacts it expects: a registry snapshot, the eager **and** compile
vertical coverage CSVs, and the unit-test rc. Standalone, you also produce the
upstream pytest-json with **and** without `--runxfail`; under the orchestrator
the central lane produces those. The verifier confirms you did not silence a test.
Item 6 (both execution paths) is gated too: the verifier grades the compile-path
CSV with its own status vocabulary (`supported`, not `runs_on_neuron`), and if
you added a decomposition it must be run under `--require-compile`. Items 8–12
are your own discipline; the reviewer checks them.

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
