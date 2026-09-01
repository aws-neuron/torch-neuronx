---
name: agent-isolation
description: Run multiple AI debug agents on the same Neuron host in parallel without collisions, and pick the right dispatch tier (parallel-streaming / parallel-with-worktrees-and-hardware-queue / sequential) for a given batch. Provides agent-run (per-agent core slice + port window + cache + temp scoping), provision-worktree (per-agent isolated checkout with editable-install MAPPING rewrite), with-hw-lock (a host-wide flock that serializes a whole-host hardware command), and the retro-derived rules (cancel-cascade trap, completion-manifest write-first, branch-tracking fix, _all_cores tests, hardcoded port literals). The ONLY use case is launching multiple independent debug agents in parallel on the same trn2 host — each with its own core slice, port window, caches, and (Tier 2) worktree + venv. Use when an orchestrator wants to fan out N debugger subagents, or when the user says "run agents in parallel" / "use agent-run" / "core slicing". Do NOT use it for anything inside a single agent, including `pytest -n` / xdist parallelism — those workers share that one agent's slice, CWD, caches, and venv and are coordinated by pytest itself, so no isolation kit is needed.
---

# Run multiple debug agents in parallel on one Neuron host

This skill is a **dispatch kit** for orchestrators. It turns "Neuron is
single-tenant, so debug-loop work has to be sequential" into a tiered
policy where the orchestrator picks the highest-throughput dispatch mode
the host can safely support, falling back as constraints kick in.

> **Only use this skill when launching multiple debug agents in
> parallel** on one host. Anything inside a *single* agent — including
> `pytest -n <N>` (xdist) — needs nothing from this kit: those workers
> share the one agent's slice, CWD, caches, and venv and pytest
> coordinates them itself.

The three tiers, top to bottom (each tier subsumes the next as fallback):

| Tier | Debug parallelism | Workdir | Hardware commands |
|---|---|---|---|
| **1 — parallel + streaming review** | N agents, disjoint core slices | shared CWD | parallel (slices are disjoint) |
| **2 — parallel + worktrees + hw queue** | N agents, disjoint core slices | per-agent git worktree | unit/coverage lanes run in parallel on disjoint `agent-run` slices; a whole-host command serializes via `with-hw-lock`; the upstream runner is single-tenant (whole host, one at a time) |
| **3 — sequential** | 1 agent at a time | shared CWD | n/a (only one agent) |

In **all three tiers**, Phase 2 review streams: as soon as a Phase-1
debugger completes for category X, the orchestrator spawns the Phase-2
review for X — no waiting for all debuggers.

## When the orchestrator picks which tier

The orchestrator probes once at batch start and picks the highest tier
that probes clean.

| Probe | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| `agent-run` exists and runs `--help` | required | required | not used |
| `flock` on PATH (host has util-linux) | nice-to-have | required | not used |
| `provision-worktree` exists and `git worktree --version` works | not used | required | not used |
| Sum of requested slices ≤ 64 cores | required | required | n/a |
| All categories' tests are slice-safe (no `_all_cores`) | required | required | required |
| No category has `--ignore-isolation` annotation from the user | required | required | n/a |

Probe happens once, before Phase 0 triage. Result is logged. Mid-batch
falls back automatically if a Tier-1 dispatch errors out for one of these
reasons (and only these — anything else is a real failure to surface):

- Slice overlap detected (the script enforces this; should never happen
  but if it does, drop to Tier 2).
- Edit-conflict between two parallel agents touching the same file in the
  shared CWD. (Tier 1 → Tier 2.)
- `EADDRINUSE` on a port outside the agent's window (a test hardcoded a
  port literal; per-test fix needed). Drop the offending category to
  Tier 3 and continue with the rest in current tier.

## Tier 1 — parallel debug + streaming review

Default for batches of 1–14 categories on a 64-core trn2.48xlarge where
no category needs more than 8 cores (gives at least 8 parallel slots).

```
spawn debug.01 in slice 0-3   ┐
spawn debug.02 in slice 4-7   │ all parallel
spawn debug.03 in slice 8-11  │ in one Agent-tool turn
...                           ┘

as each debug.NN completes:
    spawn review.NN in parallel (no hardware)
```

Per-agent invocation:

```bash
# Inside the spawned agent's prompt:
.kiro/skills/agent-isolation/scripts/agent-run "$BASE_CORE" "$NUM_CORES" -- \
    /path/to/your/agent-command-here
```

Slice sizing rules of thumb (debugger workload):

| Test type the category touches | NUM_CORES |
|---|---|
| Single-core unit tests, `tests/python_ops/`, OpInfo-driven failures | 4 |
| Distributed (NCCL/FSDP/comm) | 8 |
| Autograd (`gradcheck`, `gradgradcheck`) | 8 |
| Anything in a category needing the full machine | drop to Tier 3 |

Slices are assigned in input order: cat 1 → 0–3, cat 2 → 4–7, cat 3 →
8–11, etc. With 4-core slices that's 16 parallel agents max. Sum of
slices must be ≤ 64. Slices must not overlap (the wrapper exports
`NEURON_RT_VISIBLE_CORES=BASE-END`; two agents with overlapping ranges
fight for the runtime allocator).

### Streaming Phase 2 — the dispatch primitive

Spawn each Phase-2 review **inside the same assistant turn that observes
the corresponding Phase-1 completion**. Don't wait for all Phase-1 spawns
to drain before launching any review — that's the old strict-phase rule
and it serializes parallelism that exists.

Pattern:

```
# Turn 1: spawn all N debug agents in parallel, in_background=true
spawn debug.01 ...; spawn debug.02 ...; ...; spawn debug.NN ...

# Each subsequent turn (when one or more debug.NN completes):
for completed_NN:
    if completed_NN.status == done:
        spawn review.NN (also in_background=true so parallel reviews stay parallel)
    else:
        record blocked/no-repro/regressed; continue
```

The orchestrator never blocks on "Phase 1 fully drained" — it just
processes completions as they arrive and dispatches the matching review.

## Tier 2 — parallel debug + worktrees + hardware queue

Use when Tier 1 fails the edit-conflict probe (categories overlap on the
same source file — e.g. two categories both editing
`torch_neuronx/python_ops/torch_mlir/ops/indexing.py`), or when the
orchestrator detected an actual conflict mid-batch.

Each agent gets a private worktree from `provision-worktree`. The unit and
coverage lanes run under `agent-run` on disjoint core slices. A whole-host
command (the upstream runner, a `csrc` build) goes through `with-hw-lock`:

```bash
# Per-agent setup (orchestrator does this once per category before spawn).
# Activate the venv FIRST: provision-worktree copies the ACTIVE venv
# ($VIRTUAL_ENV), then a repo-root .venv. A shared host venv (for example
# /opt/torch-neuronx/.venv) is only found if it is active here.
eval "$(.kiro/skills/agent-isolation/scripts/provision-worktree "$SLUG")"
# now WORKTREE, BRANCH, VENV are exported

# Inside the agent's prompt:
cd "$WORKTREE"                     # load-bearing — see "CWD pinning" below
source .venv/bin/activate

# Unit / coverage lane — a disjoint slice; these trees honor the slice.
# agent-run pins each xdist worker to one core in the slice (no with-hw-lock):
.kiro/skills/agent-isolation/scripts/agent-run 0 4 -- \
    pytest tests/python_ops/test_<op>.py

.kiro/skills/agent-isolation/scripts/agent-run 8 4 -- \
    python -m pytest tests/vertical/... -k <op>

# Whole-host command (a csrc build): serialize it with with-hw-lock.
.kiro/skills/agent-isolation/scripts/with-hw-lock -- ./tools/build
```

`with-hw-lock` takes an exclusive whole-host `flock` on `/tmp/agent/hw.lock`
(override with `AGENT_HW_LOCK`), so only one wrapped command runs at a time. Wrap
a **whole-host** command in it (the upstream runner, a `csrc` build). Do NOT wrap
the unit or coverage lane in it: those lanes parallelize on disjoint `agent-run`
slices, so a host lock would serialize them for no reason. Each xdist worker gets
one core in the slice from the `agent_core_isolation` plugin that `agent-run`
loads.

**The upstream runner is single-tenant.** `run_upstream_tests.py`
self-parallelizes with xdist, and the tree's
`conftest._setup_neuron_worker_for_xdist` OVERWRITES `NEURON_RT_VISIBLE_CORES`
from each worker GLOBAL index — so the runner claims cores `0..N-1` and ignores
any `agent-run` slice. Run it on the whole host, one at a time, only after the
unit lanes release their cores. Wrap it in `with-hw-lock` so two runner instances
cannot overlap. A live unit lane holds cores but not the host lock, so gate the
runner on a device-free check first
(`.kiro/skills/aten-lowering-orchestrator/scripts/neuron_devices_free.py`) — that
gate, not the lock, is what stops the runner from colliding with a live unit lane.

### Why the worktree (and the editable-install rewrite)

The retro names this as a 30-minute pain point. Plain `git worktree add`
gives you a separate filesystem tree, but Python's editable-install
machinery has cached the *original* repo root in
`__editable___torch_neuronx_*_finder.py`'s MAPPING dict. So
`import torch_neuronx` from inside the worktree still resolves to the
canonical checkout's source — the worktree is silently wrong.

`provision-worktree` rewrites the MAPPING + nukes stale `.pyc` (which
embed absolute paths in `co_filename`) + symlinks the untracked artifacts
that a fresh checkout lacks but `import torch_neuronx` needs: `build/`,
`CMakeCache.txt`, and every in-tree compiled `.so` (`torch_neuronx/_C.so`
and the torch-mlir `_mlir_libs/*.so`). The MAPPING points imports at the
worktree, so the `.so` must exist there at the same relative path.

### CWD pinning — load-bearing

Inside an agent that's running in a worktree, **`cd` into the worktree
before any other command**. Python's `sys.path` has `''` (CWD) at the
front, ahead of the editable meta-path finder. If the agent runs from
the canonical checkout, even with the venv copied and the MAPPING
rewritten, `import torch_neuronx` finds the canonical-checkout source
first and the worktree's edits don't take effect. The `cd` is the
fix — keep it.

### Branch tracking gotcha

`git worktree add -b NEW BASE` makes the new branch track BASE upstream,
not its own remote counterpart. `gh pr create` complains. The
provisioner clears the upstream explicitly so `git push -u origin
<branch>` works at the push step — but if you provision worktrees by
hand without the script, remember this.

## Tier 3 — sequential

Fallback. Used when:

- Tier 1 and Tier 2 both probe-fail (no `agent-run`, no `flock`, no
  `provision-worktree`, or environment otherwise rejects parallel
  execution).
- Any category in the batch needs all-cores access (filenames matching
  `*_all_cores*`, e.g. `tests/device/test_device_lnc1_all_cores.py`).
  These can't run in any agent that owns less than the full 64-core
  machine, so the whole batch drops to sequential.
- The user explicitly passes a `--sequential` flag or annotates the
  batch with "single-tenant".

Sequential mode runs one debugger at a time, sharing the canonical
checkout's CWD and hardware. Streaming Phase 2 still applies — each
debugger's review starts as soon as it completes, even though the next
debugger is also serializing.

## Kit contents

| File | Purpose |
|---|---|
| `scripts/agent-run` | Per-agent core/port/cache/temp scoping + cancel-safe SIGTERM trap. Used in Tier 1 and Tier 2. |
| `scripts/provision-worktree` | Per-agent isolated checkout for Tier 2: worktree + venv copy + editable-install MAPPING rewrite + .pyc nuke + branch tracking + build-artifact symlinks. |
| `scripts/with-hw-lock` | whole-host `flock` that serializes a hardware command (the upstream runner, a `csrc` build). Tier 2 only. |
| `scripts/lib/agent_port_isolation.py` | pytest plugin auto-loaded by `agent-run` via `PYTEST_PLUGINS`. Patches `_BASE_MASTER_PORT` for the repo's `DistributedTester` and sets `MASTER_PORT` for upstream tests. No-op if `AGENT_PORT_OFFSET` is unset. |

## What the wrapper does (mechanics)

Given `BASE_CORE` and `NUM_CORES`, `agent-run` exports:

1. **Cores** — `NEURON_RT_VISIBLE_CORES=BASE-END` for the master pytest
   process. Three more vars (`PYTEST_XDIST_WORKER=gw<BASE>`,
   `PYTEST_XDIST_WORKER_COUNT=64`, `TOTAL_DEVICE_COUNT=64`) make
   `tests/utils/neuron_test_utils.get_core_start_index()` return `BASE`,
   so distributed tests' spawned ranks pin themselves to `BASE+rank`
   inside `mp.spawn` children.

2. **Ports** — `AGENT_PORT_OFFSET = 1000 + BASE*32` gives a 32-port
   window per slice starting at `30500 + BASE*32`. The pytest plugin
   reads it and rewrites the repo's `_BASE_MASTER_PORT` and the
   `MASTER_PORT` env var.

3. **Caches and temp** — `TORCH_NEURONX_HLO_CACHE_DIR`,
   `TORCH_NEURONX_NEFF_CACHE_DIR`, and `TMPDIR` all point at
   `/tmp/agent/c<BASE>/...`. TMPDIR catches torchinductor work dirs,
   NKI temp dirs, filelocks, scratch artifacts.

4. **Cancel safety** — when the parent agent is cancelled (harness
   sends SIGTERM), the trap forwards SIGTERM to the whole child
   process group so pytest descendants don't keep holding
   `/dev/neuron0`. Without this, the next agent on the same slice
   fails with "Failed to initialize Neuron Runtime: status code 1"
   until someone manually `fuser`-kills the zombies.

## Mental model

```
trn2.48xlarge (64 logical cores, 1 host network namespace, 1 /tmp)
│
├── Tier 1:
│     Agent A: cores 0-3   ports 30500-30531  /tmp/agent/c0/   shared CWD
│     Agent B: cores 4-7   ports 30628-30659  /tmp/agent/c4/   shared CWD
│     Agent C: cores 8-11  ports 30756-30787  /tmp/agent/c8/   shared CWD
│
├── Tier 2:
│     Agent A: cores 0-3   ports 30500-30531  /tmp/agent/c0/   /tmp/worktrees/<slug-A>/
│     Agent B: cores 4-7   ports 30628-30659  /tmp/agent/c4/   /tmp/worktrees/<slug-B>/
│     Agent C: cores 8-11  ports 30756-30787  /tmp/agent/c8/   /tmp/worktrees/<slug-C>/
│     (unit/coverage lanes run at the same time on disjoint agent-run slices;
│      a whole-host command serializes on /tmp/agent/hw.lock; the upstream
│      runner is single-tenant — whole host, one at a time, after the unit
│      lanes release their cores)
│
└── Tier 3:
      Agent X: full machine, canonical checkout, run to completion, then Agent Y, ...
```

## Constraints (Tier 1 and Tier 2)

The orchestrator must enforce:

1. **Sum of slice sizes ≤ 64** on trn2.48xlarge.
2. **Slices do not overlap.**
3. **One physical core per process at the bottom.** The repo's
   distributed helpers already do this. Don't try to give one pytest
   process a multi-core range and parallelize *inside* that one
   process — that path collides because the runtime stops cross-process
   coordinating once `NEURON_RT_VISIBLE_CORES` is constrained.
4. **No `_all_cores` tests in the parallel batch.** Drop those
   categories to Tier 3 or reject them from the batch.

## Failure modes (rare but real)

- **Hardcoded port literals in test source** (e.g.
  `init_method="tcp://localhost:29501"`) bypass both the
  `_BASE_MASTER_PORT` patch and the `MASTER_PORT` env var because the
  port reaches the C++ `TCPStore` directly. Per-test fix, not per-agent.
  Move the affected category to Tier 3 (or the whole batch if many
  categories share this).

- **Probe-and-close port race in `tests/distributed/utils.py
  _get_worker_port`.** With many agents launching distributed tests at
  the exact same instant, two could both probe-find the same port free
  before either calls `TCPStore`. The 32-port-per-agent windows make
  this nearly impossible; if it ever fires, retry the agent.

- **Tests that need all cores visible** (filenames containing
  `_all_cores`, e.g. `tests/device/test_device_lnc1_all_cores.py`)
  cannot run in any agent that owns less than the full 64-core machine.
  Tier-3 fallback for these.

- **Stale `.pyc` in copied venvs.** The provisioner nukes them, but if
  the orchestrator hand-rolls a worktree without the script, it must
  too — or `import torch_neuronx` keeps resolving to the original
  checkout's source.

- **Editable-install MAPPING points at the original repo root.** The
  provisioner rewrites it. Hand-rolled worktrees must too (see "Why
  the worktree" above).

- **Subagent cancel cascade.** Cancelling one stage in a `mode:
  blocking` pipeline can tear down sibling stages — even ones that
  already pushed commits. `agent-run`'s SIGTERM trap helps locally
  (frees the device) but doesn't undo the cancel cascade itself. If
  this happens, the orchestrator recovers by re-running affected
  categories — the branch and any already-pushed commits survive.

- **Subagent didn't write `completion.json` on `done`** (cancel arrived
  pre-write). The debugger skill should write the manifest first with
  `status: "in_progress"` and update it at terminal status, so the
  orchestrator can distinguish "agent crashed before writing" from
  "agent decided not to write". (See debugger skill Step 10 for the
  current contract.)

## Verification (recorded)

Two `world_size=2` distributed tests
(`tests/distributed/cache/test_cross_process_caching.py`) ran in parallel
on disjoint cores 0–1 and 2–3:

```
Agent A (cores 0-1, port offset 1000): 1 passed in 13.81s
Agent B (cores 2-3, port offset 1064): 1 passed in 23.10s
Wallclock total: 27s
```

Without the wrapper, the same scenario fails immediately with
`EADDRINUSE` on port 29500.

A subsequent batch of 14 debugger subagents ran in parallel using Tier 2
(worktrees + hw lock) and produced 14 draft PRs in one wallclock run.
See the retro at the end of `debug-orchestrator-full-run.json` for what
worked and what bit us.

## Why a plugin instead of a code change

A single-line patch to `tests/distributed/utils.py` (`_BASE_MASTER_PORT
= int(os.environ.get(...))`) would also work and would be cleaner. It
was rejected because the agent debug workflow also runs PyTorch upstream
tests, which use their own multiprocess machinery and don't read
`_BASE_MASTER_PORT`. The plugin handles both surfaces from outside the
repo.

## How orchestrators consume this skill

The orchestrator's pseudocode at batch start:

```
probe → pick tier → for each category:
    case tier 1: spawn agent (background) with `agent-run BASE 4 -- ...`
    case tier 2: provision-worktree first, then spawn agent with
                 `agent-run BASE 4 -- with-hw-lock -- ...`
    case tier 3: spawn agent (foreground) with no wrapping

per-completion:
    spawn matching pr-review subagent (always background)

aggregate at end.
```

See `.kiro/agents/upstream-debugger-orchestrator.md` for the concrete
orchestrator that drives this.
