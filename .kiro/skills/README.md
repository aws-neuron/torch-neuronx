# torch-neuronx Skills

Kiro skills for the `torch-neuronx` package. Each skill is one directory
with a `SKILL.md` at its root whose YAML frontmatter is matched against
the user's question. This follows the same convention used by
[pytorch/pytorch `.claude/skills`](https://github.com/pytorch/pytorch/tree/main/.claude/skills).

## Skills in this tree

### `torch-neuronx-docs/` — reference knowledge base

Answers "what is this repo?", "how does X work?", and "where does Y live?"
questions about the `torch-neuronx` codebase. `SKILL.md` is the index;
subsystem detail lives in the `docs/AI-*.md` set, covering overview →
public API → backends → internals → tests / examples / build.

### `pytorch-upstream-test-debugger/` — end-to-end fix workflow

Drives an 8-step debug loop (ingest → pin → reproduce → RCA+tests →
fix → verify → draft PR) for one failure category in PyTorch upstream
tests running against Neuron. Takes a pytest report (JSON / CSV /
plain log) plus a user pin (test nodeid or error-message regex);
expands the pin to every test sharing the same error signature;
scaffolds an RCA workspace with a unit-test template and a standalone
repro script; iterates a fix inside `torch_neuronx/`; re-verifies
against a baseline; opens a draft PR with the repro script attached.

`xfailed` tests are tracked failures and are in scope — they must be
fixed, not tolerated. Fixes land in `torch_neuronx` source, never in
pytest patches. One bug per PR. Hardware access is delegated to a
separate wrapper skill — this skill emits the commands, the wrapper
runs them.

### `agent-isolation/` — parallel-debug dispatch kit

Lets multiple debug agents run on the same Neuron host without
colliding on cores, ports, caches, tempdirs, or source files. Provides
the kit (`agent-run` for per-agent core/port/cache scoping,
`provision-worktree` for per-agent isolated checkouts with editable-
install MAPPING rewrite, `with-hw-lock` for serializing hardware
commands across parallel agents) and the dispatch policy that picks
between three tiers — parallel-streaming, parallel-with-worktrees-
and-hardware-queue, sequential — based on what the host supports.

Cited from `.kiro/agents/upstream-debugger-orchestrator.md` (the
orchestrator picks a tier at batch start) and from
`pytorch-upstream-test-debugger/SKILL.md` (the debugger emits the
hardware commands; this skill runs them).

### `pr-review/` — PR / branch review workflow

Reviews a GitHub PR against this repo — or a local branch before
raising one — focusing on what CI cannot check: design quality, use of
project infrastructure, test adequacy, backward compatibility of the
`torch_neuronx` public surface, thread / compilation safety, and
Neuron-specific correctness (dtype cast policy, contiguity, NEFF
cache, C++ dispatch / Python op fallback, `torch.compile` constraints).

Three modes — PR mode (fetches via `gh pr view` / `gh pr diff`),
branch mode (diffs against `origin/main`), and commit-range mode. The
review output reports problems only; nothing that CI already enforces.
Cross-references `docs/AI-*.md` for subsystem
context; evaluates BC per `bc-guidelines.md`; walks the diff against
`review-checklist.md`.

### `compiler-artifact-repro/` — compiler-bug isolation & hand-off

Turns a "test fails on this neuronx-cc version" report into a package
for the Neuron compiler team: reproduces the failure on a Neuron
(trn2-class) host, dumps the offending **HLO** (`module.mlir`) and
**NEFF** (`module.neff`) via `TORCH_NEURONX_DEBUG_DIR` +
`…PRESERVE_COMPILATION_ARTIFACTS`, and — the key step — proves the
dumped HLO reproduces **standalone** through
`torch_neuronx._C.compile_graph` / `execute_compiled_graph` (no aten
dispatch, no Python decomposition). Then localizes the bug
(shard / tile / shape / race) and packages the artifacts for hand-off.
`SKILL.md` is the flow; `reference.md` (§A–F) holds the env-var,
runtime-API, localization, and hand-off detail; `scripts/` has `artifact_dump.py`, `validate_hlo.py`
(compile-only or execute+diff), and `gen_ticket.py`.

Hardware access and the team's upload/issue-tracker routing are
delegated (not prescribed) — matching the repo convention that skills
emit host commands but don't bake in the execution mechanism. Isolates
and hands off a compiler bug; it does **not** edit `torch_neuronx/`
source — that's `pytorch-upstream-test-debugger`.

### `accuracy/` — Neuron-vs-GPU accuracy skill suite

A nested **suite** (a master `accuracy` index/router + 25 sub-skills) for
validating and debugging that a model computes the same thing on Neuron (TRN) as
on a GPU/CPU reference — from a single module up to a 10k-step training run. The
master `accuracy/SKILL.md` routes to the right sub-skill; **each sub-skill is also
usable directly**. Covers dense and MoE models (DeepSeek-V3, Qwen3, Llama4,
GPT-OSS, Llama3). Grouped:

- **Level router** — `moduscope-accuracy-levels` (module → op/NKI → 5/50/10k-step → activation compare).
- **Debugging** — `moduscope-op-bisect` (per-op fwd/bwd localization: eager
  `torch.save`, compiled return-intermediates, NKI `nl.device_print` via
  `wrap_nki` + `NEURON_RT_DEBUG_OUTPUT_DIR`), `moduscope-accuracy-debug`,
  `activation-compare` (per-layer FrobRatio/CosSim across steps + loss-spike
  X-1 param-norm workflow), `moduscope-result-triage`, `moduscope-test-roadblocks`.
- **Standards / knowledge** — `neuron-accuracy-issues` (9 known causes, precision
  RCs, CRs, exit criteria, prevention rules), `large-model-accuracy-bounds`
  (bf16-ULP perturbation envelope + grad-norm-diff, accept without fp32),
  `moduscope-reference-fidelity`.
- **Component suite (`test_00..16`)** — `moduscope-component-tests`, `-run-test`,
  `-dist-sweep`, `-add-test-module`, `-model-profile`, `-register-model`,
  `-statedict-mapping`, `-gpu-baselines`, `-trn-compare`, `-env-bootstrap`,
  `-node-select`, `-parallel-orchestration`, `-newmodel`, `-autopilot`.
- **Full-model runs** — `titan-accuracy` (GPU H200 vs TRN2 e2e), `apples-10k-trn-vs-gpu`.

Personal paths are parameterized (`$SHARED`, `$VENV`, `$LOGIN_HOST`, `$SSH_KEY`,
`$USER`, …) — see the placeholder table in `accuracy/SKILL.md`. Portable
self-tests (torch only) ship under several sub-skills' `scripts/`.

## Layout

```
.kiro/skills/
├── README.md                               (this file)
├── torch-neuronx-docs/
│   ├── SKILL.md                            (index + quick answers)
│   └── docs/
│       ├── 01-package-overview.md          what torch-neuronx is, end-to-end
│       ├── 02-runtime-api.md               torch_neuronx public Python API
│       ├── 03-dynamo-backend.md            torch.compile backend="neuron"
│       ├── 04-python-ops.md                eager-mode op registration + kernels
│       ├── 05-csrc.md                      C++ _C extension internals
│       ├── 06-distributed.md               ProcessGroupNeuron, collectives, DTensor
│       ├── 07-compilation-kernels.md       NEFF cache, neuronx-cc wrapper
│       ├── 08-testing.md                   test layout, drivers, markers
│       ├── 09-examples.md                  examples/ tour
│       └── 10-build-infra.md               Bazel/CMake build, tools/, CI
├── pytorch-upstream-test-debugger/
│   ├── SKILL.md                            8-step debug loop + rules
│   ├── references/
│   │   ├── loop-playbook.md                per-signature heuristics + traps (where the bug is)
│   │   └── dev-playbook.md                 how to write the fix (4 principles, codegraph, high-layer bias)
│   └── scripts/
│       ├── select_category_tests.py        parse report, pin, expand by signature
│       ├── rca_workspace.py                scaffold RCA notes + unit-test seeds + repro stub
│       ├── rerun_category.sh               run tests + json-report
│       └── verify_fixed.sh                 diff current run vs baseline
├── pr-review/
│   ├── SKILL.md                            PR / branch review workflow
│   ├── review-checklist.md                 per-subsystem review items
│   └── bc-guidelines.md                    backward-compat rules for torch_neuronx
├── compiler-artifact-repro/
│   ├── SKILL.md                            compiler-bug isolation & hand-off flow
│   ├── reference.md                        env vars, runtime API, localization, hand-off (§A–F)
│   └── scripts/
│       ├── artifact_dump.py                dump HLO/NEFF (set debug dir + preserve, before torch_neuronx)
│       ├── validate_hlo.py                 compile + run a dumped module.mlir standalone (no aten)
│       └── gen_ticket.py                   render a ready-to-file ticket.md into the repro folder
├── agent-isolation/
│   ├── SKILL.md                            kit + three-tier dispatch policy
│   └── scripts/
│       ├── agent-run                       per-agent core/port/cache scoping
│       ├── provision-worktree              per-agent isolated checkout
│       ├── with-hw-lock                    serialize hardware commands
│       └── lib/agent_port_isolation.py     pytest plugin for port windows
└── accuracy/                               Neuron-vs-GPU accuracy suite (25 skills)
    ├── SKILL.md                            master index + router + placeholders
    ├── moduscope-accuracy-levels/          testing-level ladder (module→op→run→activation)
    ├── moduscope-op-bisect/                per-op fwd/bwd localization (+ portable self-tests)
    ├── moduscope-accuracy-debug/           triage→attribute→localize→characterize loop
    ├── activation-compare/                 per-layer FrobRatio/CosSim across steps (+ loss-spike)
    ├── neuron-accuracy-issues/             9 causes, precision RCs, CRs, exit criteria (+ assets/)
    ├── large-model-accuracy-bounds/        bf16-ULP envelope + grad-norm-diff (no fp32)
    ├── moduscope-component-tests/ … +13 more   build & run test_00..16
    ├── titan-accuracy/ · apples-10k-trn-vs-gpu/   full-model 5/50/10k-step runs
    └── … (each sub-dir = one skill: SKILL.md [+ scripts/ / assets/])
```

## How to use

- **Reference questions about this repo** → `torch-neuronx-docs`.
  Start with `SKILL.md`, drill into `docs/AI-*.md`.
- **Fixing a failing or xfailing test** → `pytorch-upstream-test-debugger`.
  Start with its `SKILL.md`; drive through the 8 steps using the
  scripts in `scripts/` and the heuristics in `references/loop-playbook.md`.
- **Reviewing a PR or self-reviewing a branch** → `pr-review`.
  Start with its `SKILL.md`; walk the diff against
  `review-checklist.md` and evaluate BC per `bc-guidelines.md`.
  Cite `docs/AI-*.md` for subsystem context.
- **Running multiple debug agents in parallel** → `agent-isolation`.
  Read its `SKILL.md` for the three-tier dispatch policy and
  `scripts/` for the kit (`agent-run`, `provision-worktree`,
  `with-hw-lock`). Used by the orchestrator at
  `.kiro/agents/upstream-debugger-orchestrator.md`.
- **Isolating a compiler miscompile and handing it to the Neuron compiler
  team** → `compiler-artifact-repro`. Start with its `SKILL.md`, use
  `scripts/` to dump + standalone-validate the HLO/NEFF, and consult
  `reference.md` §A–F for the details.
- **Validating or debugging model numerical accuracy on Neuron** → `accuracy/`.
  Start with `accuracy/SKILL.md` (the master router) to pick the level/skill, or
  load a sub-skill directly (e.g. `moduscope-op-bisect` to localize a divergent
  op, `neuron-accuracy-issues` to recognize a known cause). Set the placeholders
  from `accuracy/SKILL.md` before running any script.

## Conventions

- One directory per skill, exactly one `SKILL.md` at its root.
- A **suite** (like `accuracy/`) may nest sub-skill directories under a folder
  that also has its own master `SKILL.md`; recursive `skill://.kiro/skills/**/SKILL.md`
  discovery finds them all, and each sub-skill stays independently usable.
- Skill names are lowercase with hyphens and match their directory name.
- `description` in frontmatter says both what the skill covers and when
  to use it, so the skill matcher picks it up.
- Paths inside skills are repo-relative where possible.
- Skills that require hardware access emit commands but don't prescribe
  the execution mechanism — that belongs in a separate
  hardware-wrapper skill.
