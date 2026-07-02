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
└── agent-isolation/
    ├── SKILL.md                            kit + three-tier dispatch policy
    └── scripts/
        ├── agent-run                       per-agent core/port/cache scoping
        ├── provision-worktree              per-agent isolated checkout
        ├── with-hw-lock                    serialize hardware commands
        └── lib/agent_port_isolation.py     pytest plugin for port windows
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

## Conventions

- One directory per skill, exactly one `SKILL.md` at its root.
- Skill names are lowercase with hyphens and match their directory name.
- `description` in frontmatter says both what the skill covers and when
  to use it, so the skill matcher picks it up.
- Paths inside skills are repo-relative where possible.
- Skills that require hardware access emit commands but don't prescribe
  the execution mechanism — that belongs in a separate
  hardware-wrapper skill.
