# Dev playbook: how to write the fix

Four principles for authoring the `torch_neuronx` change once Step 7
has you editing source. Terse on purpose. The `loop-playbook.md`
sibling tells you *where the bug is*; this tells you *how to write the
fix once you're there*.

## 1. Think before coding

Find the real boundary before you touch a line. The failing frame is a
symptom; the fix belongs at the op/dispatch/cast layer that produced
it, not the first place you can make the error disappear.

**Map it first.** If codegraph is loaded (`codegraph_status` succeeds),
use it — it's the pre-built index, faster and more complete than grep:

- `codegraph_explore "<op> dispatch registration cast policy"` — one
  call returns the verbatim source of the relevant symbols. Usually the
  only call you need.
- `codegraph_callers` / `codegraph_callees` — trace what reaches the op
  and what it reaches.
- `codegraph_impact` — before editing a shared symbol, see what else
  breaks. This is how you avoid a "fix" that regresses ten siblings.

If `codegraph_status` errors ("not initialized" / "no project loaded"),
fall back to grep + Read. Do **not** block on it or try to init it
mid-fix — degrade silently and keep moving.

Write the root cause and the target layer into `notes.md` *before*
editing. One or two lines: "float64 leaks through cpu_fallback allowlist
→ fix at Python op registration in `torch_neuronx/ops/…`". If you can't
write that sentence, you don't understand the bug yet — keep reading.

**The test:** could you point at one symbol and say "the fix goes
here, because that's where the wrong behaviour originates"? If you're
guessing, you're not ready to edit.

## 2. Fix at the highest layer that can express it

Bias every fix upward. Same correctness, less blast radius, no rebuild.

```
Python op registration / cast policy / decomposition   ← prefer
        ↓ only if the layer above genuinely can't express it
C++ csrc dispatch / kernel
        ↓ never
neuronx compiler  •  the pytest / test-overlay layer
```

- **Python first.** A cast-policy entry, a decomposition route, or a
  dispatch-key registration in `torch_neuronx/` beats a C++ edit:
  re-import picks it up (no `./tools/build` cycle), and the diff is
  legible to reviewers.
- **C++ `csrc/` only when you must** — the behaviour can't be reached
  or corrected from Python. Then accept the rebuild (loop-playbook
  "Common traps").
- **Never** patch the compiler or the test framework to go green. A
  compiler bug gets an RCA note + ticket + `xfail` with a reason
  (loop-playbook `compiler_error_neuron`). A test-framework patch hides
  the bug while production stays broken (loop-playbook "Don't chase a
  fix through the test framework").

This is a strong bias, not an absolute. If the high layer genuinely
can't express the fix, drop one level — and record *why* in `notes.md`
so the reviewer sees the reasoning, not just the lower-level diff.

**The test:** is there a higher layer where this same fix would be
smaller and need no rebuild? If yes, move it there.

## 3. Surgical changes

Edit only what this one category needs.

- **One bug per PR** (skill's standing rule). If you spot an adjacent
  bug, defer it as a single `## Long tail` bullet in the PR body (a
  future pass through this skill — see Step 9 "PR-body structure") —
  don't fix it here.
- **No drive-by refactors, renames, or cleanups** in files you're
  touching for the fix. They bloat the diff and break `git blame`.
- **Match the surrounding idiom** — naming, cast helpers, registration
  pattern, comment density. The fix should read like the code already
  there, not like a transplant. (`docs/AI-python-ops.md`
  shows the house style for op registration.)
- **No speculative generality.** Don't add a config flag, an abstraction
  layer, or handling for dtypes the category doesn't exercise.

**The test:** would a reviewer skimming the diff see only changes that
trace back to this failure signature? Anything they'd have to ask "why
is this here?" about — cut it.

## 4. Goal-driven execution

The new unit test from Step 6 is the success criterion. Loop against it,
don't eyeball it.

- Inner loop: edit → run **only** the new test (seconds) → repeat.
- Green on the unit test → run the whole op test file (catch sibling
  regressions) → only then the full-category verify (Step 8).
- **Two failed attempts on the same hypothesis = stop.** Re-read the
  frame, find a different root cause. Incremental patches on a
  misunderstood cause burn budget (Step 7 standing rule).
- **Don't claim done without a green you can point at.** "Should work"
  is not a verification. The verify diff against the Step 3 baseline is
  the proof; if you didn't run it, the fix isn't verified — say so.

**The test:** is there a command whose output proves the fix works, and
did you actually run it? If not, you're not done.
