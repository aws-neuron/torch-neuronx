# PR body format for a lowering PR

Use this exact format for every lowering PR. It follows
`.github/PULL_REQUEST_TEMPLATE.md` and keeps the body short and consistent across
the batch. Reference PRs: #1756 and #1810.

## Section order

1. `## Description` — one lead sentence, then a bullet list of the op names and
   variants. Nothing else.
2. `## Related Issue` — leave blank. Add a line only for a genuine linked issue
   or PR (for example a rebase dependency).
3. `## Motivation and Context` — one sentence: the ops fell back to CPU in eager
   but already pass the vertical `torch.compile` coverage; the native
   `@register_aten` lowering keeps the eager graph on Neuron. Add one clause for a
   bug fix that ships in the same PR.
4. `## Type of Change` — the template checkboxes.
5. `## Testing` — the two template checkboxes, then three detail bullets in the
   SAME list: `Unit tests` (result), `Vertical coverage (eager + compile)`,
   `Upstream OpInfo suite (single-tenant, source of truth)` with the nodeid
   counts.
   - `### Registry categorization` — one summary sentence with the new-xfail,
     new-skip, and removed counts. Add a `Category | Count | Root cause` table
     only when it helps. Do NOT repeat the counts in prose and the table.
   - Add one advisory line only when the verifier flags a roofline ratio over the
     threshold. Perf is a human call; do not block on it.
6. `## Compiler tickets to file` — include this section ONLY when a compiler bug
   exists. One line per ticket: the symptom and the failing input. Omit the whole
   section when there is no compiler bug. A non-compiler tracking ticket (for
   example a dispatcher bug) goes under Registry categorization, not here.
7. `## Checklist` — the template checkboxes.
8. Last line: disclose that an AI assistant (Claude) authored the change under
   human direction.

## GitHub markdown rules (avoid the two common render defects)

- **Write each paragraph and bullet on ONE line.** GitHub renders a single
  newline as a hard break (`<br>`). A paragraph hard-wrapped at ~76 chars leaves
  the right half of the box empty. Do not wrap; let the browser wrap.
- **Keep every list tight — no blank line between items of the same list.** A
  blank line inside a `-` list makes GitHub treat the whole block as a "loose"
  list and add a large vertical gap between every item. In `## Testing`, keep the
  two `- [x]` checkboxes and the three detail bullets in ONE list with no blank
  line between them.

## Style

- Write the body in Simplified Technical English (see `AGENTS.md`).
- No `Co-Authored-By` trailer.
- The PR stays draft; a human merges.
