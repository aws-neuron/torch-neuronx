# Pull request body format

Prepare this body only after the deterministic acceptance gate passes.
Use the optional `.github/PULL_REQUEST_TEMPLATE.md` if present. If it is absent, use the complete fallback below. Do not invent unavailable sections or checklist items.
Do not open or push a pull request without explicit user approval of the exact branch, title, and body.
Open it as a draft after approval.
Keep each paragraph and list item on one physical line.
Use Simplified Technical English.

## Required human section

Start the body with this section exactly.
The agent must not replace or remove the comment.
Stop and ask the human to write the commentary before any push.
The human commentary must explain why the change is needed.
It must confirm review of the agent-generated content and every accepted limitation.

```markdown
## Human commentary

<!-- Human: explain why this change is needed, confirm that you reviewed it, and confirm that you reviewed every accepted limitation -->
```

## Complete fallback body

Put all agent-generated text in one quote block after the human section.
Select only checkboxes supported by evidence.
Keep the AI disclosure line in that block.
Use this complete fallback when the optional repository template is not present:

```markdown
> ## Agent-generated contribution
>
> ### Description
>
> <AI tool name> authored this change under human direction.
>
> Adds native Neuron support for `aten::<operator>`.
>
> - `aten::<operator>`
> - `aten::<operator>.out`
> - `aten::<backward>`
>
> ### Related Issue
>
> Fixes #<issue>
>
> ### Motivation and Context
>
> <prior fallback or missing support and why the lowering is needed>
>
> ### Type of Change
>
> - [ ] Bug fix (non-breaking change which fixes an issue)
> - [ ] New feature (non-breaking change which adds functionality)
> - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
> - [ ] Documentation update
> - [ ] Refactoring (no functional changes)
>
> ### Testing
>
> **Ran locally** (no Neuron hardware needed):
>
> ```text
> <exact commands and results>
> ```
>
> **Ran on Neuron hardware:**
>
> ```text
> <exact commands and results, or state that hardware was unavailable>
> ```
>
> - Focused tests: `<command>` produced `<result>`.
> - Upstream tests: `<command>` produced `<result>`.
> - Coverage: eager `<status>` and compile `<status>`.
> - Acceptance gate: `PASS` from `verify_op_acceptance.py`.
>
> ### Registry categorization
>
> Added marks: `<count>`. Removed marks: `<count>`. Unchanged marks: `<count>`.
> Accepted limitations: `<count>`.
> - `<exact-node-id>`: `<skip-or-xfail>` limitation `<limitation-id>` matched reason `<registry-reason>` and signature `<failure-signature>` because `<root-cause>`.
>
> ### Checklist
>
> - [ ] Added or updated tests for this change
> - [ ] I've performed a self-review for my code
> - [ ] Ran the tests, or explained under Testing why I could not
> - [ ] Updated the corresponding documentation
> - [ ] My changes generate no new warnings
> - [ ] Said under Description if an AI assistant helped with this change (see `AI_POLICY.md`)
```

Omit overloads that the change does not cover.
Link the required actionable public issue with `Fixes #<issue>`.
Add a compiler issue section only when a public compiler issue exists.
State the symptom and smallest failing input there.

A new skip or xfail fails acceptance by default.
Accept it only after a human approves the exact node ID and stable limitation ID.
Pass one `--accept-limitation <node-id>=<limitation-id>` value per approved node.
Copy every `accepted_limitation` verdict check into Registry categorization.
Include the exact node ID, mark type, limitation ID, registry reason, matched signature, and root cause.
Write `Accepted limitations: 0` when no new mark was accepted.
Ask the human to review this list and confirm it in Human commentary.

Do not add an automated co-author trailer.
Do not claim that an unrun test passed.
Do not include temporary paths or host details.

## Approval boundary

Show the exact branch, title, and complete body after the human writes the commentary.
Wait for explicit approval of that exact content before each GitHub write.
Push only to the contributor's fork remote, `origin`.
Open only a draft pull request against `$REPO`.
Never merge, mark ready, force-push, or push to `main`.
After approval, replace each angle-bracket value and run:

```bash
set -euo pipefail; source "agent_space/<task-slug>/env.sh"
BRANCH="$(git branch --show-current)"
FORK_OWNER='<github-login>'
# Run only after explicit approval of the exact branch and push.
git push -u origin "$BRANCH"
# Run only after explicit approval of the exact title and body.
gh pr create --repo "$REPO" --draft --head "$FORK_OWNER:$BRANCH" --base main \
  --title '<approved title>' --body-file "agent_space/<task-slug>/pr-body.md"
```
