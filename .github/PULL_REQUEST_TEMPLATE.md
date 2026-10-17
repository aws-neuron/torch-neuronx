<!--
Thanks for contributing to torch-neuronx. Two things to know:

1. Hardware CI does not start on its own. You will see a pending "Smoke Test"
   check until a maintainer triggers it. Nothing is needed from you.
2. We squash-merge. Your commits do not need to be tidy, but the PR title does.
   It becomes the commit subject on main.
-->

## Description

<!-- What changes, in a sentence or two. -->

## Related Issue

<!-- Link the issue this PR addresses. Use "Fixes #123" or "Closes #123" to auto-close. -->

## Motivation and Context

<!-- Why is this change required? What problem does it solve? Which model, op, or workload benefits from it? -->

## Type of Change

<!-- Mark with an `x` all the checkboxes that apply -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)

## Testing

<!--
Describe the tests you ran to verify your changes.
Paste the commands you ran and what happened. Commands, not a description of them.

Say which paths you exercised — eager, torch.compile, or both — and how many
devices, if this touches distributed.

Building and testing torch-neuronx needs a Trainium instance. If you do not have
one, leave the hardware section empty and say so. A maintainer will run hardware
CI for you. That is a normal state for an external PR. Do not report a suite as
passing if you did not run it.
-->

**Ran locally** (no Neuron hardware needed):

```
```

**Ran on Neuron hardware:**

```
```

## Checklist

- [ ] Added or updated tests for this change
- [ ] I've performed a self-review for my code
- [ ] Ran the tests, or explained under Testing why I could not
- [ ] Updated the corresponding documentation
- [ ] My changes generate no new warnings
- [ ] Said under Description if an AI assistant helped with this change (see [AI_POLICY.md](../AI_POLICY.md))
