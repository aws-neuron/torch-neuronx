# Pull Request Fallback Template

Use the repository template `.github/PULL_REQUEST_TEMPLATE.md` if present. Use this fallback only when that file is absent.

## Description

Describe the change in one or two sentences.
State whether an AI assistant helped with the change.

## Related Issue

Fixes #<issue>

## Motivation and Context

Explain why the change is required and what problem it solves.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring with no functional change

## Testing

List exact commands and results.
Do not claim a command ran when it did not run.

**Ran locally** (no Neuron hardware needed):

```text
<commands and results>
```

**Ran on Neuron hardware:**

```text
<commands and results, or state that Trainium was unavailable>
```

Hardware CI checks can stay pending until a maintainer starts them.

## Checklist

- [ ] Added or updated tests for this change.
- [ ] Performed a self-review.
- [ ] Ran the tests, or explained why they could not run.
- [ ] Updated the corresponding documentation.
- [ ] Confirmed that the change creates no new warnings.
- [ ] Disclosed AI assistance under Description and followed `AI_POLICY.md`.
