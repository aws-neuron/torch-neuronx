# Fallback pull request template

Use the repository template `.github/PULL_REQUEST_TEMPLATE.md` if present. Use this template only when that file is absent.

## Description

<Describe the change in one or two sentences. State whether an AI assistant helped.>

## Related Issue

Fixes #<issue-number>

## Motivation and Context

<Explain why the change is required and what problem it solves.>

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring (no functional changes)

## Testing

**Ran locally** (no Neuron hardware needed):

```text
<exact commands and results>
```

**Ran on Neuron hardware:**

```text
<exact commands and results, or state that hardware was unavailable>
```

## Checklist

- [ ] Added or updated tests for this change
- [ ] I performed a self-review of my code
- [ ] Ran the tests, or explained under Testing why I could not
- [ ] Updated the corresponding documentation
- [ ] My changes generate no new warnings
- [ ] Said under Description if an AI assistant helped with this change
