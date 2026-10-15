# AI Policy

AI tools are welcome here. This page sets the rules for using them when you file an
issue, open a pull request, or comment.

## Disclose AI-generated content, and contain it

Put AI output in a code block or a quote block, and say which tool produced it. Add
your own commentary explaining why it is relevant. For example:

> Claude produced this analysis: `<output>`. I checked the allocator path myself and
> agree with the conclusion, because ...

## Do not answer people with raw AI text

A question from a maintainer or another contributor is a request for *your*
understanding. Pasting unreviewed AI output moves the work of verifying it onto
them, and makes the discussion slower and noisier.

## Read your own pull request before you send it

This matters most when a tool wrote or rewrote the code. Make sure you understand
the change, that the main idea is clear, and that you can defend it in review.
AI-generated code is often longer and more indirect than it needs to be — simplify
it before you ask for review. If it is not ready, open a draft pull request.

## A human is accountable for every contribution

Someone has to understand the change and answer questions about it. **We do not
accept pull requests from fully autonomous agents**, and we may close a pull request
that shows no sign of human involvement.

## Related

- [CONTRIBUTING.md](CONTRIBUTING.md) — how to build, test, and submit a change.
- [AGENTS.md](AGENTS.md) — instructions read by AI agents working in this
  repository. This page is for the humans directing them.
