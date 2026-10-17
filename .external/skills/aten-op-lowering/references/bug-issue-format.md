# Bug issue fallback

Use the optional `.github/ISSUE_TEMPLATE/bug_report.yml` if present. If it is absent, use the complete fallback below.
Show the exact title and body. Wait for explicit approval before creating the issue.
Review all text for credentials, private paths, private hostnames, account data, and personal data before submission.

```markdown
## Describe the bug

<clear description of the bug, the observed error, and the smallest failing case>

```python
<minimal code that reproduces the problem>
```

```text
<full error message or traceback>
```

## Reproduction

Save the script above as `repro.py`, then run:

```bash
python repro.py
```

Observed result:

```text
<result>
```

## Expected Behavior

<expected result and why it follows from the public API>

## Environment

Run this command and paste the reviewed output:

```bash
python -m torch_neuronx.collect_env
```

```text
<reviewed collect_env output>
```
```

After approval, create the issue against the explicit target repository:

```bash
gh issue create --repo "$REPO" --title '<approved title>' --body-file '<approved body file>'
```
