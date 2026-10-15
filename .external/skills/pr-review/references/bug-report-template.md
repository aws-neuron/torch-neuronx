# Bug Report Fallback Template

Use the repository form `.github/ISSUE_TEMPLATE/bug_report.yml` if present. Use this fallback only when that file is absent.

## Describe the bug

Describe the failure clearly.
Include the complete error or traceback after removing sensitive data.

## Reproduction

Provide the smallest script and exact command that reproduce the failure.
Use as little hardware as the failure allows.

```python
import torch
import torch_neuronx

# Add the minimal reproduction.
```

```text
<exact command>
```

## Expected Behavior

Describe the expected result.

## Environment

Run this command and paste its output after removing sensitive data:

```bash
python -m torch_neuronx.collect_env
```
