---
name: compiler-artifact-repro
description: Isolates a torch-neuronx compiler failure or numerical miscompile, captures a standalone HLO and NEFF reproduction, validates the evidence, and renders a review-gated local issue draft. Use when a Neuron failure must be reduced below the PyTorch operation or model layer.
---

# Reproduce a compiler issue

Produce one minimal reproduction bundle and one local issue draft.
Prove that one selected HLO reproduces the symptom without PyTorch operation dispatch.
Keep the draft local until a human completes the review gates.

## Prerequisites

- Use a checkout of this repository.
- Follow `CONTRIBUTING.md` section "Setup Development Environment" to build the checkout.
- Use Python 3.10 or later for the helper scripts.
- Install the exact helper and test dependencies: `uv pip install 'PyYAML==6.0.2' 'pytest==8.3.5'`.
- Use a Trainium instance for capture and standalone validation.
- Run CPU-only helper tests without Trainium hardware.
- Hardware CI checks on a pull request can stay pending until a maintainer starts them (see the pull request template); if a check fails for an unrelated reason, comment on the pull request (see CONTRIBUTING.md).
- Record the failing command and its complete output.
- Record the `torch-neuronx` and `neuronx-cc` versions.
- Record the device type and operating system.
- Keep the original failure unchanged before reduction.

Define the repository that receives a pull request or issue:

```bash
REPO="${REPO:-aws-neuron/torch-neuronx}"   # the repository that receives the pull request or issue
```

Define one artifact directory from the repository root:

```bash
export ARTDIR='agent_space/<task-slug>/artifacts'
```

Use `"${ARTDIR}/..."` for each artifact path.
The capture helper creates the artifact directory and its capture subdirectories.
The helper also records the inherited launch mode in `"${ARTDIR}/launch_blocking.txt"`.

Use only these capture variables:

- `TORCH_NEURONX_DEBUG_DIR`
- `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`
- `TORCH_NEURONX_NEFF_CACHE_DIR`

This repository reads all three variables.
Do not add compiler flags that are absent from the original command or captured logs.

## Workflow

### 1. Confirm the symptom

Run the smallest existing test or program that shows the failure.
Record whether the symptom is a compile failure or a numerical mismatch.
Reject a skipped or expected failure as new evidence.
Reject a timeout, resource failure, or device setup failure as compiler evidence.
Repeat the case once with the same inputs.
Record whether the symptom is deterministic.

Use `pytorch-upstream-test-debugger` for an upstream test failure.
Use `aten-op-lowering` for a lowering failure before HLO compilation.

### 2. Add the capture helper

Copy the helper next to the reproduction program:

```bash
mkdir -p "${ARTDIR}"
cp ".external/skills/compiler-artifact-repro/scripts/artifact_dump.py" \
  "${ARTDIR}/artifact_dump.py"
```

Import the helper before `torch` and `torch_neuronx`:

```python
import artifact_dump

import torch
import torch_neuronx
```

The helper configures the three verified capture variables.
It creates the compiler dump, NEFF cache, and artifact root directories.
It records the inherited `NEURON_LAUNCH_BLOCKING` value before runtime import.
It does not add compiler command flags.

Save the fixed inputs for an accuracy case.
Save both the reference and actual outputs:

```python
artifact_dump.save_io(
    artifact_dump.ARTDIR,
    inputs={"input0": input0.cpu()},
    outputs={"reference0": reference0.cpu(), "actual0": actual0.cpu()},
)
```

This call writes `in_input0.pt`, `out_reference0.pt`, and `out_actual0.pt`.
Use small tensors when they preserve the symptom.
Never include user data or credentials.

### 3. Capture the failing launch mode

Run the reproduction with the helper imported first:

```bash
set -o pipefail
python3 "${ARTDIR}/reproduce.py" 2>&1 | tee "${ARTDIR}/run.log"
repro_status=${PIPESTATUS[0]}
printf '%s\n' "${repro_status}" > "${ARTDIR}/run.exitcode"
```

Use `"${ARTDIR}/run.exitcode"` as the reproduction process status.
Keep the complete command in the bundle.
Keep the final minimal reproduction program in `"${ARTDIR}/reproduce.py"`.
Keep the selected compiler work directory logs.
Expect `module.mlir` for a submitted compile.
Treat `module.mlir` as opaque binary data because it can contain MLIR bytecode.
Expect `module.neff` only after a successful compile.
A compile failure can have no NEFF.

Read `"${ARTDIR}/launch_blocking.txt"` before each reproduction.
Restore the recorded value in the same shell:

```bash
launch_mode=$(cat "${ARTDIR}/launch_blocking.txt")
if [[ "${launch_mode}" == '<unset>' ]]; then
  unset NEURON_LAUNCH_BLOCKING
else
  export NEURON_LAUNCH_BLOCKING="${launch_mode}"
fi
```

Use that shell for every validation run.

The default mode reports Neuron errors asynchronously.
An error can appear at a later synchronization point.
The traceback can identify detection instead of the failing dispatch.
Value `1` waits for each submitted operation.
It changes scheduling and error attribution.
Do not change the mode only to improve the traceback.

List all candidate HLO files:

```bash
python3 ".external/skills/compiler-artifact-repro/scripts/artifact_dump.py" \
  --artdir "${ARTDIR}" --list
```

### 4. Select one HLO

Use an HLO path from the error when one exists.
Otherwise compare operation names, shapes, constants, and timestamps.
Account for delayed errors in the default launch mode.
Use standalone validation to confirm the selection.
Stop if two candidates remain equally likely.
Do not package an ambiguous HLO set.

Check whether the selected HLO compiles from the bundle alone.
Custom calls can require an external kernel or another unavailable artifact.
Use `Outcome: BLOCKED` with `Reason: EXTERNAL_DEPENDENCY` when that dependency is absent.
List the missing dependency and the exact failed command.

### 5. Minimize the case

Remove model code that does not affect the selected HLO.
Replace random inputs with fixed inputs.
Reduce tensor shapes while the same symptom remains.
Remove unrelated outputs.
Keep data types and compiler inputs unchanged.
Keep the original HLO before each reduction.

For an accuracy case, save the expected output from a CPU computation.
Also save the actual output from the failing run.
Take `atol` and `rtol` from the originating test or a cited public source.
Record that source with both values before validation.
Do not invent default tolerances.
Do not increase tolerances to make the comparison pass.

### 6. Validate the HLO

Run compile-error mode with an exact public error fragment:

```bash
set -o pipefail
python3 ".external/skills/compiler-artifact-repro/scripts/validate_hlo.py" \
  "${ARTDIR}/compiler_dump/neuronx-cc/<id>/module.mlir" \
  --artdir "${ARTDIR}" \
  --expect-error-regex '<expected-error-fragment>' \
  | tee "${ARTDIR}/validator-result.json"
validator_status=${PIPESTATUS[0]}
printf '%s\n' "${validator_status}" > "${ARTDIR}/validator-result.exitcode"
```

Run accuracy mode with each input and output description:

```bash
set -o pipefail
python3 ".external/skills/compiler-artifact-repro/scripts/validate_hlo.py" \
  "${ARTDIR}/compiler_dump/neuronx-cc/<id>/module.mlir" \
  --artdir "${ARTDIR}" \
  --input "${ARTDIR}/in_input0.pt" \
  --out-shape '8x16' --out-dtype 'float32' \
  --ref "${ARTDIR}/out_reference0.pt" \
  --atol '<absolute-tolerance>' --rtol '<relative-tolerance>' \
  | tee "${ARTDIR}/validator-result.json"
validator_status=${PIPESTATUS[0]}
printf '%s\n' "${validator_status}" > "${ARTDIR}/validator-result.exitcode"
```

Read `"${ARTDIR}/validator-result.exitcode"` before you interpret the JSON.
Exit `0` means the expected standalone symptom reproduced.
Exit `1` means the expected standalone symptom did not reproduce.
Exit `2` means the input or environment is invalid.

The validator configures `compiler_dump` and `neff_cache` below `--artdir` before runtime loading.
Pass `--artdir` or set `ARTDIR` for every validator run.
The validator calls `compile_graph` with the HLO bytes.
Its default key uses the `validate_hlo` prefix and the HLO SHA-256 value.
It calls `execute_compiled_graph` with the returned cache key in accuracy mode.
It bypasses PyTorch operation dispatch and Python decomposition.
It emits strict JSON.
It emits `null` for a nonfinite difference metric.

### 7. Validate the bundle

Keep only the selected HLO and related evidence.
Keep the NEFF when compilation completed.
Keep the compiler logs for the selected HLO.
Keep fixed inputs, actual outputs, and reference outputs for an accuracy case.
Keep the launch mode, reproduction script, and exact command.
Keep version and environment output.

Record a version command and its output:

```bash
printf '%s\n' 'Command: python3 -m pip show torch-neuronx torch neuronx-cc' \
  > "${ARTDIR}/versions.txt"
python3 -m pip show torch-neuronx torch neuronx-cc \
  >> "${ARTDIR}/versions.txt"
```

The first line must start with `Command: `.
Prepare `"${ARTDIR}/issue-spec.json"` with the reference schema.
Use artifact-relative paths inside the JSON fields.
The renderer verifies the validator state, selected HLO path, and HLO SHA-256 value.
The renderer uses the repository form `.github/ISSUE_TEMPLATE/bug_report.yml` if present.
Otherwise, it uses the shipped [fallback issue form](references/bug-report.md).
Use `--issue-template PATH` only to override both forms.

Render a local draft and a review-gated archive:

```bash
python3 ".external/skills/compiler-artifact-repro/scripts/render_issue.py" \
  --artdir "${ARTDIR}" --spec "${ARTDIR}/issue-spec.json" \
  --draft "${ARTDIR}/issue-draft.md" \
  --archive "${ARTDIR}/compiler-repro.tar.gz" \
  --max-archive-bytes '52428800'
```

Add `--boundary-rules "${ARTDIR}/rules.json"` for each extra rule file.
Use different draft and archive paths.
Do not point either output at or below a required evidence file.
The renderer stages both outputs outside `${ARTDIR}`.
It excludes final output paths and `.compiler-issue-*` staging names from the archive.
The renderer redacts non-HLO text and records the result in each manifest review field.
It archives the selected HLO and good HLO evidence without byte changes.
It stops if either required HLO contains content that requires redaction.
It runs the repository public-boundary scanner on each prepared text artifact.
It denies unlisted binary types.
It marks allowed binaries for human review before public upload.

Review every excluded file in the renderer result.
Review the archive manifest and checksums.
Open the archive on a clean CPU-only host.
Run the reproduction command when Neuron hardware is available.

### 8. Prepare human review

Show the complete local draft to the human.
Show the archive manifest to the human.
Show all redactions and exclusions to the human.
State that the helpers made no network request.
Show the empty human commentary section.
Leave every human checklist item unchecked.

Before any filing action, require these human affirmations:

- [ ] I read the complete reproducer and issue draft.
- [ ] I understand the reproducer and its evidence.
- [ ] I take responsibility for this contribution.
- [ ] I reviewed and approved the exact filing content.

The agent must not complete these items for the human.
The agent must never file the issue or upload the archive.
A human can file only after all affirmations are complete.

Default to report-only for all git and GitHub work.
Commit only when the user asks.
Before any GitHub write, show the exact content and wait for explicit approval.
Open a pull request only as a draft.
Never merge it, mark it ready, force-push, or push to `main`.
Disclose AI assistance in the pull request description.

## Verification

Run the CPU-only helper tests from the repository root:

```bash
python3 -m pytest -o addopts='' -p no:cacheprovider \
  ".external/skills/compiler-artifact-repro/tests" -q
```

Run each helper with `--help`.
Confirm that each command exits with status zero.
Run the repository public-boundary scanner when it exists.
Run the standalone validator on a supported Neuron host.
Record each command, exit code, and output.

A valid bundle has all these properties:

- One selected `module.mlir` reproduces the exact symptom.
- The selected compiler logs are present.
- The NEFF is present when compilation succeeded.
- An accuracy case contains fixed inputs, actual outputs, and reference outputs.
- The recorded launch mode matches the failing run and validation runs.
- The archive manifest matches every payload checksum.
- The draft and archive contain no private data.

## Failure handling

Use `BLOCKED` when Neuron hardware, a build, or a credential is missing.
Stop the hardware workflow after this outcome.
Report what you could not run.
Report the exact command that another user must run.
Do not claim standalone reproduction from CPU-only tests.

Use `NOT_REPRODUCED` when exit `1` shows that the selected HLO lacks the symptom.
Return to HLO selection or report that the compiler hypothesis lacks evidence.

Use `INVALID_ARGUMENTS` when exit `2` reports an invalid file or option.
Correct the arguments before you interpret compiler behavior.

Use `INVALID_ENVIRONMENT` when exit `2` reports a runtime or device failure.
Correct the environment before you interpret compiler behavior.

Use `Outcome: BLOCKED` with `Reason: AMBIGUOUS_HLO` when two HLO files remain plausible.
Collect better attribution before you prepare a draft.

Use `Outcome: BLOCKED` with `Reason: EXTERNAL_DEPENDENCY` for a missing custom-call dependency.
Report the custom call, missing artifact, and exact standalone command.
Do not claim a standalone compiler reproduction.

Exclude a file when redaction and boundary scanning cannot make it safe.
Record the file name and reason in the manifest.
Stop if a required HLO, log, tensor, or launch-mode file is excluded.

## Output

Return these results:

- Outcome: `REPRODUCED`, `NOT_REPRODUCED`, `INVALID_ARGUMENTS`, `INVALID_ENVIRONMENT`, or `BLOCKED`.
- Blocked reason when the outcome is `BLOCKED`.
- Symptom type and exact expected signature.
- Selected HLO path.
- NEFF path or the reason no NEFF exists.
- Compiler log paths.
- Recorded `NEURON_LAUNCH_BLOCKING` value.
- Standalone validator command, recorded exit code, and JSON output.
- Input, actual-output, and reference-output paths for accuracy cases.
- Archive path, byte size, SHA-256 value, and verification result.
- Excluded files and reasons.
- Local issue draft path.
- Commands that still require Neuron hardware.
- Unchecked human affirmation status.

Read [references/reproduction-reference.md](references/reproduction-reference.md) for the detailed evidence rules.
Use the [fallback issue form](references/bug-report.md) when the repository form is absent.
Use `.external/skills/compiler-artifact-repro/scripts/artifact_dump.py` for capture setup and tensor storage.
Use `.external/skills/compiler-artifact-repro/scripts/validate_hlo.py` for standalone validation.
Use `.external/skills/compiler-artifact-repro/scripts/render_issue.py` only for local output.
