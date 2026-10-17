# Compiler reproduction reference

Use this reference after the main workflow selects one candidate HLO.
Keep `../SKILL.md` as the controlling procedure.

## Verified capture settings

The capture helper configures these variables:

| Variable | Result |
|---|---|
| `TORCH_NEURONX_DEBUG_DIR` | Selects the compilation artifact root. |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1` | Preserves each compilation work directory. |
| `TORCH_NEURONX_NEFF_CACHE_DIR` | Selects an isolated NEFF cache for the run. |

Set them before you import `torch_neuronx`.
The helper creates the artifact root, compiler dump, and NEFF cache directories.
The helper records the inherited launch mode in `${ARTDIR}/launch_blocking.txt`.
Do not infer extra compiler flags from older examples.
Use only flags already present in the captured command.

The expected artifact tree has this form:

```text
${ARTDIR}/
  compiler_dump/
    torch_neuronx/
      rank<rank>/
        <job>/
          fx_graphs/
          raw/
          stablehlo/
    neuronx-cc/
      <id>/
        module.mlir
        module.neff
        <compiler log files>
  neff_cache/
  in_<name>.pt
  out_actual<index>.pt
  out_reference<index>.pt
  launch_blocking.txt
  run.log
  run.exitcode
  validator-result.json
  validator-result.exitcode
```

The exact files can vary by software version.
Use the files produced by the captured run.
Do not require a NEFF after a compile failure.
Treat `module.mlir` as opaque binary data.
An MLIR file can contain MLIR bytecode despite its suffix.

## Launch-mode attribution

Preserve the failing run's `NEURON_LAUNCH_BLOCKING` value.
Use the same value for capture, selection, and standalone validation.
The helper writes `<unset>` when the variable is absent.
Restore `<unset>` by unsetting the variable.
Do not replace the unset state with an empty value.

The default mode dispatches operations without waiting for each result.
A Neuron error can appear at a later synchronization point.
The traceback can identify error detection instead of the failing dispatch.
Value `1` waits for the submitted operation result.
This wait changes scheduling and error attribution.
Use value `1` only when the failing run used value `1`.

Repository source verifies this behavior.
See `torch_neuronx/csrc/c10/neuron/NeuronStream.cpp` and `torch_neuronx/csrc/core/utils/ErrorHandler.cpp`.

## HLO selection rules

Select the HLO with the strongest available evidence.
Use this order:

1. Use an exact HLO path from the error.
2. Match the operation name in the error and HLO.
3. Match unique input shapes or constants.
4. Match the compile timestamp to the failing operation.
5. Run standalone validation for each remaining candidate.

Account for asynchronous error detection during steps one through four.
Stop when two candidates produce the same evidence.
Report `Outcome: BLOCKED` with `Reason: AMBIGUOUS_HLO`.
Never choose the newest file without other evidence.

A selected HLO can depend on a custom call or external kernel.
Try compilation only with the bundle contents.
Stop when the dependency is not in the bundle.
Report `Outcome: BLOCKED` with `Reason: EXTERNAL_DEPENDENCY`.
Name the missing dependency and include the exact failed command.
Do not describe this result as a standalone compiler reproduction.

## Standalone validation contract

The validator reads the HLO as bytes.
It requires `--artdir` or the `ARTDIR` environment variable.
It configures the compiler dump and NEFF cache below that artifact root.
It completes this configuration before it imports `torch_neuronx`.
It passes those bytes to `compile_graph`.
Its default key uses the `validate_hlo` prefix and the HLO SHA-256 value.
The optional `--key` value replaces only that prefix.
It saves the cache key returned by compilation.
It passes that cache key to `execute_compiled_graph`.
It never substitutes a user label for the returned cache key.
It loads Neuron modules only after local argument validation.
It parses every output shape and validates every output data type before runtime loading.
It prints one strict JSON object.
A nonfinite difference metric appears as JSON `null`.
It uses these exit codes:

| Code | State | Meaning |
|---:|---|---|
| 0 | `REPRODUCED` | The exact expected error matched, or values exceeded the stated tolerances. |
| 1 | `NOT_REPRODUCED` | Compilation succeeded, another error occurred, or values stayed within tolerance. |
| 2 | `INVALID_ARGUMENTS` or `INVALID_ENVIRONMENT` | A file, option, module, device, compile, execution, shape, or data type was invalid. |

Each validator pipeline must enable `pipefail`.
Capture `${PIPESTATUS[0]}` immediately after `tee`.
Write that value to `${ARTDIR}/validator-result.exitcode`.
Interpret the status file before you interpret the JSON.

Compile-error mode requires `--expect-error-regex`.
Use a stable and public fragment from the observed message.
A different exception is not reproduction evidence.

Accuracy mode requires one or more `--input` options.
It requires equal counts of output shapes, data types, and reference files.
Specify an input cast as `FILE:TYPE` only when the HLO input requires it.

Accuracy mode requires explicit `--atol` and `--rtol` options.
Take both values from the originating test or a cited public source.
Record the source with the command.
The validator has no default numerical tolerances.

Floating outputs use `torch.isclose` with `equal_nan=True`.
A one-sided NaN is always a mismatch.
An infinity mismatch is always a mismatch.
An opposite infinity sign is always a mismatch.
Integer and Boolean outputs use exact equality.
Scalar outputs use `--out-shape scalar` or `--out-shape '[]'`.
A zero-size dimension is valid.
A negative or malformed dimension is invalid.

## Reduction rules

Preserve the selected HLO before each change.
Change one property at a time.
Use fixed values that make a mismatch easy to inspect.
Prefer small shapes that preserve the exact error or mismatch.
Repeat each candidate three times when nondeterminism is possible.

A useful reduction removes these elements:

- Model loading.
- Dataset access.
- Training loops.
- Unrelated operations.
- Random input generation.
- Unused outputs.

A valid reduction preserves these elements:

- Input order.
- Input data types.
- Output shapes and data types.
- Compiler version.
- Device type.
- Launch mode.
- The exact error fragment or numerical mismatch.

## Bundle contents

Every bundle contains:

- The selected `module.mlir`.
- Compiler logs from its work directory.
- The minimal reproduction program.
- The exact reproduction and validation commands in `issue-spec.json`.
- The complete reproduction output in `run.log`.
- The software versions and device type.
- The validator JSON result and `validator-result.exitcode` process status.
- The recorded `NEURON_LAUNCH_BLOCKING` value in `launch_blocking.txt`.

Add `module.neff` when compilation succeeded.
Add fixed input tensors for an accuracy case.
Add actual and reference output tensors for an accuracy case.
Use `out_actual*.pt` for actual outputs.
Use `out_reference*.pt` or `out_ref*.pt` for reference outputs.
Add the stated tolerances and their originating test or public citation.
Add the last known-good compiler when the case is a regression.
Explain an unknown regression status when no comparison result exists.
Add a good-compiler HLO for an accuracy regression when available.

Exclude unrelated HLO files.
Exclude caches that the reproduction does not need.
Exclude large outputs after you create a smaller equivalent case.
Exclude all user data, secrets, private paths, and private host names.

## Issue spec schema

Create this JSON object in `${ARTDIR}/issue-spec.json`:

```json
{
  "category": "Accuracy",
  "regression": "Unknown",
  "regression_unknown_explanation": "No earlier compiler result is available.",
  "summary": "The compiled HLO produces a different output",
  "description": "A standalone HLO reproduces the mismatch.",
  "selected_hlo": "compiler_dump/neuronx-cc/<id>/module.mlir",
  "exact_error": "No compile error. Output 0 differs.",
  "reproduction_command": "python3 reproduce.py",
  "validation_command": "python3 validate_hlo.py module.mlir --artdir . --input in_input0.pt --out-shape 8x16 --out-dtype float32 --ref out_reference0.pt --atol <value> --rtol <value>",
  "repro_program": "reproduce.py",
  "validator_result": "validator-result.json",
  "repro_exit_code_file": "run.exitcode",
  "versions_file": "versions.txt",
  "environment": "Paste the public package versions and device type.",
  "impact": "Describe the blocked public use case.",
  "evidence": [
    "Output 0 has mismatched values.",
    "The stated tolerances come from the originating test."
  ]
}
```

Use artifact-relative paths in this JSON object.
Use `Compile failure` or `Accuracy` for `category`.
Use `Yes`, `No`, or `Unknown` for `regression`.
Add `last_known_good_compiler` when `regression` is `Yes`.
Add `regression_unknown_explanation` when `regression` is `Unknown`.
Add `good_hlo_evidence` when a good HLO is available.
The renderer rejects extra fields.
The renderer requires each required text field to be nonempty.
The renderer requires a nonempty evidence list.
The reproduction program must be a `.py` file.
The validator result must contain the `validate_hlo.py` JSON object.
Its state must be `REPRODUCED`.
Its HLO path and SHA-256 value must match the selected HLO.
The reproduction exit code file must contain one integer.
The versions file must start with `Command: ` and contain command output.
Use a `pip show` command or a Python import command for version evidence.
The selected HLO path and good HLO path must stay in `${ARTDIR}`.

## Local issue shape

The renderer reads the bug report issue form at runtime.
It uses `--issue-template PATH` when given.
Else it uses the repository form `.github/ISSUE_TEMPLATE/bug_report.yml` if present.
If that form is absent, it uses `references/bug-report.md` from this skill.
It creates one section for each non-markdown form field.
It uses the form order.
It also lists each required field for human confirmation.

The renderer creates this draft shape:

```markdown
# Issue field plan

## Suggested issue title

> [Compiler] <summary>
> AI-assistance disclosure: ...
> Status: DRAFT_READY_FOR_HUMAN_REVIEW

## Field 1: <label> (`<id>`)
## Field 2: <label> (`<id>`)
## Required fields for human confirmation
## Human commentary (human only)
```

The renderer creates files only on the local host.
It does not open a network connection.
It does not file an issue.
The renderer puts agent analysis in quote blocks or code blocks.
The renderer leaves the human commentary section empty.

## Redaction and archive rules

The renderer redacts each UTF-8 text artifact except required HLO evidence.
It records whether redaction occurred in each text artifact's manifest review field.
It preserves the selected HLO and good HLO evidence byte-for-byte.
It stops if required HLO evidence contains content that requires redaction.
It scans each prepared text artifact with `tools/agent-skills/scan_public_boundary.py`.
Pass each extra JSON rule file with `--boundary-rules`.
The renderer denies binary files by default.
It permits only required reproduction evidence types.
It handles MLIR bytecode as binary evidence.
It marks each permitted binary for human review before upload.
It excludes symbolic links.
It normalizes each archive path.
It rejects path traversal and nonportable archive path syntax.
It stages draft and archive files outside `${ARTDIR}`.
It excludes final output paths and `.compiler-issue-*` staging names.
The draft and archive paths must differ.
Neither output may resolve to or below a required evidence file.

The archive contains `manifest.json` and `SHA256SUMS`.
The manifest lists each included and excluded file.
Each included file has its stored size and SHA-256 value.
The archive verifier rejects missing, extra, changed, linked, or unsafe members.

The renderer removes only optional payloads to meet the size limit.
It refuses the package when required evidence is removed.
Required evidence includes the selected HLO and compiler logs.
A successful compile also requires its NEFF.
An accuracy case requires input, actual-output, and reference-output tensors.
A successful result has state `DRAFT_READY_FOR_HUMAN_REVIEW`.

## Human review checklist

Leave every item unchecked until a human completes it.

- [ ] I reviewed the exact draft.
- [ ] I reviewed every manifest entry.
- [ ] I reviewed every excluded file and reason.
- [ ] I verified the archive checksum.
- [ ] I searched the extracted archive for private markers.
- [ ] I confirmed that one HLO reproduces the exact symptom.
- [ ] I confirmed that the NEFF is present after successful compilation.
- [ ] I confirmed that accuracy tensors contain no user data.
- [ ] I confirmed that the draft has no unsupported root-cause claim.
- [ ] I read and understand the complete reproducer.
- [ ] I take responsibility for the contribution.
- [ ] I approve the exact filing content.

The agent must not complete these items for the human.
The agent must never upload the archive or file the issue.
A human can file only after all checklist items are complete.
