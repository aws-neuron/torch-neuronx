# Platform limitation policy

Use this file to classify a new skip or xfail.
Fix the lowering unless evidence proves a platform limitation.
The verifier JSON uses `verdict: PASS|FAIL|BLOCKED`.
The verifier emits only `reason: ENVIRONMENT` because it can detect only runner and report infrastructure failures.
The workflow report uses `REASON: HARDWARE` or `REASON: DECISION` for failures detected before verification.

The acceptance script reads the stable IDs and regexes below.
The reason regex must match the registry reason.
For a new xfail, `--runxfail` disables the xfail mark. Require a failed or errored runxfail outcome whose failure text matches the limitation signature.
For a new skip, `--runxfail` does not disable the skip mark. Require the runxfail outcome to be exactly `skipped`. Reject `passed`, `failed`, `error`, `xfailed`, and `xpassed`. Also require the unchanged baseline report, which has no new mark, to show `failed` or `error` for the same node. Require that baseline failure text to match the limitation signature.
A failed or errored baseline with `ModuleNotFoundError`, `ImportError`, `No module named`, `device not available`, `collection error`, or `INTERNALERROR` is environment evidence, not an operator baseline. The verifier returns `BLOCKED` with `reason: ENVIRONMENT`.
Matching text never grants acceptance by itself.
Pass `--accept-limitation <node_id>=<limitation-id>` for each exact node ID.

## Documented limitations

- `complex-dtype` | reason: `complex(?: dtype)?.*(?:unsupported|not supported)` | failure signature: `complex(?: dtype)?.*(?:unsupported|not supported|not implemented)`
- `sparse-data-model` | reason: `sparse(?: tensor| layout| data model)?.*(?:unsupported|not supported|not implemented)` | failure signature: `sparse(?: tensor| layout| data model)?.*(?:unsupported|not supported|not implemented)`
- `float64-precision` | reason: `(?:float64|double).*(?:precision|unsupported|not supported|slow)` | failure signature: `(?:float64|double).*(?:precision|unsupported|not supported|slow|mismatch)`
- `quantized-data-model` | reason: `quantiz(?:ed|ation).*(?:unsupported|not supported|not implemented)` | failure signature: `quantiz(?:ed|ation).*(?:unsupported|not supported|not implemented)`
- `nested-tensor-data-model` | reason: `nested[ -]tensor.*(?:unsupported|not supported|not implemented)` | failure signature: `nested[ -]tensor.*(?:unsupported|not supported|not implemented)`
- `mkldnn-not-applicable` | reason: `(?:mkldnn|onednn).*(?:not applicable|unsupported)` | failure signature: `(?:mkldnn|onednn).*(?:not applicable|unsupported|not supported)`
- `cuda-only-test` | reason: `cuda.*(?:specific|only|not applicable)` | failure signature: `cuda.*(?:specific|only|not applicable|unavailable|required)`
- `upstream-test-defect` | reason: `upstream test defect.*https://github.com/pytorch/pytorch/(?:issues|pull)/[0-9]+` | failure signature: `upstream test defect.*(?:assert|expect|reference|fixture)`
- `missing-dependency-operator` | reason: `missing dependency operator.*https://github.com/aws-neuron/torch-neuronx/issues/[0-9]+` | failure signature: `(?:missing dependency operator|aten::[A-Za-z0-9_.]+).*(?:not implemented|unsupported|not supported)`

Use a documented reason only after reproducing the limitation.
Add a public issue link when the reason regex requires one.
Keep the mark limited to the affected dtype, layout, or test case.
Do not change an ID or signature to make one task pass.
A new limitation or changed signature requires human review.

## Denied reason patterns

- `(?:not close|wrong value|numeric mismatch)`
- `integer.*(?:fail|mismatch|wrong)`
- `(?:crash|arity|dispatch.*fail)`
- `(?:cpu fallback|falls back to cpu)`
- `(?:layout|stride|non.contiguous)`
- `flaky`
- `fewer dtype`
- `opinfo.*(?:broken|generic)`

A denied reason identifies a lowering defect or incomplete investigation.
Do not add a skip or xfail for it.
Fix the implementation or report `RESULT: BLOCKED` with `REASON: DECISION`.

## Classification rules

Use a skip when the test cannot run safely.
Use an xfail when the test runs and reaches a known platform limit.
Group entries by root cause.
Do not group unrelated failures by operator name.
Delete a complete group when its root cause is fixed.

A supported integer mismatch is a defect.
A supported floating-point mismatch is a defect until evidence proves otherwise.
A CPU fallback is incomplete lowering work.
A layout mismatch is incomplete lowering work.
A crash is incomplete lowering work.

## Required evidence

Record these items for a new mark:
- Exact node ID and stable limitation ID.
- Exact normal and `--runxfail` commands.
- Observed and unchanged-base results.
- Smallest input, dtype, shape, and root cause.
- Public issue or existing public limitation.
- Narrow registry pattern and matched failure signature.

Run the marked test normally and with `--runxfail`.
For an xfail, confirm that the runxfail report shows `failed` or `error` with the documented signature.
For a skip, confirm that the runxfail report still shows exactly `skipped`. Confirm that the unchanged baseline shows `failed` or `error` with the documented signature.
The verifier returns `FAIL` without exact-node acceptance.
After a human approves the limitation, pass one `--accept-limitation` value for that node.
The verifier accepts only a newly marked node with a matching ID, reason, and failure signature.
List every accepted skip or xfail in the draft pull request.
Ask the human to review the list before any GitHub write.

## Float64 guidance

Target execution can use a narrower internal floating-point dtype.
This fact does not justify a broad xfail.
Prove that the mismatch is limited to float64.
Confirm that supported lower-precision dtypes pass.
Mark only the affected tests.
Do not claim CPU fallback without dispatch evidence.

## Complex guidance

Do not claim partial complex support when semantics cannot be preserved.
Match the upstream exception when the public API rejects the input.
Mark only tests that require an unsupported complex result.

## Upstream test defects

Treat upstream tests as the source of truth by default.
Report `RESULT: BLOCKED` with `REASON: DECISION` when a test appears incorrect.
Require a public PyTorch issue or pull request before an upstream-defect mark.
Do not change the overlay test.

## Compiler defects

Compare the target output with the CPU reference.
Reduce the graph to the smallest reproducer.
Use `compiler-artifact-repro` to package the evidence.
Do not add operations that hide an incorrect target result.
A compiler defect can produce `RESULT: BLOCKED` with `REASON: DECISION`.
It does not automatically justify a registry mark.
Use a public issue and the narrowest node IDs.
