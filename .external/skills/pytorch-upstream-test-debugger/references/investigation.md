# Investigation guide

Use this guide after you reproduce one error-signature category.
Use each category as a starting hypothesis, not as a conclusion.

## Read source before you edit

Find the first `torch_neuronx/` frame in the traceback.
Trace the operation registration and its callers.
Find the highest product layer that creates the wrong behavior.

Prefer this order:

1. Python operation registration, decomposition, or cast policy.
2. Native dispatch code when Python cannot express the correction.
3. A compiler handoff when the product source is correct.

Do not edit a test framework to change product behavior.
Do not patch an installed PyTorch package.
Do not change a cloned upstream test.

Write this statement before an edit:

```text
The defect starts in <symbol> because <specific behavior> produces <observed signature>.
```

Continue reading if you cannot complete that statement.

## Group by stable evidence

A normalized signature removes volatile values such as addresses, line numbers, and large shape values.
It keeps the exception type and stable message text.

Inspect each selected group manually.
Two matching strings can still have different root causes.
Two different strings can still come from one bad source boundary.

Keep the proposed PR scope to one confirmed root cause.

## Dtype mismatch

A mismatch can start at a cast policy, fallback boundary, decomposition, or backward formula.
Trace the input dtype, operation dtype, result dtype, and gradient dtype separately.

Check whether the dtype is documented as supported.
Use public Neuron documentation as the authority.

If the dtype is supported, reproduce the mismatch with the smallest tensor.
Fix the source boundary that changes or preserves the wrong dtype.

If the dtype is not supported, keep the limitation visible.
Do not add a silent cast only to satisfy the test.

## Gradient or Jacobian mismatch

First compare a small forward result with a CPU reference.
Then compare the analytical gradient.
Separate a forward precision error from a backward formula error.

Do not increase a tolerance until you prove the product result meets its documented contract.
A test-only tolerance is not a source fix.

Check these causes:

- A cast occurs too early.
- An accumulation uses the wrong dtype.
- A decomposition has different edge behavior.
- The backward registration uses the wrong shape or dtype.
- A source operation takes a different dispatch path.

Use `compiler-artifact-repro` when product source produces the correct compiler input but execution is wrong.

## Timeout or hang

Reproduce with one process first.
Distinguish slow progress from a deadlock.

Check repeated compilation, changing input shapes, synchronization, and resource cleanup.
Do not reduce the test sample only to make the timeout disappear.

A test-size change is valid only when the original sample exceeds a documented test contract.
Record that evidence before you propose it.

## Compilation failure

Reduce the failing test to the smallest product-level reproduction.
Confirm the same operation succeeds on the CPU reference.

Inspect product source for an invalid graph, unsupported input, or wrong decomposition.
Fix it there when product source creates the invalid input.

Use `compiler-artifact-repro` when the product graph is valid and the compiler rejects it.
Keep the upstream test xfailed until the compiler defect is resolved.

## Unexpected success

Find whether the xfail comes from the repository registry or upstream PyTorch.

For a repository xfail, run the test with xfail handling disabled.
Remove the registry entry only after the selected test passes in final verification.

Do not edit an upstream expected-failure decorator.
Report an upstream mismatch separately when necessary.

## Overlay or environment mismatch

Suspect a mismatch when the traceback references a missing PyTorch helper.
Compare the repository overlay with the pinned PyTorch version.
Read the pin from `build-system.requires` in `pyproject.toml`.

Set `PYTORCH_ROOT` to an empty destination for an official `pytorch/pytorch` checkout at tag `v<torch-pin>`.
Extract the pin, clone the public source when needed, and accept its HTTPS or SSH remote form:

```bash
PYTORCH_PIN=$(python3 - <<'PY'
import re
import tomllib
with open("pyproject.toml", "rb") as stream:
    build_system = tomllib.load(stream)["build-system"]
    requirements = build_system.get("requires", [])
print(next(re.fullmatch(r"torch==(.+)", item).group(1) for item in requirements if item.startswith("torch==")))
PY
)
test -n "$PYTORCH_ROOT"
git clone --filter=blob:none --branch "v$PYTORCH_PIN" \
  https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT"
case "$(git -C "$PYTORCH_ROOT" remote get-url origin)" in
  https://github.com/pytorch/pytorch.git) ;;
  git@github.com:pytorch/pytorch.git) ;;
  *) echo 'BLOCKED: PYTORCH_ROOT is not the public pytorch/pytorch repository'; exit 1 ;;
esac
test "$(git -C "$PYTORCH_ROOT" describe --tags --exact-match HEAD)" = "v$PYTORCH_PIN"
```

Skip `git clone` only when `PYTORCH_ROOT` already contains that verified public checkout.
Verify both destination trees before replacement.
The destination trees are ignored by Git, so `git status` cannot verify them:

```bash
TEST_OVERLAY=tests/pytorch_tests_native/test
INTERNAL_OVERLAY=tests/pytorch_tests_native/torch/testing/_internal
if test -d "$TEST_OVERLAY"; then
  overlay_diff=$(diff -qr "$PYTORCH_ROOT/test" "$TEST_OVERLAY" || true)
  test "$overlay_diff" = "Only in $PYTORCH_ROOT/test: conftest.py"
fi
if test -d "$INTERNAL_OVERLAY"; then
  diff -qr "$PYTORCH_ROOT/torch/testing/_internal" "$INTERNAL_OVERLAY"
fi
```

Warn the user that the next command deletes and replaces both copied trees.
This includes the copied upstream test tree when present.
Show the command and require explicit confirmation immediately before running it:

```bash
python3 tests/pytorch_tests_native/clone_device_agnostic_tests.py --source-repo "$PYTORCH_ROOT"
```

Never run this helper without `--source-repo`.
If the project dependency is stale, propose a dependency update as separate scope.
Do not patch `.venv/`.

Return `BLOCKED` when the official checkout, clean destination, compatible pin, or confirmation is unavailable.

## Regression fence

Reuse an existing node ID when it is specific and fast.
Create a focused repository test when the upstream case is broad or slow.

The fence must fail before the fix.
The fence must pass after the fix.
The full affected test file must also pass.

Match the surrounding test style.
Avoid a new helper that only one short test uses.
Avoid unrelated parameter expansion.

## Surgical fix checklist

Before final verification, confirm:

- Each changed source line supports the root-cause statement.
- No test patch hides the original behavior.
- No unrelated rename or refactor is present.
- Native changes were built before the passing run.
- The narrow fence and its containing file pass.
- The source change does not add unsupported behavior by accident.
