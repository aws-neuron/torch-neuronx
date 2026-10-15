# ATen lowering implementation lessons

Use these lessons after you inspect the current repository code.
Treat current code and upstream PyTorch behavior as authoritative.

## Confirm backend dispatch

A decomposition table entry does not prove that the backend uses it.
A higher-priority Python dispatch implementation can run first.
Inspect each overload's dispatch registrations.
Test routing as well as values.

A composite operator can already decompose into supported primitives.
Do not add a backend handler unless the operator reaches the backend.

## Keep one calculation source

Put reusable calculation logic in one decomposition or helper.
Call the ATen operator from the native handler.
Let tracing reuse the registered decomposition.
Do not copy the calculation into each overload.

Keep dtype promotion in the shared calculation.
The compile path can bypass checks that exist only in the eager handler.

## Group overloads carefully

Group functional and `out` overloads when they use identical metadata.
Keep an in-place overload in a separate handler.
An in-place handler must mutate and return its input.

Use separate handlers when static arguments or output metadata differ.
Registration metadata can apply to an entire function group.
Do not copy metadata from an adjacent overload without checking its schema.

## Keep runtime values dynamic

Mark a value static only when it changes graph structure.
An output size and a mode selector are typical static values.
A scalar used only in arithmetic is usually dynamic.

Too many static values cause unnecessary compilations.
Too few static values can reuse a graph with the wrong structure.
Test cache behavior when the distinction is important.

## Use value-dependent preprocessing when required

Use value-dependent preprocessing when tensor data changes the graph.
A shape-only key cannot distinguish equal shapes with different structural values.
Test two calls with equal input metadata and different relevant values.

## Preserve reference dtype rules

Target execution can use a narrower internal dtype than the input dtype.
Take range and saturation rules from the operator's accepted dtype.
Do not infer them from a widened traced tensor.

Describe a double input accurately.
It can compute through a narrower path and cast back.
Do not call this CPU fallback unless measurement shows CPU fallback.

## Preserve layout and alias behavior

A correct value with a wrong stride is a defect.
Use the PyTorch meta function to obtain required output strides.
Test contiguous and non-contiguous `out` tensors.
Test that `out` does not corrupt an unrelated input.

For scatter-shaped calculations, compare axis choices and data movement.
Prefer an implementation that avoids unnecessary transposes and copies.

## Build constants functionally

A Python list or a new CPU tensor can become an invalid lifted constant.
Build regular maps from device-local tensor operations when possible.
Test the calculation under autograd and compile tracing.

## Test the operator family

Add tests to the established family file.
A separate file is useful only for a new family.
Run both the focused test and the upstream operator tests.
A name filter can miss a transitive caller.

After dispatch table changes, run the unfiltered decomposition registry test.
A per-operator filter can omit global registry invariants.

## Prove backward support

Probe autograd before implementation.
A forward test does not prove that a dedicated backward operator runs natively.
Test the backward operator's dispatch and values separately.

A derived backward needs no new operator when all primitives are supported.
Still run gradient tests for the complete operation.

## Match reference edge behavior

Check the CPU result before you add a correction for NaN or infinity.
Keep a correction when it reproduces defined PyTorch behavior.
Do not use a correction to hide a target compiler defect.

A reference call that only raises can be necessary.
Use it when upstream tests require exact reference error text.
Explain the operator-specific reason in one short comment.

## Maintain registry quality

Group marks by root cause.
Use a skip for a test that cannot execute.
Use an xfail for a test that executes and returns a known wrong result.

Delete a stale mark when the lowering fixes its cause.
Run the affected node ID with and without xfail handling.
Do not use a general limitation to hide a supported-dtype defect.

## Keep documentation consistent

Update `docs/torch_api_compatibility.md` from measured execution.
Use the established note format.
Use `Native MLIR impl` for a native registration.
Do not invent a new phrase when the table already has one.

## Keep the task isolated

Use a separate git worktree for the task.
Run commands from that worktree so Python imports the intended source.
Confirm the active branch before a commit or push.
Keep build and cache artifacts out of the diff.
