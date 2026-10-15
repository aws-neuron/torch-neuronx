"""Test the after-fork() guard on the Neuron runtime.

A process that initialized the runtime then fork()s leaves the child with
invalid inherited NRT handles. A device op in the child must fail fast with a
clear error, not deadlock. We run the scenario in a fresh subprocess (forking
the pytest process itself would poison the test runner) and assert the child
raised the expected error.
"""

import subprocess
import sys

# Child program: init the runtime in the parent, fork, then use the device in
# the child. Prints a sentinel line describing the child's outcome.
_FORK_PROG = r"""
import os
import sys
import torch
import torch_neuronx  # noqa: F401

# Initialize the runtime in the parent.
x = torch.ones(8, device="neuron", dtype=torch.bfloat16)
_ = (x + x).cpu()

pid = os.fork()
if pid == 0:
    # os._exit() skips atexit/stdout flushing, so flush explicitly (and use
    # flush=True) or the child's report is lost from captured stdout.
    try:
        y = torch.ones(8, device="neuron", dtype=torch.bfloat16)
        _ = (y + y).cpu()
        print("CHILD_OUTCOME: NO_ERROR", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"CHILD_OUTCOME: RAISED {type(e).__name__}: {e}", flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)
else:
    os.waitpid(pid, 0)
    print("PARENT_DONE", flush=True)
"""


def test_fork_after_init_raises_not_deadlocks():
    # Using the Neuron device in a process forked after runtime init must raise a
    # clear error naming fork(), and must return (not hang) within the timeout.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FORK_PROG],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "child did not return within 180s (device op in forked child hung)"
        ) from None
    out = proc.stdout
    assert "PARENT_DONE" in out, f"parent did not finish; stderr:\n{proc.stderr}"
    assert (
        "CHILD_OUTCOME: RAISED" in out
    ), f"child did not raise as expected; got:\n{out}\nstderr:\n{proc.stderr}"
    # The error must name fork() as the cause and point at the workaround.
    assert "fork()" in out
    assert "spawn" in out or "num_workers=0" in out
