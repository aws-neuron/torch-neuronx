"""XFAIL groups for PT2.12 nightly upstream failures.

Observed on the PT2.12 nightly (GitHub Actions run 28958874830) in the
``test_foreach`` upstream suite. These are pre-existing PT2.12 Neuron gaps
(foreach parity / cross-device / dtype handling). xfail (strict=False) to keep
the nightly green; delete a group when its root cause is fixed.
"""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[PT2.12] test_foreach parity / cross-device / dtype failures on Neuron "
            "(autodiff, binary/unary op cross-device, reduce, fastpath parity)."
        ),
        tests=[
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_add_neuron_bool",
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_div_neuron_float16",
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_mul_neuron_bfloat16",
        ],
    ),
]
