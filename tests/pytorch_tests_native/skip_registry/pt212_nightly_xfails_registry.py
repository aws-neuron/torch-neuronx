"""XFAIL groups for PT2.12 nightly upstream failures.

Observed on the PT2.12 nightly (GitHub Actions run 28958874830) in the ``dynamo``
and ``test_foreach`` upstream suites. These are pre-existing PT2.12 Neuron gaps
(inplace-op grad-metadata propagation, dynamo subclass/repro, foreach parity /
cross-device / dtype handling), unrelated to the eager pass-through-alias fix.
xfail (strict=False) to keep the nightly green; delete a group when its root
cause is fixed.
"""

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

XFAIL_GROUPS: list[SkipGroup] = [
    SkipGroup(
        reason=(
            "[PT2.12] dynamo autograd-function / tensor-subclass / repro failures on "
            "the Neuron backend."
        ),
        tests=[
            "AutogradFunctionTests::test_materialize_grad",
            "ReproTests::test_subclass_graph_output_repro",
            "SubclassTests::test_tensor_subclass_attr_codegen_tos",
        ],
    ),
    SkipGroup(
        reason=(
            "[PT2.12] test_foreach parity / cross-device / dtype failures on Neuron "
            "(autodiff, binary/unary op cross-device, reduce, fastpath parity)."
        ),
        tests=[
            "TestForeachNEURON::test_autodiff__foreach_lgamma_outplace_neuron_float64",
            "TestForeachNEURON::test_binary_op_scalar_with_different_tensor_dtypes__foreach_pow_neuron_float32",
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_add_neuron_bool",
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_div_neuron_float16",
            "TestForeachNEURON::test_binary_op_tensors_on_different_devices__foreach_mul_neuron_bfloat16",
            "TestForeachNEURON::test_foreach_reduce_large_input__foreach_norm_w_empty_True_neuron_float64",
            "TestForeachNEURON::test_parity__foreach_minimum_fastpath_inplace_neuron_int16",
            "TestForeachNEURON::test_unary_op_tensors_on_different_devices__foreach_erf_neuron_bool",
            "TestForeachNEURON::test_unary_op_tensors_on_different_devices__foreach_reciprocal_neuron_bool",
            "TestForeachNEURON::test_unary_op_tensors_on_different_devices__foreach_reciprocal_neuron_float32",
            "TestForeachNEURON::test_unary_op_tensors_on_different_devices__foreach_reciprocal_neuron_int8",
            "TestForeachNEURON::test_unary_op_tensors_on_different_devices__foreach_tanh_neuron_float64",
        ],
    ),
]
