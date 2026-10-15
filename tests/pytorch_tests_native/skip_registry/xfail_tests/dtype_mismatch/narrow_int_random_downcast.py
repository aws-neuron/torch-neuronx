# Bare/full-range random_ on signed narrow ints (int8/int16), C++ dispatch path.
#
# DowncastArgs widens int8/int16 -> int32 before random_'s factory runs, so the
# bare-range modulus becomes 2**31 instead of iinfo(dtype).max+1; casting the int32
# result back to the narrow dtype wraps negative and fails the [0, max] checks.
# Bounded random_(from, to) is unaffected (modulus from explicit args). Same class as
# DTYPE_MISMATCH_FLOAT64_DOWNCAST; proper fix is op-aware DowncastArgs that skips
# narrow-int widening for random_.
DTYPE_MISMATCH_NARROW_INT_RANDOM_DOWNCAST = [
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_default_neuron_int8",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_default_neuron_int16",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_full_range_neuron_int8",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_full_range_neuron_uint8",
]
