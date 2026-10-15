# random_ range-coverage xfails. The upstream range-coverage checks assert the
# drawn min/max fall within 10 percent of the range ends, but the on-device NKI
# RNG uses an int32 datapath, so draws cannot span past the int32 window: uint32
# upper half (draws cap near 2**31) and wide bf16/fp32/fp64 spans miss the tolerance
# deterministically (fp64 also runs on the 32-bit datapath). int8/uint8 (tiny) and
# int64 (CPU fallback) pass alone, miss only under -n64.
RANDOM_RANGE_COVERAGE = [
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_from_to_neuron_int32",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_from_to_neuron_int64",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_from_to_neuron_int8",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_from_to_neuron_uint32",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_from_to_neuron_uint8",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_full_range_neuron_float64",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_full_range_neuron_int64",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_full_range_neuron_uint32",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_to_neuron_bfloat16",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_to_neuron_float32",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_to_neuron_float64",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_to_neuron_int64",
    "test_tensor_creation_ops.py::TestTensorCreationNEURON::test_random_to_neuron_uint32",
]
