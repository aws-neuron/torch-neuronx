"""Skip and expected failure lists for Neuron ops."""

# Use tests=None to skip for all test classes
NEURON_SKIP_OPS: dict[str, dict] = {
    # Broadcast/shape issues
    "clamp_min": {"tests": None, "reason": "StableHLO broadcast_dimensions mismatch"},
    "clamp_max": {"tests": None, "reason": "StableHLO broadcast_dimensions mismatch"},
    "remainder": {"tests": None, "reason": "neuronx-cc compilation error"},
    "as_strided": {"tests": None, "reason": "storage size out of bounds"},
    # sample.input is list not tensor
    "cat": {"tests": ["TestNeuronOpsBackward"], "reason": "sample.input is list not tensor"},
    "stack": {"tests": ["TestNeuronOpsBackward"], "reason": "sample.input is list not tensor"},
    # Storage/stride issues
    "unfold": {"tests": ["TestNeuronOpsBackward"], "reason": "storage size out of bounds"},
    # Gradient mismatches
    "sin": {"tests": ["TestNeuronOpsBackward"], "reason": "bfloat16 gradient mismatch"},
    "erfinv": {"tests": ["TestNeuronOpsBackward"], "reason": "bfloat16 gradient mismatch"},
    "_log_softmax": {"tests": ["TestNeuronOpsBackward"], "reason": "bfloat16 gradient mismatch"},
    "native_layer_norm": {"tests": None, "reason": "dtype mismatch bfloat16 vs float32"},
    # Correctness issues
    "addmm": {"tests": ["TestNeuronOpsCorrectness"], "reason": "correctness/dtype mismatch"},
    "constant_pad_nd": {
        "tests": ["TestNeuronOpsCorrectness"],
        "reason": "correctness/dtype mismatch",
    },
    "mm": {"tests": ["TestNeuronOpsCorrectness"], "reason": "dtype mismatch bfloat16 vs float32"},
    "floor_divide": {
        "tests": ["TestNeuronOpsCorrectness"],
        "reason": "correctness mismatch bfloat16",
    },
    "addcdiv": {"tests": ["TestNeuronOpsCorrectness"], "reason": "correctness mismatch bfloat16"},
    "div": {"tests": ["TestNeuronOpsCorrectness"], "reason": "correctness mismatch bfloat16"},
    "histc": {"tests": ["TestNeuronOpsCorrectness"], "reason": "correctness mismatch bfloat16"},
    "_native_multi_head_attention": {
        "tests": ["TestNeuronOpsCorrectness"],
        "reason": "correctness mismatch",
    },
    "_grouped_mm": {"tests": ["TestNeuronOpsCorrectness"], "reason": "NaN in output"},
    "nonzero_static": {"tests": ["TestNeuronOpsCorrectness"], "reason": "correctness mismatch"},
    # Uninitialized memory / device issues
    "new_empty": {"tests": None, "reason": "uninitialized memory / device mismatch"},
    "empty": {"tests": ["TestNeuronOpsCorrectness"], "reason": "uninitialized memory comparison"},
    "empty_strided": {
        "tests": ["TestNeuronOpsCorrectness"],
        "reason": "uninitialized memory comparison",
    },
    "ones_like": {"tests": None, "reason": "output storage not on Neuron device"},
    # Random output
    "native_dropout": {"tests": ["TestNeuronOpsCorrectness"], "reason": "random output comparison"},
}

NEURON_XFAIL_OPS: dict[str, dict] = {
    # Example:
    # "flaky_op": {"tests": ["TestNeuronOps", "TestNeuronOpsBackward"], "reason": ""},
}


def get_skip_reason(op_name: str, test_class_name: str) -> str | None:
    """Get skip reason if op should be skipped for given test class."""
    if op_name not in NEURON_SKIP_OPS:
        return None
    config = NEURON_SKIP_OPS[op_name]
    tests = config.get("tests")
    if tests is None or test_class_name in tests:
        return config.get("reason", "Skipped")
    return None


def get_xfail_reason(op_name: str, test_class_name: str) -> str | None:
    """Get xfail reason if op should be marked as expected failure."""
    if op_name not in NEURON_XFAIL_OPS:
        return None
    config = NEURON_XFAIL_OPS[op_name]
    tests = config.get("tests")
    if tests is None or test_class_name in tests:
        return config.get("reason", "Expected failure")
    return None
