"""Utility functions for TorchNeuronx C++ tests."""

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

# Common dependencies and compiler options for all TorchNeuronx C++ tests
_TORCH_NEURONX_TEST_COPTS = [
    "-I.",
    "-Imocks",
    "-isystemexternal/torch",
]

_TORCH_NEURONX_TEST_DEPS = [
    "//tests/csrc:test_mocks",
    "@nrt//:nrt",
    "@nrt//:libnrt",
    "@torch//:headers",
    "@torch//:runtime_headers",
    "@torch//:libc10",
    "@torch//:libtorch",
    "@torch//:libtorch_cpu",
    "@torch//:libtorch_python",
    "@googletest//:gtest_main",
    "@rules_python//python/cc:current_py_cc_headers",
    "@rules_python//python/cc:current_py_cc_libs",
]

def torch_neuronx_cc_test(copts = [], deps = [], size = "small", timeout = "short", **kwargs):
    """Macro to create a TorchNeuronx C++ unit test with common configuration.

    Args:
        name: Name of the test target
        srcs: Source files for the test
        size: Test size (small, medium, large)
        timeout: Test timeout (short, moderate, long, eternal)
        **kwargs: Additional arguments passed to cc_test
    """
    native.cc_test(
        size = size,
        timeout = timeout,
        copts = copts + _TORCH_NEURONX_TEST_COPTS,
        deps = deps + _TORCH_NEURONX_TEST_DEPS,
        **kwargs
    )
