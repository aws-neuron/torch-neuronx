"""Shared Python configuration for all Bazel files."""

PythonVersion = struct(
    PY310 = "3.10",
    PY311 = "3.11",
    PY312 = "3.12",
    PY313 = "3.13",
)

DEFAULT_PYTHON_VERSION = PythonVersion.PY310

# SHA256 for PyTorch 2.11.0 CPU wheels (manylinux_2_28_x86_64)
TORCH_SHA256 = {
    PythonVersion.PY310: "f378df648bf7fb94bbc820dfec37d3d346bd1703c692a2215edebf6edcef8b75",
    PythonVersion.PY311: "8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485",
    PythonVersion.PY312: "f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338",
    PythonVersion.PY313: "45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86",
}

def format_python_wheel_tag(python_version):
    """Format Python version for wheel tags (e.g., '3.10' -> 'cp310-cp310')."""
    version_no_dot = python_version.replace(".", "")
    return "cp{}-cp{}".format(version_no_dot, version_no_dot)

def get_torch_sha256(python_version):
    """Get SHA256 for PyTorch wheel for given Python version."""
    return TORCH_SHA256.get(python_version, TORCH_SHA256.get(DEFAULT_PYTHON_VERSION))
