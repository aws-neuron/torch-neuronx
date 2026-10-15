"""Shared Python configuration for all Bazel files."""

PythonVersion = struct(
    PY310 = "3.10",
    PY311 = "3.11",
    PY312 = "3.12",
    PY313 = "3.13",
)

DEFAULT_PYTHON_VERSION = PythonVersion.PY310

# SHA256 for PyTorch 2.13.0 CPU wheels (manylinux_2_28_x86_64)
TORCH_SHA256 = {
    PythonVersion.PY310: "f5cbb61180a9793d9e12fe115a2310d2600bd449dfb9a01ec5640e21359fa5ea",
    PythonVersion.PY311: "6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b",
    PythonVersion.PY312: "4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2",
    PythonVersion.PY313: "3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6",
}

def format_python_wheel_tag(python_version):
    """Format Python version for wheel tags (e.g., '3.10' -> 'cp310-cp310')."""
    version_no_dot = python_version.replace(".", "")
    return "cp{}-cp{}".format(version_no_dot, version_no_dot)

def get_torch_sha256(python_version):
    """Get SHA256 for PyTorch wheel for given Python version."""
    return TORCH_SHA256.get(python_version, TORCH_SHA256.get(DEFAULT_PYTHON_VERSION))
