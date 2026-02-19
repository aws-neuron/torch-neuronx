"""Shared Python configuration for all Bazel files."""

PythonVersion = struct(
    PY310 = "3.10",
    PY311 = "3.11",
    PY312 = "3.12",
    PY313 = "3.13",
)

DEFAULT_PYTHON_VERSION = PythonVersion.PY310

# SHA256 for PyTorch 2.10.0 CPU wheels (manylinux_2_28_x86_64)
TORCH_SHA256 = {
    PythonVersion.PY310: "a280ffaea7b9c828e0c1b9b3bd502d9b6a649dc9416997b69b84544bd469f215",
    PythonVersion.PY311: "b7cb1ec66cefb90fd7b676eac72cfda3b8d4e4d0cacd7a531963bc2e0a9710ab",
    PythonVersion.PY312: "ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d",
    PythonVersion.PY313: "8d316e5bf121f1eab1147e49ad0511a9d92e4c45cc357d1ab0bee440da71a095",
}

def format_python_wheel_tag(python_version):
    """Format Python version for wheel tags (e.g., '3.10' -> 'cp310-cp310')."""
    version_no_dot = python_version.replace(".", "")
    return "cp{}-cp{}".format(version_no_dot, version_no_dot)

def get_torch_sha256(python_version):
    """Get SHA256 for PyTorch wheel for given Python version."""
    return TORCH_SHA256.get(python_version, TORCH_SHA256.get(DEFAULT_PYTHON_VERSION))
