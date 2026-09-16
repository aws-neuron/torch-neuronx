"""Shared Python configuration for all Bazel files."""

PythonVersion = struct(
    PY310 = "3.10",
    PY311 = "3.11",
    PY312 = "3.12",
    PY313 = "3.13",
)

DEFAULT_PYTHON_VERSION = PythonVersion.PY310

# SHA256 for PyTorch 2.14.0 CPU wheels (manylinux_2_28_x86_64)
TORCH_SHA256 = {
    PythonVersion.PY310: "019463a7566401db8b1ed142026086db8fa680abda49ec41a60a8718d88e891d",
    PythonVersion.PY311: "673dbf5c9bbadfffab7a386b6dd7a0c219f1408a328b7b4e86d0ae551cdafa42",
    PythonVersion.PY312: "a09987c95ec4cffdb6df798d3d641558110a334cfccce22f2f046d83142bc260",
    PythonVersion.PY313: "160e1bc46aeded3111d2801f8ae10dc9a1b946843a7e126b4dbf5e19c5706e95",
}

def format_python_wheel_tag(python_version):
    """Format Python version for wheel tags (e.g., '3.10' -> 'cp310-cp310')."""
    version_no_dot = python_version.replace(".", "")
    return "cp{}-cp{}".format(version_no_dot, version_no_dot)

def get_torch_sha256(python_version):
    """Get SHA256 for PyTorch wheel for given Python version."""
    return TORCH_SHA256.get(python_version, TORCH_SHA256.get(DEFAULT_PYTHON_VERSION))
