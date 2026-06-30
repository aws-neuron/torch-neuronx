"""Shared Python configuration for all Bazel files."""

PythonVersion = struct(
    PY310 = "3.10",
    PY311 = "3.11",
    PY312 = "3.12",
    PY313 = "3.13",
)

DEFAULT_PYTHON_VERSION = PythonVersion.PY310

# SHA256 for PyTorch 2.12.0 CPU wheels (manylinux_2_28_x86_64)
TORCH_SHA256 = {
    PythonVersion.PY310: "5d8513a19dcf28c5dc338f3c8ee5a692b07699e8dbd01dbabce6c74a75474f70",
    PythonVersion.PY311: "d85bdbc271bf22ef1931375a81b0366ab11081509728c58df730cf194a090818",
    PythonVersion.PY312: "5e3dc83725581fa38b7b2e45c58692e30b2a3cde19191af54b675ffcac3840a6",
    PythonVersion.PY313: "ada78018bdfa30d1c766596cd32d910dbf5b03424cd859231b6d2a00533de922",
}

def format_python_wheel_tag(python_version):
    """Format Python version for wheel tags (e.g., '3.10' -> 'cp310-cp310')."""
    version_no_dot = python_version.replace(".", "")
    return "cp{}-cp{}".format(version_no_dot, version_no_dot)

def get_torch_sha256(python_version):
    """Get SHA256 for PyTorch wheel for given Python version."""
    return TORCH_SHA256.get(python_version, TORCH_SHA256.get(DEFAULT_PYTHON_VERSION))
