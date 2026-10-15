"""Print the environment details a torch-neuronx bug report needs.

python -m torch_neuronx.collect_env
"""

import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version

PYTHON_PACKAGES = ("torch-neuronx", "torch", "neuronx-cc", "nki")
SYSTEM_PACKAGES = (
    "aws-neuronx-runtime-lib",
    "aws-neuronx-collectives",
    "aws-neuronx-tools",
)


def run(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def pip_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def system_package_version(name):
    # Ubuntu uses dpkg. Amazon Linux uses rpm.
    return (
        run(["dpkg-query", "-W", "-f=${Version}", name])
        or run(["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", name])
        or "not installed"
    )


def driver_version():
    # The driver is a kernel module on the host. In a container, the package
    # database can disagree with the module that is actually loaded.
    try:
        with open("/sys/module/neuron/version") as f:
            return f.read().strip()
    except OSError:
        return "not loaded"


def device_usable():
    try:
        import torch

        import torch_neuronx  # noqa: F401

        torch.zeros(1, device="neuron")
        return "yes"
    except Exception as exc:
        return f"no ({type(exc).__name__})"


def main():
    ls = run(["neuron-ls"])

    def ls_field(key):
        match = re.search(rf"^{key}:\s*(.+)$", ls, re.MULTILINE)
        return match.group(1) if match else "unknown"

    rows = [(name, pip_version(name)) for name in PYTHON_PACKAGES]
    rows += [("Neuron driver", driver_version())]
    rows += [(name, system_package_version(name)) for name in SYSTEM_PACKAGES]
    rows += [
        ("instance type", ls_field("instance-type")),
        ("logical NeuronCore config", ls_field("logical-neuroncore-config")),
        ("device usable from torch", device_usable()),
        ("OS", platform.platform()),
        ("Python", platform.python_version()),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")


if __name__ == "__main__":
    main()
