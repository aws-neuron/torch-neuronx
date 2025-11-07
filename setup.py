import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

PACKAGE_NAME = "torch_neuron"
version = "0.1.0"

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "torch_neuron/csrc"


class BuildPyCommand(build_ext):
    """Custom build command to compile proto files."""

    def fix_import(self, file_path):
        with open(file_path) as f:
            content = f.read()

        # Adjust the path so that we can import the files correctly
        content = re.sub(r"from xla import xla_data_pb2", "from .xla import xla_data_pb2", content)

        with open(file_path, "w") as f:
            f.write(content)

    def run(self):
        builder_path = "torch_neuron/protos"
        # Compile proto files
        proto_files = [f"{builder_path}/hlo.proto", f"{builder_path}/xla/xla_data.proto"]
        command = [
            "python",
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={builder_path}",
            f"--python_out={builder_path}",
            f"--grpc_python_out={builder_path}",
            *proto_files,
        ]

        if subprocess.call(command) != 0:
            raise RuntimeError("Error: Unable to build proto files")

        self.fix_import(f"{builder_path}/hlo_pb2.py")


# Check if we should use CMake for building C++ extension
# Can be controlled by environment variable or command line argument
USE_CMAKE = os.environ.get("USE_CMAKE", "0") == "1"

# Check if we're in the actual build phase (not just getting requirements)
is_building = any(
    arg in ["build", "build_ext", "install", "develop", "bdist_wheel"] for arg in sys.argv
)

if is_building:
    try:
        if USE_CMAKE:
            print("Using CMake Build for C++ Extension")

            class CMakeExtension(Extension):
                """A CMake extension module"""

                def __init__(self, name):
                    Extension.__init__(self, name, sources=[])

            class BuildCMakeExtension(build_ext):
                """A command that runs CMake to build a C/C++ extension."""

                def run(self):
                    # First run proto compilation
                    self.run_command("generate_proto")

                    try:
                        subprocess.check_output(["cmake", "--version"])
                    except OSError as e:
                        raise RuntimeError(
                            f"CMake must be installed to build the extension {self.__class__}"
                        ) from e
                    build_ext.run(self)

                def build_extension(self, ext):
                    if not isinstance(ext, CMakeExtension):
                        return super().build_extension(ext)

                    build_type = "Debug" if "develop" in sys.argv else "Release"

                    cwd = Path().absolute()
                    extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
                    extdir.mkdir(parents=True, exist_ok=True)
                    cmake_args = [
                        f"-DCMAKE_BUILD_TYPE={build_type}",
                        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                    ]

                    try:
                        import torch

                        os.environ["LOCAL_TORCH_PATH"] = torch.__path__[0]
                    except ImportError:
                        pass

                    build_args = ["--config", build_type]

                    if hasattr(self, "parallel") and self.parallel:
                        build_args += [f"-j{self.parallel}"]
                    else:
                        build_args += ["-j4"]  # Default to 4 parallel jobs

                    env = os.environ.copy()
                    env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
                        env.get("CXXFLAGS", ""), self.distribution.get_version()
                    )

                    build_dir = Path(self.build_temp)
                    build_dir.mkdir(parents=True, exist_ok=True)

                    # Configure
                    subprocess.check_call(["cmake", f"{cwd}", *cmake_args], cwd=build_dir, env=env)

                    # Build
                    subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=build_dir)

                def get_ext_filename(self, fullname):
                    # Override build_ext 'get_ext_filename' to get original
                    # shared library name without python ABI suffix.
                    # Equivalent to PyTorch BuildExtension's 'no_python_abi_suffix=True'
                    ext_filename = super().get_ext_filename(fullname)
                    # The parts will be e.g. ["my_extension", "cpython-310m-x86_64-linux-gnu", "so"]
                    ext_filename_parts = ext_filename.split(".")
                    # Omit the second to last element.
                    without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                    return ".".join(without_abi)

            # CMake Extension module with _C library name
            ext_modules = [CMakeExtension("torch_neuron._C")]

            cmdclass = {
                "generate_proto": BuildPyCommand,
                "build_ext": BuildCMakeExtension,
            }
        else:
            print("Using PyTorch C++ Extension")
            from torch.utils.cpp_extension import BuildExtension, CppExtension

            class CustomBuildExt(BuildExtension):
                def run(self):
                    self.run_command("generate_proto")
                    BuildExtension.run(self)

            # GCC warns about ignored [[maybe_unused]] attributes in upstream PyTorch
            # headers; suppress that specific warning while keeping other -Werror
            # checks active.
            CXX_FLAGS = {"cxx": ["-g", "-Wall", "-Werror", "-Wno-error=attributes"]}

            # Collect all C++ source files
            sources = []
            # Add files from torch_neuron/csrc recursively
            sources.extend(CSRC_DIR.glob("**/*.cpp"))
            # Add files from torch_neuron/c10/neuron
            c10_dir = ROOT_DIR / "torch_neuron/c10"
            sources.extend(c10_dir.glob("**/*.cpp"))

            # Convert to strings and remove duplicates while preserving order
            sources = list(dict.fromkeys(str(s) for s in sources))

            # Neuron Runtime paths
            NRT_INCLUDE = "/opt/aws/neuron/include"
            NRT_LIB = "/opt/aws/neuron/lib"

            # Get torch library path
            import torch

            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

            ext_modules = [
                CppExtension(
                    name="torch_neuron._C",
                    sources=sorted(str(s) for s in sources),
                    include_dirs=[str(ROOT_DIR), str(CSRC_DIR), NRT_INCLUDE],
                    library_dirs=[NRT_LIB, torch_lib_path],
                    libraries=["nrt"],
                    runtime_library_dirs=[NRT_LIB, torch_lib_path],
                    extra_compile_args=CXX_FLAGS,
                    extra_link_args=[
                        f"-Wl,-rpath,{NRT_LIB}",
                        f"-Wl,-rpath,{torch_lib_path}",
                    ],
                )
            ]

            cmdclass = {
                "generate_proto": BuildPyCommand,
                "build_ext": CustomBuildExt.with_options(no_python_abi_suffix=True),
            }
    except ImportError:
        ext_modules = []
        cmdclass = {}
else:
    ext_modules = []
    cmdclass = {}

setup(
    name=PACKAGE_NAME,
    version=version,
    author="AWS Neuron Team",
    description="Neuron backend for PyTorch",
    packages=find_packages(exclude=("test",)),
    package_data={PACKAGE_NAME: ["*.dll", "*.dylib", "*.so", "protos/**/*.py"]},
    ext_modules=ext_modules,
    python_requires=">=3.10",
    cmdclass=cmdclass,
)
