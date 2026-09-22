import os
import posixpath
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

from setuptools import Extension, find_namespace_packages, find_packages, setup
from setuptools.command.build_ext import build_ext

PACKAGE_NAME = "torch_neuronx"

# Prevent an installed backend from loading during the build.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


def _get_pytorch_version():
    """Get the PyTorch version that this wheel uses."""
    selected_version = os.environ.get("TORCH_NEURONX_BUILD_PYTORCH_VERSION")
    try:
        import torch

        installed_version = torch.__version__.split("+", 1)[0]
    except ImportError:
        installed_version = None

    version = selected_version or installed_version
    if not version:
        raise RuntimeError("PyTorch is not installed. Install PyTorch before the build.")

    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z0-9.]*)?", version):
        raise ValueError(f"PyTorch version {version!r} is invalid. Use a PEP 440 version.")

    if selected_version and installed_version and selected_version != installed_version:
        raise RuntimeError(
            f"Selected PyTorch {selected_version} does not match installed PyTorch "
            f"{installed_version}. Install the selected version."
        )
    return version


def _get_git_sha():
    sha = os.environ.get("GIT_SHA")
    if sha:
        return sha
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _get_build_counter():
    """Get build counter from env (CI) or git commit count (local)."""
    build = os.environ.get("VERSION_BUILD")
    if build:
        return build
    try:
        return (
            subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "0"


def _is_release_branch():
    """Check if current branch is a release branch."""
    if os.environ.get("RELEASE_BRANCH"):
        return True
    branch = os.environ.get("GIT_BRANCH", "")
    if not branch:
        try:
            branch = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            branch = ""
    return branch.startswith("release")


def _get_version(pytorch_version):
    """Build version string per versioning scheme.

    Format: {TORCH_MAJOR}.{TORCH_MINOR}.{NEURON_MAJOR}.{NEURON_MINOR}.{BUILD}+{suffix}

    Mainline: 2.14.3.0.4527+8e870898.dev
    Release:  2.14.3.0.4527+8e870898
    """
    version_info = runpy.run_path(Path(__file__).parent / "torch_neuronx" / "_version.py")
    neuron_version = version_info["NEURON_VERSION"]
    if not re.fullmatch(r"\d+\.\d+", neuron_version):
        raise ValueError(
            f"Neuron version {neuron_version!r} is invalid. Use a major.minor version."
        )

    torch_parts = pytorch_version.split(".")
    neuron_parts = neuron_version.split(".")
    version_prefix = ".".join([*torch_parts[:2], *neuron_parts])

    sha = _get_git_sha() or "unknown"
    suffix = sha if _is_release_branch() else f"{sha}.dev"

    return f"{version_prefix}.{_get_build_counter()}+{suffix}"


pytorch_version = _get_pytorch_version()
version = _get_version(pytorch_version)

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "torch_neuronx/csrc"


def generate_build_config(is_release):
    """Update build configuration file with build-time IS_RELEASE_BUILD value."""
    config_path = ROOT_DIR / "torch_neuronx" / "_build_config.py"

    # Read the template
    with open(config_path) as f:
        content = f.read()

    # Replace BUILD_TYPE value
    build_type = "RELEASE" if is_release else "DEV"
    content = re.sub(r'BUILD_TYPE = "(RELEASE|DEV)"', f'BUILD_TYPE = "{build_type}"', content)
    print(content)

    # Write back
    with open(config_path, "w") as f:
        f.write(content)


def is_ninja_available():
    """Check if Ninja build system is available."""
    try:
        subprocess.check_output(["ninja", "--version"], stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


class BuildTimer:
    """Utility class for timing build operations."""

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        print(f"[TIMING] Starting {self.operation_name}...")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_t):
        self.end_time = time.time()
        if self.start_time is not None:
            duration = self.end_time - self.start_time
            print(f"[TIMING] {self.operation_name} completed in {duration:.2f} seconds")
            if exc_type is not None:
                print(f"[TIMING] {self.operation_name} failed after {duration:.2f} seconds")


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
        builder_path = "torch_neuronx/protos"
        # Compile proto files
        proto_files = [
            f"{builder_path}/hlo.proto",
            f"{builder_path}/xla/xla_data.proto",
        ]
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
# Check if we should use Bazel for building C++ extension
# Can be controlled by environment variable or command line argument
USE_BAZEL = os.environ.get("USE_BAZEL", "0") == "1"


# Check if this is a release build (via env var or command line arg)
is_release = "release" in sys.argv

# Remove 'release' from argv so setuptools doesn't complain
if "release" in sys.argv:
    sys.argv.remove("release")

generate_build_config(is_release)

# Check if we're in the actual build phase (not just getting requirements)
is_building = any(
    arg in ["build", "build_ext", "install", "develop", "bdist_wheel"] for arg in sys.argv
)

if is_building:
    try:
        if USE_BAZEL:
            print("Using Bazel Build for C++ Extension")
            import shutil

            class BazelExtension(Extension):
                """A C/C++ extension that is defined as a Bazel BUILD target."""

                def __init__(self, bazel_target):
                    self.bazel_target = bazel_target
                    self.relpath, self.target_name = posixpath.relpath(bazel_target, "//").split(
                        ":"
                    )
                    ext_name = os.path.join(
                        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name
                    )
                    if ext_name.endswith(".so"):
                        ext_name = ext_name[:-3]
                    Extension.__init__(self, ext_name, sources=[])

            class BuildBazelExtension(build_ext):
                """A command that runs Bazel to build a C/C++ extension."""

                def run(self):
                    with BuildTimer("Bazel Extension Build"):
                        # First run proto compilation
                        self.run_command("generate_proto")
                        try:
                            subprocess.check_output(["bazel", "--version"])
                        except OSError as e:
                            raise RuntimeError(
                                f"Bazel must be installed to build the extension {self.__class__}."
                                "Please run prerequisites.sh or install_bazelisk.sh to install."
                            ) from e
                        # Run custom extension
                        build_ext.run(self)

                def _fix_rpath(self, so_path):
                    """Rewrite RPATH for the installed wheel layout.

                    patchelf --set-rpath REPLACES the RPATH, so every runtime
                    dependency dir must be listed. EFA/libfabric is loaded via
                    dlopen at runtime (not linked), so its path is not needed here.
                    """
                    new_rpath = (
                        "$ORIGIN"
                        ":$ORIGIN/../../../../../torch_neuronx"
                        ":$ORIGIN/../../torch_neuronx"
                        ":$ORIGIN/../torch/lib:/opt/aws/neuron/lib"
                    )

                    # Check if patchelf is available
                    patchelf = shutil.which("patchelf")
                    if not patchelf:
                        raise RuntimeError(
                            "patchelf is required to build wheels but was not found.\n"
                            "Please install patchelf on your system."
                        )

                    # Make file writable (Bazel outputs are read-only)
                    so_path = Path(so_path)
                    so_path.chmod(so_path.stat().st_mode | 0o200)

                    try:
                        subprocess.run(
                            [patchelf, "--set-rpath", new_rpath, str(so_path)],
                            check=True,
                            capture_output=True,
                        )
                        print(f"Fixed RPATH on {so_path}: {new_rpath}")
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(
                            f"Failed to fix RPATH on {so_path}.\n"
                            f"patchelf stderr: {e.stderr.decode() if e.stderr else 'none'}\n"
                            f"The wheel will not work without this fix."
                        ) from e

                def build_extension(self, ext: Extension) -> None:
                    """
                    This method is called by setuptools to build a single extension.
                    We override it to implement our custom Bazel build logic.
                    """

                    if not isinstance(ext, BazelExtension):
                        # If it's not our custom extension type, let setuptools handle it.
                        super().build_extension(ext)
                        return

                    build_dir = Path(self.build_temp)
                    build_dir.mkdir(parents=True, exist_ok=True)

                    # Prepare the Bazel command
                    python_version = (
                        os.environ.get("PYTHON_VERSION")
                        or Path(".python-version").read_text().strip()
                    )
                    bazel_argv = [
                        "bazel",
                        "build",
                        ext.bazel_target,
                        f"--symlink_prefix={os.path.join(self.build_temp, 'bazel-')}",
                        f"--@rules_python//python/config_settings:python_version={python_version}",
                        f"--action_env=PYTHON_VERSION={python_version}",
                    ]

                    # Set env variable for pytorch local path
                    import torch

                    os.environ["PYTORCH_LOCAL_PATH"] = os.path.dirname(torch.__file__)

                    # Run the Bazel build
                    self.spawn(bazel_argv)

                    # Copy the Bazel-built extension to the expected location
                    bazel_extension = build_dir / "bazel-bin" / ext.relpath / ext.target_name
                    if ext.target_name in (
                        "_torchMlir.so",
                        "_mlir.so",
                        "_stablehlo.so",
                    ):
                        target_extension = (
                            ROOT_DIR
                            / "third_party"
                            / "torch-mlir"
                            / "python"
                            / "torch_mlir"
                            / "_mlir_libs"
                            / ext.target_name
                        )
                    else:
                        target_extension = ROOT_DIR / "torch_neuronx" / ext.target_name

                    if bazel_extension.exists():
                        # Remove existing file if it exists to avoid permission issues
                        if target_extension.exists():
                            try:
                                target_extension.unlink()
                            except OSError as e:
                                print(f"Warning: Could not remove existing {target_extension}: {e}")
                                # Try to make it writable and remove again
                                try:
                                    target_extension.chmod(0o666)
                                    target_extension.unlink()
                                except OSError as e:
                                    raise RuntimeError(
                                        f"Cannot overwrite existing {target_extension}. "
                                        f"Please remove it manually."
                                    ) from e

                        shutil.copy2(str(bazel_extension), str(target_extension))
                        print(f"Copied Bazel extension: {bazel_extension} -> {target_extension}")

                        # Fix RPATH - Bazel sets incorrect paths, we need torch/lib relative path
                        self._fix_rpath(target_extension)
                    else:
                        raise RuntimeError(f"Bazel extension not found at {bazel_extension}")

                def get_ext_filename(self, fullname):
                    # Override build_ext 'get_ext_filename' to get original shared library name
                    # without python ABI suffix.
                    # Equivalent to PyTorch BuildExtension's 'no_python_abi_suffix=True'
                    ext_filename = super().get_ext_filename(fullname)
                    # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
                    ext_filename_parts = ext_filename.split(".")
                    # Omit the second to last element.
                    without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                    return ".".join(without_abi)

            # Bazel Extension modules
            ext_modules = [
                BazelExtension("//:_C.so"),
                BazelExtension("//third_party/torch-mlir:_torchMlir.so"),
                BazelExtension("//third_party/torch-mlir:_mlir.so"),
                BazelExtension("//third_party/torch-mlir:_stablehlo.so"),
            ]

            cmdclass = {
                "generate_proto": BuildPyCommand,
                "build_ext": BuildBazelExtension,
            }

        elif USE_CMAKE:
            print("Using CMake Build for C++ Extension")

            class CMakeExtension(Extension):
                """A CMake extension module"""

                def __init__(self, name):
                    Extension.__init__(self, name, sources=[])

            class BuildCMakeExtension(build_ext):
                """A command that runs CMake to build a C/C++ extension."""

                def run(self):
                    with BuildTimer("CMake Extension Build"):
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

                    # Check if Ninja is available and use it for faster builds
                    # Speeds up build time 2-3x
                    use_ninja = is_ninja_available()
                    generator_args = ["-GNinja"] if use_ninja else []

                    cmake_args = [
                        f"-DCMAKE_BUILD_TYPE={build_type}",
                        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                        *generator_args,
                    ]

                    if use_ninja:
                        print("Using Ninja build system for faster compilation")
                    else:
                        print("Ninja not available, using default build system")

                    try:
                        import torch

                        os.environ["LOCAL_TORCH_PATH"] = torch.__path__[0]
                    except ImportError:
                        pass

                    build_args = ["--config", build_type]

                    # Set parallel build arguments
                    if use_ninja:
                        # Ninja automatically uses all available cores by default
                        # but we can still specify parallel jobs if needed
                        if hasattr(self, "parallel") and self.parallel:
                            build_args += [f"-j{self.parallel}"]
                        # Let Ninja decide optimal parallelism if not specified
                    else:
                        # For Make specify parallel jobs
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
                    # Override build_ext 'get_ext_filename' to get original shared library name
                    # without python ABI suffix.
                    # Equivalent to PyTorch BuildExtension's 'no_python_abi_suffix=True'
                    ext_filename = super().get_ext_filename(fullname)
                    # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
                    ext_filename_parts = ext_filename.split(".")
                    # Omit the second to last element.
                    without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                    return ".".join(without_abi)

            ext_modules = [CMakeExtension("torch_neuronx._C")]
            cmdclass = {
                "generate_proto": BuildPyCommand,
                "build_ext": BuildCMakeExtension,
            }

        else:
            print("Using PyTorch C++ Extension")
            from torch.utils.cpp_extension import BuildExtension, CppExtension

            class CustomBuildExt(BuildExtension):
                def run(self):
                    with BuildTimer("PyTorch C++ Extension Build"):
                        self.run_command("generate_proto")
                        BuildExtension.run(self)

            # GCC warns about ignored [[maybe_unused]] attributes in upstream PyTorch
            # headers; suppress that specific warning while keeping other -Werror
            # checks active.
            CXX_FLAGS = {"cxx": ["-g", "-Wall", "-Werror", "-Wno-error=attributes"]}

            # Collect all C++ source files
            sources = []
            # Add files from torch_neuronx/csrc recursively
            sources.extend(CSRC_DIR.glob("**/*.cpp"))
            # Add files from torch_neuronx/c10/neuron
            c10_dir = ROOT_DIR / "torch_neuronx/c10"
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
                    name="torch_neuronx._C",
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

install_requires = [
    f"torch=={pytorch_version}",
    "numpy<2; python_version < '3.12'",
    "numpy>=2; python_version >= '3.12'",
    "grpcio-tools",
    "wheel",
]

setup(
    name=PACKAGE_NAME,
    version=version,
    author="AWS Neuron Team",
    description="Neuron backend for PyTorch",
    packages=find_packages(exclude=("test", "tests", "tests.*"))
    + find_namespace_packages(
        where="third_party/torch-mlir/python", exclude=("test", "tests", "tests.*")
    ),
    package_dir={"torch_mlir": "third_party/torch-mlir/python/torch_mlir"},
    package_data={
        PACKAGE_NAME: ["*.dll", "*.dylib", "*.so", "protos/**/*.py", "**/*.jinja"],
        "torch_mlir": ["*.so", "_mlir_libs/*.so"],
    },
    ext_modules=ext_modules,
    python_requires=">=3.10",
    cmdclass=cmdclass,
    license_files=["LICENSE", "third_party/torch-mlir/LICENSE"],
    install_requires=install_requires,
)
