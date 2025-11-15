"""Module extension for PyTorch dependency.
See https://bazel.build/rules/lib/repo/local.
"""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:python_config.bzl", "format_python_wheel_tag")


def _torch_extension_impl(module_ctx):
    """Implementation of the torch extension."""
    torch_path = module_ctx.getenv("PYTORCH_LOCAL_PATH")
    use_external = module_ctx.getenv("PYTORCH_USE_EXTERNAL") == "1"

    if torch_path and not use_external:
        # Use local PyTorch installation
        new_local_repository(
            name = "torch",
            build_file = Label("//bazel:torch.BUILD"),
            path = torch_path,
        )
    else:
        torch_version = module_ctx.getenv("PYTORCH_VERSION")
        python_version = module_ctx.getenv("PYTHON_VERSION")
        python_wheel_tag = format_python_wheel_tag(python_version)
        # Download and extract PyTorch wheel
        http_archive(
            name = "torch",
            urls = ["https://download.pytorch.org/whl/cpu-cxx11-abi/torch-{}%2Bcpu.cxx11.abi-{}-linux_x86_64.whl".format(torch_version, python_wheel_tag)],
            sha256 = "e8ab5a935b33526df0ccae43f4ec1bf6656c5be9c1238af412e0e87dc8345f04",
            build_file = Label("//bazel:torch.BUILD"),
            strip_prefix = "torch",
            type = "zip",
        )

    return module_ctx.extension_metadata(
        root_module_direct_deps = ["torch"],
        root_module_direct_dev_deps = [],
    )

torch_extension = module_extension(
    implementation = _torch_extension_impl,
)
