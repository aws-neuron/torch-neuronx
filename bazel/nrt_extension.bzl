"""Module extension for NRT (Neuron Runtime) dependency.
See https://bazel.build/rules/lib/repo/local.
"""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _nrt_extension_impl(module_ctx):
    """Implementation of the nrt extension."""
    nrt_path = module_ctx.getenv("NRT_LOCAL_PATH")
    new_local_repository(
        name = "nrt",
        build_file = Label("//bazel:nrt.BUILD"),
        path = nrt_path,
    )
    return module_ctx.extension_metadata(
        root_module_direct_deps = ["nrt"],
        root_module_direct_dev_deps = [],
    )

nrt_extension = module_extension(
    implementation = _nrt_extension_impl,
)
