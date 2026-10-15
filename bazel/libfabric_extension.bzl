"""Module extension for libfabric (vendored headers, runtime dlopen).

The libfabric v1.22.0 public headers are vendored at third_party/libfabric/include/
because EFA is not available in non-Trainium environments (compile-only mode).
The RDMA code still needs the header type definitions (structs, enums, function
signatures) to compile. The actual libfabric.so.1 is loaded via dlopen() at runtime
only when RDMA is needed on EFA-equipped machines.

To update: replace third_party/libfabric/include/rdma/ from a new release tarball.
"""

def _libfabric_repo_impl(rctx):
    # Symlink the vendored headers from the main repo into the external repo workspace.
    rctx.symlink(rctx.attr.headers_path, "include")
    rctx.file("BUILD", """package(default_visibility = ["//visibility:public"])

# Headers-only target for compile-time struct/enum definitions.
# libfabric.so is NOT linked - it is loaded via dlopen at runtime.
cc_library(
    name = "libfabric_hdrs",
    hdrs = glob(["include/rdma/**/*.h"]),
    strip_include_prefix = "include",
)
""")

_libfabric_repo = repository_rule(
    implementation = _libfabric_repo_impl,
    attrs = {
        "headers_path": attr.label(default = Label("//:third_party/libfabric/include")),
    },
    local = True,
)

def _libfabric_extension_impl(module_ctx):
    _libfabric_repo(name = "libfabric")
    return module_ctx.extension_metadata(
        root_module_direct_deps = ["libfabric"],
        root_module_direct_dev_deps = [],
    )

libfabric_extension = module_extension(
    implementation = _libfabric_extension_impl,
)
