"""Module extension for libfabric from system EFA installation.

Looks up libfabric headers and runtime under `LIBFABRIC_LOCAL_PATH` (default
`/opt/amazon/efa`). The rpath of the built extension is set to the same path
so the override applies end-to-end (build + runtime).
"""

def _libfabric_repo_impl(rctx):
    local_path = rctx.os.environ.get("LIBFABRIC_LOCAL_PATH", "/opt/amazon/efa")
    rctx.symlink(local_path, "efa")
    rctx.file("BUILD", """package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libfabric",
    hdrs = glob(["efa/include/**/*.h"]),
    srcs = glob(["efa/lib/libfabric.so*"]),
    linkopts = ["-Wl,-rpath,{rpath}"],
    strip_include_prefix = "efa/include",
)
""".format(rpath = local_path + "/lib"))

_libfabric_repo = repository_rule(
    implementation = _libfabric_repo_impl,
    environ = ["LIBFABRIC_LOCAL_PATH"],
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
