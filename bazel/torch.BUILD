package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ], exclude = [
        "include/google/protobuf/**/*.h",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

# Runtime headers, for importing <torch/torch.h>.
cc_library(
    name = "runtime_headers",
    hdrs = glob(["include/torch/csrc/api/include/**/*.h"]),
    strip_include_prefix = "include/torch/csrc/api/include",
)


cc_import(
    name = "libtorch",
    shared_library = "lib/libtorch.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "lib/libtorch_python.so",
)

cc_import(
    name = "libc10",
    shared_library = "lib/libc10.so",
)
