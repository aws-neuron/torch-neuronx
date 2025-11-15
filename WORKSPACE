load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

################################ StableHLO Setup ################################

# StableHLO dependency
# Using v1.13.1 which has proper MODULE.bazel support
http_archive(
    name = "stablehlo",
    strip_prefix = "stablehlo-1.13.1",
    urls = ["https://github.com/openxla/stablehlo/archive/v1.13.1.tar.gz"],
    integrity = "sha256-1uhmEP0EUcuU1+bkTduSVRWoht0m4DothVAZoSMLCg8=",
)

# Set up LLVM dependencies the same way StableHLO does
LLVM_COMMIT = "113f01aa82d055410f22a9d03b3468fa68600589"
LLVM_SHA256 = "9aee00a35aa76639746589c6d09e8c18249be16b5b6aa6b788a570a4bc6c4543"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
)

http_archive(
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)

http_archive(
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project")
