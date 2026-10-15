load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

################################ StableHLO Setup ################################

# StableHLO dependency — pinned to the same commit as torch-mlir's externals/stablehlo
STABLEHLO_COMMIT = "4c0d4841519aed22e3689c30b72a0e4228051249"

http_archive(
    name = "stablehlo",
    sha256 = "88fc2c0ef370e1b4d20043cb0f0db7027a7b9299cf9084b184ab9f5d9fe175c4",
    strip_prefix = "stablehlo-" + STABLEHLO_COMMIT,
    urls = ["https://github.com/openxla/stablehlo/archive/{commit}.tar.gz".format(commit = STABLEHLO_COMMIT)],
)

# LLVM — pinned to the same commit as torch-mlir's externals/llvm-project
LLVM_COMMIT = "41f65666f6378bba7266be7c662c70074f04ed75"
LLVM_SHA256 = "bbcc839033dad5fe6a7ee115c38fa29e4831a51500373853d405761ca29d730e"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    patch_cmds = [
        # GCC 15 default-constructor fix. Remove when LLVM pin includes 05dc519011cd.
        (
            "python3 -c 'from pathlib import Path; "
            + "p = Path(\"llvm/include/llvm/IR/CFG.h\"); s = p.read_text(); "
            + "old = \"  explicit inline SuccIterator(InstructionT *Inst)\"; "
            + "assert s.count(old) == 1; "
            + "p.write_text(s.replace(old, \"  inline SuccIterator() : Inst(nullptr), Idx(0) {}\\n\" + old, 1))'"
        ),
        # GCC 15 indexed-accessor fix. Remove when LLVM pin includes bad300bda1b9.
        (
            "python3 -c 'from pathlib import Path; "
            + "p = Path(\"llvm/include/llvm/ADT/STLExtras.h\"); s = p.read_text(); "
            + "old = \"  public:\\n    // Index into this iterator, invoking a static method on the derived type.\"; "
            + "new = \"  public:\\n    iterator() : iterator::indexed_accessor_iterator(nullptr, 0) {}\\n"
            + "    // Index into this iterator, invoking a static method on the derived type.\"; "
            + "assert s.count(old) == 1; p.write_text(s.replace(old, new, 1))'"
        ),
        # GCC destructor template-id fix — remove when LLVM pin includes 8f85bb16548c
        "sed -i 's/~OptionValue<OpPassManager>()/~OptionValue()/' mlir/lib/Pass/PassRegistry.cpp",
        # GCC 11 ambiguous operator== fix — remove when LLVM pin includes 17f5f5ba5597
        "sed -i 's/!llvm::is_contained(PushedRegs, Reg)/!llvm::is_contained(PushedRegs, Reg.id())/' llvm/lib/Target/X86/X86WinEHUnwindV2.cpp",
    ],
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
llvm_configure(
    name = "llvm-project",
    targets = ["X86", "NVPTX"],
)

http_archive(
    name = "pybind11",
    build_file = "@llvm-raw//utils/bazel/third_party_build:pybind.BUILD",
    sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
    strip_prefix = "pybind11-2.10.3",
    url = "https://github.com/pybind/pybind11/archive/v2.10.3.zip",
)

http_archive(
    name = "nanobind",
    build_file = "@llvm-raw//utils/bazel/third_party_build:nanobind.BUILD",
    sha256 = "8ce3667dce3e64fc06bfb9b778b6f48731482362fb89a43da156632266cd5a90",
    strip_prefix = "nanobind-2.9.2",
    url = "https://github.com/wjakob/nanobind/archive/refs/tags/v2.9.2.tar.gz",
)

http_archive(
    name = "robin_map",
    build_file = "@llvm-raw//utils/bazel/third_party_build:robin_map.BUILD",
    sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
    strip_prefix = "robin-map-1.3.0",
    url = "https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz",
)
