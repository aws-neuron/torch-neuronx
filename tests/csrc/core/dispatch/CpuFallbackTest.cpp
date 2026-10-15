/**
 * Unit tests for CpuFallback.cpp.
 *
 * The end-to-end CPU-fallback behavior (out=, inplace, multi-return) is
 * driven by ``at::native::cpu_fallback`` (PyTorch) and exercised by the
 * Python integration tests in
 * ``tests/python_ops/test_int64_integration.py::TestCpuFallback*``.
 *
 * These C++ tests pin only the project-specific glue: op-name parsing
 * and OperatorHandle resolution. The pybind layer is intentionally not
 * tested here — it requires a Python interpreter; the Python tests
 * already cover that path.
 */

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>

#include "torch_neuronx/csrc/core/dispatch/CpuFallback.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

// ═══════════════════════════════════════════════════════════════
//  detail::SplitOpName
// ═══════════════════════════════════════════════════════════════

TEST(CpuFallbackSplitOpNameTest, WithOverload) {
  auto [name, overload] = detail::SplitOpName("aten::mm.out");
  EXPECT_EQ(name, "aten::mm");
  EXPECT_EQ(overload, "out");
}

TEST(CpuFallbackSplitOpNameTest, WithoutOverload) {
  auto [name, overload] = detail::SplitOpName("aten::add");
  EXPECT_EQ(name, "aten::add");
  EXPECT_EQ(overload, "");
}

TEST(CpuFallbackSplitOpNameTest, MultiDottedOverloadSplitsAtFirstDot) {
  // Real ATen overload names sometimes contain ``.`` (e.g. dim_IntList).
  // SplitOpName must split at the first dot only.
  auto [name, overload] = detail::SplitOpName("aten::sum.dim_IntList");
  EXPECT_EQ(name, "aten::sum");
  EXPECT_EQ(overload, "dim_IntList");
}

TEST(CpuFallbackSplitOpNameTest, EmptyString) {
  auto [name, overload] = detail::SplitOpName("");
  EXPECT_EQ(name, "");
  EXPECT_EQ(overload, "");
}

// ═══════════════════════════════════════════════════════════════
//  Schema resolution via SplitOpName + findSchemaOrThrow
// ═══════════════════════════════════════════════════════════════

TEST(CpuFallbackHandleResolutionTest, KnownOpResolves) {
  auto [name, overload] = detail::SplitOpName("aten::mm.out");
  EXPECT_NO_THROW({
    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
    EXPECT_EQ(handle.schema().name(), "aten::mm");
    EXPECT_EQ(handle.schema().overload_name(), "out");
  });
}

TEST(CpuFallbackHandleResolutionTest, UnknownOpThrows) {
  auto [name, overload] = detail::SplitOpName("aten::definitely_not_a_real_op");
  EXPECT_THROW(c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str()),
               c10::Error);
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
