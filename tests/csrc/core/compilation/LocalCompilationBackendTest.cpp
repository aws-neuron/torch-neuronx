#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/CacheEntry.h"  // CompileResult
#include "torch_neuronx/csrc/core/compilation/CompilationBackend.h"
#include "torch_neuronx/csrc/core/compilation/LocalCompilationBackend.h"

using namespace at::neuron;

TEST(LocalCompilationBackendTest, NameIsLocal) {
  LocalCompilationBackend backend;
  EXPECT_EQ(backend.GetName(), "local");
}

TEST(LocalCompilationBackendTest, CompileFnInvoked) {
  LocalCompilationBackend backend;
  std::vector<uint8_t> expected_neff = {0xCA, 0xFE};

  auto result =
      backend.Compile({0x01}, "XLA", "op_name", "cache_key", "-O1", "--model-type transformer",
                      [&]() { return CompileResult{expected_neff, {}}; });

  // The local backend returns the CompileFn's CompileResult verbatim; a non-WLO
  // compile carries the NEFF and an empty wrapped_neff.hlo sidecar.
  EXPECT_EQ(result.neff_bytes, expected_neff);
  EXPECT_TRUE(result.wrapped_neff_hlo.empty());
}

TEST(LocalCompilationBackendTest, CompileFnWrappedNeffHloPassedThrough) {
  LocalCompilationBackend backend;
  std::vector<uint8_t> expected_neff = {0xCA, 0xFE};
  std::vector<uint8_t> expected_hlo = {0x01, 0x02, 0x03};  // WLO sidecar

  auto result = backend.Compile({0x01}, "StableHLO", "op_name", "cache_key", "-O2", "",
                                [&]() { return CompileResult{expected_neff, expected_hlo}; });

  // A WLO compile's wrapped_neff.hlo must survive the backend unchanged.
  EXPECT_EQ(result.neff_bytes, expected_neff);
  EXPECT_EQ(result.wrapped_neff_hlo, expected_hlo);
}

TEST(LocalCompilationBackendTest, CompileFailureReturnsEmpty) {
  LocalCompilationBackend backend;

  auto result = backend.Compile({0x01}, "XLA", "op_name", "cache_key", "-O1", "",
                                []() -> CompileResult { return {}; });

  EXPECT_TRUE(result.neff_bytes.empty());
  EXPECT_TRUE(result.wrapped_neff_hlo.empty());
}

TEST(LocalCompilationBackendTest, CompileFnExceptionPropagates) {
  LocalCompilationBackend backend;

  EXPECT_THROW(backend.Compile({0x01}, "XLA", "op_name", "cache_key", "-O1", "",
                               []() -> CompileResult { throw std::runtime_error("fail"); }),
               std::runtime_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
