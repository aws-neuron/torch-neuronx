#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

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

  auto result = backend.Compile({0x01}, "pkey", "XLA", "-O1", "--model-type transformer",
                                [&]() { return expected_neff; });

  EXPECT_EQ(result, expected_neff);
}

TEST(LocalCompilationBackendTest, CompileFailureReturnsEmpty) {
  LocalCompilationBackend backend;

  auto result = backend.Compile({0x01}, "pkey", "XLA", "-O1", "",
                                []() -> std::vector<uint8_t> { return {}; });

  EXPECT_TRUE(result.empty());
}

TEST(LocalCompilationBackendTest, CompileFnExceptionPropagates) {
  LocalCompilationBackend backend;

  EXPECT_THROW(backend.Compile({0x01}, "pkey", "XLA", "-O1", "",
                               []() -> std::vector<uint8_t> { throw std::runtime_error("fail"); }),
               std::runtime_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
