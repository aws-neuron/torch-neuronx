#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/compilation/experimental/TensorizerCompiler.h"

namespace at::neuron {

class TensorizerCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temp directory for test files
    test_dir_ = std::filesystem::temp_directory_path() / "tensorizer_test";
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  // Wrappers to access private methods via friend
  static void RenameBirTensors(const std::string& path) {
    TensorizerCompiler::RenameBirTensors(path);
  }

  static std::string GetNeuronOptPath() { return TensorizerCompiler::GetNeuronOptPath(); }

  static std::string GetNkiOptPath() { return TensorizerCompiler::GetNkiOptPath(); }

  std::string WriteTestFile(const std::string& filename, const std::string& content) {
    std::string path = (test_dir_ / filename).string();
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
  }

  std::string ReadFile(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }

  std::filesystem::path test_dir_;
};

TEST_F(TensorizerCompilerTest, RenameBirTensorsBasic) {
  std::string path =
      WriteTestFile("bir.json", R"({"in_tensor_0": "data", "out_tensor_0": "result"})");
  RenameBirTensors(path);
  std::string result = ReadFile(path);
  EXPECT_EQ(result, R"({"input0": "data", "output0": "result"})");
}

TEST_F(TensorizerCompilerTest, RenameBirTensorsMultiple) {
  std::string path = WriteTestFile("bir.json", "in_tensor_0 in_tensor_1 out_tensor_0 out_tensor_1");
  RenameBirTensors(path);
  EXPECT_EQ(ReadFile(path), "input0 input1 output0 output1");
}

TEST_F(TensorizerCompilerTest, RenameBirTensorsNoMatch) {
  std::string path = WriteTestFile("bir.json", "no_replacements_here");
  RenameBirTensors(path);
  EXPECT_EQ(ReadFile(path), "no_replacements_here");
}

TEST_F(TensorizerCompilerTest, RenameBirTensorsInvalidPath) {
  EXPECT_THROW(RenameBirTensors("/nonexistent/path/bir.json"), std::runtime_error);
}

TEST_F(TensorizerCompilerTest, GetNeuronOptPathDefault) {
  EXPECT_EQ(GetNeuronOptPath(), "neuron-opt");
}

TEST_F(TensorizerCompilerTest, GetNkiOptPathDefault) { EXPECT_EQ(GetNkiOptPath(), "nki-opt"); }

TEST_F(TensorizerCompilerTest, CompileLinalgToNeffThrowsOnInvalidInput) {
  std::vector<uint8_t> invalid_bytes = {0x00, 0x01, 0x02};
  EXPECT_THROW(TensorizerCompiler::CompileLinalgToNeff(invalid_bytes, "trn1"), std::runtime_error);
}

TEST_F(TensorizerCompilerTest, CompileLinalgToNeffThrowsOnEmptyInput) {
  std::vector<uint8_t> empty_bytes;
  EXPECT_THROW(TensorizerCompiler::CompileLinalgToNeff(empty_bytes, "trn1"), std::exception);
}

}  // namespace at::neuron
