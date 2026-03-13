#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "torch_neuronx/csrc/core/compilation/CompilerDebugUtils.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"

using namespace at::neuron;

class CompilerDebugUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = "/tmp/compiler_debug_test_" + std::to_string(getpid());
    setenv("TORCH_NEURONX_DUMP", test_dir_.c_str(), 1);
    setenv("TORCH_NEURONX_DUMP_HLO_PB", "1", 1);
  }

  void TearDown() override {
    unsetenv("TORCH_NEURONX_DUMP");
    unsetenv("TORCH_NEURONX_DUMP_HLO_PB");
    std::filesystem::remove_all(test_dir_);
  }

  std::string test_dir_;
};

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsCreatesDirectory) {
  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03};
  CompilationConfig config;
  config.framework = "XLA";
  config.platform_target = "trn2";

  debug::DumpIRDetails(hlo, config, "XLA", "test_cache_key", "TestKernel");

  std::string expected_dir = test_dir_ + "/tn_op_dumps/TestKernel-test_cache_key";
  EXPECT_TRUE(std::filesystem::exists(expected_dir));
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsWritesHloFile) {
  std::vector<uint8_t> hlo = {0xDE, 0xAD, 0xBE, 0xEF};
  CompilationConfig config;
  config.framework = "XLA";

  debug::DumpIRDetails(hlo, config, "XLA", "key", "Kernel");

  std::string hlo_file = test_dir_ + "/tn_op_dumps/Kernel-key/module.pb";
  ASSERT_TRUE(std::filesystem::exists(hlo_file));

  std::ifstream ifs(hlo_file, std::ios::binary);
  std::vector<uint8_t> read_bytes((std::istreambuf_iterator<char>(ifs)),
                                  std::istreambuf_iterator<char>());
  EXPECT_EQ(read_bytes, hlo);
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsSkipsOnEmptyHlo) {
  std::vector<uint8_t> empty_hlo;
  CompilationConfig config;

  debug::DumpIRDetails(empty_hlo, config, "key", "Kernel");

  std::string dump_root = test_dir_ + "/tn_op_dumps";
  EXPECT_FALSE(std::filesystem::exists(dump_root));
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsSkipsWhenEnvNotSet) {
  unsetenv("TORCH_NEURONX_DUMP_HLO_PB");

  std::vector<uint8_t> hlo = {0x01};
  CompilationConfig config;

  debug::DumpIRDetails(hlo, config, "XLA", "key", "Kernel");

  std::string dump_root = test_dir_ + "/tn_op_dumps";
  EXPECT_FALSE(std::filesystem::exists(dump_root));
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsHandlesLongCacheKey) {
  std::string long_key(500, 'x');
  std::vector<uint8_t> hlo = {0x01};
  CompilationConfig config;

  ASSERT_NO_THROW(debug::DumpIRDetails(hlo, config, "XLA", long_key, "Kernel"));

  std::string dump_root = test_dir_ + "/tn_op_dumps";
  EXPECT_TRUE(std::filesystem::exists(dump_root));
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsWritesMlirForStableHLO) {
  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config;
  config.framework = "XLA";

  debug::DumpIRDetails(hlo, config, "StableHLO", "key", "Kernel");

  std::string mlir_file = test_dir_ + "/tn_op_dumps/Kernel-key/module.mlir";
  EXPECT_TRUE(std::filesystem::exists(mlir_file));

  // Verify .pb does NOT exist
  std::string pb_file = test_dir_ + "/tn_op_dumps/Kernel-key/module.pb";
  EXPECT_FALSE(std::filesystem::exists(pb_file));
}

TEST_F(CompilerDebugUtilsTest, DumpIRDetailsHashesLongCacheKey) {
  // Cache keys longer than filesystem NAME_MAX (255) should be hashed
  std::string long_key(500, 'a');
  std::vector<uint8_t> hlo = {0x01};
  CompilationConfig config;

  debug::DumpIRDetails(hlo, config, "XLA", long_key, "Kernel");

  std::string dump_root = test_dir_ + "/tn_op_dumps";
  ASSERT_TRUE(std::filesystem::exists(dump_root));

  for (auto& entry : std::filesystem::directory_iterator(dump_root)) {
    std::string dirname = entry.path().filename().string();
    EXPECT_LE(dirname.size(), 255) << "Directory name must fit in NAME_MAX";
    EXPECT_LT(dirname.size(), 500) << "Long key should be hashed, not used verbatim";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
