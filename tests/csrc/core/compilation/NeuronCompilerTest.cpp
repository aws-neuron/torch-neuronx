#include <ftw.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "tests/csrc/mocks/MockNeuronDevice.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"

using namespace at::neuron;

CompilationConfig CreateTestConfig(const std::string& additional_args = "",
                                   const std::string& opt_level = "-O2") {
  CompilationConfig config;
  config.framework = "XLA";
  config.platform_target = "trn2";
  config.logical_neuron_cores = "2";
  config.optimization_level = opt_level;
  config.additional_args = additional_args;
  return config;
}

class MockCompilerHelper {
 public:
  static std::string CreateMockScript(const std::string& temp_dir, int exit_code,
                                      const std::string& output_content = "") {
    std::string script_path = temp_dir + "/neuronx-cc";
    std::ofstream script(script_path);

    // Mock neuronx-cc script
    script << "#!/bin/bash\n";
    script << "# Mock neuronx-cc compiler\n";
    if (exit_code == 0 && !output_content.empty()) {
      script << "for ((i=1; i<=$#; i++)); do\n"
             << "  [[ \"${!i}\" == \"--output\" ]] && { j=$((i+1)); echo -n '" << output_content
             << "' > \"${!j}\"; break; }\n"
             << "done\n";
    }
    script << "exit " << exit_code << "\n";
    script.close();

    chmod(script_path.c_str(), 0755);
    return script_path;
  }
};

class NeuronCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SaveEnvironment();
    CreateTempDirectory();
    torch_neuronx::SetMockInstanceType("trn1");
  }

  void TearDown() override {
    RestoreEnvironment();
    CleanupTempDirectory();
  }

  void SetupMockCompiler(int exit_code = 0, const std::string& output = "mock_neff") {
    MockCompilerHelper::CreateMockScript(temp_dir_, exit_code, output);
    setenv("PATH", (temp_dir_ + ":" + original_path_).c_str(), 1);
  }

 private:
  void SaveEnvironment() {
    original_path_ = getenv("PATH") ? std::string(getenv("PATH")) : "";
    original_lnc_ =
        getenv("NEURON_LOGICAL_NC_CONFIG") ? std::string(getenv("NEURON_LOGICAL_NC_CONFIG")) : "";
  }

  void RestoreEnvironment() {
    setenv("PATH", original_path_.c_str(), 1);
    if (!original_lnc_.empty()) {
      setenv("NEURON_LOGICAL_NC_CONFIG", original_lnc_.c_str(), 1);
    } else {
      unsetenv("NEURON_LOGICAL_NC_CONFIG");
    }
  }

  void CreateTempDirectory() {
    char temp_template[] = "/tmp/neuron_test_XXXXXX";
    char* dir = mkdtemp(temp_template);
    ASSERT_NE(dir, nullptr);
    temp_dir_ = dir;
  }

  void CleanupTempDirectory() {
    if (!temp_dir_.empty()) {
      // Use POSIX nftw instead of std::filesystem to avoid PyTorch 2.9 bug
      nftw(
          temp_dir_.c_str(),
          [](const char* fpath, const struct stat*, int, struct FTW*) { return remove(fpath); }, 64,
          FTW_DEPTH | FTW_PHYS);
    }
  }

 protected:
  std::string original_path_;
  std::string original_lnc_;
  std::string temp_dir_;
};

TEST_F(NeuronCompilerTest, CompileHloToNeff_Success) {
  SetupMockCompiler(0, "compiled_neff_data");

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "compiled_neff_data");
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_CompilerNotFound) {
  setenv("PATH", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  EXPECT_THROW(NeuronCompiler::CompileHloToNeff(hlo, config), std::runtime_error);
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_CompilerFailsWithNonZeroExit) {
  SetupMockCompiler(1);  // Exit code 1

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  EXPECT_THROW(
      {
        try {
          NeuronCompiler::CompileHloToNeff(hlo, config);
        } catch (const std::runtime_error& e) {
          EXPECT_THAT(e.what(), ::testing::HasSubstr("failed"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_EmptyHloSucceedsIfCompilerAcceptsIt) {
  SetupMockCompiler(0, "empty_neff");

  std::vector<uint8_t> empty_hlo;
  CompilationConfig config = CreateTestConfig();

  // Empty HLO is written to file and passed to compiler
  // If compiler accepts it (returns 0), we get the output
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(empty_hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "empty_neff");
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_UsesConfigForLogicalCores) {
  // Create mock neuronx-cc that verifies --lnc 1 is in args
  std::string path = temp_dir_ + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "if [[ \"$*\" == *\"--lnc 1\"* ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'lnc1_neff' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "else\n"
    << "  exit 1\n"
    << "fi\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (temp_dir_ + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  config.logical_neuron_cores = "1";
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "lnc1_neff");
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_PassesAdditionalArgs) {
  // Mock neuronx-cc that verifies specific args are present
  std::string path = temp_dir_ + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "if [[ \"$*\" == *\"--verbose\"* ]] && [[ \"$*\" == *\"--debug\"* ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'args_passed' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "else\n"
    << "  exit 1\n"
    << "fi\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (temp_dir_ + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig("--verbose --debug");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "args_passed");
}

TEST_F(NeuronCompilerTest, CompileHloToNeff_PassesOptimizationLevel) {
  // Mock neuronx-cc that verifies -O2 is in args
  std::string path = temp_dir_ + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "if [[ \"$*\" == *\"-O2\"* ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'O2_neff' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "else\n"
    << "  exit 1\n"
    << "fi\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (temp_dir_ + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig("", "-O2");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "O2_neff");
}

class ExitCodeTest : public NeuronCompilerTest, public ::testing::WithParamInterface<int> {};

TEST_P(ExitCodeTest, CompileHloToNeff_HandlesNonZeroExitCodes) {
  SetupMockCompiler(GetParam());

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  EXPECT_THROW(NeuronCompiler::CompileHloToNeff(hlo, config), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(ExitCodes, ExitCodeTest, ::testing::Values(1, 2, 127, 255));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
