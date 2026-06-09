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
  // Creates a mock neuronx-cc script in a bin/ subdirectory to avoid collision
  // with the neuronx-cc/ workdir hierarchy created by GetCompilationWorkdir.
  static std::string CreateMockScript(const std::string& temp_dir, int exit_code,
                                      const std::string& output_content = "") {
    std::string bin_dir = temp_dir + "/bin";
    std::filesystem::create_directories(bin_dir);
    std::string script_path = bin_dir + "/neuronx-cc";
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
    // Use bin/ subdirectory in PATH to avoid collision with neuronx-cc/ workdir
    setenv("PATH", (temp_dir_ + "/bin:" + original_path_).c_str(), 1);
  }

  // Finds workdir(s) created under {debug_dir}/neuronx-cc/
  std::vector<std::string> FindWorkdirs(const std::string& debug_dir) {
    std::vector<std::string> result;
    std::string neuronx_cc_dir = debug_dir + "/neuronx-cc";
    if (!std::filesystem::exists(neuronx_cc_dir)) return result;
    for (auto& entry : std::filesystem::directory_iterator(neuronx_cc_dir)) {
      if (entry.is_directory()) result.push_back(entry.path().string());
    }
    return result;
  }

 private:
  void SaveEnvironment() {
    original_path_ = getenv("PATH") ? std::string(getenv("PATH")) : "";
    original_lnc_ =
        getenv("NEURON_LOGICAL_NC_CONFIG") ? std::string(getenv("NEURON_LOGICAL_NC_CONFIG")) : "";
    original_debug_dir_ =
        getenv("TORCH_NEURONX_DEBUG_DIR") ? std::string(getenv("TORCH_NEURONX_DEBUG_DIR")) : "";
    original_preserve_ = getenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS")
                             ? std::string(getenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"))
                             : "";
  }

  void RestoreEnvironment() {
    setenv("PATH", original_path_.c_str(), 1);
    if (!original_lnc_.empty()) {
      setenv("NEURON_LOGICAL_NC_CONFIG", original_lnc_.c_str(), 1);
    } else {
      unsetenv("NEURON_LOGICAL_NC_CONFIG");
    }
    if (!original_preserve_.empty()) {
      setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", original_preserve_.c_str(), 1);
    } else {
      unsetenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS");
    }
    if (!original_debug_dir_.empty()) {
      setenv("TORCH_NEURONX_DEBUG_DIR", original_debug_dir_.c_str(), 1);
    } else {
      unsetenv("TORCH_NEURONX_DEBUG_DIR");
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
      // Use system command instead of nftw to avoid hang
      std::string cmd = "rm -rf " + temp_dir_;
      std::system(cmd.c_str());
    }
  }

 protected:
  std::string original_path_;
  std::string original_lnc_;
  std::string original_preserve_;
  std::string original_debug_dir_;
  std::string temp_dir_;

  // Helper to verify all metadata files exist in workdir
  void VerifyMetadataFiles(const std::string& workdir) {
    EXPECT_TRUE(std::filesystem::exists(workdir + "/metadata.txt"))
        << "metadata.txt not found in " << workdir;
    EXPECT_TRUE(std::filesystem::exists(workdir + "/compiler.txt"))
        << "compiler.txt not found in " << workdir;
  }
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

// ============================================================================
// Debug Artifact Tests - verify workdir creation and file paths
// ============================================================================

TEST_F(NeuronCompilerTest, DebugArtifacts_CreatesWorkdirInDebugDir) {
  SetupMockCompiler(0, "debug_neff_data");
  std::string debug_dir = temp_dir_ + "/subdir";
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", debug_dir.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "debug_neff_data");

  auto workdirs = FindWorkdirs(debug_dir);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));
}

TEST_F(NeuronCompilerTest, DebugArtifacts_CleansUpWorkdir_WhenPreserveFalse) {
  SetupMockCompiler(0, "cleanup_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "0", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "cleanup_neff");

  EXPECT_TRUE(std::filesystem::exists(temp_dir_)) << "Debug dir not found";
  auto workdirs = FindWorkdirs(temp_dir_);
  EXPECT_TRUE(workdirs.empty()) << "Workdir should be cleaned up with preserve=0";
}

TEST_F(NeuronCompilerTest, DebugArtifacts_PreservesWorkdir_WhenPreserveTrue) {
  SetupMockCompiler(0, "preserve_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "true", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "preserve_neff");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
}

TEST_F(NeuronCompilerTest, DebugArtifacts_PreservesWorkdir_WhenPreserveYes) {
  SetupMockCompiler(0, "preserve_yes_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "yes", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "preserve_yes_neff");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
}

TEST_F(NeuronCompilerTest, DebugArtifacts_PreservesWorkdir_WhenPreserve1) {
  SetupMockCompiler(0, "preserve_1_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "preserve_1_neff");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
}

TEST_F(NeuronCompilerTest, DebugArtifacts_StableHLO_UsesCorrectFilename) {
  SetupMockCompiler(0, "stablehlo_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config, "StableHLO");

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "stablehlo_neff");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));
}

TEST_F(NeuronCompilerTest, PreserveArtifacts_MetadataTxtContainsExpectedFields) {
  SetupMockCompiler(0, "metadata_test_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig("--verbose", "-O2");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(
      hlo, config, "StableHLO", "test_op", "test_cache_key", "TestKernel");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);

  // Verify metadata.txt content - each field on its own line
  std::ifstream metadata_file(workdirs[0] + "/metadata.txt");
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(metadata_file, line)) {
    lines.push_back(line);
  }

  ASSERT_EQ(lines.size(), 8u);
  EXPECT_EQ(lines[0], "test_op");
  EXPECT_EQ(lines[1], "test_cache_key");
  EXPECT_EQ(lines[2], "TestKernel");
  EXPECT_EQ(lines[3], "XLA");
  EXPECT_EQ(lines[4], "trn2");
  EXPECT_EQ(lines[5], "2");
  EXPECT_EQ(lines[6], "-O2");
  EXPECT_EQ(lines[7], "StableHLO");

  // Verify compiler.txt content
  std::ifstream compiler_args_file(workdirs[0] + "/compiler.txt");
  std::string compiler_args_content((std::istreambuf_iterator<char>(compiler_args_file)),
                                    std::istreambuf_iterator<char>());
  EXPECT_NE(compiler_args_content.find("neuronx-cc"), std::string::npos);
  EXPECT_NE(compiler_args_content.find("--verbose"), std::string::npos);
}

TEST_F(NeuronCompilerTest, PreserveArtifacts_StableHloFileContainsInputData) {
  SetupMockCompiler(0, "stablehlo_test_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x11, 0x22, 0x33, 0x44, 0x55};
  CompilationConfig config = CreateTestConfig();
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config, "StableHLO");

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
  std::string stablehlo_path = workdirs[0] + "/module.mlir";

  ASSERT_TRUE(std::filesystem::exists(stablehlo_path)) << "module.mlir not found";

  std::ifstream stablehlo_file(stablehlo_path, std::ios::binary);
  std::vector<uint8_t> stablehlo_content((std::istreambuf_iterator<char>(stablehlo_file)),
                                         std::istreambuf_iterator<char>());

  EXPECT_EQ(stablehlo_content, hlo);
}

TEST_F(NeuronCompilerTest, PreserveArtifacts_AllFilesCreatedTogether) {
  SetupMockCompiler(0, "all_files_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03};
  CompilationConfig config = CreateTestConfig();

  NeuronCompiler::CompileHloToNeff(hlo, config);

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));
}

TEST_F(NeuronCompilerTest, DebugDir_RelativePath_CreatesDirectoryAndWorkdir) {
  SetupMockCompiler(0, "relative_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  std::string relative_debug_dir = "debug_output/neuron_artifacts";

  // Clean up any existing directory from previous runs
  if (std::filesystem::exists(relative_debug_dir)) {
    std::string cmd = "rm -rf " + relative_debug_dir;
    std::system(cmd.c_str());
  }

  setenv("TORCH_NEURONX_DEBUG_DIR", relative_debug_dir.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "relative_path_neff");

  auto workdirs = FindWorkdirs(relative_debug_dir);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));

  std::string cmd = "rm -rf " + relative_debug_dir;
  std::system(cmd.c_str());
}

TEST_F(NeuronCompilerTest, DebugDir_AbsolutePath_CreatesDirectoryAndWorkdir) {
  SetupMockCompiler(0, "absolute_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  std::string absolute_debug_dir = temp_dir_ + "/absolute/path/to/debug";
  setenv("TORCH_NEURONX_DEBUG_DIR", absolute_debug_dir.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "absolute_path_neff");

  auto workdirs = FindWorkdirs(absolute_debug_dir);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));
}

TEST_F(NeuronCompilerTest, DebugDir_NonExistentPath_CreatesEntireHierarchy) {
  SetupMockCompiler(0, "nonexistent_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  std::string nonexistent_debug_dir =
      temp_dir_ + "/does/not/exist/yet/deeply/nested/debug/artifacts";

  ASSERT_FALSE(std::filesystem::exists(nonexistent_debug_dir));

  setenv("TORCH_NEURONX_DEBUG_DIR", nonexistent_debug_dir.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "nonexistent_path_neff");

  EXPECT_TRUE(std::filesystem::exists(nonexistent_debug_dir));

  auto workdirs = FindWorkdirs(nonexistent_debug_dir);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
  EXPECT_TRUE(std::filesystem::exists(workdirs[0] + "/module.mlir"));
}

TEST_F(NeuronCompilerTest, DebugDir_RelativePathWithDotDot_WorksCorrectly) {
  SetupMockCompiler(0, "dotdot_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  std::string subdir = temp_dir_ + "/subdir";
  std::filesystem::create_directories(subdir);

  std::string debug_dir_with_dotdot = subdir + "/../actual_debug";
  setenv("TORCH_NEURONX_DEBUG_DIR", debug_dir_with_dotdot.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "dotdot_path_neff");

  auto workdirs = FindWorkdirs(debug_dir_with_dotdot);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
}

TEST_F(NeuronCompilerTest, DebugDir_PathWithSpaces_WorksCorrectly) {
  SetupMockCompiler(0, "spaces_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  std::string debug_dir_with_spaces = temp_dir_ + "/path with spaces/debug dir";
  setenv("TORCH_NEURONX_DEBUG_DIR", debug_dir_with_spaces.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "spaces_path_neff");

  auto workdirs = FindWorkdirs(debug_dir_with_spaces);
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);
}

TEST_F(NeuronCompilerTest, DebugDir_SingleDotPath_WorksCorrectly) {
  SetupMockCompiler(0, "dot_path_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);

  // Clean up any existing neuronx-cc directory from previous runs
  if (std::filesystem::exists("./neuronx-cc")) {
    std::system("rm -rf ./neuronx-cc");
  }

  setenv("TORCH_NEURONX_DEBUG_DIR", ".", 1);

  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  CompilationConfig config = CreateTestConfig();

  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "dot_path_neff");

  auto workdirs = FindWorkdirs(".");
  ASSERT_EQ(workdirs.size(), 1u);
  VerifyMetadataFiles(workdirs[0]);

  std::system("rm -rf ./neuronx-cc");
}

// ============================================================================
// Additional Args Tokenization Tests - verify single-quote-aware parsing
// ============================================================================

TEST_F(NeuronCompilerTest, AdditionalArgs_SingleQuotedValuePassedAsSingleArg) {
  // Verify that a single-quoted value with an '=' is passed as one argument
  // with quotes stripped. The mock script checks that the argument arrives intact.
  std::string bin_dir = temp_dir_ + "/bin";
  std::filesystem::create_directories(bin_dir);
  std::string path = bin_dir + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "for arg in \"$@\"; do\n"
    << "  if [[ \"$arg\" == \"--options=--foo=bar\" ]]; then\n"
    << "    for ((i=1; i<=$#; i++)); do\n"
    << "      if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "        j=$((i+1)); echo -n 'quoted_ok' > \"${!j}\"; break\n"
    << "      fi\n"
    << "    done\n"
    << "    exit 0\n"
    << "  fi\n"
    << "done\n"
    << "exit 1\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (bin_dir + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig("--options='--foo=bar'");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "quoted_ok");
}

TEST_F(NeuronCompilerTest, AdditionalArgs_QuotedValueWithSpacesIsOneArg) {
  // Verify that spaces inside single quotes do NOT split the token.
  std::string bin_dir = temp_dir_ + "/bin";
  std::filesystem::create_directories(bin_dir);
  std::string path = bin_dir + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "for arg in \"$@\"; do\n"
    << "  if [[ \"$arg\" == \"--msg=hello world\" ]]; then\n"
    << "    for ((i=1; i<=$#; i++)); do\n"
    << "      if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "        j=$((i+1)); echo -n 'space_ok' > \"${!j}\"; break\n"
    << "      fi\n"
    << "    done\n"
    << "    exit 0\n"
    << "  fi\n"
    << "done\n"
    << "exit 1\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (bin_dir + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig("--msg='hello world'");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "space_ok");
}

TEST_F(NeuronCompilerTest, AdditionalArgs_DeepseekV3FlagsTokenizedCorrectly) {
  // Reproduces the exact deepseek_v3_16b case from the bug:
  // "--internal-disable-fma-on-ios
  // --internal-hlo2tensorizer-options='--partitioner-bnt-labels=false'" Should produce exactly 2
  // args (besides the standard ones).
  std::string bin_dir = temp_dir_ + "/bin";
  std::filesystem::create_directories(bin_dir);
  std::string path = bin_dir + "/neuronx-cc";
  std::ofstream f(path);
  // Script checks that both args arrive correctly as separate tokens
  f << "#!/bin/bash\n"
    << "found_fma=0\n"
    << "found_tensorizer=0\n"
    << "for arg in \"$@\"; do\n"
    << "  if [[ \"$arg\" == \"--internal-disable-fma-on-ios\" ]]; then\n"
    << "    found_fma=1\n"
    << "  fi\n"
    << "  if [[ \"$arg\" == \"--internal-hlo2tensorizer-options=--partitioner-bnt-labels=false\" "
       "]]; then\n"
    << "    found_tensorizer=1\n"
    << "  fi\n"
    << "done\n"
    << "if [[ $found_fma -eq 1 && $found_tensorizer -eq 1 ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'deepseek_ok' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "fi\n"
    << "exit 1\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (bin_dir + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig(
      "--internal-disable-fma-on-ios "
      "--internal-hlo2tensorizer-options='--partitioner-bnt-labels=false'");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "deepseek_ok");
}

TEST_F(NeuronCompilerTest, AdditionalArgs_EmptyStringProducesNoExtraArgs) {
  // Empty additional_args should not add any extra arguments.
  // The mock script counts total args: compile, input, --framework, XLA,
  // --target, trn2, --lnc, 2, -O2, --output, module.neff = 11 args
  std::string bin_dir = temp_dir_ + "/bin";
  std::filesystem::create_directories(bin_dir);
  std::string path = bin_dir + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "# With empty additional_args, expect exactly 11 args\n"
    << "if [[ $# -eq 11 ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'empty_ok' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "fi\n"
    << "exit 1\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (bin_dir + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig("");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "empty_ok");
}

TEST_F(NeuronCompilerTest, AdditionalArgs_MultipleSpacesBetweenArgsIgnored) {
  // Multiple spaces between args should not produce empty tokens.
  std::string bin_dir = temp_dir_ + "/bin";
  std::filesystem::create_directories(bin_dir);
  std::string path = bin_dir + "/neuronx-cc";
  std::ofstream f(path);
  f << "#!/bin/bash\n"
    << "found_a=0\n"
    << "found_b=0\n"
    << "for arg in \"$@\"; do\n"
    << "  if [[ \"$arg\" == \"--aaa\" ]]; then found_a=1; fi\n"
    << "  if [[ \"$arg\" == \"--bbb\" ]]; then found_b=1; fi\n"
    << "  if [[ \"$arg\" == \"\" ]]; then exit 1; fi\n"  // fail on empty arg
    << "done\n"
    << "if [[ $found_a -eq 1 && $found_b -eq 1 ]]; then\n"
    << "  for ((i=1; i<=$#; i++)); do\n"
    << "    if [[ \"${!i}\" == \"--output\" ]]; then\n"
    << "      j=$((i+1)); echo -n 'spaces_ok' > \"${!j}\"; break\n"
    << "    fi\n"
    << "  done\n"
    << "  exit 0\n"
    << "fi\n"
    << "exit 1\n";
  f.close();
  chmod(path.c_str(), 0755);
  setenv("PATH", (bin_dir + ":" + original_path_).c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig("  --aaa    --bbb  ");
  std::vector<uint8_t> result = NeuronCompiler::CompileHloToNeff(hlo, config);

  std::string result_str(result.begin(), result.end());
  EXPECT_EQ(result_str, "spaces_ok");
}

TEST_F(NeuronCompilerTest, CompilerTxt_QuotesPreservedForArgsWithSpaces) {
  // Verify that compiler.txt wraps args containing spaces in single quotes.
  // This is the exact pattern from NEURON_CC_FLAGS for multi-core compilation:
  //   --internal-hlo2tensorizer-options='--partition --hierarchy-splitter'
  SetupMockCompiler(0, "quotes_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config =
      CreateTestConfig("--internal-hlo2tensorizer-options='--partition --hierarchy-splitter'");
  NeuronCompiler::CompileHloToNeff(hlo, config);

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);

  std::ifstream compiler_file(workdirs[0] + "/compiler.txt");
  std::string content((std::istreambuf_iterator<char>(compiler_file)),
                      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("'--internal-hlo2tensorizer-options=--partition --hierarchy-splitter'"),
            std::string::npos)
      << "compiler.txt should preserve single quotes around args with spaces. Got: " << content;
}

TEST_F(NeuronCompilerTest, CompilerTxt_NoQuotesForSimpleArgs) {
  // Args without spaces should not be quoted in compiler.txt.
  SetupMockCompiler(0, "noquotes_neff");
  setenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "1", 1);
  setenv("TORCH_NEURONX_DEBUG_DIR", temp_dir_.c_str(), 1);

  std::vector<uint8_t> hlo = {0x01, 0x02};
  CompilationConfig config = CreateTestConfig("--model-type transformer --auto-cast=none");
  NeuronCompiler::CompileHloToNeff(hlo, config);

  auto workdirs = FindWorkdirs(temp_dir_);
  ASSERT_EQ(workdirs.size(), 1u);

  std::ifstream compiler_file(workdirs[0] + "/compiler.txt");
  std::string content((std::istreambuf_iterator<char>(compiler_file)),
                      std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("--model-type transformer --auto-cast=none"), std::string::npos)
      << "Simple args should appear without quotes. Got: " << content;
  EXPECT_EQ(content.find("'--model-type"), std::string::npos)
      << "Simple args should not be quoted. Got: " << content;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  _exit(result);
}
