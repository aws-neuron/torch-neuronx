#include "NeuronCompiler.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "torch_neuronx/csrc/core/compilation/CompilerDebugUtils.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

namespace at::neuron {

CompilationConfig CompilationConfig::FromEnvironment() {
  CompilationConfig config;
  config.framework = "XLA";
  config.platform_target = utils::GetPlatformTarget();
  config.logical_neuron_cores = utils::GetLogicalNeuronCores();
  const char* opt_env = std::getenv("NEURON_COMPILER_OPT_LEVEL");
  config.optimization_level = opt_env ? opt_env : "-O1";
  const char* flags_env = std::getenv("NEURON_CC_FLAGS");
  if (flags_env) config.additional_args = flags_env;
  return config;
}

namespace {

// File names used during compilation process.
constexpr const char* kHloFilename = "module.hlo.pb";
constexpr const char* kMlirFilename = "module.mlir";
constexpr const char* kBirFilename = "bir.json";
constexpr const char* kNeffFilename = "module.neff";

// Compiler executable constant.
constexpr const char* kNeuronxCCExecutable = "neuronx-cc";

// Command line arguments for neuronx-cc.
constexpr const char* kCompileCommand = "compile";
constexpr const char* kFrameworkFlag = "--framework";
constexpr const char* kTargetFlag = "--target";
constexpr const char* kLogicalNeuronCoresFlag = "--lnc";
constexpr const char* kOutputFlag = "--output";
constexpr const char* kWorkingDirFlag = "--workdir";

}  // namespace

std::vector<uint8_t> NeuronCompiler::CompileHloToNeff(const std::vector<uint8_t>& hlo_bytes,
                                                      const CompilationConfig& config,
                                                      const std::string& ir_type,
                                                      const std::string& op_name,
                                                      const std::string& cache_key,
                                                      const std::string& kernel_class_name) {
  // Create temporary directory using RAII wrapper
  TempDirectory temp_compilation_dir("neuron_compile_");

  // Determine file format based on ir_type parameter
  // "BIR" requires sg00/ subdirectory, "StableHLO" uses MLIR format, "XLA" (default) uses HLO
  // protobuf
  std::string hlo_input_file;
  if (ir_type == "BIR") {
    std::string sg00_dir = temp_compilation_dir.path() + "/sg00";
    std::filesystem::create_directories(sg00_dir);
    hlo_input_file = sg00_dir + "/" + kBirFilename;
  } else {
    hlo_input_file =
        temp_compilation_dir.path() + "/" + (ir_type == "StableHLO" ? kMlirFilename : kHloFilename);
  }
  std::string neff_output_file = temp_compilation_dir.path() + "/" + kNeffFilename;

  // Write HLO bytes to temporary file
  temp_compilation_dir.WriteFile(hlo_input_file.substr(temp_compilation_dir.path().size() + 1),
                                 hlo_bytes);

  // Build neuronx-cc command arguments using config
  // execvp will search PATH automatically
  std::vector<std::string> compiler_args = {kNeuronxCCExecutable,  kCompileCommand,  hlo_input_file,
                                            kFrameworkFlag,        config.framework, kTargetFlag,
                                            config.platform_target};

  if (!config.logical_neuron_cores.empty()) {
    compiler_args.push_back(kLogicalNeuronCoresFlag);
    compiler_args.push_back(config.logical_neuron_cores);
  }
  if (!config.optimization_level.empty()) {
    compiler_args.push_back(config.optimization_level);
  }

  // Parse and add additional compiler arguments
  std::istringstream additional_args_stream(config.additional_args);
  std::string single_arg;
  while (additional_args_stream >> single_arg) {
    compiler_args.push_back(single_arg);
  }

  // Add output specification
  compiler_args.push_back(kOutputFlag);
  compiler_args.push_back(neff_output_file);

  // Execute neuronx-cc in the temporary directory
  ProcessExecutor::Execute(compiler_args, std::string(kNeuronxCCExecutable) + " compilation",
                           temp_compilation_dir.path());

  return temp_compilation_dir.ReadFile(kNeffFilename);
}

bool NeuronCompiler::IsStableHLOEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TORCH_NEURONX_ENABLE_STABLEHLO");
    // Default to enabled (true). Only disable if explicitly set to "0" or "false"
    if (!env) {
      return true;  // Default: enabled
    }
    std::string env_str(env);
    return !(env_str == "0" || env_str == "false");
  }();
  return enabled;
}

}  // namespace at::neuron
