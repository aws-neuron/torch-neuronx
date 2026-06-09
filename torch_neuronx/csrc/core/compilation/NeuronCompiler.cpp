#include "NeuronCompiler.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "torch_neuronx/csrc/core/cache/CacheUtils.h"
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
  std::string workdir = cache_utils::GetCompilationWorkdir();
  bool preserve = cache_utils::ShouldPreserveCompilationArtifacts();
  TempDirectory temp_compilation_dir(workdir, /*is_full_path=*/true, preserve);
  const std::string& workdir_str = temp_compilation_dir.path();

  // "BIR" requires sg00/ subdirectory; everything else is StableHLO (MLIR).
  std::string hlo_input_file;
  if (ir_type == "BIR") {
    std::filesystem::create_directories(temp_compilation_dir.path() + "/sg00");
    hlo_input_file = std::string("sg00/") + kBirFilename;
  } else {
    hlo_input_file = kMlirFilename;
  }

  // Write HLO bytes to temporary file
  temp_compilation_dir.WriteFile(hlo_input_file, hlo_bytes);

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

  // Parse and add additional compiler arguments (handles single-quoted values)
  {
    const std::string& args = config.additional_args;
    std::string current;
    bool in_quotes = false;
    for (size_t i = 0; i < args.size(); ++i) {
      char c = args[i];
      if (c == '\'') {
        in_quotes = !in_quotes;
      } else if (c == ' ' && !in_quotes) {
        if (!current.empty()) {
          compiler_args.push_back(current);
          current.clear();
        }
      } else {
        current += c;
      }
    }
    if (!current.empty()) {
      compiler_args.push_back(current);
    }
  }

  compiler_args.push_back(kOutputFlag);
  compiler_args.push_back(kNeffFilename);

  if (preserve) {
    std::string metadata_path = workdir_str + "/metadata.txt";
    std::ofstream metadata(metadata_path);
    if (metadata) {
      metadata << op_name << "\n";
      metadata << cache_key << "\n";
      metadata << kernel_class_name << "\n";
      metadata << config.framework << "\n";
      metadata << config.platform_target << "\n";
      metadata << config.logical_neuron_cores << "\n";
      metadata << config.optimization_level << "\n";
      metadata << ir_type << "\n";
    }

    std::string compiler_args_path = workdir_str + "/compiler.txt";
    std::ofstream compiler_args_file(compiler_args_path);
    if (compiler_args_file) {
      for (const auto& arg : compiler_args) {
        if (arg.find(' ') != std::string::npos) {
          compiler_args_file << "'" << arg << "' ";
        } else {
          compiler_args_file << arg << " ";
        }
      }
    }
  }

  // Execute neuronx-cc in the workdir
  ProcessExecutor::Execute(compiler_args, std::string(kNeuronxCCExecutable) + " compilation",
                           workdir_str);

  return temp_compilation_dir.ReadFile(kNeffFilename);
}

}  // namespace at::neuron
