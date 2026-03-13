#include "TensorizerCompiler.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/utils/ProcessExecutor.h"
#include "torch_neuronx/csrc/core/utils/TempDirectory.h"

namespace at::neuron {

namespace {

constexpr const char* kLinalgFilename = "module.linalg.mlir";
constexpr const char* kNisaFilename = "model.nisa.mlir";
constexpr const char* kBirFilename = "bir.json";

constexpr const char* kDefaultNeuronOptPath = "neuron-opt";
constexpr const char* kDefaultNkiOptPath = "nki-opt";

}  // namespace

std::string TensorizerCompiler::GetNeuronOptPath() {
  static const std::string path = []() {
    const char* env = std::getenv("NEURON_OPT_PATH");
    return env ? std::string(env) : std::string(kDefaultNeuronOptPath);
  }();
  return path;
}

std::string TensorizerCompiler::GetNkiOptPath() {
  static const std::string path = []() {
    const char* env = std::getenv("NKI_OPT_PATH");
    return env ? std::string(env) : std::string(kDefaultNkiOptPath);
  }();
  return path;
}

void TensorizerCompiler::CompileLinalgToNisa(const std::string& linalg_path,
                                             const std::string& nisa_path) {
  std::vector<std::string> args = {GetNeuronOptPath(), linalg_path, "--neuron-tensor-opt-pipeline",
                                   "-o", nisa_path};
  ProcessExecutor::Execute(args, "neuron-opt (Linalg -> NISA)");
}

void TensorizerCompiler::CompileNisaToBir(const std::string& nisa_path,
                                          const std::string& bir_path) {
  std::string pipeline_arg =
      "--nki-opt-pipeline=enable-linear-scan-allocation=true output-path=" + bir_path;
  std::vector<std::string> args = {GetNkiOptPath(), nisa_path, pipeline_arg};
  ProcessExecutor::Execute(args, "nki-opt (NISA -> BIR)");
}

void TensorizerCompiler::RenameBirTensors(const std::string& bir_path) {
  // Read BIR file content
  std::ifstream in(bir_path);
  if (!in) {
    throw std::runtime_error("Failed to read BIR file: " + bir_path);
  }
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();

  // Replace in_tensor_X -> inputX and out_tensor_X -> outputX
  // Simple prefix-based replacement: scan for "in_tensor_" and "out_tensor_" prefixes
  std::string result;
  result.reserve(content.size());

  for (size_t i = 0; i < content.size();) {
    if (content.compare(i, 10, "in_tensor_") == 0) {
      result += "input";
      i += 10;  // Skip "in_tensor_", keep the digit(s) that follow
    } else if (content.compare(i, 11, "out_tensor_") == 0) {
      result += "output";
      i += 11;  // Skip "out_tensor_", keep the digit(s) that follow
    } else {
      result += content[i];
      ++i;
    }
  }

  // Write modified content back
  std::ofstream out(bir_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to write BIR file: " + bir_path);
  }
  out << result;
}

std::vector<uint8_t> TensorizerCompiler::CompileBirToNeff(const std::string& bir_path,
                                                          const std::string& platform_target) {
  // Rename tensor names to match executor naming convention
  RenameBirTensors(bir_path);

  // Read BIR file as bytes
  std::ifstream bir_in(bir_path, std::ios::binary | std::ios::ate);
  if (!bir_in) {
    throw std::runtime_error("Failed to read BIR file: " + bir_path);
  }
  std::streamsize bir_size = bir_in.tellg();
  bir_in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bir_bytes(bir_size);
  if (!bir_in.read(reinterpret_cast<char*>(bir_bytes.data()), bir_size)) {
    throw std::runtime_error("Failed to read BIR data: " + bir_path);
  }

  // Build config with BIR-specific flags in additional_args
  std::string additional_args =
      "--verbose=info --pipeline compile SaveTemps "
      "--enable-internal-bir-e2e-compilation "
      "--internal-backend-options=--skip-pass=unroll";
  if (const char* cc_flags = std::getenv("NEURON_CC_FLAGS")) {
    additional_args += " ";
    additional_args += cc_flags;
  }

  CompilationConfig config{.framework = "XLA",
                           .platform_target = platform_target,
                           .logical_neuron_cores = "",
                           .optimization_level = "",
                           .additional_args = additional_args};

  return NeuronCompiler::CompileHloToNeff(bir_bytes, config, "BIR");
}

std::vector<uint8_t> TensorizerCompiler::CompileLinalgToNeff(
    const std::vector<uint8_t>& linalg_bytes, const std::string& platform_target) {
  // Create temporary directory for compilation artifacts
  TempDirectory temp_dir("neuron_tensorizer_");

  // Write linalg bytes to file
  temp_dir.WriteFile(kLinalgFilename, linalg_bytes);
  std::string linalg_path = temp_dir.path() + "/" + kLinalgFilename;

  // Run tensorizer pipeline: Linalg -> NISA -> BIR -> NEFF
  std::string nisa_path = temp_dir.path() + "/" + kNisaFilename;
  std::string bir_path = temp_dir.path() + "/" + kBirFilename;

  CompileLinalgToNisa(linalg_path, nisa_path);
  CompileNisaToBir(nisa_path, bir_path);
  return CompileBirToNeff(bir_path, platform_target);
}

}  // namespace at::neuron
