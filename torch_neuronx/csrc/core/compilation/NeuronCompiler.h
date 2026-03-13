#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace at::neuron {

// Configuration for neuronx-cc compilation.
struct CompilationConfig {
  // Framework type for compilation (e.g., "XLA", "NKI")
  std::string framework;

  // Platform target (e.g., "trn1", "trn2", "inf1")
  std::string platform_target;

  // Logical neuron cores setting (e.g., "1", "2")
  std::string logical_neuron_cores;

  // Compiler optimization level (e.g., "-O1", "-O2")
  std::string optimization_level;

  // Additional compiler arguments
  std::string additional_args;
};

// NeuronCompiler provides functionality to compile modules to NEFF format
// using the neuronx-cc compiler subprocess. This class handles the entire
// compilation pipeline from input bytes to executable NEFF files.
class NeuronCompiler {
 public:
  // Compiles HLO bytes to NEFF format using the neuronx-cc compiler.
  //
  // Args:
  //   hlo_bytes: The HLO module serialized as protobuf bytes or MLIR bytes
  //   config: Compilation configuration with framework, platform, and optimization settings
  //   ir_type: Type of IR being compiled ("XLA" for HLO protobuf, "StableHLO" for MLIR)
  //   op_name: Optional operation name for IR dumping
  //   cache_key: Optional cache key for IR dumping
  //   kernel_class_name: Optional kernel class name for IR dumping
  //
  // Returns:
  //   The compiled NEFF file as a byte vector
  //
  // Throws:
  //   std::runtime_error if compilation fails or neuronx-cc is not found
  static std::vector<uint8_t> CompileHloToNeff(
      const std::vector<uint8_t>& hlo_bytes, const CompilationConfig& config,
      const std::string& ir_type = "XLA", const std::string& op_name = "",
      const std::string& cache_key = "", const std::string& kernel_class_name = "UnknownKernel");

  // Checks if StableHLO lowering is enabled via the TORCH_NEURONX_ENABLE_STABLEHLO environment
  // variable.
  //
  // Returns:
  //   true by default. false only if TORCH_NEURONX_ENABLE_STABLEHLO is explicitly set to "0" or
  //   "false"
  static bool IsStableHLOEnabled();
};

}  // namespace at::neuron
