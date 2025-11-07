#include "KernelExecution.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "torch_neuron/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuron/csrc/core/utils/PlatformUtils.h"

namespace at::neuron {

// NeuronKernelExecution implementation
torch::Tensor NeuronKernelExecution::ConvertToNeuronDtype(const torch::Tensor& tensor) const {
  const auto dtype = tensor.scalar_type();
  if (!NeedsDtypeConversion(dtype)) {
    return tensor;
  }

  torch::ScalarType target_dtype;
  switch (dtype) {
    case torch::kInt64:
      target_dtype = torch::kInt32;
      break;
    case torch::kFloat64:
      target_dtype = torch::kFloat32;
      break;
    default:
      throw std::logic_error("Unexpected dtype conversion request for type: " +
                             std::string(c10::toString(dtype)) + " in op '" + GetOpName() + "'");
  }

  try {
    return tensor.to(target_dtype, /*non_blocking=*/true);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to convert tensor dtype from " + std::string(c10::toString(dtype)) + " to " +
        std::string(c10::toString(target_dtype)) + " for op '" + GetOpName() + "': " + e.what());
  }
}

// CompilableKernelExecution implementation

std::string CompilableKernelExecution::CreateCompilerExtensionHash(
    const std::string& additional_args, const std::string& optimization_level) {
  // Build extension from compiler-specific parameters (no hashing for readability)
  std::string platform_target = utils::GetPlatformTarget();
  std::string logical_cores = utils::GetLogicalNeuronCores();

  // Remove leading dash from optimization level for cleaner key
  std::string opt_level_suffix = optimization_level;
  if (!opt_level_suffix.empty() && opt_level_suffix[0] == '-') {
    opt_level_suffix = opt_level_suffix.substr(1);
  }

  // Sanitize additional_args for use in cache key (replace spaces with underscores)
  std::string sanitized_args = additional_args;
  std::replace(sanitized_args.begin(), sanitized_args.end(), ' ', '_');
  std::replace(sanitized_args.begin(), sanitized_args.end(), '=', '-');

  // Build extension: <target>_<lnc>_<opt_level>_<additional_args>
  std::string extension = platform_target + "_" + logical_cores + "_" + opt_level_suffix;

  if (!sanitized_args.empty()) {
    extension += "_" + sanitized_args;
  }

  return extension;
}

std::string CompilableKernelExecution::GenerateCacheKey(const std::string& base_key,
                                                        const std::string& additional_args,
                                                        const std::string& optimization_level) {
  std::string compiler_extension = CreateCompilerExtensionHash(additional_args, optimization_level);
  return base_key + "_" + compiler_extension;
}

bool CompilableKernelExecution::ValidateImpl() const {
  return !cache_key_.empty() && cache_key_[0] != '_';
}

// XLACompilableKernelExecution implementation

std::string XLACompilableKernelExecution::GetXLAAdditionalArgs() {
  // Base XLA compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided XLA flags from environment
  if (const char* cc_flags_env = std::getenv("NEURON_CC_FLAGS")) {
    args += " " + std::string(cc_flags_env);
  }

  return args;
}

std::string XLACompilableKernelExecution::GetXLAOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O1";
}

XLACompilableKernelExecution::XLACompilableKernelExecution(
    const std::string& op_name, const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs, const std::string& base_cache_key,
    const std::vector<uint8_t>& hlo_bytes, bool has_collectives)
    : CompilableKernelExecution(
          op_name, inputs, outputs,
          GenerateCacheKey(base_cache_key, GetXLAAdditionalArgs(), GetXLAOptimizationLevel()),
          GetXLAAdditionalArgs(), GetXLAOptimizationLevel(), has_collectives),
      hlo_bytes_(hlo_bytes) {}

bool XLACompilableKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !hlo_bytes_.empty();
}

std::vector<uint8_t> XLACompilableKernelExecution::CompileToNeff() const {
  if (hlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no HLO bytes available");
  }

  try {
    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(hlo_bytes_, config);
  } catch (const std::exception& e) {
    throw std::runtime_error("HLO to NEFF compilation failed for op '" + GetOpName() +
                             "': " + e.what());
  }
}

// NKICompilableKernelExecution implementation

std::string NKICompilableKernelExecution::GetNKIAdditionalArgs() {
  // Base NKI compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided NKI flags from environment
  if (const char* nki_flags_env = std::getenv("NEURON_NKI_COMPILE_FLAGS")) {
    args += " " + std::string(nki_flags_env);
  }

  return args;
}

std::string NKICompilableKernelExecution::GetNKIOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O1";
}

NKICompilableKernelExecution::NKICompilableKernelExecution(
    const std::string& op_name, const std::string& kernel_name, const std::string& module_path,
    const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& outputs,
    const std::string& base_cache_key, bool has_collectives)
    : CompilableKernelExecution(
          op_name, inputs, outputs,
          GenerateCacheKey(base_cache_key, GetNKIAdditionalArgs(), GetNKIOptimizationLevel()),
          GetNKIAdditionalArgs(), GetNKIOptimizationLevel(), has_collectives),
      nki_kernel_name_(kernel_name),
      nki_module_path_(module_path) {}

bool NKICompilableKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !nki_kernel_name_.empty() &&
         !nki_module_path_.empty();
}

std::vector<uint8_t> NKICompilableKernelExecution::CompileToNeff() const {
  // TODO: Implement NKI compilation
  throw std::logic_error("NKI compilation not yet implemented for kernel: " + nki_kernel_name_);
}

// CollectiveDirectKernelExecution implementation

CollectiveDirectKernelExecution::CollectiveDirectKernelExecution(
    const std::string& op_name, CollectiveType collective_type,
    const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& outputs,
    c10d::ReduceOp reduce_op)
    : DirectKernelExecution(op_name, inputs, outputs),
      collective_type_(collective_type),
      reduce_op_(reduce_op) {}

bool CollectiveDirectKernelExecution::ValidateImpl() const {
  return !GetInputs().empty() && !GetOutputs().empty();
}

}  // namespace at::neuron
