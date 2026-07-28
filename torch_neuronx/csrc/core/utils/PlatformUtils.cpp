#include "PlatformUtils.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// Include Neuron Runtime headers
extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {
namespace utils {

namespace {

// Platform configuration for Neuron compilation.
// Contains target architecture and logical core settings for each instance type.
struct PlatformConfig {
  std::string target;
  std::string logical_cores;
};

// Queries NRT for instance information. Throws on failure.
nrt_instance_info_t QueryInstanceInfo() {
  nrt_instance_info_t info{};
  NRT_STATUS status = nrt_get_instance_info(&info, sizeof(info));
  if (status != NRT_SUCCESS) {
    throw std::runtime_error("Failed to get Neuron instance information. Status: " +
                             std::to_string(status));
  }
  return info;
}

// Maps NRT instance family to a readable instance type name.
std::string FamilyToInstanceType(uint32_t family) {
  switch (family) {
    case NRT_INSTANCE_TRN1:
      return "trn1";
    case NRT_INSTANCE_TRN1N:
      return "trn1n";
    case NRT_INSTANCE_INF1:
      return "inf1";
    case NRT_INSTANCE_INF2:
      return "inf2";
    case NRT_INSTANCE_TRN2:
      return "trn2";
    case NRT_INSTANCE_TRN2N:
      return "trn2n";
    case NRT_INSTANCE_TRN2U:
      return "trn2u";
    case NRT_INSTANCE_TRN3:
      return "trn3";
    case NRT_INSTANCE_TRN3PDS98:
      return "trn3u";
    default:
      throw std::runtime_error(
          "Unknown Neuron instance family: " + std::to_string(family) +
          ". This version of torch-neuronx may need to be updated to support this device.");
  }
}

// Gets the platform configuration for the given instance type.
const PlatformConfig& GetPlatformConfigFor(const std::string& instance_type) {
  static const std::unordered_map<std::string, PlatformConfig> platform_configs = {
      {"inf1", {"inf1", "1"}},  {"trn1", {"trn1", "1"}}, {"trn1n", {"trn1", "1"}},
      {"inf2", {"trn1", "1"}},  {"trn2", {"trn2", "2"}}, {"trn2n", {"trn2", "2"}},
      {"trn2u", {"trn2", "2"}}, {"trn3", {"trn3", "2"}}, {"trn3u", {"trn3", "2"}}};

  auto it = platform_configs.find(instance_type);
  if (it == platform_configs.end()) {
    throw std::runtime_error(
        "Unknown instance type: " + instance_type +
        ". Supported types: inf1, trn1, trn1n, inf2, trn2, trn2n, trn2u, trn3, trn3u");
  }
  return it->second;
}

}  // namespace

std::string GetInstanceType() { return FamilyToInstanceType(QueryInstanceInfo().family); }

std::string GetPlatformTarget() {
  const char* env_target = std::getenv("NEURON_PLATFORM_TARGET_OVERRIDE");
  if (env_target && env_target[0] != '\0') {
    return std::string(env_target);
  }
  const auto info = QueryInstanceInfo();
  std::string instance_type = FamilyToInstanceType(info.family);
  return GetPlatformConfigFor(instance_type).target;
}

std::string GetLogicalNeuronCores() {
  // Check environment variable override first
  const char* env_lnc = std::getenv("NEURON_LOGICAL_NC_CONFIG");
  if (env_lnc) {
    return std::string(env_lnc);
  }
  return GetPlatformConfigFor(GetInstanceType()).logical_cores;
}

std::string GetCompilerVersion() {
  const char* env_version = std::getenv("NEURONXCC_VERSION");
  if (env_version && env_version[0] != '\0') {
    return std::string(env_version);
  }
  return "";
}

bool IsHloContentKeyEnabled() {
  const char* v = std::getenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
  return v && std::string(v) == "1";
}

}  // namespace utils
}  // namespace at::neuron
