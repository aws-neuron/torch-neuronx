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

// Gets the platform configuration for the current instance type.
const PlatformConfig& GetPlatformConfig() {
  static const std::unordered_map<std::string, PlatformConfig> platform_configs = {
      {"inf1", {"inf1", "1"}},  {"trn1", {"trn1", "1"}},  {"trn1n", {"trn1", "1"}},
      {"inf2", {"trn1", "1"}},  {"inf2e", {"trn1", "1"}}, {"trn2", {"trn2", "2"}},
      {"trn2n", {"trn2", "2"}}, {"trn2p", {"trn2", "2"}}, {"trn2u", {"trn2", "2"}}};

  std::string instance = GetInstanceType();
  auto it = platform_configs.find(instance);
  if (it == platform_configs.end()) {
    throw std::runtime_error(
        "Unknown instance type: " + instance +
        ". Supported types: inf1, trn1, trn1n, inf2, inf2e, trn2, trn2n, trn2p, trn2u");
  }
  return it->second;
}

}  // namespace

std::string GetInstanceType() {
  // Get instance information
  nrt_instance_info_t instance_info;
  NRT_STATUS status = nrt_get_instance_info(&instance_info, sizeof(instance_info));

  if (status == NRT_SUCCESS) {
    // Map instance family to a readable name
    switch (instance_info.family) {
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
      case NRT_INSTANCE_INF2E:
        return "inf2e";
      case NRT_INSTANCE_TRN2P:
        return "trn2p";
      case NRT_INSTANCE_TRN2U:
        return "trn2u";
      // todo: Mariana support
      // case NRT_INSTANCE_TRN3:
      //   return "trn3";
      default:
        throw std::runtime_error(
            "Unknown Neuron instance family: " + std::to_string(instance_info.family) +
            ". This version of torch-neuron may need to be updated to support this device.");
    }
  } else {
    throw std::runtime_error("Failed to get Neuron instance information. Status: " +
                             std::to_string(status));
  }
}

std::string GetPlatformTarget() { return GetPlatformConfig().target; }

std::string GetLogicalNeuronCores() {
  // Check environment variable override first
  const char* env_lnc = std::getenv("NEURON_RT_VIRTUAL_CORE_SIZE");
  if (env_lnc) {
    return std::string(env_lnc);
  }

  return GetPlatformConfig().logical_cores;
}

}  // namespace utils
}  // namespace at::neuron
