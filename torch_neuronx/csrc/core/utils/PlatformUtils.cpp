#include "PlatformUtils.h"

#include <cstdlib>
#include <cstring>
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
      {"inf1", {"inf1", "1"}},  {"trn1", {"trn1", "1"}}, {"trn1n", {"trn1", "1"}},
      {"inf2", {"trn1", "1"}},  {"trn2", {"trn2", "2"}}, {"trn2n", {"trn2", "2"}},
      {"trn2u", {"trn2", "2"}}, {"trn3", {"trn3", "2"}}, {"trn3u", {"trn3", "2"}}};

  std::string instance = GetInstanceType();
  auto it = platform_configs.find(instance);
  if (it == platform_configs.end()) {
    throw std::runtime_error(
        "Unknown instance type: " + instance +
        ". Supported types: inf1, trn1, trn1n, inf2, trn2, trn2n, trn2u, trn3");
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
      case NRT_INSTANCE_TRN2U:
        return "trn2u";
      case NRT_INSTANCE_TRN3:
        return "trn3";
      case NRT_INSTANCE_TRN3PDS98:
        return "trn3u";
      default:
        throw std::runtime_error(
            "Unknown Neuron instance family: " + std::to_string(instance_info.family) +
            ". This version of torch-neuronx may need to be updated to support this device.");
    }
  } else {
    throw std::runtime_error("Failed to get Neuron instance information. Status: " +
                             std::to_string(status));
  }
}

std::string GetPlatformTarget() { return GetPlatformConfig().target; }

std::string GetLogicalNeuronCores() {
  // Check environment variable override first
  const char* env_lnc = std::getenv("NEURON_LOGICAL_NC_CONFIG");
  if (env_lnc) {
    return std::string(env_lnc);
  }

  return GetPlatformConfig().logical_cores;
}

bool IsSyncModeEnabled() {
  static bool is_sync_mode_enabled = []() {
    const char* sync_mode = std::getenv("NEURON_LAUNCH_BLOCKING");
    return sync_mode != nullptr && strcmp(sync_mode, "1") == 0;
  }();
  return is_sync_mode_enabled;
}

}  // namespace utils
}  // namespace at::neuron
