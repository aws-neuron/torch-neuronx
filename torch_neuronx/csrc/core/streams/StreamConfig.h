#pragma once

#include <c10/util/Exception.h>

#include <cstdlib>
#include <string_view>

namespace at::neuron::config {

// streams=0, host_cc=0: alias user streams to the default stream.
// streams=1, host_cc=0: keep distinct streams; submit compiled collectives to the default stream.
// streams=1, host_cc=1: keep distinct streams and execute collectives through HostCC.
struct StreamConfig {
  bool streams_enabled;
  bool host_cc_enabled;
};

inline constexpr const char* kStreamsEnabledEnv = "TORCH_NEURONX_ENABLE_STREAMS";
inline constexpr const char* kHostCCEnabledEnv = "TORCH_NEURONX_ENABLE_HOST_CC";

inline bool ParseBooleanEnvironment(const char* name) {
  const char* value = std::getenv(name);
  TORCH_CHECK(
      value != nullptr && value[0] != '\0', name,
      " is missing or empty. Import torch_neuronx before C++ stream configuration is read.");

  const std::string_view configured(value);
  if (configured == "0") return false;
  if (configured == "1") return true;

  TORCH_CHECK(false, name, " must be 0 or 1; got '", configured, "'");
  return false;
}

inline const StreamConfig& GetStreamConfig() {
  static const StreamConfig config = []() {
    StreamConfig configured{ParseBooleanEnvironment(kStreamsEnabledEnv),
                            ParseBooleanEnvironment(kHostCCEnabledEnv)};
    TORCH_CHECK(!configured.host_cc_enabled || configured.streams_enabled, kHostCCEnabledEnv,
                "=1 requires ", kStreamsEnabledEnv, "=1");
    return configured;
  }();
  return config;
}

inline bool IsHostCCEnabled() { return GetStreamConfig().host_cc_enabled; }

inline bool AreStreamsEnabled() { return GetStreamConfig().streams_enabled; }

}  // namespace at::neuron::config
