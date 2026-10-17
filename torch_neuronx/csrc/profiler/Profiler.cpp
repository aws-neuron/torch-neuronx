#include "Profiler.h"

#include <algorithm>
#include <cstdlib>
#include <optional>

#include "LogWrapper.h"
#include "NConfigParser.h"
#include "Session.h"

namespace at::neuron {

namespace {

std::string envOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? value : fallback;
}

std::optional<std::string> defaultSessionConfig() {
  if (envOr("TORCH_NEURONX_PROFILE_DEFAULT_SESSION", "1") == "0") {
    return std::nullopt;
  }
  const auto ranks =
      torch_neuronx::profiler::detail::splitAndTrim(envOr("TORCH_NEURONX_PROFILE_RANKS", "0"), ',');
  if (std::find(ranks.begin(), ranks.end(), envOr("RANK", "0")) == ranks.end()) {
    return std::nullopt;
  }
  std::string config =
      "neuron_profile:" + envOr("TORCH_NEURONX_PROFILE_MODES", "runtime,cpu_util,host_memory");
  config += ";max_events_per_nc:" + envOr("TORCH_NEURONX_PROFILE_MAX_EVENTS_PER_NC", "10000000");
  config += ";capture_enabled_for_nc:" + envOr("TORCH_NEURONX_PROFILE_CAPTURE_NC", "0,1");
  config += ";profile_output_dir:" + envOr("TORCH_NEURONX_PROFILE_OUTPUT_DIR", "neuron_profile");
  config += ";neff_cache_dir:" + envOr("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache");
  return config;
}

}  // namespace

NeuronActivityProfiler::NeuronActivityProfiler() { NPROF_INFO("Profiler CREATED"); }

NeuronActivityProfiler::~NeuronActivityProfiler() { NPROF_INFO("Profiler DESTROYED"); }

const std::string& NeuronActivityProfiler::name() const { return name_; }

const std::set<libkineto::ActivityType>& NeuronActivityProfiler::availableActivities() const {
  return availableActivities_;
}

bool NeuronActivityProfiler::hasNeuronProfileKey(const std::string& customConfig) {
  // NeuronConfig always serializes "neuron_profile:..." as the first key in the
  // config string, so a prefix check is sufficient.
  constexpr const char* kKey = "neuron_profile:";
  return customConfig.rfind(kKey, 0) == 0;
}

std::unique_ptr<libkineto::IActivityProfilerSession> NeuronActivityProfiler::configure(
    const std::set<libkineto::ActivityType>& activityTypes, const libkineto::Config& config) {
  std::set<libkineto::ActivityType> supportedTypes;

  // Filter in only the Neuron supported activities.
  for (auto type : activityTypes) {
    NPROF_INFO("  - ActivityType:", static_cast<int>(type));
    if (availableActivities_.count(type)) {
      NPROF_INFO("    - Is supported");
      supportedTypes.insert(type);
    } else {
      NPROF_INFO("    - NOT supported");
    }
  }

  const std::string& customConfig = config.getCustomConfig();
  if (hasNeuronProfileKey(customConfig)) {
    return std::make_unique<NeuronActivityProfilerSession>(supportedTypes, config);
  }

  // PyTorch passes PRIVATEUSE1_RUNTIME/DRIVER even for CPU-only profiling. It adds
  // CONCURRENT_KERNEL only for ProfilerActivity.PrivateUse1.
  if (!activityTypes.count(libkineto::ActivityType::CONCURRENT_KERNEL)) {
    NPROF_INFO("No neuron_profile config and no device activity; skipping session creation");
    return nullptr;
  }
  const auto defaultConfig = defaultSessionConfig();
  if (!defaultConfig) {
    NPROF_INFO("No neuron_profile config and default session disabled; skipping session creation");
    return nullptr;
  }
  std::string sessionCustomConfig =
      customConfig.empty() ? *defaultConfig : *defaultConfig + ";" + customConfig;
  NPROF_INFO("No neuron_profile config; starting default session:", sessionCustomConfig);
  auto sessionConfig = config.clone();
  sessionConfig->handleOption("CUSTOM_CONFIG", sessionCustomConfig);
  return std::make_unique<NeuronActivityProfilerSession>(supportedTypes, *sessionConfig);
}

std::unique_ptr<libkineto::IActivityProfilerSession> NeuronActivityProfiler::configure(
    int64_t ts_ms, int64_t duration_ms, const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  NPROF_INFO("Asked to configure async.");
  return configure(activityTypes, config);
}
}  // namespace at::neuron
