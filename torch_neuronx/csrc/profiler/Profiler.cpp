#include "Profiler.h"

#include "LogWrapper.h"
#include "Session.h"

namespace at::neuron {

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
  // Gate: only start if custom config contains neuron_profile key
  const std::string& customConfig = config.getCustomConfig();
  if (!hasNeuronProfileKey(customConfig)) {
    NPROF_INFO("No neuron_profile config; skipping session creation");
    return nullptr;
  }

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

  return std::make_unique<NeuronActivityProfilerSession>(supportedTypes, config);
}

std::unique_ptr<libkineto::IActivityProfilerSession> NeuronActivityProfiler::configure(
    int64_t ts_ms, int64_t duration_ms, const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  NPROF_INFO("Asked to configure async.");
  return configure(activityTypes, config);
}
}  // namespace at::neuron
