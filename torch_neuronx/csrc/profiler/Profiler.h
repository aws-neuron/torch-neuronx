/*
 * Neuron Activity Profiler
 *
 * Implements IActivityProfiler and IActivityProfilerSession interfaces
 * for integrating Neuron hardware profiling with libkineto/PyTorch.
 *
 * This implementation integrates with NRT inspect APIs to capture
 * hardware profiling data from Neuron devices.
 */

#pragma once

#include <ActivityType.h>
#include <IActivityProfiler.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace at::neuron {
class NeuronActivityProfiler : public libkineto::IActivityProfiler {
 public:
  NeuronActivityProfiler();
  ~NeuronActivityProfiler() override;

  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms, int64_t duration_ms, const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

 private:
  static inline const std::string name_{"__neuron_profiler__"};

  // The subset of the libkineto::ActivityType that Neuron supports.
  static inline const std::set<libkineto::ActivityType> availableActivities_{
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
      libkineto::ActivityType::PRIVATEUSE1_DRIVER,
  };

  // Check if custom config string contains the neuron_profile key.
  static bool hasNeuronProfileKey(const std::string& customConfig);
};

}  // namespace at::neuron
