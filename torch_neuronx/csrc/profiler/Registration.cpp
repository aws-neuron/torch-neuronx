#include <libkineto.h>

#include <mutex>

#include "LogWrapper.h"
#include "Profiler.h"

namespace at::neuron {

namespace {
std::once_flag g_registration_flag;
}  // namespace

bool registerNeuronProfiler() {
  std::call_once(g_registration_flag, []() {
    if (!libkineto::api().isProfilerRegistered()) {
      NPROF_INFO("libkineto profiler not registered.",
                 "This may be expected if PyTorch profiling is not initialized yet.");
    }

    libkineto::api().registerProfilerFactory([]() -> std::unique_ptr<libkineto::IActivityProfiler> {
      NPROF_INFO("Factory called - creating profiler instance");
      return std::make_unique<NeuronActivityProfiler>();
    });

    NPROF_INFO("Registration complete");
  });
  return true;
}

}  // namespace at::neuron
