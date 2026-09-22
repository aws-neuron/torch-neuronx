#include "NeuronGuardImpl.h"

#include "torch_neuronx/csrc/core/Coexistence.h"

namespace c10_neuron {

// Expanded C10_REGISTER_GUARD_IMPL; gated on PrivateUse1 ownership.
namespace {
const bool g_neuron_guard_registered = []() {
  if (torch_neuronx::privateuse1_owned_by_other()) {
    return false;
  }
  static c10::impl::DeviceGuardImplRegistrar registrar(c10::DeviceType::PrivateUse1,
                                                       new NeuronGuardImpl());
  return true;
}();
}  // namespace

}  // namespace c10_neuron
