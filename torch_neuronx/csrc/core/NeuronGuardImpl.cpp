#include "NeuronGuardImpl.h"

namespace c10_neuron {

// Register the guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, NeuronGuardImpl);

}  // namespace c10_neuron
