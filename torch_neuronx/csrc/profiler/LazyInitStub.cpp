// Weak symbol stub for maybe_lazy_init().
// This allows the runtime library to link independently of neuron_bindings.
// When neuron_bindings is also linked (e.g., in production builds), its
// strong symbol definition will override this weak stub.

namespace torch_neuronx {

// Weak symbol - will be overridden by the real implementation in NeuronBindings.cpp
__attribute__((weak)) void maybe_lazy_init() {
  // No-op stub - the real implementation initializes NRT
}

}  // namespace torch_neuronx
