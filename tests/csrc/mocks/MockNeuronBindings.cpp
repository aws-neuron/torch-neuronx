#include "MockNeuronBindings.h"

// Flag to check if mocks are initialized
static bool g_neuron_bindings_mocks_initialized = false;

namespace torch_neuronx {

void maybe_lazy_init() {
  if (g_neuron_bindings_mocks_initialized) {
    testing::MockNeuronBindings::GetInstance()->maybe_lazy_init();
    return;
  }
  // Default: no-op (skip initialization in test environment)
}

namespace testing {

MockNeuronBindingsSession::MockNeuronBindingsSession() {
  if (!g_neuron_bindings_mocks_initialized) {
    g_neuron_bindings_mocks_initialized = true;
    initialized_ = true;

    auto* mock = MockNeuronBindings::GetInstance();
    ON_CALL(*mock, maybe_lazy_init()).WillByDefault(::testing::Return());
  }
}

MockNeuronBindingsSession::~MockNeuronBindingsSession() {
  if (initialized_) {
    g_neuron_bindings_mocks_initialized = false;
    ::testing::Mock::VerifyAndClearExpectations(MockNeuronBindings::GetInstance());
  }
}

}  // namespace testing
}  // namespace torch_neuronx
