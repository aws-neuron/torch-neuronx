#pragma once

#include <gmock/gmock.h>

namespace torch_neuronx {
namespace testing {

class MockNeuronBindings {
 public:
  static MockNeuronBindings* GetInstance() {
    static MockNeuronBindings instance;
    return &instance;
  }

  MOCK_METHOD(void, maybe_lazy_init, ());

 private:
  MockNeuronBindings() = default;
};

// Helper class to manage mock session lifecycle
class MockNeuronBindingsSession {
 public:
  MockNeuronBindingsSession();
  ~MockNeuronBindingsSession();
  MockNeuronBindingsSession(const MockNeuronBindingsSession&) = delete;
  MockNeuronBindingsSession& operator=(const MockNeuronBindingsSession&) = delete;

 private:
  bool initialized_ = false;
};

}  // namespace testing
}  // namespace torch_neuronx
