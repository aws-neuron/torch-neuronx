#pragma once

#include <gmock/gmock.h>
#include <torch/torch.h>

extern "C" {
#include <nrt/nrt.h>
}

namespace torch_neuronx {
namespace utils {
namespace testing {

class MockCopyUtils {
 public:
  static MockCopyUtils* GetInstance() {
    static MockCopyUtils instance;
    return &instance;
  }

  MOCK_METHOD(void, copy_cpu_to_neuron, (const at::Tensor&, at::Tensor&, bool));
  MOCK_METHOD(void, copy_neuron_to_cpu, (const at::Tensor&, at::Tensor&, bool));
  MOCK_METHOD(void, copy_neuron_to_neuron, (const at::Tensor&, at::Tensor&, bool));
  MOCK_METHOD(nrt_tensor_t*, get_nrt_tensor, (const at::Tensor&, bool));

 private:
  MockCopyUtils() = default;
};

// Helper class to manage mock session lifecycle
class MockCopyUtilsSession {
 public:
  MockCopyUtilsSession();
  ~MockCopyUtilsSession();
  MockCopyUtilsSession(const MockCopyUtilsSession&) = delete;
  MockCopyUtilsSession& operator=(const MockCopyUtilsSession&) = delete;

 private:
  bool initialized_ = false;
};

}  // namespace testing
}  // namespace utils
}  // namespace torch_neuronx
