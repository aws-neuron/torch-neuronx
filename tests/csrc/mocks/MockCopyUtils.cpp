#include "MockCopyUtils.h"

// Flag to check if mocks are initialized
static bool g_copy_utils_mocks_initialized = false;

namespace torch_neuronx {
namespace utils {

// Override CopyUtils symbols
void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  if (g_copy_utils_mocks_initialized) {
    testing::MockCopyUtils::GetInstance()->copy_cpu_to_neuron(src, dst, non_blocking);
    return;
  }
  // Default: just copy the data
  dst.copy_(src, non_blocking);
}

void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  if (g_copy_utils_mocks_initialized) {
    testing::MockCopyUtils::GetInstance()->copy_neuron_to_cpu(src, dst, non_blocking);
    return;
  }
  // Default: just copy the data
  dst.copy_(src, non_blocking);
}

void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking) {
  if (g_copy_utils_mocks_initialized) {
    testing::MockCopyUtils::GetInstance()->copy_neuron_to_neuron(src, dst, non_blocking);
    return;
  }
  // Default: just copy the data
  dst.copy_(src, non_blocking);
}

nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking) {
  if (g_copy_utils_mocks_initialized) {
    return testing::MockCopyUtils::GetInstance()->get_nrt_tensor(tensor, non_blocking);
  }
  // Default: return nullptr (no NRT tensor)
  return nullptr;
}

}  // namespace utils
}  // namespace torch_neuronx

namespace torch_neuronx {
namespace utils {
namespace testing {

MockCopyUtilsSession::MockCopyUtilsSession() {
  if (!g_copy_utils_mocks_initialized) {
    g_copy_utils_mocks_initialized = true;
    initialized_ = true;

    // Set up default behaviors
    auto* mock = MockCopyUtils::GetInstance();

    // Default: copy operations just copy the data
    ON_CALL(*mock, copy_cpu_to_neuron(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke([](const at::Tensor& src, at::Tensor& dst,
                                            bool non_blocking) { dst.copy_(src, non_blocking); }));

    ON_CALL(*mock, copy_neuron_to_cpu(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke([](const at::Tensor& src, at::Tensor& dst,
                                            bool non_blocking) { dst.copy_(src, non_blocking); }));

    ON_CALL(*mock, copy_neuron_to_neuron(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke([](const at::Tensor& src, at::Tensor& dst,
                                            bool non_blocking) { dst.copy_(src, non_blocking); }));

    ON_CALL(*mock, get_nrt_tensor(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(nullptr));
  }
}

MockCopyUtilsSession::~MockCopyUtilsSession() {
  if (initialized_) {
    g_copy_utils_mocks_initialized = false;
    // Clear any expectations
    ::testing::Mock::VerifyAndClearExpectations(MockCopyUtils::GetInstance());
  }
}

}  // namespace testing
}  // namespace utils
}  // namespace torch_neuronx
