#pragma once

#include <gmock/gmock.h>

#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"

namespace at::neuron::testing {

// GMock-based mock for NeuronKernelExecution (for OperationContext tests)
class MockNeuronKernelExecution : public NeuronKernelExecution {
 public:
  MockNeuronKernelExecution(const std::string& op_name = "mock_op",
                            const std::vector<torch::Tensor>& inputs = {},
                            const std::vector<torch::Tensor>& outputs = {}, int device_id = 0)
      : NeuronKernelExecution(op_name, get_tensor_data_ptrs(inputs), get_tensor_data_ptrs(outputs),
                              device_id) {}

  MOCK_METHOD(KernelTypeEnum, GetKernelType, (), (const, override));
  MOCK_METHOD(bool, RequiresCompilation, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
  MOCK_METHOD(void, ExecuteOrSchedule, (nrt::ErrorTrackerHandle*, nrt::SequenceId*, c10::StreamId),
              (const, override));
  MOCK_METHOD(bool, RequiresPrepare, (), (const, override));
  MOCK_METHOD(void, Prepare, (), (const, override));
};

// GMock-based mock for CompilableKernelExecution (for OperationContext tests)
class MockCompilableKernelExecution
    : public CompilableKernelExecution,
      public NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                        c10::DeviceType::PrivateUse1> {
 public:
  MockCompilableKernelExecution(const std::string& cache_key = "mock_cache_key",
                                const std::string& compiler_args = "",
                                const std::string& opt_level = "", bool has_collectives = false,
                                int device_id = 0)
      : CompilableKernelExecution(cache_key, compiler_args, opt_level, has_collectives),
        NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                   c10::DeviceType::PrivateUse1>("mock_op", {}, {}, device_id) {
    // Set up default mock behavior
    ON_CALL(*this, GetKernelType()).WillByDefault(::testing::Return(KernelTypeEnum::kHLO));
    ON_CALL(*this, CompileToNeff())
        .WillByDefault(::testing::Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));
    ON_CALL(*this, ValidateImpl()).WillByDefault(::testing::Return(true));
    ON_CALL(*this, Execute()).WillByDefault(::testing::Return());
  }

  MOCK_METHOD(KernelTypeEnum, GetKernelType, (), (const, override));
  MOCK_METHOD(std::vector<uint8_t>, CompileToNeff, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
  MOCK_METHOD(void, ExecuteOrSchedule, (nrt::ErrorTrackerHandle*, nrt::SequenceId*, c10::StreamId),
              (const, override));
};

// GMock-based mock for XLACompilableKernelExecution (for concatenation tests)
class MockXLACompilableKernelExecution : public XLACompilableKernelExecution {
 public:
  // Constructor with TensorDataRef vectors (matching actual XLACompilableKernelExecution signature)
  MockXLACompilableKernelExecution(
      const std::string& op_name, std::vector<TensorDataRef>&& input_refs,
      std::vector<TensorDataRef>&& output_refs, const std::vector<TensorContext>& input_contexts,
      const std::vector<TensorContext>& output_contexts, const std::string& cache_key,
      const std::vector<uint8_t>& hlo_bytes, bool has_collectives, int device_id)
      : XLACompilableKernelExecution(op_name, std::move(input_refs), std::move(output_refs),
                                     input_contexts, output_contexts, cache_key,
                                     hlo_bytes.empty() ? GetDefaultValidStableHloIR() : hlo_bytes,
                                     has_collectives, device_id) {
    SetupDefaultMockBehavior();
  }

  // Simplified constructor for basic tests (op_name only)
  explicit MockXLACompilableKernelExecution(const std::string& op_name = "mock_op")
      : XLACompilableKernelExecution(op_name, std::vector<TensorDataRef>{},
                                     std::vector<TensorDataRef>{}, std::vector<TensorContext>{},
                                     std::vector<TensorContext>{}, "mock_cache_key",
                                     GetDefaultValidStableHloIR(), false, 0) {
    SetupDefaultMockBehavior();
  }

  // Constructor with torch::Tensor vectors for convenience (converts to TensorDataRef)
  MockXLACompilableKernelExecution(const std::string& op_name,
                                   const std::vector<torch::Tensor>& inputs,
                                   const std::vector<torch::Tensor>& outputs)
      : XLACompilableKernelExecution(op_name, CreateTensorDataRefs(inputs),
                                     CreateTensorDataRefs(outputs), std::vector<TensorContext>{},
                                     std::vector<TensorContext>{}, "mock_cache_key",
                                     GetDefaultValidStableHloIR(), false, 0) {
    SetupDefaultMockBehavior();
  }

  MOCK_METHOD(std::vector<uint8_t>, CompileToNeff, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
  MOCK_METHOD(bool, HasCollectives, (), (const));

 private:
  void SetupDefaultMockBehavior() {
    ON_CALL(*this, CompileToNeff())
        .WillByDefault(::testing::Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));
    ON_CALL(*this, ValidateImpl()).WillByDefault(::testing::Return(true));
    ON_CALL(*this, Execute()).WillByDefault(::testing::Return());
    ON_CALL(*this, HasCollectives()).WillByDefault(::testing::Return(false));
  }

  // Helper function to provide valid minimal StableHLO IR for testing
  static std::vector<uint8_t> GetDefaultValidStableHloIR() {
    // Minimal valid StableHLO module with a simple add operation
    const std::string valid_stablehlo = R"(
module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
)";
    return std::vector<uint8_t>(valid_stablehlo.begin(), valid_stablehlo.end());
  }

  // Helper to convert torch::Tensor vectors to TensorDataRef vectors
  static std::vector<TensorDataRef> CreateTensorDataRefs(
      const std::vector<torch::Tensor>& tensors) {
    std::vector<TensorDataRef> refs;
    refs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      refs.emplace_back(tensor.data_ptr());
    }
    return refs;
  }
};

}  // namespace at::neuron::testing
