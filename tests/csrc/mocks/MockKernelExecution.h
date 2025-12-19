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
    ON_CALL(*this, CompileToNeff())
        .WillByDefault(::testing::Return(std::vector<uint8_t>{0x01, 0x02, 0x03}));
    ON_CALL(*this, ValidateImpl()).WillByDefault(::testing::Return(true));
    ON_CALL(*this, Execute()).WillByDefault(::testing::Return());
  }

  MOCK_METHOD(std::vector<uint8_t>, CompileToNeff, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
};

}  // namespace at::neuron::testing
