#pragma once

#include <gmock/gmock.h>

#include "torch_neuronx/csrc/core/KernelExecution.h"

namespace at::neuron::testing {

// GMock-based mock for DirectKernelExecution
class MockDirectKernelExecution : public DirectKernelExecution {
 public:
  MockDirectKernelExecution(const std::string& op_name = "mock_direct_op",
                            const std::vector<torch::Tensor>& inputs = {},
                            const std::vector<torch::Tensor>& outputs = {})
      : DirectKernelExecution(op_name, inputs, outputs) {}

  MOCK_METHOD(KernelType, GetKernelType, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
};

// GMock-based mock for NeuronKernelExecution (for OperationContext tests)
class MockNeuronKernelExecution : public NeuronKernelExecution {
 public:
  MockNeuronKernelExecution(const std::string& op_name = "mock_op",
                            const std::vector<torch::Tensor>& inputs = {},
                            const std::vector<torch::Tensor>& outputs = {})
      : NeuronKernelExecution(op_name, inputs, outputs) {}

  MOCK_METHOD(bool, RequiresCompilation, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
  MOCK_METHOD(void, Execute, (), (const, override));
};

// GMock-based mock for CompilableKernelExecution (for OperationContext tests)
class MockCompilableKernelExecution : public CompilableKernelExecution {
 public:
  MockCompilableKernelExecution(const std::string& op_name = "mock_compilable_op",
                                const std::vector<torch::Tensor>& inputs = {},
                                const std::vector<torch::Tensor>& outputs = {},
                                const std::string& cache_key = "mock_cache_key",
                                bool has_collectives = false)
      : CompilableKernelExecution(op_name, inputs, outputs, cache_key, "", "", has_collectives) {}

  MOCK_METHOD(std::vector<uint8_t>, CompileToNeff, (), (const, override));
  MOCK_METHOD(bool, ValidateImpl, (), (const, override));
};

}  // namespace at::neuron::testing
