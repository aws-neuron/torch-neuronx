#include "MockNeuronDevice.h"

// Mock implementation of GetInstanceType for testing
namespace torch_neuronx {

static std::string mock_instance_type = "trn1";

std::string GetInstanceType() { return mock_instance_type; }

// Test helper to set mock instance type
void SetMockInstanceType(const std::string& instance_type) { mock_instance_type = instance_type; }

}  // namespace torch_neuronx
