#pragma once

#include <string>

// Mock declarations for testing - avoids PyTorch dependencies
namespace torch_neuron {

// Get the instance type string (e.g., "trn1", "trn2", "inf1")
std::string GetInstanceType();

// Test helper to set mock instance type
void SetMockInstanceType(const std::string& instance_type);

}  // namespace torch_neuron
