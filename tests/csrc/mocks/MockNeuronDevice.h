#pragma once

#include <string>

// Mock declarations for testing - avoids PyTorch dependencies
namespace torch_neuronx {

// Get the instance type string (e.g., "trn1", "trn2", "inf1")
std::string GetInstanceType();

// Test helper to set mock instance type
void SetMockInstanceType(const std::string& instance_type);

}  // namespace torch_neuronx

// Mock device count function for c10_neuron namespace
namespace c10_neuron::testing {

// Get the number of devices available
int device_count();

// Test helper to set mock device count
void SetMockDeviceCount(int count);

}  // namespace c10_neuron::testing
