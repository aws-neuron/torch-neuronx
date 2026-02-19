#pragma once

#include <string>

namespace at::neuron {

// Platform configuration utilities for Neuron compilation.
// Provides platform-specific settings like target architecture and logical core configuration.
namespace utils {

// Gets the current Neuron instance type from the runtime.
//
// Returns:
//   Instance type string (e.g., "trn1", "trn2", "inf1", "inf2", etc.)
//
// Throws:
//   std::runtime_error if the instance type cannot be determined or is unsupported
std::string GetInstanceType();

// Gets the compilation target platform based on the current instance type.
// Maps instance types (inf1, trn1, trn2, etc.) to compiler targets.
//
// Returns:
//   Target platform string for neuronx-cc (e.g., "trn1", "trn2", "inf1")
//
// Throws:
//   std::runtime_error if the instance type is unknown or unsupported
std::string GetPlatformTarget();

// Gets the logical neuron cores setting for compilation.
// Checks environment variables first, then falls back to instance defaults.
// The NEURON_LOGICAL_NC_CONFIG environment variable can override defaults.
//
// Returns:
//   Logical neuron core count as string (e.g., "1", "2")
//
// Throws:
//   std::runtime_error if the instance type is unknown or unsupported
std::string GetLogicalNeuronCores();

// TODO(Runtime): Remove once it is defaulted.
// Checks if synchronous execution mode is enabled.
// When enabled, operations execute synchronously instead of being queued asynchronously.
// Controlled by the NEURON_LAUNCH_BLOCKING environment variable.
//
// Returns:
//   true if NEURON_LAUNCH_BLOCKING is set to "1", false otherwise
bool IsSyncModeEnabled();

}  // namespace utils

}  // namespace at::neuron
