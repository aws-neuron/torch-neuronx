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

// Gets the neuronx-cc compiler version for cache key composition.
// Reads NEURONXCC_VERSION environment variable, set by torch_neuronx Python
// init from neuronxcc.__version__. Returns empty string if not set, in which
// case callers should treat compiler version as unknown for keying purposes.
//
// Returns:
//   Compiler version string (e.g., "2.18.123") or empty string if unset
std::string GetCompilerVersion();

// Returns true when the persistent NEFF cache key comes from the HLO/MLIR bytes.
// Such a key invalidates itself: a different lowering, compiler, or IR gives different
// bytes and a different key. One entry then persists across versions and sessions.
//
// Default on. Set TORCH_NEURONX_NEFF_KEY_FROM_HLO=0 for the legacy base_key key, which
// breaks on any source or version change.
bool IsHloContentKeyEnabled();

}  // namespace utils

}  // namespace at::neuron
