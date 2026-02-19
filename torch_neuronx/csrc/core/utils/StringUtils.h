#pragma once

#include <string>

namespace at::neuron::utils {

// Simplifies a string for use in filenames by removing problematic characters.
// Replaces characters like <, >, :, /, etc. with underscores and collapses
// multiple consecutive underscores into one.
//
// Args:
//   str: The input string to sanitize
//
// Returns:
//   A filename-safe version of the input string
std::string SimplifyForFilename(const std::string& str);

}  // namespace at::neuron::utils
