#pragma once

#include <torch/torch.h>

#include <vector>

namespace at::neuron {

/**
 * Encapsulates all metadata needed for CPU fallback execution.
 * Groups static argument info and list argument reconstruction data.
 *
 * This struct bundles four pieces of information previously passed separately:
 * - static_args: Values of arguments marked as static (not compiled into NEFF)
 * - static_arg_positions: Original positions of static arguments
 * - list_arg_positions: Positions where list/tuple arguments were flattened
 * - list_arg_lengths: Number of elements in each flattened list/tuple
 */
struct CPUFallbackContext {
  // Static arguments (constants not compiled into NEFF)
  std::vector<c10::IValue> static_args;
  std::vector<int64_t> static_arg_positions;

  // List argument reconstruction metadata
  std::vector<int64_t> list_arg_positions;
  std::vector<int64_t> list_arg_lengths;

  // Default constructor creates empty context
  CPUFallbackContext() = default;

  // Full constructor
  CPUFallbackContext(std::vector<c10::IValue> args, std::vector<int64_t> arg_positions,
                     std::vector<int64_t> list_positions, std::vector<int64_t> list_lengths)
      : static_args(std::move(args)),
        static_arg_positions(std::move(arg_positions)),
        list_arg_positions(std::move(list_positions)),
        list_arg_lengths(std::move(list_lengths)) {}

  // Check if context has any fallback metadata
  bool HasData() const { return !static_args.empty() || !list_arg_positions.empty(); }

  // Check if context is empty
  bool IsEmpty() const {
    return static_args.empty() && static_arg_positions.empty() && list_arg_positions.empty() &&
           list_arg_lengths.empty();
  }
};

}  // namespace at::neuron
