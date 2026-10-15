#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>

#include <string>
#include <vector>

namespace torch_neuronx {
namespace dispatch {

/**
 * Try to dispatch a structural operation on a complex tensor through an
 * interleaved real view. Returns true when the operation was handled.
 */
bool TryDispatchStructuralComplexOp(const std::string& lookup_name,
                                    const std::vector<c10::IValue>& args, torch::jit::Stack* stack,
                                    const std::string& op_name);

}  // namespace dispatch
}  // namespace torch_neuronx
