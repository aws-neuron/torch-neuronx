#include "torch_neuronx/csrc/core/dispatch/CpuFallback.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <stdexcept>

#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/utils/TypeConversion.h"

namespace torch_neuronx {
namespace dispatch {

namespace detail {

// Split "aten::mm.out" → ("aten::mm", "out"); "aten::add" → ("aten::add", "").
std::pair<std::string, std::string> SplitOpName(const std::string& op_name) {
  auto dot_pos = op_name.find('.');
  if (dot_pos == std::string::npos) {
    return {op_name, ""};
  }
  return {op_name.substr(0, dot_pos), op_name.substr(dot_pos + 1)};
}

}  // namespace detail

py::object RunCpuFallback(const std::string& op_name, py::args args, py::kwargs kwargs) {
  // ── 1. Resolve the OperatorHandle from the op name + overload. ──
  auto [name_part, overload_part] = detail::SplitOpName(op_name);
  auto handle =
      c10::Dispatcher::singleton().findSchemaOrThrow(name_part.c_str(), overload_part.c_str());
  const auto& schema = handle.schema();

  // ── 2. Build a JIT stack from Python args/kwargs in schema order. ──
  // createStackForSchema handles defaults, kwarg-only args, and
  // schema-based filtering. Anything not matching the schema raises here
  // (which is the correct behavior — surfacing a bad call early beats
  // silently dropping kwargs in CPU fallback).
  torch::jit::Stack stack =
      torch::jit::createStackForSchema(schema, args, kwargs, /*self=*/c10::nullopt);

  // ── 3. Log the fallback once per op for observability. ──
  if (torch_neuronx::shouldLogFallback(op_name)) {
    torch_neuronx::markFallbackLogged(op_name);
  }

  // ── 4. Run PyTorch's CPU fallback. ──
  // Pops inputs from `stack`, moves all tensors to CPU, redispatches to
  // CPU, and pushes outputs back. For Tensor(a!) write-aliased args
  // (e.g. `out=`, inplace `self`), the CPU result is copied back to the
  // original device so the caller's tensor is mutated as the schema
  // promises.
  at::native::cpu_fallback(handle, &stack);

  // ── 5. Convert outputs back to Python. ──
  // Use IValueToPyObjectWithSchema so optional/list/scalar types unbox the
  // way the schema declares (matches what torch.ops.aten.* calls would
  // produce).
  const auto& returns = schema.returns();
  const size_t n = returns.size();
  if (n == 0) return py::none();
  if (n == 1) {
    return utils::IValueToPyObjectWithSchema(stack[0], returns[0].type());
  }
  py::tuple result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = utils::IValueToPyObjectWithSchema(stack[i], returns[i].type());
  }
  return result;
}

}  // namespace dispatch
}  // namespace torch_neuronx
