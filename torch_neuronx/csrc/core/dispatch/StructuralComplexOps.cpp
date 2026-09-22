#include "torch_neuronx/csrc/core/dispatch/StructuralComplexOps.h"

#include <ATen/ops/gather.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>

#include <optional>
#include <unordered_map>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

using StructuralComplexHandler = std::optional<at::Tensor> (*)(const std::vector<c10::IValue>&,
                                                               const at::Tensor&);

/**
 * Wrap `dim` against the complex rank, before view_as_real appends the
 * [real, imag] axis. An out-of-range dim, or rank 0, would address that axis
 * instead of raising, so both return nullopt for the generic path.
 */
std::optional<int64_t> WrapComplexDim(int64_t dim, const at::Tensor& self) {
  const int64_t rank = self.dim();
  if (rank == 0 || dim < -rank || dim >= rank) return std::nullopt;
  return dim < 0 ? dim + rank : dim;
}

std::optional<at::Tensor> DispatchContiguous(const std::vector<c10::IValue>& args,
                                             const at::Tensor& self) {
  auto memory_format = c10::MemoryFormat::Contiguous;
  if (args.size() > 1 && args[1].isInt()) {
    memory_format = static_cast<c10::MemoryFormat>(args[1].toInt());
  }
  if (memory_format != c10::MemoryFormat::Contiguous) return std::nullopt;

  return at::view_as_real(self).contiguous();
}

std::optional<at::Tensor> DispatchGather(const std::vector<c10::IValue>& args,
                                         const at::Tensor& self) {
  if (args.size() < 3 || !args[1].isInt() || !args[2].isTensor()) return std::nullopt;

  const std::optional<int64_t> dim = WrapComplexDim(args[1].toInt(), self);
  if (!dim.has_value()) return std::nullopt;

  const at::Tensor& index = args[2].toTensor();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  index_sizes.push_back(2);
  at::Tensor real_index = index.unsqueeze(-1).expand(index_sizes);
  bool sparse_grad = args.size() > 3 && args[3].isBool() && args[3].toBool();
  return at::gather(at::view_as_real(self), *dim, real_index, sparse_grad);
}

std::optional<at::Tensor> DispatchIndexSelect(const std::vector<c10::IValue>& args,
                                              const at::Tensor& self) {
  if (args.size() < 3 || !args[1].isInt() || !args[2].isTensor()) return std::nullopt;

  const std::optional<int64_t> dim = WrapComplexDim(args[1].toInt(), self);
  if (!dim.has_value()) return std::nullopt;

  return at::index_select(at::view_as_real(self), *dim, args[2].toTensor());
}

std::optional<at::Tensor> DispatchIndex(const std::vector<c10::IValue>& args,
                                        const at::Tensor& self) {
  if (args.size() < 2 || !args[1].isList()) return std::nullopt;

  // A surplus entry would index the [real, imag] axis on the real view instead of
  // raising "too many indices". Rank 0 is rejected by the same check.
  if (static_cast<int64_t>(args[1].toListRef().size()) > self.dim()) return std::nullopt;

  c10::List<c10::optional<at::Tensor>> indices;
  for (const auto& elem : args[1].toListRef()) {
    if (elem.isNone()) {
      indices.push_back(c10::nullopt);
    } else if (elem.isTensor()) {
      indices.push_back(elem.toTensor());
    } else {
      return std::nullopt;
    }
  }
  return at::index(at::view_as_real(self), indices);
}

const std::unordered_map<std::string, StructuralComplexHandler>& StructuralComplexHandlers() {
  static const std::unordered_map<std::string, StructuralComplexHandler> kHandlers = {
      {"contiguous", DispatchContiguous},
      {"gather", DispatchGather},
      {"index_select", DispatchIndexSelect},
      {"index.Tensor", DispatchIndex},
  };
  return kHandlers;
}

}  // namespace

bool TryDispatchStructuralComplexOp(const std::string& lookup_name,
                                    const std::vector<c10::IValue>& args, torch::jit::Stack* stack,
                                    const std::string& op_name) {
  if (!ComplexDecompositionEnabled() || args.empty() || !args[0].isTensor()) return false;

  const auto handler = StructuralComplexHandlers().find(lookup_name);
  if (handler == StructuralComplexHandlers().end()) return false;

  const at::Tensor& self = args[0].toTensor();
  if (!self.is_complex()) return false;

  // view_as_real rejects these, and there is no exception-to-fallback route here.
  // Materializing the bit belongs to copy_ and clone, which ATen registers as
  // fallthroughs at the Conjugate and Negative keys.
  if (self.is_conj() || self.is_neg()) return false;

  std::optional<at::Tensor> moved = handler->second(args, self);
  if (!moved.has_value()) return false;

  // Structural outputs preserve the trailing interleaved [real, imag] axis.
  // Materialize only if the nested real op did not already return a compatible
  // contiguous layout.
  at::Tensor result = at::view_as_complex(moved->contiguous());
  torch::jit::push(*stack, result);
  logOperationExecuted(op_name);
  TORCH_NEURONX_DISPATCH_DEBUG("[StructuralComplex] interleaved real-view path", " op=", op_name);
  return true;
}

}  // namespace dispatch
}  // namespace torch_neuronx
