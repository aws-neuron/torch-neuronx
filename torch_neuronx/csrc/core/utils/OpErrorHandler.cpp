#include "torch_neuronx/csrc/core/utils/OpErrorHandler.h"

#include <torch/torch.h>

#include <utility>

#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/utils/CPUFallbackContext.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/utils/TensorContext.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {

namespace {
// Single C++ -> Python invoker, set once at import. Invoked with (op_name, ctx)
// on a fault; runs the op's handler if one is registered.
OpErrorHandler::Invoker g_invoker;
}  // namespace

void OpErrorHandler::RegisterInvoker(Invoker invoker) { g_invoker = std::move(invoker); }

size_t OpErrorContext::TensorOffset(size_t arg_index) const {
  const auto& ctx = op_->cpu_fallback_context;
  TORCH_CHECK(arg_index < ctx.input_metadata.size(), "OpErrorContext: arg index ", arg_index,
              " out of range (", ctx.input_metadata.size(), " args)");
  size_t offset = 0;
  for (size_t i = 0; i < arg_index; ++i) {
    offset += GetTensorCount(ctx.input_metadata[i]);
  }
  return offset;
}

torch::Tensor OpErrorContext::CopyTensorToCpu(size_t tensor_index) const {
  const auto& ctx = op_->cpu_fallback_context;
  TORCH_CHECK(tensor_index < ctx.tensor_metadata.size(), "OpErrorContext: tensor index ",
              tensor_index, " out of range");
  const TensorContext& tctx = ctx.tensor_metadata[tensor_index];
  nrt_tensor_t* nrt_ptr = ctx.tensor_data_ptrs[tensor_index];
  if (!nrt_ptr) {
    nrt_ptr = c10_neuron::NeuronCachingAllocator::findTensor(ctx.tensor_storage_ptrs[tensor_index]);
  }
  TORCH_CHECK(nrt_ptr, "OpErrorContext: null tensor pointer for tensor index ", tensor_index);
  auto cpu_tensor = torch::empty(tctx.get_shape(), tctx.get_options().device(torch::kCPU));
  const size_t offset_bytes = tctx.storage_offset * tctx.element_size;
  torch_neuronx::utils::nrt_copy_neuron_to_cpu(nrt_ptr, cpu_tensor.data_ptr(), offset_bytes,
                                               tctx.get_size_bytes());
  return cpu_tensor;
}

std::vector<int64_t> OpErrorContext::Shape(size_t arg_index) const {
  const auto& meta = op_->cpu_fallback_context.input_metadata.at(arg_index);
  TORCH_CHECK(IsTensor(meta), "OpErrorContext::Shape: arg ", arg_index, " is not a tensor");
  return op_->cpu_fallback_context.tensor_metadata.at(TensorOffset(arg_index)).get_shape();
}

c10::IValue OpErrorContext::Scalar(size_t arg_index) const {
  const auto& meta = op_->cpu_fallback_context.input_metadata.at(arg_index);
  TORCH_CHECK(IsScalar(meta), "OpErrorContext::Scalar: arg ", arg_index, " is not a scalar");
  return GetScalarValue(meta);
}

torch::Tensor OpErrorContext::Tensor(size_t arg_index) const {
  const auto& meta = op_->cpu_fallback_context.input_metadata.at(arg_index);
  TORCH_CHECK(IsTensor(meta), "OpErrorContext::Tensor: arg ", arg_index, " is not a tensor");
  return CopyTensorToCpu(TensorOffset(arg_index));
}

std::vector<torch::Tensor> OpErrorContext::TensorList(size_t arg_index) const {
  const auto& meta = op_->cpu_fallback_context.input_metadata.at(arg_index);
  TORCH_CHECK(IsTensorList(meta), "OpErrorContext::TensorList: arg ", arg_index,
              " is not a tensor list");
  const size_t offset = TensorOffset(arg_index);
  const size_t count = GetTensorCount(meta);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    tensors.emplace_back(CopyTensorToCpu(offset + i));
  }
  return tensors;
}

void OpErrorHandler::SurfaceException(const OperationContext* op) {
  // Nothing to reconstruct, or no handler mechanism registered -> the caller
  // falls through to the generic error path.
  if (op == nullptr || op->cpu_fallback_context.IsEmpty() || !g_invoker) {
    return;
  }
  OpErrorContext ctx(op);
  // Raises the op's precise exception if a handler is registered; otherwise the
  // Python invoker no-ops and we return.
  g_invoker(op->GetOpName(), ctx);
}

}  // namespace at::neuron
