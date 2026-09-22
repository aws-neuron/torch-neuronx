#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/core/OperationContext.h"

namespace at::neuron {

/**
 * Read-only, lazy view over a faulting op's inputs, handed to the op's Python
 * error handler (see torch_neuronx/python_ops/op_error_handlers.py).
 *
 * Shapes and scalars come from the OperationContext metadata with no copy;
 * tensors are copied to CPU only when the handler explicitly asks for them.
 * Because the op-specific handler knows its own argument layout and decides
 * what to pull, nothing here has to guess which argument is an index.
 *
 * Argument indices are positions in the op's positional arg list (the same
 * order as OperationContext::cpu_fallback_context.input_metadata).
 */
class OpErrorContext {
 public:
  explicit OpErrorContext(const OperationContext* op) : op_(op) {}

  // Shape of the arg_index-th positional arg (must be a single tensor).
  // Metadata only; no data transfer.
  std::vector<int64_t> Shape(size_t arg_index) const;

  // Value of the arg_index-th positional arg (must be a scalar/non-tensor).
  c10::IValue Scalar(size_t arg_index) const;

  // Copy the arg_index-th positional arg (a single tensor) to CPU.
  torch::Tensor Tensor(size_t arg_index) const;

  // Copy the arg_index-th positional arg (a tensor list) to CPU.
  std::vector<torch::Tensor> TensorList(size_t arg_index) const;

 private:
  // First index into tensor_metadata/tensor_data_ptrs for the arg_index-th
  // positional arg (sum of tensor counts of the preceding args).
  size_t TensorOffset(size_t arg_index) const;
  // Copy a single tensor (by flat tensor index) to CPU.
  torch::Tensor CopyTensorToCpu(size_t tensor_index) const;

  const OperationContext* op_;
};

/**
 * Surfaces the exact op-specific exception for a faulting op by invoking its
 * registered Python error handler with an OpErrorContext.
 *
 * Self-contained: it does NOT run the op and does not go through the CPU
 * fallback executor. Tensors are materialized to CPU lazily, and only the ones
 * the handler asks for.
 */
class OpErrorHandler {
 public:
  // (op_name, ctx) -> runs the op's Python handler, which raises the precise
  // exception, or no-ops if no handler is registered for the op.
  using Invoker = std::function<void(const std::string&, OpErrorContext&)>;

  // Registered once at import (via NeuronBindings _register_error_handler_invoker).
  static void RegisterInvoker(Invoker invoker);

  // Invoked on a device fault: builds the context and calls the invoker. The
  // handler's exception propagates to the caller. No-op if no invoker/handler
  // is registered or the op carries no reconstruction context.
  static void SurfaceException(const OperationContext* op);
};

}  // namespace at::neuron
