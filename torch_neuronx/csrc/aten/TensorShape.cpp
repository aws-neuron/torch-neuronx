#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "torch_neuronx/csrc/aten/NeuronNativeFunctions.h"
#include "torch_neuronx/csrc/utils/SymIntUtils.h"

namespace torch_neuronx {

// Helper function to create a view tensor with new sizes and strides
static at::Tensor alias_with_sizes_and_strides(const at::Tensor& self, at::IntArrayRef sizes,
                                               at::IntArrayRef strides) {
  // Create a new tensor that shares storage with the original
  at::Tensor result =
      at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW,         // Mark as view
                                              c10::Storage(self.storage()),  // Share storage
                                              self.key_set(), self.dtype());

  // Set the metadata
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(sizes, strides);

  // Propagate names if any
  at::namedinference::propagate_names(result, self);

  return result;
}

// View operation - reshapes tensor without copying data
at::Tensor view_neuron(const at::Tensor& self, at::IntArrayRef size) {
  // Infer size (handle -1 in size specification)
  auto inferred_size = at::infer_size(size, self.numel());

  // Compute strides for the new size
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);

  TORCH_CHECK(stride.has_value(),
              "view size is not compatible with input tensor's size and stride "
              "(at least one dimension spans across two contiguous subspaces). "
              "Use .reshape(...) instead.");

  // Create the view
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

at::Tensor unfold_neuron(const at::Tensor& self, int64_t d, int64_t size, int64_t step) {
  // Handle special case when self.dim() == 0, allow d == 0
  auto ndim = self.dim();
  d = at::maybe_wrap_dim(d, ndim, /*wrap_scalar=*/true);

  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  int64_t max_size = self.dim() == 0 ? 1 : sizes[d];

  // Validate parameters
  TORCH_CHECK(size >= 0, "size is ", size, " but must be >= 0");
  TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", d, " is ", max_size,
              " but size is ", size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");

  // Add new dimension for the unfolded elements
  sizes.push_back(size);
  strides.push_back(self.dim() == 0 ? 1 : strides[d]);
  // The if handles the self.dim() == 0 case
  if (d < ndim) {
    sizes[d] = (sizes[d] - size) / step + 1;
    strides[d] *= step;
  }

  // Use neuron as_strided implementation
  return as_strided_neuron(self, sizes, strides, c10::nullopt);
}

// as_strided - the fundamental view operation
at::Tensor as_strided_neuron(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
                             c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());

  // Validate the new view doesn't go out of bounds
  at::native::checkInBoundsForStorage(size, stride, storage_offset, self.dtype(), self.storage());

  // Create the strided view
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(), self.dtype());

  at::native::setStrided(result, size, stride, storage_offset);
  at::namedinference::propagate_names(result, self);

  return result;
}

// _reshape_alias - used internally by PyTorch for view operations
at::Tensor _reshape_alias_neuron(const at::Tensor& self, at::IntArrayRef sizes,
                                 at::IntArrayRef strides) {
  return view_neuron(self, sizes);
}

at::Tensor select_backward_neuron(const at::Tensor& grad, c10::SymIntArrayRef input_sizes,
                                  int64_t dim, c10::SymInt index) {
  auto sizes_vec = utils::symint_to_sizes(input_sizes);
  auto index_i = index.expect_int();

  auto options = grad.options();

  auto cpu_options = options.device(at::kCPU);
  auto grad_cpu = grad.to(at::kCPU);
  auto grad_input_cpu = at::zeros(sizes_vec, cpu_options);
  grad_input_cpu.select(dim, index_i).copy_(grad_cpu);
  return grad_input_cpu.to(grad.device(), grad.scalar_type());
}

at::Tensor slice_backward_neuron(const at::Tensor& grad, c10::SymIntArrayRef input_sizes,
                                 int64_t dim, c10::SymInt start, c10::SymInt end,
                                 c10::SymInt step) {
  auto sizes_vec = utils::symint_to_sizes(input_sizes);
  auto start_i = start.expect_int();
  auto end_i = end.expect_int();
  auto step_i = step.expect_int();

  auto options = grad.options();

  auto cpu_options = options.device(at::kCPU);
  auto grad_cpu = grad.to(at::kCPU);
  auto grad_input_cpu = at::zeros(sizes_vec, cpu_options);
  grad_input_cpu.slice(dim, start_i, end_i, step_i).copy_(grad_cpu);
  return grad_input_cpu.to(grad.device(), grad.scalar_type());
}

// argsort.stable to workaround limitation on writing to non-contiguous destination on device
at::Tensor argsort_stable_neuron(const at::Tensor& self, bool stable, int64_t dim,
                                 bool descending) {
  auto self_cpu = self.to(at::kCPU);
  auto result_cpu = self_cpu.argsort(stable, dim, descending);
  return result_cpu.to(self.device());
}

}  // namespace torch_neuronx
