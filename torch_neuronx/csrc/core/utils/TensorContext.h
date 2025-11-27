#pragma once

#include <c10/core/TensorOptions.h>
#include <torch/torch.h>

#include <vector>

namespace at::neuron {

// TensorContext stores tensor metadata without holding references to the actual tensor.
struct TensorContext {
  size_t numel;                // Number of elements
  size_t size_bytes;           // Total size in bytes
  size_t element_size;         // Size of each element in bytes
  std::vector<int64_t> shape;  // Tensor shape/dimensions
  size_t storage_offset;       // Offset within storage
  c10::TensorOptions options;  // Tensor options (dtype, device, layout, requires_grad)

  // Factory method to create TensorContext from a PyTorch tensor
  static TensorContext FromTensor(const at::Tensor& tensor) {
    return TensorContext{static_cast<size_t>(tensor.numel()),
                         static_cast<size_t>(tensor.numel() * tensor.element_size()),
                         static_cast<size_t>(tensor.element_size()),
                         tensor.sizes().vec(),
                         static_cast<size_t>(tensor.storage_offset()),
                         tensor.options().requires_grad(tensor.requires_grad())};
  }

  // Getter methods for convenient access
  size_t get_numel() const { return numel; }
  size_t get_size_bytes() const { return size_bytes; }
  size_t get_element_size() const { return element_size; }
  const std::vector<int64_t>& get_shape() const { return shape; }
  size_t get_storage_offset() const { return storage_offset; }
  const c10::TensorOptions& get_options() const { return options; }

  // Convenience getters that delegate to options
  c10::ScalarType get_dtype() const { return options.dtype().toScalarType(); }
  c10::Device get_device() const { return options.device(); }
  c10::Layout get_layout() const { return options.layout(); }
  bool get_requires_grad() const { return options.requires_grad(); }
};

}  // namespace at::neuron
