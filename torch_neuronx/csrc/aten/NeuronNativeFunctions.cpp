#include "torch_neuronx/csrc/aten/NeuronNativeFunctions.h"

#include <ATen/EmptyTensor.h>
#include <ATen/core/jit_type.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <nrt/nrt.h>

#include <algorithm>

#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
// For calling Python contiguous op without dispatcher recursion
#include "torch_neuronx/csrc/ops/InternalOps.h"

namespace {

// Helper to get strides for a given memory format and size
std::vector<int64_t> get_strides_for_memory_format(c10::IntArrayRef size,
                                                   c10::MemoryFormat format) {
  if (format == c10::MemoryFormat::ChannelsLast && size.size() == 4) {
    // NHWC: stride order is (HWC, 1, WC, C) for dims (N, C, H, W)
    auto c = size[1], h = size[2], w = size[3];
    return {h * w * c, 1, w * c, c};
  } else if (format == c10::MemoryFormat::ChannelsLast3d && size.size() == 5) {
    // NDHWC: stride order for dims (N, C, D, H, W)
    auto c = size[1], d = size[2], h = size[3], w = size[4];
    return {d * h * w * c, 1, h * w * c, w * c, c};
  }
  // Default: contiguous strides
  return c10::TensorType::contiguousStridesOf(size);
}

// Helper to create Neuron storage
c10::Storage make_neuron_storage(size_t size_bytes, int device_index) {
  c10::Allocator* allocator = c10_neuron::NeuronCachingAllocator::get();
  c10_neuron::set_device(device_index);
  auto data_ptr = allocator->allocate(size_bytes);

  auto storage_impl = c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, true);

  return c10::Storage(std::move(storage_impl));
}

// Helper to resize neuron storage
static inline void resize_storage_neuron(at::TensorImpl* self, size_t new_size_bytes,
                                         bool skip_empty_resize = false) {
  // For set_ operations, don't resize storage for zero elements
  if (skip_empty_resize && self->numel() == 0) {
    return;
  }

  // Get current storage size
  size_t current_size = self->unsafe_storage().nbytes();

  // Check if we need to reallocate
  if (new_size_bytes > current_size) {
    // Get underlying storage
    auto& storage = self->unsafe_storage();

    // Get the Neuron hooks interface
    auto hooks = torch_neuronx::get_neuron_hooks();
    TORCH_CHECK(hooks != nullptr, "Neuron hooks interface not available");

    // Use the hooks interface to resize the storage
    // This will use set_data_ptr_noswap internally to preserve storage identity
    hooks->resizePrivateUse1Bytes(storage, new_size_bytes);
  }
}

inline at::TensorImpl* resize_impl_neuron_(at::TensorImpl* self, c10::IntArrayRef size,
                                           at::OptionalIntArrayRef stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }
  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);
  }

  resize_storage_neuron(self, storage_size, /*skip_empty_resize=*/true);

  return self;
}

}  // anonymous namespace

// Forward declaration for lazy init
namespace torch_neuronx {
extern void maybe_lazy_init();
}

namespace at {
namespace native {

at::Tensor& set_storage_neuron(at::Tensor& result, c10::Storage storage, int64_t storage_offset,
                               c10::IntArrayRef size, c10::IntArrayRef stride) {
  const std::string op_name = "aten::set_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }
  at::native::checkSetStorage(result, storage, storage_offset, size, stride);

  auto* impl = result.unsafeGetTensorImpl();
  impl->set_storage_offset(storage_offset);

  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;

  // Use native resize implementation
  resize_impl_neuron_(impl, size, stride_opt);

  return result;
}

at::Tensor& set_neuron(at::Tensor& result, c10::Storage source) {
  const std::string op_name = "aten::set_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }
  TORCH_CHECK(source.device_type() == c10::DeviceType::PrivateUse1,
              "set_: expected neuron storage, but got storage on device ", source.device_type());

  auto itemsize = result.dtype().itemsize();
  TORCH_CHECK(itemsize > 0, "set_: tensor dtype has invalid itemsize ", itemsize);

  auto storage_bytes = source.nbytes();
  TORCH_CHECK(storage_bytes % itemsize == 0, "set_: storage size (", storage_bytes,
              ") not divisible by dtype itemsize (", itemsize, ")");

  int64_t new_size = static_cast<int64_t>(storage_bytes / itemsize);

  // Use the standard PyTorch pattern: set_(storage, offset, size, stride)
  return set_storage_neuron(result, std::move(source), 0, {new_size}, {});
}

at::Tensor& set_tensor_neuron(at::Tensor& result, const at::Tensor& source) {
  const std::string op_name = "aten::set_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate devices
  TORCH_CHECK(result.device().type() == c10::DeviceType::PrivateUse1,
              "set_: expected neuron tensor, but got tensor on device ", result.device());
  TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1,
              "set_: expected neuron source tensor, but got tensor on device ", source.device());

  return set_storage_neuron(result, source.storage(), source.storage_offset(), source.sizes(),
                            source.strides());
}

at::Tensor& set_empty_neuron(at::Tensor& result) {
  const std::string op_name = "aten::set_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  TORCH_CHECK(result.device().type() == c10::DeviceType::PrivateUse1,
              "set_: expected neuron tensor, but got tensor on device ", result.device());

  // Create empty storage efficiently
  int device_index = result.device().index();
  auto storage = make_neuron_storage(0, device_index);

  return set_storage_neuron(result, std::move(storage), 0, {0}, {});
}

at::Tensor& set_tensor_storage_offset_neuron(at::Tensor& result, const at::Tensor& source,
                                             c10::SymInt storage_offset, c10::SymIntArrayRef size,
                                             c10::SymIntArrayRef stride) {
  const std::string op_name = "aten::set_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate inputs
  TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1,
              "set_: expected neuron source tensor, but got tensor on device ", source.device());

  auto offset_val = storage_offset.expect_int();
  TORCH_CHECK(offset_val >= 0, "set_: storage_offset must be non-negative, got ", offset_val);

  return set_storage_neuron(result, source.storage(), offset_val + source.storage_offset(),
                            C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor empty_neuron(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
                        c10::optional<c10::Layout> layout_opt,
                        c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt,
                        c10::optional<c10::MemoryFormat> memory_format_opt) {
  const std::string op_name = "aten::empty";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  if (memory_format == c10::MemoryFormat::Preserve) {
    memory_format = c10::MemoryFormat::Contiguous;
  }
  auto strides = get_strides_for_memory_format(size, memory_format);
  return empty_strided_neuron(size, strides, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor new_empty_neuron(const at::Tensor& self, at::IntArrayRef size,
                            c10::optional<c10::ScalarType> dtype_opt,
                            c10::optional<c10::Layout> layout_opt,
                            c10::optional<c10::Device> device_opt,
                            c10::optional<bool> pin_memory_opt) {
  // Mark for executed on neuron device
  const std::string op_name = "aten::new_empty";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Inherit dtype and device from original tensor if not set
  return empty_strided_neuron(size, c10::TensorType::contiguousStridesOf(size),
                              dtype_opt.value_or(self.scalar_type()), layout_opt,
                              device_opt.value_or(self.device()), pin_memory_opt);
}

at::Tensor empty_strided_neuron(at::IntArrayRef size, at::IntArrayRef stride,
                                c10::optional<c10::ScalarType> dtype_opt,
                                c10::optional<c10::Layout> layout_opt,
                                c10::optional<c10::Device> device_opt,
                                c10::optional<bool> pin_memory_opt) {
  // Mark for executed on neuron device
  const std::string op_name = "aten::empty_strided";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate device
  auto device = c10::device_or_default(device_opt);
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
              "Expected neuron device but got: ", device);

  // Initialize Neuron runtime if needed (lazy initialization)
  torch_neuronx::maybe_lazy_init();

  // Get device index
  int device_index = device.has_index() ? device.index() : c10_neuron::current_device();

  // Get dtype
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));

  // Calculate total size
  int64_t nelements = c10::multiply_integers(size);
  int64_t size_bytes = nelements * dtype.itemsize();

  // Handle zero-size tensors
  if (size_bytes == 0) {
    // For zero-size tensors, we still need to create a valid tensor
    // but with minimal allocation
    size_bytes = dtype.itemsize();
  }

  // Create storage
  auto storage = make_neuron_storage(size_bytes, device_index);

  // Create tensor
  auto tensor = at::detail::make_tensor<c10_neuron::NeuronTensorImpl>(std::move(storage), dtype);

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

const at::Tensor& resize_neuron(const at::Tensor& self, c10::IntArrayRef size,
                                std::optional<c10::MemoryFormat> memory_format) {
  // Mark for executed on neuron device
  const std::string op_name = "aten::resize_";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate device
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "resize_: expected neuron tensor, but got: ", self.device());

  auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
  if (format == c10::MemoryFormat::Preserve) {
    format = c10::MemoryFormat::Contiguous;
  }

  // Calculate new total elements and size
  int64_t new_nelements = c10::multiply_integers(size);
  auto dtype = self.scalar_type();
  size_t itemsize = c10::elementSize(dtype);
  size_t new_size_bytes = new_nelements * itemsize;

  resize_storage_neuron(self.unsafeGetTensorImpl(), new_size_bytes);

  // Update sizes and strides based on memory format
  auto strides = get_strides_for_memory_format(size, format);
  self.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);

  return self;
}

at::Tensor clone_neuron(const at::Tensor& src, c10::optional<c10::MemoryFormat> memory_format) {
  const std::string op_name = "aten::clone";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate device
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "clone: expected neuron tensor, but got: ", src.device());

  auto format = memory_format.value_or(c10::MemoryFormat::Preserve);
  if (format == c10::MemoryFormat::Preserve) {
    format = c10::MemoryFormat::Contiguous;
  }

  // Ensure source is contiguous to avoid Neuron->Neuron copy with layout change
  at::Tensor src_to_copy = src;
  if (!src.is_contiguous()) {
    src_to_copy = torch_neuronx::ops::contiguous_internal(src, c10::MemoryFormat::Contiguous);
  }

  auto strides = get_strides_for_memory_format(src_to_copy.sizes(), format);
  auto cloned = empty_strided_neuron(src_to_copy.sizes(), strides, src.scalar_type(), src.layout(),
                                     src.device(), c10::nullopt);

  // Copy data
  cloned.copy_(src_to_copy);

  return cloned;
}

at::Scalar _local_scalar_dense_neuron(const at::Tensor& self) {
  const std::string op_name = "aten::_local_scalar_dense";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Validate device
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "_local_scalar_dense: expected neuron tensor, but got: ", self.device());

  // Check that tensor has exactly one element
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense: tensor must have exactly one element, got ",
              self.numel());

  // Copy to CPU to extract scalar value
  auto cpu_tensor = self.cpu();
  return cpu_tensor.item();
}

// Override _has_compatible_shallow_copy_type for Neuron backend
bool _has_compatible_shallow_copy_type_neuron(const at::Tensor& self, const at::Tensor& from) {
  const std::string op_name = "aten::_has_compatible_shallow_copy_type";
  if (torch_neuronx::shouldLogExecuted(op_name)) {
    torch_neuronx::markExecutedLogged(op_name);
  }

  // Allow compatibility between CPU and Neuron tensors
  auto self_device = self.device().type();
  auto from_device = from.device().type();

  return (self_device == from_device) ||
         (self_device == at::kCPU && from_device == at::kPrivateUse1) ||
         (self_device == at::kPrivateUse1 && from_device == at::kCPU);
}

}  // namespace native
}  // namespace at
