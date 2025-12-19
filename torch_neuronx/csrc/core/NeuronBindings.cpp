#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Include our native functions
#include "torch_neuronx/csrc/aten/NeuronNativeFunctions.h"
// Include the allocator
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
// Include logging
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
// Include device management
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronGuardImpl.h"
#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/core/utils/PythonTypeConverter.h"
// Include Streams
#include "torch_neuronx/csrc/Stream.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/streams/StreamManager.h"
// Copy helpers
#include "torch_neuronx/csrc/utils/CopyUtils.h"
// Operation execution
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
// Slice utilities
#include "torch_neuronx/csrc/utils/SliceUtils.h"
// Lazy transformation management
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistration.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TypeUtils.h"
// Internal operations
#include "torch_neuronx/csrc/ops/InternalOps.h"

// Include Neuron Runtime headers
extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
}

namespace torch_neuronx {

// Forward declarations
bool are_python_ops_registered();

// Helper struct to hold input transformation processing results
struct InputTransformationResult {
  std::vector<torch::Tensor> final_inputs;
  std::vector<std::vector<c10_neuron::lazy::TensorTransformation>> kernel_transforms;
  bool has_input_transforms = false;
};

// Helper function to process inputs and setup transformations
InputTransformationResult ProcessInputTransformations(const std::string& op_name,
                                                      const std::vector<torch::Tensor>& inputs,
                                                      bool has_collectives) {
  InputTransformationResult result;
  result.final_inputs.reserve(inputs.size());
  result.kernel_transforms.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];

    // Check if tensor is non-contiguous
    if (!input.is_contiguous()) {
      // TODO: Add timing counter here for prologue perf measurement
      TORCH_NEURONX_DEBUG("Detected non-contiguous input", "op=", op_name, "input_index=", i,
                          "input_shape=", input.sizes(), "has_collectives=", has_collectives);

      // TODO(apoorvgu): enable prologues for collectives when they lower to StableHLO
      if (has_collectives) {
        TORCH_NEURONX_DEBUG(
            "Collective operation: using contiguous_internal for non-contiguous input",
            "op=", op_name, "input_index=", i);

        torch::Tensor contiguous_tensor = torch_neuronx::ops::contiguous_internal(input);
        result.final_inputs.push_back(contiguous_tensor);
        // kernel_transforms[i] remains empty for this input
      } else {
        // For non-collective operations, try to create a transformation
        auto& registry = c10_neuron::lazy::TransformationRegistry::Get();
        auto transform_result = registry.TryCreateAnyTransformation(input, op_name, i);

        if (transform_result.has_value()) {
          // SUPPORTED pattern - use the transformation
          TORCH_NEURONX_DEBUG("Using supported pattern transformation", "op=", op_name,
                              "input_index=", i, "pattern=", transform_result->pattern_name);

          result.kernel_transforms[i].push_back(transform_result->transformation);
          result.has_input_transforms = true;
          result.final_inputs.push_back(transform_result->processed_input);
        } else {
          // UNSUPPORTED pattern - apply contiguous_internal as fallback
          TORCH_NEURONX_DEBUG("Unsupported pattern, using contiguous_internal fallback",
                              "op=", op_name, "input_index=", i);

          torch::Tensor contiguous_tensor = torch_neuronx::ops::contiguous_internal(input);
          result.final_inputs.push_back(contiguous_tensor);
          // kernel_transforms[i] remains empty for this input
        }
      }
    } else {
      // Already contiguous - use as-is
      result.final_inputs.push_back(input);
    }
  }

  return result;
}

// Track slice tensors per tensor set for cleanup
static std::unordered_map<nrt_tensor_set_t*, std::vector<nrt_tensor_t*>> slice_tensors_map;
static std::mutex slice_tensors_mutex;

// Whitelist of view operations that only manipulate metadata
static const std::unordered_set<std::string> VIEW_OPS_WHITELIST = {
    // Basic reshaping
    "aten::view",
    "aten::reshape",
    "aten::flatten",

    // Dimension manipulation
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::transpose",
    "aten::permute",

    // Slicing and indexing
    "aten::select",
    "aten::slice",
    "aten::narrow",
    "aten::split",
    "aten::chunk",

    // Advanced views
    "aten::expand",
    "aten::expand_as",
    "aten::as_strided",
    "aten::unfold",

    // Diagonal operations
    "aten::diagonal",
    "aten::diagonal_backward",

    // Complex/real views
    "aten::view_as_real",
    "aten::view_as_complex",
    "aten::real",
    "aten::imag",

    // Conjugate/neg views
    "aten::conj",
    "aten::_conj",
    "aten::_neg_view",

    // Alias operations
    "aten::alias",
    "aten::detach",

    // Shape operations
    "aten::view_as",
    "aten::reshape_as",

    // Matrix operations
    "aten::t",   // 2D transpose
    "aten::mT",  // Matrix transpose (batched)
    "aten::mH",  // Matrix Hermitian transpose
    "aten::adjoint",

    // Dimension rearrangement
    "aten::movedim",
    "aten::moveaxis",
    "aten::swapaxes",
    "aten::swapdims",
};

// Custom CPU fallback with warning system
void neuron_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto op_name = c10::toString(op.schema().operator_name());

  // Check if this is a whitelisted view operation
  if (VIEW_OPS_WHITELIST.find(op_name) == VIEW_OPS_WHITELIST.end()) {
    // Not whitelisted - warn once per operation
    if (torch_neuronx::shouldLogFallback(op_name)) {
      TORCH_NEURONX_FALLBACK_WARN(op_name);
      torch_neuronx::markFallbackLogged(op_name);
    }
  }

  // Fallback
  at::native::cpu_fallback(op, stack);
}

// Device properties structure
struct __attribute__((visibility("hidden"))) NeuronDeviceProperties {
  std::string name;
  int64_t total_memory;
};

// Forward declarations
static void neuron_cleanup_atexit();

// Global initialization state
static bool g_initialized = false;
static std::mutex g_init_mutex;

// Check if runtime is initialized
bool is_initialized() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  return g_initialized;
}

// Runtime initialization
void neuron_init() {
  std::lock_guard<std::mutex> lock(g_init_mutex);

  if (g_initialized) {
    return;
  }

  // Initialize Neuron Runtime
  NRT_STATUS status = nrt_init(NRT_FRAMEWORK_TYPE_PYTORCH,  // Framework type
                               "2.0",                       // Framework version (PyTorch version)
                               "1.0"                        // FAL version
  );

  if (status != NRT_SUCCESS) {
    std::stringstream ss;
    ss << "Failed to initialize Neuron Runtime: status code " << status;
    throw std::runtime_error(ss.str());
  }

  // Guard against a device count that PyTorch cannot represent.
  //
  // Context:
  // - PyTorch defines c10::DeviceIndex as an 8-bit integer (int8_t); see:
  //   pytorch/c10/core/Device.h (typedef of DeviceIndex)
  // - The autograd engine obtains device counts from the registered
  //   DeviceGuardImpl implementations (via c10::impl::VirtualGuardImpl) and
  //   uses them to size per-device data structures, e.g. when starting device
  //   threads and when stashing current streams; see:
  //   - pytorch/torch/csrc/autograd/engine.cpp (start_device_threads)
  //   - pytorch/torch/csrc/autograd/engine.cpp (GraphTask::stash_current_streams)
  // - If a backend exposes >= 128 devices, converting that count to
  //   c10::DeviceIndex overflows (128 -> -128), which leads autograd to size
  //   vectors with a negative count that becomes a huge size_t, ultimately
  //   triggering std::vector growth errors during backward (e.g.,
  //   "vector::_M_default_append").
  //
  // To fail fast with a clear message instead of crashing deep in autograd,
  // we detect this condition right after the Neuron runtime is initialized and
  // bail out.
  {
    int actual_devices = c10_neuron::device_count();
    const auto max_representable = std::numeric_limits<c10::DeviceIndex>::max();
    if (actual_devices > static_cast<int>(max_representable)) {
      std::stringstream ss;
      ss << "Detected " << actual_devices
         << " Neuron devices (virtual neuron cores), but PyTorch's DeviceIndex"
            " is limited to "
         << static_cast<int>(max_representable)
         << ". This exceeds what PyTorch's PrivateUse1/autograd integration can"
            " safely handle and would cause failures during backward. "
            "Reduce the number of visible Neuron devices (e.g., via"
            " NEURON_RT_NUM_CORES or NEURON_RT_VISIBLE_CORES) or avoid"
            " configurations that expose >= 128 devices.";
      throw std::runtime_error(ss.str());
    }
  }

  g_initialized = true;

  // Register cleanup function
  std::atexit(neuron_cleanup_atexit);
}

// Maybe initialize - this is what native functions should call
void maybe_lazy_init() {
  if (!is_initialized()) {
    neuron_init();
  }
}

// Register 'neuron' as the name for PrivateUse1 device type
void register_neuron_device() {
  static bool registered = false;
  if (!registered) {
    c10::register_privateuse1_backend("neuron");

    // Register the allocator - needed for GetAllocator to work
    c10::SetAllocator(c10::DeviceType::PrivateUse1, c10_neuron::NeuronCachingAllocator::get());

    // Register custom StorageImpl creator for storage creation operations
    c10::SetStorageImplCreate(
        c10::DeviceType::PrivateUse1,
        [](c10::StorageImpl::use_byte_size_t use_byte_size, c10::SymInt size_bytes,
           c10::DataPtr data_ptr, c10::Allocator* allocator,
           bool resizable) -> c10::intrusive_ptr<c10::StorageImpl> {
          if (!allocator) {
            allocator = c10_neuron::NeuronCachingAllocator::get();
          }

          c10::intrusive_ptr<c10::StorageImpl> neuron_storage_impl;
          if (data_ptr.get() == nullptr) {
            // Constructor without data_ptr - let allocator handle allocation
            neuron_storage_impl = c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
                use_byte_size, size_bytes.as_int_unchecked(), allocator, resizable);
          } else {
            // Constructor with data_ptr
            neuron_storage_impl = c10::make_intrusive<c10_neuron::NeuronStorageImpl>(
                use_byte_size, size_bytes.as_int_unchecked(), std::move(data_ptr), allocator,
                resizable);
          }

          return neuron_storage_impl;
        });

    // Register the hooks interface for autograd support
    at::RegisterPrivateUse1HooksInterface(torch_neuronx::get_neuron_hooks());
    registered = true;
  }
}

// Device management functions that delegate to NeuronDevice.cpp
int64_t neuron_getDeviceCount() { return static_cast<int64_t>(c10_neuron::device_count()); }

int64_t neuron_getCurrentDevice() { return static_cast<int64_t>(c10_neuron::current_device()); }

void neuron_setDevice(int64_t device_id) { c10_neuron::set_device(static_cast<int>(device_id)); }

// Get device properties for a specific device
NeuronDeviceProperties neuron_getDeviceProperties(int64_t device_id) {
  NeuronDeviceProperties props;

  props.name = at::neuron::utils::GetInstanceType();

  // Get memory info from runtime using VNC stats
  nrt_vnc_memory_stats_t memory_stats;
  size_t stats_size_out;
  NRT_STATUS status;

  if (device_id < 0 || device_id > std::numeric_limits<uint32_t>::max()) {
    status = NRT_FAILURE;
  } else {
    status = nrt_get_vnc_memory_stats(static_cast<uint32_t>(device_id), &memory_stats,
                                      sizeof(memory_stats), &stats_size_out);
  }

  if (status == NRT_SUCCESS) {
    props.total_memory = static_cast<int64_t>(memory_stats.bytes_limit);
  } else {
    torch_neuronx::NeuronLogger::getInstance().log(
        torch_neuronx::LogLevel::WARNING, torch_neuronx::LogCategory::MEMORY,
        "Failed to get VNC memory stats for device " + std::to_string(device_id) +
            ", status: " + std::to_string(status) + ". Falling back to total_memory=0");
    props.total_memory = 0;
  }

  return props;
}

// Cleanup function
void neuron_close() { nrt_close(); }

// Static cleanup function for atexit
static void neuron_cleanup_atexit() { neuron_close(); }

}  // namespace torch_neuronx

// Register native functions with PyTorch dispatch
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &at::native::empty_neuron);
  m.impl("empty_strided", &at::native::empty_strided_neuron);
  m.impl("new_empty", &at::native::new_empty_neuron);
  m.impl("resize_", &at::native::resize_neuron);
  m.impl("clone", &at::native::clone_neuron);
  m.impl("_local_scalar_dense", &at::native::_local_scalar_dense_neuron);
  // Copy operations
  // Moved to Python implementation via @neuron_op and JAX; C++ impl disabled.
  // m.impl("copy_", &torch_neuronx::copy_neuron);
  // m.impl("_to_copy", &torch_neuronx::_to_copy_neuron);
  m.impl("_copy_from_and_resize", &torch_neuronx::_copy_from_and_resize_neuron);
  // View operations
  m.impl("view", &torch_neuronx::view_neuron);
  m.impl("unfold", &torch_neuronx::unfold_neuron);
  m.impl("as_strided", &torch_neuronx::as_strided_neuron);
  m.impl("_reshape_alias", &torch_neuronx::_reshape_alias_neuron);
  m.impl("select_backward", &torch_neuronx::select_backward_neuron);
  m.impl("slice_backward", &torch_neuronx::slice_backward_neuron);
  m.impl("argsort.stable", &torch_neuronx::argsort_stable_neuron);
  // Set Operations
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L8031-L8078
  m.impl("set_.source_Storage", &at::native::set_neuron);
  m.impl("set_.source_Storage_storage_offset", &at::native::set_storage_neuron);
  m.impl("set_.source_Tensor", &at::native::set_tensor_neuron);
  m.impl("set_.source_Tensor_storage_offset", &at::native::set_tensor_storage_offset_neuron);
  m.impl("set_", &at::native::set_empty_neuron);
  // Streams
  m.impl("record_stream", &torch_neuronx::record_stream_neuron);
  // Type Properties
  m.impl("_has_compatible_shallow_copy_type",
         &at::native::_has_compatible_shallow_copy_type_neuron);
}

namespace torch_neuronx {

// Flag to track if Python ops are registered
static bool g_python_ops_registered = false;

void set_python_ops_registered(bool registered) { g_python_ops_registered = registered; }

bool are_python_ops_registered() { return g_python_ops_registered; }

// Simple autograd fallback that redispatches to the actual implementation
// This prevents the CPU fallback warning for operations that have implementations
void autograd_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys,
                       torch::jit::Stack* stack) {
  // Remove autograd keys and redispatch to the actual implementation
  auto new_keys = dispatch_keys & c10::after_autograd_keyset;
  op.redispatchBoxed(new_keys, stack);
}

}  // namespace torch_neuronx

// Register CPU fallback for unimplemented operations
// This happens at module load time, but can be overridden by Python ops
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&torch_neuronx::neuron_cpu_fallback>());
}

// Register autograd fallback that just redispatches
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&torch_neuronx::autograd_fallback>());
}

// Register autocast fallback
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) { m.fallback(torch::CppFunction::makeFallthrough()); }

namespace torch_neuronx {

// Python contiguous operation instance (set from Python)
// Use raw PyObject* to avoid destructor issues during Python finalization
static PyObject* python_contiguous_op = nullptr;

// Function to call Python contiguous operation from C++
namespace ops {

at::Tensor call_python_contiguous_op(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (!python_contiguous_op) {
    TORCH_CHECK(false, "Python contiguous operation not registered");
  }

  py::gil_scoped_acquire gil;
  py::object op = py::reinterpret_borrow<py::object>(python_contiguous_op);
  py::object result = op(self, memory_format);
  return result.cast<at::Tensor>();
}

}  // namespace ops

}  // namespace torch_neuronx

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Register transformations explicitly (static initializer may not run)
  c10_neuron::lazy::RegisterTransformations();

  // Initialize Stream and Event base classes
  THNPStream_init(m.ptr());
  THNPEvent_init(m.ptr());

  // Define the device properties class for Python
  py::class_<torch_neuronx::NeuronDeviceProperties>(m, "NeuronDeviceProperties")
      .def_readonly("name", &torch_neuronx::NeuronDeviceProperties::name)
      .def_readonly("total_memory", &torch_neuronx::NeuronDeviceProperties::total_memory)
      .def("__repr__", [](const torch_neuronx::NeuronDeviceProperties& props) {
        return "<NeuronDeviceProperties(name='" + props.name +
               "', total_memory=" + std::to_string(props.total_memory) + ")>";
      });

  m.def("_register_device", &torch_neuronx::register_neuron_device,
        "Register neuron as a device type in PyTorch");
  m.def("_set_python_ops_registered", &torch_neuronx::set_python_ops_registered,
        "Set flag indicating Python operations are registered");
  m.def(
      "_set_python_contiguous_op",
      [](py::object op) {
        // Release old reference if any
        if (torch_neuronx::python_contiguous_op) {
          Py_XDECREF(torch_neuronx::python_contiguous_op);
        }
        // Store new reference (increment ref count)
        torch_neuronx::python_contiguous_op = op.ptr();
        Py_XINCREF(torch_neuronx::python_contiguous_op);
      },
      "Set the Python contiguous operation instance");

  // Raw NRT memory operations for CPU fallback
  m.def(
      "_nrt_copy_raw_to_cpu",
      [](const at::Tensor& src, at::Tensor& dst, int device_id, bool non_blocking = false) {
        TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                    "Expected Neuron tensor, got ", src.device());
        TORCH_CHECK(dst.device().is_cpu(), "Destination tensor must be on CPU, got ", dst.device());

        void* dst_data_ptr = dst.data_ptr();
        size_t src_offset = src.storage_offset() * src.element_size();
        size_t size = dst.nbytes();

        if (!at::neuron::utils::IsSyncModeEnabled()) {
          void* src_ptr = at::neuron::utils::GetBaseTensorPtr(src.data_ptr(), src_offset);
          auto read_kernel = std::make_unique<at::neuron::ReadDirectKernelExecution>(
              "neuron::copy::neuron_to_cpu", src_ptr, dst_data_ptr, src_offset, size, device_id);
          auto context = std::make_unique<at::neuron::OperationContext>(std::move(read_kernel));
          auto current_stream = at::neuron::getCurrentNeuronStream(device_id);
          at::neuron::SubmitOperationContext(current_stream, std::move(context));
          if (!non_blocking) {
            at::neuron::synchronize(device_id);
          }
          return;
        }

        // Perform the copy
        auto storage_impl = src.storage().unsafeGetStorageImpl();
        auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);
        nrt_tensor_t* nrt_tensor = neuron_storage->neuron_tensor();
        torch_neuronx::utils::nrt_copy_neuron_to_cpu(nrt_tensor, dst_data_ptr, src_offset, size);
      },
      "Copy raw data from Neuron device to CPU tensor");

  m.def(
      "_nrt_copy_cpu_to_raw",
      [](const at::Tensor& src, const at::Tensor& dst, int device_id, bool non_blocking = false) {
        TORCH_CHECK(src.device().is_cpu(), "Source tensor must be on CPU, got ", src.device());
        TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                    "Destination tensor must be on Neuron device, got ", dst.device());
        TORCH_CHECK(dst.is_contiguous(), "Destination tensor must be contiguous");

        void* src_data_ptr = src.data_ptr();
        size_t size = src.nbytes();
        size_t dst_offset = dst.storage_offset() * dst.element_size();

        auto storage_impl = dst.storage().unsafeGetStorageImpl();
        auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);

        if (!at::neuron::utils::IsSyncModeEnabled()) {
          void* dst_ptr = at::neuron::utils::GetBaseTensorPtr(dst.data_ptr(), dst_offset);
          auto write_kernel = std::make_unique<at::neuron::WriteDirectKernelExecution>(
              "neuron::copy::cpu_to_neuron", src_data_ptr, dst_ptr, dst_offset, size, device_id);
          auto context = std::make_unique<at::neuron::OperationContext>(std::move(write_kernel));
          auto current_stream = at::neuron::getCurrentNeuronStream(device_id);
          at::neuron::SubmitOperationContext(current_stream, std::move(context));
          if (!non_blocking) {
            at::neuron::synchronize(device_id);
          }
          return;
        }

        nrt_tensor_t* nrt_tensor = neuron_storage->neuron_tensor();
        torch_neuronx::utils::nrt_copy_cpu_to_neuron(src_data_ptr, nrt_tensor, dst_offset, size);
      },
      "Copy raw data from CPU tensor to Neuron device");

  m.def(
      "_get_nrt_tensor_capsule",
      [](const at::Tensor& tensor) -> py::capsule {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "Expected Neuron tensor, got ", tensor.device());

        // Get the storage implementation
        auto storage_impl = tensor.storage().unsafeGetStorageImpl();
        auto neuron_storage = static_cast<c10_neuron::NeuronStorageImpl*>(storage_impl);
        nrt_tensor_t* nrt_tensor = neuron_storage->neuron_tensor();

        // Check for null pointer before creating PyCapsule
        TORCH_CHECK(nrt_tensor != nullptr,
                    "Cannot get NRT tensor capsule: tensor data pointer is null. "
                    "This may indicate the tensor was not properly allocated or "
                    "the allocation operation has not completed yet. "
                    "Tensor shape: ",
                    tensor.sizes(), ", dtype: ", tensor.dtype(), ", device: ", tensor.device());

        // Return as a PyCapsule to avoid exposing raw pointer
        return py::capsule(nrt_tensor, "nrt_tensor_t");
      },
      "Get NRT tensor handle as a capsule");

  // High-level tensor copy helpers using NRT directly (no Tensor.to used)
  m.def(
      "_nrt_copy_cpu_to_neuron_tensor",
      [](const at::Tensor& src_cpu, at::Tensor& dst_neuron, bool non_blocking = false) {
        torch_neuronx::utils::copy_cpu_to_neuron(src_cpu, dst_neuron, non_blocking);
      },
      "Copy from CPU tensor to Neuron tensor using NRT", py::arg("src_cpu"), py::arg("dst_neuron"),
      py::arg("non_blocking") = false);
  m.def(
      "_nrt_copy_neuron_to_cpu_tensor",
      [](const at::Tensor& src_neuron, at::Tensor& dst_cpu, bool non_blocking = false) {
        torch_neuronx::utils::copy_neuron_to_cpu(src_neuron, dst_cpu, non_blocking);
      },
      "Copy from Neuron tensor to CPU tensor using NRT", py::arg("src_neuron"), py::arg("dst_cpu"),
      py::arg("non_blocking") = false);
  m.def(
      "_nrt_copy_neuron_to_neuron_tensor",
      [](const at::Tensor& src_neuron, at::Tensor& dst_neuron, bool non_blocking = false) {
        torch_neuronx::utils::copy_neuron_to_neuron(src_neuron, dst_neuron, non_blocking);
      },
      "Copy from Neuron tensor to Neuron tensor (same device) using NRT", py::arg("src_neuron"),
      py::arg("dst_neuron"), py::arg("non_blocking") = false);

  m.def("version", []() { return "0.1.0"; });
  m.def("_is_neuron_runtime_initialized", &torch_neuronx::is_initialized,
        "Check if the runtime is initialized");
  m.def("_neuron_init", &torch_neuronx::neuron_init, "Initialize Neuron runtime");
  m.def("_neuron_getDeviceCount", &torch_neuronx::neuron_getDeviceCount,
        "Get the number of Neuron devices");
  m.def("_neuron_getCurrentDevice", &torch_neuronx::neuron_getCurrentDevice,
        "Get the current Neuron device");
  m.def("_neuron_setDevice", &torch_neuronx::neuron_setDevice, "Set the current Neuron device",
        py::arg("device_id"));
  m.def("_neuron_getDeviceProperties", &torch_neuronx::neuron_getDeviceProperties,
        "Get properties of a Neuron device", py::arg("device_id"));
  m.def("_neuron_close", &torch_neuronx::neuron_close,
        "Close Neuron runtime and clean up resources");

  // Local rank to vnc_id mapping functions
  m.def(
      "_set_local_world_size", [](int size) { c10_neuron::set_local_world_size(size); },
      "Set the local world size for vnc_id mapping", py::arg("size"));
  m.def(
      "_set_local_device_start_index", [](int id) { c10_neuron::set_local_device_start_index(id); },
      "Set the local start index for vnc_id mapping", py::arg("id"));
  m.def(
      "_set_world_size", [](int size) { c10_neuron::set_world_size(size); }, "Set the world size",
      py::arg("size"));
  m.def("_set_rank", [](int rank) { c10_neuron::set_rank(rank); }, "Set the rank", py::arg("rank"));
  m.def(
      "_get_vnc_id", [](int device_id) { return c10_neuron::get_vnc_id(device_id); },
      "Get vnc_id from device_id", py::arg("device_id"));
  m.def(
      "_vnc_count", []() { return c10_neuron::vnc_count(); },
      "Get the number of visible neuron cores");

  // Stream management functions
  m.def(
      "_neuron_getCurrentStream",
      [](c10::DeviceIndex device_index) -> py::tuple {
        auto stream = at::neuron::getCurrentNeuronStream(device_index);
        return py::make_tuple(stream.id(), stream.device_index(),
                              static_cast<int64_t>(stream.device().type()));
      },
      "Get current stream for device", py::arg("device_index") = -1);

  m.def(
      "_neuron_getDefaultStream",
      [](c10::DeviceIndex device_index) -> py::tuple {
        auto stream = at::neuron::getDefaultNeuronStream(device_index);
        return py::make_tuple(stream.id(), stream.device_index(),
                              static_cast<int64_t>(stream.device().type()));
      },
      "Get default stream for device", py::arg("device_index") = -1);

  m.def(
      "_neuron_setStream",
      [](c10::StreamId stream_id, c10::DeviceIndex device_index, int64_t device_type) {
        TORCH_CHECK(device_type == static_cast<int64_t>(c10::DeviceType::PrivateUse1),
                    "Expected Neuron device type");
        auto stream =
            at::neuron::NeuronStream(stream_id, device_index, at::neuron::NeuronStream::UNCHECKED);
        at::neuron::setCurrentNeuronStream(stream);
      },
      "Set current stream", py::arg("stream_id"), py::arg("device_index"), py::arg("device_type"));

  m.def(
      "_neuron_synchronize",
      [](c10::DeviceIndex device_index) { at::neuron::synchronize(device_index); },
      "Synchronize all streams on device", py::arg("device_index") = -1);

  // Expose allocator submodule temporarily for testing
  // ToDo: Remove this later
  py::module allocator_m = m.def_submodule("NeuronCachingAllocator", "Neuron memory allocator");
  allocator_m.def(
      "get",
      []() -> py::object {
        c10::Allocator* allocator = c10_neuron::NeuronCachingAllocator::get();
        // Return as opaque pointer - Python will hold a reference but can't call methods directly
        return py::cast(allocator, py::return_value_policy::reference);
      },
      "Get the Neuron allocator instance");
  allocator_m.def("emptyCache", &c10_neuron::NeuronCachingAllocator::emptyCache,
                  "Empty the allocator cache");
  allocator_m.def("getCachedBlocks", &c10_neuron::NeuronCachingAllocator::getCachedBlocks,
                  "Get the number of cached memory blocks");
  allocator_m.def(
      "findTensor",
      [](py::int_ ptr) -> py::object {
        uintptr_t int_ptr = ptr.cast<uintptr_t>();
        void* data_ptr = reinterpret_cast<void*>(int_ptr);
        nrt_tensor_t* tensor = c10_neuron::NeuronCachingAllocator::findTensor(data_ptr);
        if (tensor) {
          // Return as opaque pointer (integer) that can be passed to krtlib
          return py::int_(reinterpret_cast<uintptr_t>(tensor));
        }
        return py::none();
      },
      "Find the nrt_tensor_t* for a given data pointer", py::arg("data_ptr"));

  // NRT functions for NKI kernel execution
  m.def(
      "_nrt_load",
      [](py::bytes neff_bytes, py::int_ current_device_id) -> py::object {
        torch_neuronx::maybe_lazy_init();
        std::string neff_data = neff_bytes;
        nrt_model_t* model = nullptr;

        int vnc_id = c10_neuron::get_vnc_id(current_device_id);

        // Load model on the current device
        NRT_STATUS status = nrt_load(neff_data.data(), neff_data.size(), vnc_id, 1, &model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to load NEFF. Status: " + std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(model));
      },
      "Load a NEFF and return model handle", py::arg("neff_bytes"), py::arg("current_device_id"));

  m.def(
      "_nrt_load_collectives",
      [](py::bytes neff_bytes, py::int_ global_device_id, py::int_ global_device_count,
         py::int_ current_device_id) -> py::object {
        torch_neuronx::maybe_lazy_init();
        std::string neff_data = neff_bytes;
        nrt_model_t* model = nullptr;

        int vnc_id = c10_neuron::get_vnc_id(current_device_id);

        // Load model on the current device
        NRT_STATUS status = nrt_load_collectives(neff_data.data(), neff_data.size(), vnc_id, 1,
                                                 global_device_id, global_device_count, &model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to load NEFF. Status: " + std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(model));
      },
      "Load a NEFF and return model handle", py::arg("neff_bytes"), py::arg("global_device_id"),
      py::arg("global_device_count"), py::arg("current_device_id"));

  m.def(
      "_nrt_unload",
      [](py::int_ model_handle) {
        nrt_model_t* model = reinterpret_cast<nrt_model_t*>(model_handle.cast<uintptr_t>());
        NRT_STATUS status = nrt_unload(model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to unload model. Status: " + std::to_string(status));
        }
      },
      "Unload a model", py::arg("model_handle"));

  m.def(
      "_nrt_allocate_tensor_set",
      []() -> py::object {
        nrt_tensor_set_t* tensor_set = nullptr;
        NRT_STATUS status = nrt_allocate_tensor_set(&tensor_set);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to allocate tensor set. Status: " +
                                   std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(tensor_set));
      },
      "Allocate a new tensor set");

  m.def(
      "_nrt_destroy_tensor_set",
      [](py::int_ tensor_set_handle) {
        nrt_tensor_set_t* tensor_set =
            reinterpret_cast<nrt_tensor_set_t*>(tensor_set_handle.cast<uintptr_t>());

        // Clean up slice tensors before destroying tensor set
        {
          std::lock_guard<std::mutex> lock(torch_neuronx::slice_tensors_mutex);
          auto it = torch_neuronx::slice_tensors_map.find(tensor_set);
          if (it != torch_neuronx::slice_tensors_map.end()) {
            for (nrt_tensor_t* slice_tensor : it->second) {
              nrt_tensor_free(&slice_tensor);
            }
            torch_neuronx::slice_tensors_map.erase(it);
          }
        }

        nrt_destroy_tensor_set(&tensor_set);
      },
      "Destroy a tensor set", py::arg("tensor_set_handle"));

  m.def(
      "_nrt_add_tensor_to_tensor_set",
      [](py::int_ tensor_set_handle, py::object tensor_obj, const std::string& tensor_name) {
        nrt_tensor_set_t* tensor_set =
            reinterpret_cast<nrt_tensor_set_t*>(tensor_set_handle.cast<uintptr_t>());

        // Get PyTorch tensor using THPVariable_Unpack
        torch::Tensor torch_tensor = THPVariable_Unpack(tensor_obj.ptr());

        // Verify tensor is on neuron device
        if (torch_tensor.device().type() != c10::DeviceType::PrivateUse1) {
          throw std::runtime_error("Tensor must be on neuron device, but got: " +
                                   torch_tensor.device().str());
        }

        if (!torch_tensor.is_contiguous()) {
          throw std::runtime_error("Cannot add non-contiguous tensor to NRT tensor set");
        }

        // Get data pointer - adjust for storage offset to get base storage pointer
        void* data_ptr = torch_tensor.data_ptr();
        size_t storage_offset_bytes = torch_tensor.storage_offset() * torch_tensor.element_size();
        void* base_ptr = static_cast<char*>(data_ptr) - storage_offset_bytes;

        // Find the corresponding nrt_tensor_t using the allocator with base pointer
        nrt_tensor_t* base_tensor = c10_neuron::NeuronCachingAllocator::findTensor(base_ptr);
        if (!base_tensor) {
          throw std::runtime_error(
              "Could not find NRT tensor for PyTorch tensor. Ensure tensor is on neuron device.");
        }

        // Create a slice tensor that points to the correct offset within the base tensor
        // The slice size must match the view's byte size, not (base_size - offset)
        nrt_tensor_t* nrt_tensor = nullptr;
        size_t view_size_bytes = torch_tensor.numel() * torch_tensor.element_size();
        size_t base_size_bytes = nrt_tensor_get_size(base_tensor);
        if (storage_offset_bytes > base_size_bytes ||
            base_size_bytes - storage_offset_bytes < view_size_bytes) {
          throw std::runtime_error(
              "Tensor view exceeds base storage bounds for NRT slice allocation");
        }
        NRT_STATUS offset_status = nrt_tensor_allocate_slice(
            base_tensor, storage_offset_bytes, view_size_bytes, tensor_name.c_str(), &nrt_tensor);
        if (offset_status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to create tensor slice with offset. Status: " +
                                   std::to_string(offset_status));
        }

        // Add to tensor set
        NRT_STATUS status =
            nrt_add_tensor_to_tensor_set(tensor_set, tensor_name.c_str(), nrt_tensor);
        if (status != NRT_SUCCESS) {
          // Avoid leak of the allocated slice tensor on failure
          nrt_tensor_free(&nrt_tensor);
          throw std::runtime_error("Failed to add tensor to tensor set. Status: " +
                                   std::to_string(status));
        }

        // Track slice tensor for cleanup
        {
          std::lock_guard<std::mutex> lock(torch_neuronx::slice_tensors_mutex);
          torch_neuronx::slice_tensors_map[tensor_set].push_back(nrt_tensor);
        }
      },
      "Add a PyTorch tensor to tensor set", py::arg("tensor_set_handle"), py::arg("tensor"),
      py::arg("tensor_name"));

  m.def(
      "_nrt_execute",
      [](py::int_ model_handle, py::int_ input_set_handle, py::int_ output_set_handle) {
        nrt_model_t* model = reinterpret_cast<nrt_model_t*>(model_handle.cast<uintptr_t>());
        nrt_tensor_set_t* input_set =
            reinterpret_cast<nrt_tensor_set_t*>(input_set_handle.cast<uintptr_t>());
        nrt_tensor_set_t* output_set =
            reinterpret_cast<nrt_tensor_set_t*>(output_set_handle.cast<uintptr_t>());

        NRT_STATUS status = nrt_execute(model, input_set, output_set);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to execute model. Status: " + std::to_string(status));
        }
      },
      "Execute a model", py::arg("model_handle"), py::arg("input_set_handle"),
      py::arg("output_set_handle"));

  m.def(
      "_nrt_barrier",
      [](py::int_ device_id, py::int_ global_device_id, py::int_ global_device_count) {
        torch_neuronx::maybe_lazy_init();
        at::neuron::synchronize(global_device_id);
        NRT_STATUS status = nrt_barrier(device_id, global_device_id, global_device_count);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to execute the device barrier " +
                                   std::to_string(status));
        }
      },
      "Execute barrier across all the devices", py::arg("device_id"), py::arg("global_device_id"),
      py::arg("global_device_count"));

  // Expose logging functions
  m.def(
      "_log_executed_op",
      [](const std::string& op_name) {
        // Log only once per unique operation
        if (torch_neuronx::shouldLogExecuted(op_name)) {
          torch_neuronx::NeuronLogger::getInstance().log(
              torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::OPERATOR_EXECUTED,
              "Operator '" + op_name + "' executed on Neuron");
          torch_neuronx::markExecutedLogged(op_name);
        }
      },
      "Log an operation that was executed on Neuron device (deduplicated)", py::arg("op_name"));

  m.def(
      "_log_offloaded_op",
      [](const std::string& op_name) {
        // Log only once per unique operation
        if (torch_neuronx::shouldLogFallback(op_name)) {
          torch_neuronx::NeuronLogger::getInstance().log(
              torch_neuronx::LogLevel::WARNING, torch_neuronx::LogCategory::OPERATOR_FALLBACK,
              "Operator '" + op_name + "' fell back to CPU");
          torch_neuronx::markFallbackLogged(op_name);
        }
      },
      "Log an operation that fell back to CPU (deduplicated)", py::arg("op_name"));

  // NEFF cache logging functions
  m.def(
      "_log_neff_cache_hit",
      [](const std::string& cache_key) {
        torch_neuronx::NeuronLogger::getInstance().log(
            torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE,
            "[NEFF_CACHE] Cache HIT for key: " + cache_key);
      },
      "Log a NEFF cache hit", py::arg("cache_key"));

  m.def(
      "_log_neff_cache_miss",
      [](const std::string& cache_key) {
        torch_neuronx::NeuronLogger::getInstance().log(
            torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE,
            "[NEFF_CACHE] Cache MISS for key: " + cache_key);
      },
      "Log a NEFF cache miss", py::arg("cache_key"));

  m.def(
      "_log_neff_cache_store",
      [](const std::string& cache_key) {
        torch_neuronx::NeuronLogger::getInstance().log(
            torch_neuronx::LogLevel::INFO, torch_neuronx::LogCategory::NEFF_CACHE,
            "[NEFF_CACHE] Storing NEFF for key: " + cache_key);
      },
      "Log a NEFF cache store operation", py::arg("cache_key"));

  // Op tracking API for testing
  m.def(
      "_get_fallback_ops",
      []() {
        auto ops = torch_neuronx::getFallbackOps();
        return py::list(py::cast(ops));
      },
      "Get the list of operations that fell back to CPU");

  m.def(
      "_get_executed_ops",
      []() {
        auto ops = torch_neuronx::getExecutedOps();
        return py::list(py::cast(ops));
      },
      "Get the list of operations that executed on Neuron");

  m.def("_clear_op_tracking", &torch_neuronx::clearOpTracking,
        "Clear both fallback and executed operation tracking");

  m.def(
      "_get_instance_type", []() -> std::string { return at::neuron::utils::GetInstanceType(); },
      "Get the instance info from Neuron runtime");

  m.def(
      "_get_platform_target",
      []() -> std::string { return at::neuron::utils::GetPlatformTarget(); },
      "Get the platform target");

  m.def(
      "_get_logical_neuron_cores",
      []() -> std::string { return at::neuron::utils::GetLogicalNeuronCores(); },
      "Get the logical neuron cores setting");

  // Compilation cache statistics API
  m.def(
      "_get_compilation_cache_stats",
      []() -> py::dict {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        auto stats = cache.GetStatistics();

        py::dict result;
        result["total_entries"] = stats.total_entries;
        result["cache_hits"] = stats.cache_hits;
        result["cache_misses"] = stats.cache_misses;
        result["hit_rate"] = stats.hit_rate;
        result["memory_usage_bytes"] = stats.memory_usage_bytes;
        result["total_compilation_time_ms"] = stats.total_compilation_time.count();
        result["evictions_performed"] = stats.evictions_performed;
        result["max_cache_size_bytes"] = stats.max_cache_size_bytes;
        result["max_cache_entries"] = stats.max_cache_entries;
        result["memory_utilization_ratio"] = stats.memory_utilization_ratio;

        return result;
      },
      "Get compilation cache statistics");

  m.def(
      "_clear_compilation_cache",
      []() {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        cache.Clear();
      },
      "Clear the compilation cache");

  m.def(
      "_get_all_cache_keys",
      []() -> py::set {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        auto keys = cache.GetAllCacheKeys();
        py::set result;
        for (const auto& key : keys) {
          result.add(key);
        }
        return result;
      },
      "Get all cache keys");

  m.def(
      "_submit_xla_task_to_pipeline",
      [](const std::string& op_name, const std::vector<torch::Tensor>& inputs,
         const std::vector<torch::Tensor>& outputs, py::bytes lowered_ir_bytes,
         py::object stream_obj, const std::string& cache_key, const std::string& python_stack_trace,
         const bool has_collectives, const py::object& cpu_fallback_context = py::none()) {
        auto transform_result =
            torch_neuronx::ProcessInputTransformations(op_name, inputs, has_collectives);

        at::neuron::NeuronStream stream;
        if (!stream_obj.is_none() && THNPStream_Check(stream_obj.ptr())) {
          stream = THNPStream_Unpack(stream_obj.ptr());
        } else {
          stream = at::neuron::getCurrentNeuronStream();
        }

        std::string_view ir_str = lowered_ir_bytes.cast<std::string_view>();
        std::vector<uint8_t> ir_bytes(
            reinterpret_cast<const uint8_t*>(ir_str.data()),
            reinterpret_cast<const uint8_t*>(ir_str.data()) + ir_str.size());

        // Store tensor metadata as TensorContext
        std::vector<at::neuron::TensorContext> input_tensors_context;
        input_tensors_context.reserve(inputs.size());
        for (const auto& tensor : inputs) {
          input_tensors_context.push_back(at::neuron::TensorContext::FromTensor(tensor));
        }

        std::vector<at::neuron::TensorContext> output_tensors_context;
        output_tensors_context.reserve(outputs.size());
        for (const auto& tensor : outputs) {
          output_tensors_context.push_back(at::neuron::TensorContext::FromTensor(tensor));
        }

        // Convert Python CPU fallback context to C++ struct
        at::neuron::CPUFallbackContext fallback_ctx =
            at::neuron::utils::ConvertToCPUFallbackContext(cpu_fallback_context);

        int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs, outputs);
        // Process tensors with slicing - updates fallback_ctx.tensor_data_ptrs in-place
        std::vector<void*> src_ptrs = at::neuron::utils::ProcessTensorsWithSlicing(
            transform_result.final_inputs, fallback_ctx, stream, device_id);
        std::vector<void*> dst_ptrs =
            at::neuron::utils::ProcessTensorsWithSlicing(outputs, fallback_ctx, stream, device_id);

        auto xla_kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
            op_name, src_ptrs, dst_ptrs, cache_key, ir_bytes, has_collectives, device_id);

        if (transform_result.has_input_transforms) {
          xla_kernel->SetInputTransformations(std::move(transform_result.kernel_transforms));
        }

        auto context = std::make_unique<at::neuron::OperationContext>(
            std::move(xla_kernel), input_tensors_context, output_tensors_context,
            python_stack_trace, std::move(fallback_ctx));
        at::neuron::SubmitOperationContext(stream, std::move(context));
      },
      "Submit XLA task with pre-lowered IR for asynchronous execution", py::arg("op_name"),
      py::arg("inputs"), py::arg("outputs"), py::arg("lowered_ir_bytes"), py::arg("stream"),
      py::arg("cache_key"), py::arg("python_stack_trace"), py::arg("has_collectives"),
      py::arg("cpu_fallback_context") = py::none());

  // torch.compile: compile StableHLO to NEFF and store in cache
  m.def(
      "compile_graph",
      [](const std::string& base_cache_key, py::bytes stablehlo_bytes,
         bool has_collectives) -> std::string {
        // Convert StableHLO bytes
        std::string_view stablehlo_str = stablehlo_bytes.cast<std::string_view>();
        std::vector<uint8_t> stablehlo_vec(stablehlo_str.begin(), stablehlo_str.end());

        // Create compile kernel and get final cache key
        auto compile_kernel = std::make_unique<at::neuron::CompileOnlyKernelExecution>(
            base_cache_key, stablehlo_vec, has_collectives);

        // Get final cache key before moving kernel
        std::string cache_key = compile_kernel->GetCacheKey();

        // Create OperationContext for compilation
        std::vector<at::neuron::TensorContext> empty_input_context;
        std::vector<at::neuron::TensorContext> empty_output_context;
        auto context = std::make_unique<at::neuron::OperationContext>(
            std::move(compile_kernel), empty_input_context, empty_output_context, "");

        // Submit for compilation via proper stream infrastructure
        auto stream = at::neuron::getCurrentNeuronStream();
        at::neuron::SubmitOperationContext(stream, std::move(context));
        return cache_key;
      },
      "Compile StableHLO to NEFF and store in cache, return final cache key",
      py::arg("base_cache_key"), py::arg("stablehlo_bytes"), py::arg("has_collectives"));

  // torch.compile: execute compiled graph using cache key
  m.def(
      "execute_compiled_graph",
      [](const std::string& graph_name, const std::string& cache_key,
         const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& outputs,
         bool has_collectives) {
        TORCH_NEURONX_DEBUG("execute_compiled_graph invoked.", "graph_name=", graph_name,
                            "cache_key=", cache_key);

        // Get current stream
        auto stream = at::neuron::getCurrentNeuronStream();

        // Extract device_id from tensors
        int target_device_id = c10_neuron::GetTargetDeviceId(inputs, outputs);

        // Create empty fallback context
        at::neuron::CPUFallbackContext fallback_ctx;

        // Get tensor data pointers with slicing support
        std::vector<void*> input_ptrs = at::neuron::utils::ProcessTensorsWithSlicing(
            inputs, fallback_ctx, stream, target_device_id);
        std::vector<void*> output_ptrs = at::neuron::utils::ProcessTensorsWithSlicing(
            outputs, fallback_ctx, stream, target_device_id);

        // Create NeffDirectKernelExecution with NEFF bytes
        auto kernel = std::make_unique<at::neuron::NeffDirectKernelExecution>(
            graph_name, cache_key, input_ptrs, output_ptrs, target_device_id, has_collectives);

        // Store tensor metadata
        std::vector<at::neuron::TensorContext> input_tensors_context;
        input_tensors_context.reserve(inputs.size());
        for (const auto& tensor : inputs) {
          input_tensors_context.push_back(at::neuron::TensorContext::FromTensor(tensor));
        }

        std::vector<at::neuron::TensorContext> output_tensors_context;
        output_tensors_context.reserve(outputs.size());
        for (const auto& tensor : outputs) {
          output_tensors_context.push_back(at::neuron::TensorContext::FromTensor(tensor));
        }

        // Create and submit operation context
        auto context = std::make_unique<at::neuron::OperationContext>(
            std::move(kernel), input_tensors_context, output_tensors_context, "",
            std::move(fallback_ctx));
        at::neuron::SubmitOperationContext(stream, std::move(context));
      },
      "Execute compiled graph using cache key", py::arg("graph_name"), py::arg("cache_key"),
      py::arg("inputs"), py::arg("outputs"), py::arg("has_collectives"));

  // Register cleanup function that runs during module destruction
  // This happens before Python's atexit handlers and before C++ destructors
  static auto cleanup_streams_func = +[]() {
    try {
      // Synchronize all devices to ensure pending operations complete
      // This runs while Python services are still available for C++ -> Python calls
      auto& manager = at::neuron::NeuronResourceManager::Instance().GetStreamManager();
      int device_count = torch_neuronx::neuron_getDeviceCount();
      for (int device_idx = 0; device_idx < device_count; ++device_idx) {
        manager.SynchronizeDevice(device_idx);
      }
    } catch (const std::exception& e) {
      // Log error but don't throw during cleanup
      std::cerr << "Warning: Stream cleanup failed during module destruction: " << e.what()
                << std::endl;
    } catch (...) {
      std::cerr << "Warning: Stream cleanup failed during module destruction: unknown error"
                << std::endl;
    }
  };

  // Create a capsule that will call cleanup_streams_func when the module is destroyed
  auto cleanup_capsule = py::capsule(cleanup_streams_func);
  m.add_object("_stream_cleanup_capsule", cleanup_capsule);
}
