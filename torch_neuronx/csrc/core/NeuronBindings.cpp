#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
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
#include "torch_neuronx/csrc/core/NeuronBarrier.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronGuardImpl.h"
#include "torch_neuronx/csrc/core/NeuronHooksInterface.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/core/cache/PersistentCacheBackend.h"
#include "torch_neuronx/csrc/core/compilation/CompilationCache.h"
#include "torch_neuronx/csrc/core/utils/NeuronResourceManager.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/core/utils/PythonTypeConverter.h"
// Include Streams
#include "torch_neuronx/csrc/Stream.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
// Metrics
#include "torch_neuronx/csrc/core/metrics/NeuronMetrics.h"
// Include Distributed Work and Watchdog (for async collective completion tracking)
#include "torch_neuronx/csrc/distributed/NeuronWatchdog.h"
#include "torch_neuronx/csrc/distributed/NeuronWork.h"
// Copy helpers
#include "torch_neuronx/csrc/utils/CopyUtils.h"
// Runtime utilities (including profiling)
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
// Operation execution
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
// Profiler mapping collector
#include "torch_neuronx/csrc/core/ProfilerMappingCollector.h"
// Lazy transformation management
#include "torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistration.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TransformationRegistry.h"
#include "torch_neuronx/csrc/core/lazy_materialization/TypeUtils.h"
// Internal operations
#include "torch_neuronx/csrc/ops/InternalOps.h"
#include "torch_neuronx/csrc/profiler/Registration.h"

// Include Neuron Runtime headers
extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>
}

namespace torch_neuronx {

// Forward declarations
bool are_python_ops_registered();

// Helper function to convert c10d::OpType to CollectiveDirectKernelExecution::CollectiveType
at::neuron::CollectiveDirectKernelExecution::CollectiveType ConvertOpTypeToCollectiveType(
    c10d::OpType op_type) {
  using CollectiveType = at::neuron::CollectiveDirectKernelExecution::CollectiveType;
  switch (op_type) {
    case c10d::OpType::ALLGATHER:
    case c10d::OpType::_ALLGATHER_BASE:
      return CollectiveType::kAllGather;
    case c10d::OpType::ALLREDUCE:
      return CollectiveType::kAllReduce;
    case c10d::OpType::REDUCE_SCATTER:
      return CollectiveType::kReduceScatter;
    case c10d::OpType::ALLTOALL:
    case c10d::OpType::ALLTOALL_BASE:
      return CollectiveType::kAllToAll;
    case c10d::OpType::BROADCAST:
      return CollectiveType::kBroadcast;
    case c10d::OpType::REDUCE:
      return CollectiveType::kReduce;
    default:
      TORCH_CHECK(false, "Unsupported collective operation type: ", static_cast<int>(op_type));
  }
}

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

// Prepares the tensors for kernel execution
std::pair<std::vector<at::neuron::TensorDataRef>, std::vector<at::neuron::TensorContext>>
PrepareKernelTensors(const std::vector<torch::Tensor>& tensors) {
  std::vector<at::neuron::TensorDataRef> refs;
  std::vector<at::neuron::TensorContext> contexts;
  refs.reserve(tensors.size());
  contexts.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_contiguous()) {
      throw std::runtime_error("Cannot process non-contiguous tensors");
    }
    void* ptr = const_cast<void*>(t.storage().data());
    contexts.push_back(at::neuron::TensorContext::FromTensor(t));
    auto tensor_ptr = c10_neuron::NeuronCachingAllocator::findTensorPtr(ptr);
    refs.emplace_back(std::move(tensor_ptr), ptr);
  }

  return {std::move(refs), std::move(contexts)};
}

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

// Runtime initialization
void neuron_init() {
  if (c10_neuron::is_initialized()) {
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
    if (actual_devices > static_cast<int>(max_representable) + 1) {
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

  at::neuron::InitializeStreamPools();

  c10_neuron::set_initialized(true);

  // Register cleanup function
  std::atexit(neuron_cleanup_atexit);
}

// Maybe initialize - this is what native functions should call
void maybe_lazy_init() {
  if (!c10_neuron::is_initialized()) {
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

// Contiguous implementation for AutogradPrivateUse1
at::Tensor contiguous_autograd_neuron(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.unsafeGetTensorImpl()->is_python_dispatch()) {
    // During dynamo tracing, just return a clone or the tensor itself
    if (self.is_contiguous(memory_format)) {
      return self;
    }
    return at::clone(self, memory_format);
  }

  return torch_neuronx::ops::contiguous_internal(self, memory_format);
}

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

// Register specific implementations for AutogradPrivateUse1
// contiguous gets its own implementation to avoid going through the fallback dispatch
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("contiguous", &torch_neuronx::contiguous_autograd_neuron);
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

namespace {
// Helper function for PYBIND11 bindings to extract and validate nrt_inspect_config_t from Python
// capsule Using anonymous namespace to keep it file-local without polluting global namespace
static nrt_inspect_config_t* extract_inspect_config_from_capsule(
    const py::capsule& config_capsule) {
  nrt_inspect_config_t* options = static_cast<nrt_inspect_config_t*>(config_capsule.get_pointer());
  if (!options) {
    throw std::runtime_error("Invalid config capsule");
  }
  return options;
}
}  // anonymous namespace

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Initialize the logger
  torch_neuronx::NeuronLogger::getInstance();
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
  m.def("_register_profiler", &at::neuron::registerNeuronProfiler,
        "Register Neuron profiler with libkineto");
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
      [](const at::Tensor& src, at::Tensor& dst, bool non_blocking = false) {
        torch_neuronx::utils::copy_neuron_to_cpu(src, dst, non_blocking);
      },
      "Copy raw data from Neuron device to CPU tensor");

  m.def(
      "_nrt_copy_cpu_to_raw",
      [](const at::Tensor& src, at::Tensor& dst, bool non_blocking = false) {
        torch_neuronx::utils::copy_cpu_to_neuron(src, dst, non_blocking);
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
  m.def(
      "_get_default_generator",
      [](c10::DeviceIndex device_index) -> py::object {
        return py::reinterpret_steal<py::object>(THPGenerator_initDefaultGenerator(
            torch_neuronx::getDefaultNeuronGenerator(device_index)));
      },
      "Get the default generator for a Neuron device", py::arg("device_index") = -1);
  m.def("_is_neuron_runtime_initialized", &c10_neuron::is_initialized,
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
  m.def(
      "_reset_vnc_count", []() { c10_neuron::reset_vnc_count(); },
      "Reset the cached vnc count so it re-queries NRT");

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
        nrt_barrier_impl(device_id, global_device_id, global_device_count);
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
        auto* metrics_arena = at::neuron::metrics::NeuronMetricsArena::Get();

        py::dict result;

        // Get counter values from centralized metrics
        auto in_memory_hits_data = metrics_arena->GetCounter("CompilationCache.InMemoryHits");
        auto in_memory_misses_data = metrics_arena->GetCounter("CompilationCache.InMemoryMisses");
        auto persistent_hits_data = metrics_arena->GetCounter("CompilationCache.PersistentHits");
        auto persistent_misses_data =
            metrics_arena->GetCounter("CompilationCache.PersistentMisses");
        auto evictions_data = metrics_arena->GetCounter("CompilationCache.Evictions");
        auto total_compilations_data =
            metrics_arena->GetCounter("CompilationCache.TotalCompilations");

        int64_t in_memory_hits = in_memory_hits_data ? in_memory_hits_data->Value() : 0;
        int64_t in_memory_misses = in_memory_misses_data ? in_memory_misses_data->Value() : 0;
        int64_t persistent_hits = persistent_hits_data ? persistent_hits_data->Value() : 0;
        int64_t persistent_misses = persistent_misses_data ? persistent_misses_data->Value() : 0;
        int64_t evictions = evictions_data ? evictions_data->Value() : 0;
        int64_t total_compilations = total_compilations_data ? total_compilations_data->Value() : 0;

        // Compute aggregate hits/misses for backward compatibility
        int64_t cache_hits = in_memory_hits + persistent_hits;
        int64_t cache_misses = in_memory_misses + persistent_misses;

        size_t total_entries = cache.GetCacheSize();

        // Get metric data for timing and memory
        auto compilation_time_metric = metrics_arena->GetMetric("CompilationCache.CompilationTime");
        auto memory_usage_metric = metrics_arena->GetMetric("CompilationCache.MemoryUsage");

        double total_compilation_time_us = 0.0;
        size_t total_samples = 0;
        if (compilation_time_metric) {
          compilation_time_metric->Samples(&total_compilation_time_us, &total_samples);
        }

        double memory_usage_bytes = 0.0;
        size_t memory_samples = 0;
        if (memory_usage_metric) {
          memory_usage_metric->Samples(&memory_usage_bytes, &memory_samples);
        }

        // Calculate derived statistics
        int64_t total_requests = cache_hits + cache_misses;
        double hit_rate = total_requests > 0 ? (double)cache_hits / total_requests : 0.0;

        // Get cache configuration (these are still stored in the cache object)
        size_t max_cache_size_bytes = cache.GetMaxCacheSizeBytes();
        size_t max_cache_entries = cache.GetMaxCacheEntries();

        // Get current memory usage from cache directly
        size_t current_memory_usage = cache.GetMemoryUsageBytes();
        double memory_utilization_ratio =
            max_cache_size_bytes > 0 ? (double)current_memory_usage / max_cache_size_bytes : 0.0;

        // Populate result dictionary
        result["total_entries"] = total_entries;
        result["in_memory_hits"] = in_memory_hits;
        result["in_memory_misses"] = in_memory_misses;
        result["persistent_hits"] = persistent_hits;
        result["persistent_misses"] = persistent_misses;
        result["cache_hits"] = cache_hits;
        result["cache_misses"] = cache_misses;
        result["hit_rate"] = hit_rate;
        result["memory_usage_bytes"] = current_memory_usage;
        result["total_compilation_time_ms"] =
            total_compilation_time_us / 1000.0;  // Convert µs to ms
        result["evictions_performed"] = evictions;
        result["total_compilations"] = total_compilations;
        result["max_cache_size_bytes"] = max_cache_size_bytes;
        result["max_cache_entries"] = max_cache_entries;
        result["memory_utilization_ratio"] = memory_utilization_ratio;
        return result;
      },
      "Get compilation cache statistics from metrics arena");

  m.def(
      "_clear_compilation_cache",
      []() {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        cache.Clear();
      },
      "Clear the compilation cache");
  m.def(
      "_clear_compilation_memory_cache",
      []() {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        cache.ClearInMemoryCache();
      },
      "Clear the compilation cache");
  m.def(
      "_get_compilation_cache_entries",
      []() -> py::list {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        auto entries = cache.GetCacheEntries();

        py::list result;
        for (const auto& entry : entries) {
          py::dict d;
          d["cache_key"] = entry.cache_key;
          d["neff_size_bytes"] = entry.neff_size_bytes;
          d["access_count"] = entry.access_count;
          d["compilation_time_ms"] = entry.compilation_time.count();
          result.append(d);
        }
        return result;
      },
      "Get compilation cache entries with per-graph compilation times, for torch.compile graph "
      "metrics tracking");

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

  // ============================================================================
  // Metrics Bindings
  // ============================================================================

  m.def(
      "_neuron_counter_names",
      []() { return at::neuron::metrics::NeuronMetricsArena::Get()->GetCounterNames(); },
      "Get list of all counter names that have data");

  m.def(
      "_neuron_counter_value",
      [](const std::string& name) -> py::object {
        auto* data = at::neuron::metrics::NeuronMetricsArena::Get()->GetCounter(name);
        return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
      },
      "Get the value of a specific counter");

  m.def(
      "_neuron_metric_names",
      []() { return at::neuron::metrics::NeuronMetricsArena::Get()->GetMetricNames(); },
      "Get list of all metric names that have data");

  m.def(
      "_neuron_metric_data",
      [](const std::string& name) -> py::object {
        auto* data = at::neuron::metrics::NeuronMetricsArena::Get()->GetMetric(name);
        if (data == nullptr) {
          return py::none();
        }

        double accumulator = 0.0;
        size_t total_samples = 0;
        auto samples = data->Samples(&accumulator, &total_samples);

        py::dict result;
        result["accumulator"] = accumulator;
        result["total_samples"] = total_samples;
        result["repr"] = data->Repr(accumulator);

        py::list sample_list;
        for (const auto& sample : samples) {
          py::dict sample_dict;
          sample_dict["timestamp_ns"] = sample.timestamp_ns;
          sample_dict["value"] = sample.value;
          sample_list.append(sample_dict);
        }
        result["samples"] = sample_list;

        return result;
      },
      "Get detailed data for a specific metric");

  m.def(
      "_neuron_metrics_report", []() { return at::neuron::metrics::CreateMetricReport(); },
      "Generate a comprehensive metrics report");

  m.def(
      "_neuron_metrics_report_filtered",
      [](const py::list& counter_names, const py::list& metric_names) {
        std::vector<std::string> counter_vec;
        std::vector<std::string> metric_vec;

        for (auto& counter : counter_names) {
          counter_vec.push_back(counter.cast<std::string>());
        }
        for (auto& metric : metric_names) {
          metric_vec.push_back(metric.cast<std::string>());
        }

        return at::neuron::metrics::CreateMetricReport(counter_vec, metric_vec);
      },
      "Generate a filtered metrics report with specific counters and metrics");

  m.def(
      "_neuron_clear_counters",
      []() { at::neuron::metrics::NeuronMetricsArena::Get()->ClearCounters(); },
      "Clear all counter values");

  m.def(
      "_neuron_clear_metrics",
      []() { at::neuron::metrics::NeuronMetricsArena::Get()->ClearMetrics(); },
      "Clear all metric data");

  m.def(
      "_neuron_metrics_enabled", []() { return at::neuron::metrics::IsMetricsEnabled(); },
      "Check if metrics collection is enabled");

  m.def(
      "_neuron_set_metrics_enabled",
      [](bool enabled) { at::neuron::metrics::SetMetricsEnabled(enabled); },
      "Enable or disable metrics collection at runtime", py::arg("enabled"));

  // ============================================================================
  // Memory Statistics Bindings
  // ============================================================================

  m.def(
      "_get_memory_stats",
      [](int device_index) -> py::dict {
        auto stats =
            at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().GetMemoryStats(device_index);

        auto statToDict = [](const at::neuron::metrics::MemoryStatInfo& stat) {
          py::dict d;
          d["current"] = stat.current;
          d["peak"] = stat.peak;
          d["allocated"] = stat.allocated;
          d["freed"] = stat.freed;
          return d;
        };

        py::dict result;
        result["allocated_bytes"] = statToDict(stats.allocated_bytes);
        result["reserved_bytes"] = statToDict(stats.reserved_bytes);
        result["active_bytes"] = statToDict(stats.active_bytes);
        result["num_alloc_retries"] = stats.num_alloc_retries;
        result["num_ooms"] = stats.num_ooms;
        result["num_tensor_frees"] = stats.num_tensor_frees;
        result["allocation_requests"] = stats.allocation_requests;
        return result;
      },
      "Get memory statistics for a device", py::arg("device_index"));

  m.def(
      "_reset_peak_memory_stats",
      [](int device_index) {
        at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().ResetPeakStats(device_index);
      },
      "Reset peak memory statistics", py::arg("device_index"));

  m.def(
      "_reset_accumulated_memory_stats",
      [](int device_index) {
        at::neuron::metrics::DeviceMemoryStatsRegistry::Instance().ResetAccumulatedStats(
            device_index);
      },
      "Reset accumulated memory statistics", py::arg("device_index"));

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

        // Convert Python CPU fallback context to C++ struct
        at::neuron::CPUFallbackContext fallback_ctx =
            at::neuron::utils::ConvertToCPUFallbackContext(cpu_fallback_context);

        int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs, outputs);

        // Extract tensor data with contiguity validation
        auto [input_refs, input_contexts] =
            torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
        auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(outputs);

        auto xla_kernel = std::make_unique<at::neuron::XLACompilableKernelExecution>(
            op_name, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
            cache_key, ir_bytes, has_collectives, device_id);

        if (transform_result.has_input_transforms) {
          xla_kernel->SetInputTransformations(std::move(transform_result.kernel_transforms));
        }

        auto context = std::make_unique<at::neuron::OperationContext>(
            std::move(xla_kernel), python_stack_trace, std::move(fallback_ctx));
        at::neuron::SubmitOperationContext(stream, std::move(context));
      },
      "Submit XLA task with pre-lowered IR for asynchronous execution", py::arg("op_name"),
      py::arg("inputs"), py::arg("outputs"), py::arg("lowered_ir_bytes"), py::arg("stream"),
      py::arg("cache_key"), py::arg("python_stack_trace"), py::arg("has_collectives"),
      py::arg("cpu_fallback_context") = py::none());

  // Submit collective operation for asynchronous execution
  m.def(
      "_submit_collective_task_to_pipeline",
      [](const std::string& op_name, const std::vector<torch::Tensor>& inputs,
         const std::vector<torch::Tensor>& outputs, int64_t op_type, int64_t reduce_op,
         const std::vector<std::vector<int>>& replica_groups, py::object stream_obj,
         const std::string& python_stack_trace,
         const py::object& cpu_fallback_context = py::none()) {
        // Process input transformations (has_collectives=true makes non-contiguous inputs
        // contiguous)
        auto transform_result =
            torch_neuronx::ProcessInputTransformations(op_name, inputs, /*has_collectives=*/true);

        // Resolve stream
        at::neuron::NeuronStream stream;
        if (!stream_obj.is_none() && THNPStream_Check(stream_obj.ptr())) {
          stream = THNPStream_Unpack(stream_obj.ptr());
        } else {
          stream = at::neuron::getCurrentNeuronStream();
        }

        // Convert Python CPU fallback context to C++ struct
        at::neuron::CPUFallbackContext fallback_ctx =
            at::neuron::utils::ConvertToCPUFallbackContext(cpu_fallback_context);

        int device_id = c10_neuron::GetTargetDeviceId(transform_result.final_inputs, outputs);

        // Extract tensor data with contiguity validation
        auto [input_refs, input_contexts] =
            torch_neuronx::PrepareKernelTensors(transform_result.final_inputs);
        auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(outputs);

        // Convert c10d::OpType to CollectiveType
        auto collective_type =
            torch_neuronx::ConvertOpTypeToCollectiveType(static_cast<c10d::OpType>(op_type));

        // Extract dtype from input tensors
        TORCH_CHECK(!transform_result.final_inputs.empty(),
                    "Collective operation requires at least one input tensor");
        c10::ScalarType dtype = transform_result.final_inputs[0].scalar_type();

        // Create CollectiveDirectKernelExecution with v3 constructor signature
        auto collective_kernel = std::make_unique<at::neuron::CollectiveDirectKernelExecution>(
            op_name, collective_type, std::move(input_refs), std::move(output_refs), input_contexts,
            output_contexts, dtype,
            c10d::ReduceOp(static_cast<c10d::ReduceOp::RedOpType>(reduce_op)), replica_groups,
            device_id);

        // Create and submit OperationContext
        auto context = std::make_unique<at::neuron::OperationContext>(
            std::move(collective_kernel), python_stack_trace, std::move(fallback_ctx));

        at::neuron::SubmitOperationContext(stream, std::move(context));
      },
      "Submit collective operation for asynchronous execution", py::arg("op_name"),
      py::arg("inputs"), py::arg("outputs"), py::arg("op_type"), py::arg("reduce_op"),
      py::arg("replica_groups"), py::arg("stream"), py::arg("python_stack_trace"),
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
        auto context =
            std::make_unique<at::neuron::OperationContext>(std::move(compile_kernel), "");

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

        // Extract tensor data with contiguity validation
        auto [input_refs, input_contexts] = torch_neuronx::PrepareKernelTensors(inputs);
        auto [output_refs, output_contexts] = torch_neuronx::PrepareKernelTensors(outputs);

        auto kernel = std::make_unique<at::neuron::NeffDirectKernelExecution>(
            graph_name, cache_key, std::move(input_refs), std::move(output_refs), input_contexts,
            output_contexts, target_device_id, has_collectives);

        // Create and submit operation context
        auto context = std::make_unique<at::neuron::OperationContext>(std::move(kernel), "");
        at::neuron::SubmitOperationContext(stream, std::move(context));
      },
      "Execute compiled graph using cache key", py::arg("graph_name"), py::arg("cache_key"),
      py::arg("inputs"), py::arg("outputs"), py::arg("has_collectives"));

  // torch.compile MegaCache: write NEFF to persistent cache
  m.def(
      "put_neff_cache",
      [](const std::string& key, py::bytes neff_bytes) -> bool {
        at::neuron::PersistentCacheBackend cache;

        std::string_view neff_str = neff_bytes.cast<std::string_view>();
        // Copy required: Put() takes const ref, needs owned data
        std::vector<uint8_t> neff_vec(neff_str.begin(), neff_str.end());

        return cache.Put(key, neff_vec);
      },
      "Write NEFF to persistent cache", py::arg("key"), py::arg("neff_bytes"));

  // torch.compile MegaCache: read NEFF from persistent cache
  m.def(
      "get_neff_cache",
      [](const std::string& key) -> py::object {
        at::neuron::PersistentCacheBackend cache;

        auto result = cache.Get(key);
        if (result.has_value()) {
          return py::bytes(reinterpret_cast<const char*>(result->data()), result->size());
        }
        return py::none();
      },
      "Read NEFF from persistent cache, returns None if not found", py::arg("key"));

  // torch.compile MegaCache: get NEFF info from in-memory compilation cache
  m.def(
      "get_neff_info",
      [](const std::string& cache_key) -> py::object {
        auto& cache = at::neuron::NeuronResourceManager::Instance().GetCompilationCache();
        auto entry = cache.GetEntry(cache_key);
        if (!entry) {
          return py::none();
        }
        return py::make_tuple(entry->persistent_cache_key,
                              py::bytes(reinterpret_cast<const char*>(entry->neff_bytes.data()),
                                        entry->neff_bytes.size()));
      },
      "Get (persistent_key, neff_bytes) from compilation cache, returns None if not found",
      py::arg("cache_key"));

  // Register cleanup function that runs during module destruction
  // This happens before Python's atexit handlers and before C++ destructors
  static auto cleanup_streams_func = +[]() {
    try {
      auto devices = c10_neuron::get_visible_device_indices();
      // Synchronize all initialized devices to ensure pending operations complete
      for (auto device : devices) {
        at::neuron::synchronize(device);
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: Stream cleanup failed during module destruction: " << e.what()
                << std::endl;
    } catch (...) {
      std::cerr << "Warning: Stream cleanup failed during module destruction: unknown error"
                << std::endl;
    }
  };

  // Profiler mapping collector functions
  m.def(
      "_set_profiler_mapping_enabled",
      [](bool enabled) { at::neuron::ProfilerMappingCollector::Instance().SetEnabled(enabled); },
      "Enable/disable profiler mapping collection", py::arg("enabled"));

  m.def(
      "_get_profiler_mappings",
      []() -> py::dict {
        auto mappings = at::neuron::ProfilerMappingCollector::Instance().GetAndClear();
        py::list data;
        for (const auto& [seq_id, fw_ids] : mappings) {
          py::list fw_list;
          for (const auto& fw_id : fw_ids) {
            py::dict fw_dict;
            fw_dict["seq_nr"] = fw_id.seq_nr;
            fw_dict["th_id"] = fw_id.th_id;
            fw_dict["stream_id"] = fw_id.stream_id;
            fw_dict["os_tid"] = fw_id.os_tid;
            fw_list.append(fw_dict);
          }
          py::dict entry;
          entry["nrta_seq_id"] = seq_id;
          entry["framework_op_exec_ids"] = fw_list;
          data.append(entry);
        }
        py::dict result;
        result["version"] = 1;
        result["data"] = data;
        return result;
      },
      "Get and clear profiler mappings (nrta_sequence_id -> framework_op_exec_ids)");

  // NRT profiling functions

  m.def(
      "_nrt_inspect_stop", []() -> int { return static_cast<int>(at::neuron::nrt::StopInspect()); },
      "Stop NRT profiling/tracing");

  m.def(
      "_nrt_inspect_config_allocate",
      []() -> py::capsule {
        nrt_inspect_config_t* options = nullptr;
        NRT_STATUS status = at::neuron::nrt::AllocateInspectConfig(&options);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to allocate NRT inspect config. Status: " +
                                   std::to_string(status));
        }
        // Return as a PyCapsule with automatic cleanup using wrapper function
        return py::capsule(options, "nrt_inspect_config_t", [](void* ptr) {
          nrt_inspect_config_t* config = static_cast<nrt_inspect_config_t*>(ptr);
          if (config) {
            at::neuron::nrt::FreeInspectConfig(config);
          }
        });
      },
      "Allocate NRT inspect config structure");

  m.def(
      "_nrt_inspect_config_set_defaults",
      [](py::capsule config_capsule) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(at::neuron::nrt::SetInspectConfigDefaults(options));
      },
      "Set NRT inspect config to defaults", py::arg("config_capsule"));

  m.def(
      "_nrt_inspect_config_set_output_dir",
      [](py::capsule config_capsule, const std::string& output_dir) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(
            at::neuron::nrt::SetInspectConfigOutputDir(options, output_dir.c_str()));
      },
      "Set NRT inspect output directory", py::arg("config_capsule"), py::arg("output_dir"));

  m.def(
      "_nrt_inspect_config_set_capture_enabled_for_nc",
      [](py::capsule config_capsule, uint32_t nc_idx, bool enabled) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(
            at::neuron::nrt::SetInspectConfigCaptureEnabledForNC(options, nc_idx, enabled));
      },
      "Enable/disable capture for specific NeuronCore", py::arg("config_capsule"),
      py::arg("nc_idx"), py::arg("enabled"));

  m.def(
      "_nrt_inspect_config_set_capture_enabled_for_event_type_string",
      [](py::capsule config_capsule, const std::string& event_type, bool enabled) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(at::neuron::nrt::SetInspectConfigCaptureEnabledForEventType(
            options, event_type.c_str(), enabled));
      },
      "Enable/disable capture for specific event type", py::arg("config_capsule"),
      py::arg("event_type"), py::arg("enabled"));

  m.def(
      "_nrt_inspect_config_set_enable_inspect",
      [](py::capsule config_capsule, bool enable_inspect) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(
            at::neuron::nrt::SetInspectConfigEnableInspect(options, enable_inspect));
      },
      "Enable/disable inspect profiling", py::arg("config_capsule"), py::arg("enable_inspect"));

  m.def(
      "_nrt_inspect_config_set_device_profile_mode",
      [](py::capsule config_capsule, int device_profile_mode) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(
            at::neuron::nrt::SetInspectConfigDeviceProfileMode(options, device_profile_mode));
      },
      "Set device profile mode for inspect profiling", py::arg("config_capsule"),
      py::arg("device_profile_mode"));

  m.def(
      "_nrt_inspect_config_get_all_activity_types",
      []() -> py::list {
        const char** activity_types = nullptr;
        size_t count = 0;
        NRT_STATUS status = at::neuron::nrt::GetAllActivityTypes(&activity_types, &count);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to get all activity types. Status: " +
                                   std::to_string(status));
        }

        py::list result;
        for (size_t i = 0; i < count; ++i) {
          result.append(py::str(activity_types[i]));
        }

        // Clean up allocated memory using wrapper function
        at::neuron::nrt::FreeActivityTypes(activity_types, count);

        return result;
      },
      "Get all available activity types");

  m.def(
      "_nrt_inspect_begin_with_options",
      [](py::capsule config_capsule) -> int {
        nrt_inspect_config_t* options = extract_inspect_config_from_capsule(config_capsule);
        return static_cast<int>(at::neuron::nrt::BeginInspectWithOptions(options));
      },
      "Begin NRT profiling with configuration options", py::arg("config_capsule"));

  // Create a capsule that will call cleanup_streams_func when the module is destroyed
  auto cleanup_capsule = py::capsule(cleanup_streams_func);
  m.add_object("_stream_cleanup_capsule", cleanup_capsule);

  // ============== Distributed Bindings ==============
  // NeuronWork class for tracking async collective operations
  // Inherit from c10d::Work and use intrusive_ptr so PyTorch's distributed
  // code can cast NeuronWork back to c10d::Work when needed
  py::class_<torch_neuronx::distributed::NeuronWork, c10d::Work,
             c10::intrusive_ptr<torch_neuronx::distributed::NeuronWork>>(m, "NeuronWork")
      .def(py::init([](const std::string& pg_uid, const std::string& pg_desc, int device_index,
                       int rank, const std::string& op_type, uint64_t seq_num,
                       std::vector<at::Tensor> outputs, bool enable_timing, float timeout_ms,
                       py::object stream_obj) {
             at::Device device(c10::DeviceType::PrivateUse1, device_index);
             at::neuron::NeuronStream* stream_ptr = nullptr;
             at::neuron::NeuronStream stream;
             if (!stream_obj.is_none() && THNPStream_Check(stream_obj.ptr())) {
               stream = THNPStream_Unpack(stream_obj.ptr());
               stream_ptr = &stream;
             }
             return c10::make_intrusive<torch_neuronx::distributed::NeuronWork>(
                 pg_uid, pg_desc, device, rank, op_type, seq_num, std::move(outputs), enable_timing,
                 timeout_ms, stream_ptr);
           }),
           py::arg("pg_uid"), py::arg("pg_desc"), py::arg("device_index"), py::arg("rank"),
           py::arg("op_type"), py::arg("seq_num"), py::arg("outputs") = std::vector<at::Tensor>(),
           py::arg("enable_timing") = false, py::arg("timeout_ms") = 300000.0f,
           py::arg("stream") = py::none())
      .def("is_completed", &torch_neuronx::distributed::NeuronWork::isCompleted,
           "Check if work has completed (non-blocking)")
      .def(
          "wait",
          [](torch_neuronx::distributed::NeuronWork& self, int64_t timeout_ms) {
            if (timeout_ms < 0) {
              return self.wait();
            }
            return self.wait(std::chrono::milliseconds(timeout_ms));
          },
          "Block until work completes", py::arg("timeout_ms") = -1)
      .def("synchronize", &torch_neuronx::distributed::NeuronWork::synchronize,
           "Make current stream wait for this work")
      .def(
          "get_future",
          [](torch_neuronx::distributed::NeuronWork& self)
              -> std::shared_ptr<torch::jit::PythonFutureWrapper> {
            return std::make_shared<torch::jit::PythonFutureWrapper>(self.getFuture());
          },
          "Get future for async completion tracking")
      .def(
          "record_start_event",
          [](torch_neuronx::distributed::NeuronWork& self, py::object stream_obj) {
            at::neuron::NeuronStream* stream_ptr = nullptr;
            at::neuron::NeuronStream stream;
            if (!stream_obj.is_none() && THNPStream_Check(stream_obj.ptr())) {
              stream = THNPStream_Unpack(stream_obj.ptr());
              stream_ptr = &stream;
            }
            self.recordStartEvent(stream_ptr);
          },
          "Record start event for timing", py::arg("stream") = py::none())
      .def(
          "record_end_event",
          [](torch_neuronx::distributed::NeuronWork& self, py::object stream_obj) {
            at::neuron::NeuronStream* stream_ptr = nullptr;
            at::neuron::NeuronStream stream;
            if (!stream_obj.is_none() && THNPStream_Check(stream_obj.ptr())) {
              stream = THNPStream_Unpack(stream_obj.ptr());
              stream_ptr = &stream;
            }
            self.recordEndEvent(stream_ptr);
          },
          "Record end event for completion tracking", py::arg("stream") = py::none())
      .def("stash", &torch_neuronx::distributed::NeuronWork::stash,
           "Stash tensors to keep them alive until work completes", py::arg("tensors"))
      .def("unstash_tensors", &torch_neuronx::distributed::NeuronWork::unstashTensors,
           "Release stashed tensor references")
      .def("register_with_tensors", &torch_neuronx::distributed::NeuronWork::registerWithTensors,
           "Register work with output tensors for wait_tensor() support")
      .def("get_sequence_number", &torch_neuronx::distributed::NeuronWork::getSequenceNumber,
           "Get operation sequence number")
      .def("get_op_type", &torch_neuronx::distributed::NeuronWork::getOpType, "Get operation type")
      .def(
          "get_device",
          [](const torch_neuronx::distributed::NeuronWork& self) {
            return self.getDevice().index();
          },
          "Get device index")
      .def(
          "check_timeout",
          [](torch_neuronx::distributed::NeuronWork& self, std::optional<int64_t> timeout_ms) {
            std::optional<std::chrono::milliseconds> timeout;
            if (timeout_ms.has_value()) {
              timeout = std::chrono::milliseconds(timeout_ms.value());
            }
            return self.checkTimeout(timeout);
          },
          "Check if work has timed out", py::arg("timeout_ms") = py::none())
      .def("result", &torch_neuronx::distributed::NeuronWork::result, "Get output tensors")
      .def("get_duration", &torch_neuronx::distributed::NeuronWork::getDuration,
           "Get operation duration (requires timing enabled)")
      .def_property_readonly(
          "_end_event",
          [](const torch_neuronx::distributed::NeuronWork& self) {
            // Use THNPEvent_Wrap as in Stream.cpp line 162
            at::neuron::NeuronEvent event = self.getEndEvent();
            return py::reinterpret_steal<py::object>(THNPEvent_Wrap(std::move(event)));
          },
          "Get end event for external synchronization")
      .def_property_readonly(
          "_stream",
          [](const torch_neuronx::distributed::NeuronWork& self) {
            // Use THNPStream_Wrap as in Stream.cpp line 473
            at::neuron::NeuronStream stream = self.getStream();
            return py::reinterpret_steal<py::object>(THNPStream_Wrap(stream));
          },
          "Get stream this work runs on");

  // NeuronWatchdog class for monitoring async operations
  py::class_<torch_neuronx::distributed::NeuronWatchdog,
             std::unique_ptr<torch_neuronx::distributed::NeuronWatchdog>>(m, "NeuronWatchdog")
      .def(py::init<>())
      .def("start", &torch_neuronx::distributed::NeuronWatchdog::start, "Start watchdog thread")
      .def("stop", &torch_neuronx::distributed::NeuronWatchdog::stop, "Stop watchdog thread")
      .def("notify", &torch_neuronx::distributed::NeuronWatchdog::notify,
           "Notify watchdog of new work")
      .def("enqueue_work", &torch_neuronx::distributed::NeuronWatchdog::enqueueWork,
           "Add work to monitoring queue", py::arg("work"));
}
