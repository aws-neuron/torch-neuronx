#include "CPUFallbackExecutor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/TensorImpl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <sstream>

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/NeuronStorageImpl.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

// Include Neuron Runtime headers
extern "C" {
#include <nrt/nrt.h>
}

namespace at::neuron {

namespace {
// Common overload names to try when looking up operators
constexpr std::array<const char*, 5> kCommonOverloads = {"", "out", "Tensor", "Scalar", "_"};
}  // namespace

CPUFallbackExecutor::CPUFallbackExecutor() : enabled_(true) {
  LoadConfiguration();

  TORCH_NEURONX_DEBUG("CPUFallbackExecutor initialized", "enabled=", enabled_);
}

OperationContextResult CPUFallbackExecutor::ExecuteCpuFallback(OperationContext* op) {
  std::string operation_name = op->GetOpName();
  if (!enabled_) {
    std::string error_msg = "CPU fallback is disabled for operation=" + operation_name;
    return OperationContextResult::CreateError(error_msg);
  }

  const auto start_time = std::chrono::steady_clock::now();

  TORCH_NEURONX_DEBUG("Starting CPU fallback", "operation=", operation_name);

  try {
    ExecuteCpuFallbackImpl(op);
    const auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);
    TORCH_NEURONX_DEBUG("CPU fallback completed", "operation=", operation_name,
                        "time_us=", total_time.count());
    return OperationContextResult::CreateSuccess();
  } catch (const std::exception& e) {
    std::string error_msg =
        "CPU fallback failed for '" + operation_name + "' failed: " + std::string(e.what());
    TORCH_NEURONX_ERROR(error_msg);
    return OperationContextResult::CreateError(error_msg);
  }
}

bool CPUFallbackExecutor::CanExecuteOnCpu(const std::string& op_name) const {
  if (!enabled_) {
    return false;
  }

  // Try to get the CPU operator to see if it exists
  auto cpu_op = GetCpuOperator(op_name);
  return cpu_op.has_value();
}

std::vector<torch::Tensor> CPUFallbackExecutor::TransferTensorsToCpu(
    const std::vector<nrt_tensor_t*>& data_ptrs,
    const std::vector<TensorContext>& tensor_contexts) {
  // Validate input sizes match
  TORCH_CHECK(data_ptrs.size() == tensor_contexts.size(), "Mismatch between data_ptrs size (",
              data_ptrs.size(), ") and tensor_contexts size (", tensor_contexts.size(), ")");

  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensor_contexts.size());

  for (size_t i = 0; i < tensor_contexts.size(); ++i) {
    try {
      const auto& context = tensor_contexts[i];
      auto cpu_tensor = CreateCpuTensorLike(context);
      TORCH_CHECK(cpu_tensor.is_contiguous(), "Destination CPU tensor must be contiguous");
      nrt_tensor_t* data_ptr = data_ptrs[i];
      size_t offset_bytes = context.storage_offset * context.element_size;
      torch_neuronx::utils::nrt_copy_neuron_to_cpu(data_ptr, cpu_tensor.data_ptr(), offset_bytes,
                                                   context.get_size_bytes());
      cpu_tensors.emplace_back(std::move(cpu_tensor));
    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("Failed to transfer tensor to CPU", "index=", i, "error=", e.what());
      throw std::runtime_error("CPU transfer failed for tensor at index " + std::to_string(i) +
                               ": " + std::string(e.what()));
    }
  }

  TORCH_NEURONX_DEBUG("Transferred tensors to CPU", "count=", cpu_tensors.size());
  return cpu_tensors;
}

std::vector<torch::Tensor> CPUFallbackExecutor::ExecuteOperationOnCpu(
    const std::string& op_name, const std::vector<torch::Tensor>& cpu_inputs,
    const std::vector<torch::Tensor>& cpu_outputs, const OperationContext* op) {
  // Look up the CPU operator implementation
  const auto cpu_op = GetCpuOperator(op_name);
  TORCH_CHECK(cpu_op.has_value(), "No CPU implementation found for operation: ", op_name);

  // Analyze the operation's schema to understand its signature
  const auto& schema = cpu_op->schema();
  TORCH_NEURONX_DEBUG("The schema for cpu_op -", "operation=", op_name, "schema=", schema);
  const bool has_out_params = SchemaHasOutParameters(schema);

  TORCH_NEURONX_DEBUG("Executing CPU operation", "operation=", op_name, "schema=", schema,
                      "has_out_params=", has_out_params);

  // Construct the argument stack in the order expected by PyTorch
  std::vector<c10::IValue> stack =
      ConstructNativeStack(cpu_inputs, cpu_outputs, schema, has_out_params, op);

  // Validate stack size matches schema expectations
  TORCH_CHECK(stack.size() == schema.arguments().size(), "Stack size mismatch for operation '",
              op_name, "': expected ", schema.arguments().size(), " arguments, got ", stack.size());

  TORCH_NEURONX_DEBUG("Calling CPU operator", "operation=", op_name, "stack_size=", stack.size());

  try {
    // Execute the operation on CPU using PyTorch's dispatcher
    // We call directly instead of using at::native::cpu_fallback to have better
    // control over in-place operations and error handling
    cpu_op->callBoxed(&stack);

    // Log fallback operations for monitoring (once per operation type)
    if (torch_neuronx::shouldLogFallback(op_name)) {
      torch_neuronx::NeuronLogger::getInstance().log(
          torch_neuronx::LogLevel::WARNING, torch_neuronx::LogCategory::OPERATOR_FALLBACK,
          "Operator '" + op_name + "' executed via CPU fallback");
      torch_neuronx::markFallbackLogged(op_name);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("CPU fallback execution failed for '" + op_name + "': " + e.what());
  }

  // Return results based on operation type
  if (has_out_params) {
    // For operations with output parameters, results are written in-place to cpu_outputs
    TORCH_NEURONX_DEBUG("Returning in-place modified outputs", "operation=", op_name,
                        "count=", cpu_outputs.size());
    return cpu_outputs;
  } else {
    // For operations without output parameters, extract new results from the stack
    return ExtractResultsFromStack(stack, cpu_outputs, op_name);
  }
}

c10::optional<c10::OperatorHandle> CPUFallbackExecutor::GetCpuOperator(
    const std::string& op_name) const {
  try {
    // Ensure operation name has "aten::" prefix
    std::string aten_name = op_name;
    if (aten_name.find("aten::") != 0) {
      // If it doesn't start with "aten::", try to find the full name
      aten_name = "aten::" + op_name;
    }

    // Split operator name and overload (e.g., "aten::fill_.Scalar" -> "aten::fill_", "Scalar")
    const size_t dot_pos = aten_name.find_last_of('.');
    std::string base_name = aten_name;
    std::string existing_overload;

    // Extract existing overload if present (must be after "aten::" prefix)
    constexpr size_t aten_prefix_len = 6;  // Length of "aten::"
    if (dot_pos != std::string::npos && dot_pos > aten_prefix_len) {
      base_name = aten_name.substr(0, dot_pos);
      existing_overload = aten_name.substr(dot_pos + 1);
    }

    // Build list of overloads to try, prioritizing existing overload
    std::vector<std::string> overload_attempts;
    overload_attempts.reserve(kCommonOverloads.size() + 1);

    if (!existing_overload.empty()) {
      overload_attempts.push_back(std::move(existing_overload));
    }

    // Add common overloads
    overload_attempts.insert(overload_attempts.end(), kCommonOverloads.begin(),
                             kCommonOverloads.end());

    // Try each overload until we find a match
    for (const auto& overload : overload_attempts) {
      try {
        auto op_handle =
            c10::Dispatcher::singleton().findOp(c10::OperatorName(base_name, overload));
        if (op_handle.has_value()) {
          TORCH_NEURONX_DEBUG("Found CPU operator", "operation=", op_name, "base_name=", base_name,
                              "overload=", overload);
          return op_handle;
        }
      } catch (...) {
        // Silently continue to next overload attempt
      }
    }

    TORCH_NEURONX_DEBUG("Could not find CPU operator after trying all overloads",
                        "operation=", op_name, "base_name=", base_name);
    return c10::nullopt;

  } catch (const std::exception& e) {
    TORCH_NEURONX_DEBUG("Exception while finding CPU operator", "operation=", op_name,
                        "error=", e.what());
    return c10::nullopt;
  }
}

torch::Tensor CPUFallbackExecutor::CreateCpuTensorLike(const TensorContext& tensor_context) {
  // Create a CPU tensor with the same properties from TensorContext
  auto options = tensor_context.get_options().device(torch::kCPU);
  return torch::empty(tensor_context.get_shape(), options);
}

void CPUFallbackExecutor::LoadConfiguration() {
  // Load CPU fallback configuration from environment variables
  const char* disable_cpu_fallback_recovery =
      std::getenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  if (disable_cpu_fallback_recovery) {
    enabled_ = !(std::string(disable_cpu_fallback_recovery) == "1");
  }

  TORCH_NEURONX_DEBUG("CPUFallbackExecutor configuration loaded", "enabled=", enabled_);
}

void CPUFallbackExecutor::ExecuteCpuFallbackImpl(OperationContext* operation) {
  const auto& fallback_ctx = operation->cpu_fallback_context;

  // Convert void* to nrt_tensor_t* using allocator
  TORCH_NEURONX_DEBUG("size of tensors_data_ptrs in fallback context - ",
                      fallback_ctx.tensor_data_ptrs.size());
  std::vector<nrt_tensor_t*> src_ptrs;
  src_ptrs.reserve(fallback_ctx.tensor_data_ptrs.size());
  for (void* ptr : fallback_ctx.tensor_data_ptrs) {
    nrt_tensor_t* nrt_ptr = c10_neuron::NeuronCachingAllocator::findTensor(ptr);
    TORCH_CHECK(nrt_ptr, "Invalid input data pointer for CPU fallback");
    src_ptrs.push_back(nrt_ptr);
  }

  auto& hlo_kernel = operation->GetKernel<KernelTypeEnum::kHLO>();
  std::vector<nrt_tensor_t*> dst_ptrs = hlo_kernel.GetDstDataPtrs();

  const auto& input_contexts = fallback_ctx.tensor_metadata;
  const auto cpu_inputs = TransferTensorsToCpu(src_ptrs, input_contexts);

  // Create CPU output tensors using TensorContext from kernel
  const auto output_contexts = hlo_kernel.GetOutputContexts();
  std::vector<torch::Tensor> cpu_outputs;
  cpu_outputs.reserve(output_contexts.size());
  for (const auto& context : output_contexts) {
    cpu_outputs.emplace_back(CreateCpuTensorLike(context));
  }

  // Execute on CPU with native PyTorch out parameter handling
  const auto cpu_results =
      ExecuteOperationOnCpu(operation->GetOpName(), cpu_inputs, cpu_outputs, operation);
  // Transfer results back using data pointers and contexts
  TransferTensorsFromCpuToNeuron(cpu_results, dst_ptrs, output_contexts);
  TORCH_NEURONX_DEBUG("CPU fallback execution completed", "operation=", operation->GetOpName());
}

std::vector<torch::Tensor> CPUFallbackExecutor::ExtractResultsFromStack(
    const std::vector<c10::IValue>& stack, const std::vector<torch::Tensor>& expected_outputs,
    const std::string& op_name) {
  TORCH_CHECK(!stack.empty(), "CPU operation '", op_name, "' returned empty stack");

  std::vector<torch::Tensor> results;
  const size_t max_results = std::min(stack.size(), expected_outputs.size());
  results.reserve(max_results);

  // Handle single tensor result (common case)
  if (stack.size() == 1 && stack[0].isTensor()) {
    results.emplace_back(stack[0].toTensor());
    TORCH_NEURONX_DEBUG("Extracted single tensor result", "operation=", op_name);
    return results;
  }

  // Handle multiple results - extract only tensor values
  for (size_t i = 0; i < stack.size() && results.size() < expected_outputs.size(); ++i) {
    if (stack[i].isTensor()) {
      results.emplace_back(stack[i].toTensor());
    }
  }

  TORCH_NEURONX_DEBUG("Extracted CPU results", "operation=", op_name, "count=", results.size(),
                      "expected=", expected_outputs.size());
  return results;
}

bool CPUFallbackExecutor::OperationHasOutParameters(const std::string& op_name) const {
  try {
    const auto cpu_op = GetCpuOperator(op_name);
    if (cpu_op.has_value()) {
      const auto& schema = cpu_op->schema();
      return SchemaHasOutParameters(schema);
    }
  } catch (...) {
    // Schema inspection failed
  }
  return false;
}

bool CPUFallbackExecutor::SchemaHasOutParameters(const c10::FunctionSchema& schema) const {
  for (const auto& arg : schema.arguments()) {
    if (arg.is_out()) {
      return true;
    }
  }
  return false;
}

std::vector<c10::IValue> CPUFallbackExecutor::ConstructNativeStack(
    const std::vector<torch::Tensor>& cpu_inputs, const std::vector<torch::Tensor>& cpu_outputs,
    const c10::FunctionSchema& schema, bool has_out_params, const OperationContext* op) const {
  std::vector<c10::IValue> stack;
  stack.reserve(schema.arguments().size());

  const auto& fallback_ctx = op ? op->cpu_fallback_context : CPUFallbackContext();
  size_t cpu_input_idx = 0;
  size_t output_idx = 0;

  // Process each argument in schema order
  for (size_t arg_pos = 0; arg_pos < schema.arguments().size(); ++arg_pos) {
    const auto& arg = schema.arguments()[arg_pos];
    const std::string& arg_name = arg.name();

    // Handle output arguments (for in-place operations)
    if (arg.is_out()) {
      TORCH_CHECK(output_idx < cpu_outputs.size(), "Missing output argument '", arg_name,
                  "' at position ", arg_pos);
      stack.emplace_back(cpu_outputs[output_idx++]);
      continue;
    }

    // Check if this argument is provided as a tensor kwarg
    if (op && fallback_ctx.IsKwargTensor(arg_name)) {
      // Tensor kwarg - get from cpu_inputs at the appropriate index
      size_t kwarg_tensor_idx = fallback_ctx.GetKwargTensorIndex(arg_name);
      TORCH_CHECK(kwarg_tensor_idx < cpu_inputs.size(), "Missing CPU input for tensor kwarg '",
                  arg_name, "'");
      stack.emplace_back(cpu_inputs[kwarg_tensor_idx]);
      continue;
    }

    // Check if this argument is provided as a scalar kwarg
    if (op && !fallback_ctx.original_kwargs.empty() &&
        fallback_ctx.original_kwargs.contains(arg_name)) {
      const auto& kwarg_value = fallback_ctx.original_kwargs.at(arg_name);
      stack.emplace_back(kwarg_value);
      continue;
    }

    // Get input from metadata at this position
    if (op && arg_pos < fallback_ctx.input_metadata.size()) {
      const auto& input_meta = fallback_ctx.input_metadata[arg_pos];
      const auto& arg_type = arg.type();

      if (IsTensor(input_meta)) {
        TORCH_CHECK(cpu_input_idx < cpu_inputs.size(), "Missing CPU input for tensor '", arg_name,
                    "' at position ", arg_pos);
        stack.emplace_back(cpu_inputs[cpu_input_idx++]);
      } else if (IsTensorList(input_meta)) {
        size_t list_count = GetListSize(input_meta);
        c10::List<at::Tensor> cpu_tensor_list;
        for (size_t i = 0; i < list_count; ++i) {
          TORCH_CHECK(cpu_input_idx < cpu_inputs.size(),
                      "Missing CPU input for tensor list element at position ", arg_pos, "[", i,
                      "]");
          cpu_tensor_list.push_back(cpu_inputs[cpu_input_idx++]);
        }
        stack.emplace_back(cpu_tensor_list);
      } else if (IsScalar(input_meta)) {
        const auto& scalar_val = GetScalarValue(input_meta);
        // Check if schema expects a Tensor but we have a scalar
        if (arg_type->kind() == c10::TypeKind::TensorType && scalar_val.isScalar()) {
          // Convert scalar to tensor
          stack.emplace_back(torch::scalar_tensor(scalar_val.toScalar(), torch::kCPU));
        } else {
          // Use the IValue directly (handles None, int, double, bool, lists, etc.)
          stack.emplace_back(scalar_val);
        }
      }
    } else if (arg.default_value().has_value()) {
      // Use default value if available
      stack.emplace_back(arg.default_value().value());
    } else if (arg.type()->kind() == c10::TypeKind::OptionalType) {
      // Optional arguments default to None
      stack.emplace_back(c10::IValue());
    } else {
      TORCH_CHECK(false, "Missing required input argument '", arg_name, "' at position ", arg_pos);
    }
  }

  TORCH_NEURONX_DEBUG("Constructed stack", "size=", stack.size(),
                      "expected=", schema.arguments().size());
  return stack;
}

void CPUFallbackExecutor::TransferTensorsFromCpuToNeuron(
    const std::vector<torch::Tensor>& cpu_results, const std::vector<nrt_tensor_t*>& data_ptrs,
    const std::vector<TensorContext>& tensor_contexts) {
  TORCH_CHECK(data_ptrs.size() == cpu_results.size(), "Mismatch output tensor sizes (",
              data_ptrs.size(), ") and (", cpu_results.size(), ")");

  for (size_t i = 0; i < cpu_results.size(); ++i) {
    const auto& cpu_result = cpu_results[i];
    nrt_tensor_t* data_ptr = data_ptrs[i];

    try {
      TORCH_CHECK(cpu_result.is_contiguous(), "CPU result tensor must be contiguous");
      size_t size_bytes = cpu_result.numel() * cpu_result.element_size();
      size_t offset_bytes = tensor_contexts[i].storage_offset * tensor_contexts[i].element_size;
      torch_neuronx::utils::nrt_copy_cpu_to_neuron(cpu_result.data_ptr(), data_ptr, offset_bytes,
                                                   size_bytes);

      TORCH_NEURONX_DEBUG("Copied CPU result to Neuron output tensor", "index=", i,
                          "final_shape=", cpu_result.sizes());

    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("Failed to copy CPU result to Neuron output tensor", "index=", i,
                          "error=", e.what());
      throw std::runtime_error("Neuron output tensor copy failed for index " + std::to_string(i) +
                               ": " + std::string(e.what()));
    }
  }
}

}  // namespace at::neuron
