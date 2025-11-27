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
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/utils/CopyUtils.h"

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
  if (!enabled_) {
    std::string error_msg = "CPU fallback is disabled for operation=" + op->GetOpName();
    return OperationContextResult::CreateError(error_msg);
  }

  const auto start_time = std::chrono::steady_clock::now();
  IncrementStatsCounter(&Stats::total_fallback_attempts);

  TORCH_NEURONX_DEBUG("Starting CPU fallback", "operation=", op->GetOpName());

  try {
    ExecuteCpuFallbackImpl(op);
    const auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);
    TORCH_NEURONX_DEBUG("CPU fallback completed", "operation=", op->GetOpName(),
                        "time_us=", total_time.count());
    IncrementStatsCounter(&Stats::successful_fallbacks);
    return OperationContextResult::CreateSuccess();
  } catch (const std::exception& e) {
    IncrementStatsCounter(&Stats::failed_fallbacks);
    std::string error_msg =
        "CPU fallback failed for '" + op->GetOpName() + "' failed: " + std::string(e.what());
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
    const std::vector<torch::Tensor>& neuron_tensors,
    const std::vector<TensorContext>& tensor_contexts) {
  // Validate input sizes match
  TORCH_CHECK(neuron_tensors.size() == tensor_contexts.size(),
              "Mismatch between neuron_tensors size (", neuron_tensors.size(),
              ") and tensor_contexts size (", tensor_contexts.size(), ")");

  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensor_contexts.size());

  for (size_t i = 0; i < tensor_contexts.size(); ++i) {
    try {
      const auto& context = tensor_contexts[i];
      const auto& neuron_tensor = neuron_tensors[i];

      // Validate tensor is on Neuron device
      TORCH_CHECK(!context.get_device().is_cpu(), "Tensor at index ", i,
                  " is already on CPU, expected Neuron device");

      // Create CPU tensor with matching properties
      auto cpu_tensor = CreateCpuTensorLike(context);
      TORCH_CHECK(cpu_tensor.is_contiguous(), "Created CPU tensor at index ", i,
                  " is not contiguous");

      // Find the NRT tensor for the Neuron data
      void* src_data_ptr = neuron_tensor.data_ptr();
      nrt_tensor_t* nrt_tensor = c10_neuron::NeuronCachingAllocator::findTensor(src_data_ptr);
      TORCH_CHECK(nrt_tensor != nullptr, "Failed to find NRT tensor for data pointer at index ", i);

      // Calculate byte offset for storage
      const size_t byte_offset = context.get_storage_offset() * context.get_element_size();
      const size_t size_bytes = context.get_size_bytes();

      // Perform the actual copy from Neuron to CPU
      torch_neuronx::utils::nrt_copy_neuron_to_cpu(nrt_tensor, cpu_tensor.data_ptr(), byte_offset,
                                                   size_bytes);

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

CPUFallbackExecutor::Stats CPUFallbackExecutor::GetStatistics() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

void CPUFallbackExecutor::ResetStatistics() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stats_ = Stats{};
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

void CPUFallbackExecutor::ExecuteCpuFallbackImpl(OperationContext* op) {
  const auto& input_contexts = op->input_tensors_context;
  const auto cpu_inputs = TransferTensorsToCpu(op->kernel_execution->GetInputs(), input_contexts);

  // Create CPU output tensors using TensorContext
  const auto& output_contexts = op->output_tensors_context;
  std::vector<torch::Tensor> cpu_outputs;
  cpu_outputs.reserve(output_contexts.size());
  for (const auto& context : output_contexts) {
    cpu_outputs.emplace_back(CreateCpuTensorLike(context));
  }

  // Execute on CPU with native PyTorch out parameter handling
  const auto cpu_results = ExecuteOperationOnCpu(op->GetOpName(), cpu_inputs, cpu_outputs, op);
  TransferTensorsFromCpuToNeuron(cpu_results, op);

  TORCH_NEURONX_DEBUG("CPU fallback execution completed", "operation=", op->GetOpName());
}

void CPUFallbackExecutor::IncrementStatsCounter(uint64_t Stats::* counter) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  (stats_.*counter)++;
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

  size_t input_idx = 0;
  size_t output_idx = 0;
  size_t static_idx = 0;
  size_t list_idx = 0;

  // Process each argument in schema order
  for (size_t arg_pos = 0; arg_pos < schema.arguments().size(); ++arg_pos) {
    const auto& arg = schema.arguments()[arg_pos];
    const std::string& arg_name = arg.name();

    // Handle static arguments (constants provided at operation definition time)
    const auto& fallback_ctx = op ? op->cpu_fallback_context : CPUFallbackContext();
    if (op && static_idx < fallback_ctx.static_arg_positions.size() &&
        fallback_ctx.static_arg_positions[static_idx] == static_cast<int64_t>(arg_pos)) {
      const auto& static_arg = fallback_ctx.static_args[static_idx];

      if (static_arg.isTensor()) {
        // Static tensor arguments are pre-transferred to CPU and included in cpu_inputs
        TORCH_CHECK(input_idx < cpu_inputs.size(), "Missing CPU input for static tensor argument '",
                    arg_name, "' at position ", arg_pos);
        stack.emplace_back(cpu_inputs[input_idx++]);
      } else {
        // Non-tensor static arguments can be used directly
        stack.emplace_back(static_arg);
      }
      static_idx++;
      continue;
    }

    // Handle output arguments (for in-place operations)
    if (arg.is_out()) {
      TORCH_CHECK(output_idx < cpu_outputs.size(), "Missing output argument '", arg_name,
                  "' at position ", arg_pos);
      stack.emplace_back(cpu_outputs[output_idx++]);
      continue;
    }

    // Handle list arguments (tensor lists that were flattened)
    if (op && list_idx < fallback_ctx.list_arg_positions.size() &&
        fallback_ctx.list_arg_positions[list_idx] == static_cast<int64_t>(arg_pos)) {
      const int64_t list_len = fallback_ctx.list_arg_lengths[list_idx];
      c10::List<torch::Tensor> tensor_list;

      for (int64_t i = 0; i < list_len; ++i) {
        TORCH_CHECK(input_idx < cpu_inputs.size(),
                    "Not enough inputs to reconstruct tensor list at position ", arg_pos, " (need ",
                    list_len, " tensors)");
        tensor_list.push_back(cpu_inputs[input_idx++]);
      }

      stack.emplace_back(std::move(tensor_list));
      list_idx++;
      continue;
    }

    // Handle regular input arguments
    if (input_idx < cpu_inputs.size()) {
      stack.emplace_back(cpu_inputs[input_idx++]);
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
    const std::vector<torch::Tensor>& cpu_results, OperationContext* op) {
  const auto& output_contexts = op->output_tensors_context;
  const auto& original_neuron_outputs = op->kernel_execution->GetOutputs();

  // Validate sizes match
  TORCH_CHECK(output_contexts.size() == cpu_results.size(),
              "Output tensor count mismatch for operation '", op->GetOpName(), "': expected ",
              output_contexts.size(), " but got ", cpu_results.size());

  TORCH_CHECK(original_neuron_outputs.size() == cpu_results.size(),
              "Original Neuron output count mismatch for operation '", op->GetOpName(),
              "': expected ", original_neuron_outputs.size(), " but got ", cpu_results.size());

  // Transfer each CPU result back to its corresponding Neuron output tensor
  for (size_t i = 0; i < output_contexts.size(); ++i) {
    const auto& output_context = output_contexts[i];
    const auto& cpu_result = cpu_results[i];
    const auto& neuron_output = original_neuron_outputs[i];

    try {
      // Validate CPU result is contiguous for efficient copy
      TORCH_CHECK(cpu_result.is_contiguous(), "CPU result tensor at index ", i,
                  " must be contiguous for copy to Neuron device");

      // Find the NRT tensor for the destination
      void* dst_ptr = neuron_output.data_ptr();
      nrt_tensor_t* nrt_tensor = c10_neuron::NeuronCachingAllocator::findTensor(dst_ptr);
      TORCH_CHECK(nrt_tensor != nullptr, "Failed to find NRT tensor for output at index ", i);

      // Calculate byte offset for storage
      const size_t byte_offset =
          output_context.get_storage_offset() * output_context.get_element_size();
      const size_t size_bytes = output_context.get_size_bytes();

      // Perform the actual copy from CPU to Neuron
      torch_neuronx::utils::nrt_copy_cpu_to_neuron(cpu_result.data_ptr(), nrt_tensor, byte_offset,
                                                   size_bytes);

      TORCH_NEURONX_DEBUG("Copied CPU result to Neuron output tensor",
                          "operation=", op->GetOpName(), "index=", i, "shape=", cpu_result.sizes(),
                          "size_bytes=", size_bytes);

    } catch (const std::exception& e) {
      TORCH_NEURONX_ERROR("Failed to copy CPU result to Neuron output tensor",
                          "operation=", op->GetOpName(), "index=", i, "error=", e.what());
      throw std::runtime_error("Neuron output tensor copy failed for index " + std::to_string(i) +
                               ": " + std::string(e.what()));
    }
  }
}

}  // namespace at::neuron
