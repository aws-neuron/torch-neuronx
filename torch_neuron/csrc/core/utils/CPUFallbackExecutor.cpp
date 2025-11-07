#include "CPUFallbackExecutor.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/TensorImpl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <sstream>

#include "torch_neuron/csrc/core/NeuronLogging.h"
#include "torch_neuron/csrc/core/NeuronOpTracking.h"
#include "torch_neuron/csrc/utils/CopyUtils.h"

namespace at::neuron {

CPUFallbackExecutor::CPUFallbackExecutor() : enabled_(true) {
  LoadConfiguration();

  TORCH_NEURON_DEBUG("CPUFallbackExecutor initialized", "enabled=", enabled_);
}

OperationContextResult CPUFallbackExecutor::ExecuteCpuFallback(OperationContext* op) {
  if (!enabled_) {
    std::string error_msg = "CPU fallback is disabled for operation=" + op->GetOpName();
    return OperationContextResult::CreateError(error_msg);
  }

  const auto start_time = std::chrono::steady_clock::now();
  IncrementStatsCounter(&Stats::total_fallback_attempts);

  TORCH_NEURON_DEBUG("Starting CPU fallback", "operation=", op->GetOpName());

  try {
    ExecuteCpuFallbackImpl(op);
    const auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time);
    TORCH_NEURON_DEBUG("CPU fallback completed", "operation=", op->GetOpName(),
                       "time_us=", total_time.count());
    IncrementStatsCounter(&Stats::successful_fallbacks);
    return OperationContextResult::CreateSuccess();
  } catch (const std::exception& e) {
    IncrementStatsCounter(&Stats::failed_fallbacks);
    std::string error_msg =
        "CPU fallback failed for '" + op->GetOpName() + "' failed: " + std::string(e.what());
    TORCH_NEURON_ERROR(error_msg);
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
    const std::vector<torch::Tensor>& neuron_tensors) {
  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(neuron_tensors.size());

  for (const auto& neuron_tensor : neuron_tensors) {
    try {
      auto cpu_tensor =
          torch::empty(neuron_tensor.sizes(), neuron_tensor.options().device(torch::kCPU));
      torch_neuron::utils::copy_neuron_to_cpu(neuron_tensor, cpu_tensor, /*non_blocking=*/true);
      cpu_tensors.emplace_back(std::move(cpu_tensor));
    } catch (const std::exception& e) {
      TORCH_NEURON_ERROR("Failed to transfer tensor to CPU", "error=", e.what());
      throw std::runtime_error("CPU transfer failed for tensor: " + std::string(e.what()));
    }
  }

  TORCH_NEURON_DEBUG("Transferred tensors to CPU", "count=", cpu_tensors.size());
  return cpu_tensors;
}

std::vector<torch::Tensor> CPUFallbackExecutor::ExecuteOperationOnCpu(
    const std::string& op_name, const std::vector<torch::Tensor>& cpu_inputs,
    const std::vector<torch::Tensor>& cpu_outputs) {
  const auto cpu_op = GetCpuOperator(op_name);
  if (!cpu_op.has_value()) {
    throw std::runtime_error("No CPU implementation found for operation: " + op_name);
  }

  // Use PyTorch's schema to understand the operation signature
  const auto& schema = cpu_op->schema();
  TORCH_NEURON_DEBUG("The schema for cpu_op -", "operation=", op_name, "schema=", schema);
  const bool has_out_params = SchemaHasOutParameters(schema);

  // Construct stack according to PyTorch's expected argument order
  std::vector<c10::IValue> stack =
      ConstructNativeStack(cpu_inputs, cpu_outputs, schema, has_out_params);

  TORCH_NEURON_DEBUG("Calling CPU fallback", "operation=", op_name, "stack_size=", stack.size(),
                     "has_out_params=", has_out_params, "schema_args=", schema.arguments().size());

  if (stack.size() != schema.arguments().size()) {
    throw std::runtime_error("Stack size mismatch for operation '" + op_name + "': expected " +
                             std::to_string(schema.arguments().size()) + " arguments, got " +
                             std::to_string(stack.size()));
  }

  try {
    at::native::cpu_fallback(*cpu_op, &stack);
    // log ops which fell back to CPU
    if (torch_neuron::shouldLogFallback(op_name)) {
      torch_neuron::NeuronLogger::getInstance().log(torch_neuron::LogLevel::WARNING,
                                                    torch_neuron::LogCategory::OPERATOR_FALLBACK,
                                                    "Operator '" + op_name + "' fell back to CPU");
      torch_neuron::markFallbackLogged(op_name);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("CPU fallback execution failed for '" + op_name + "': " + e.what());
  }

  // Return results based on whether output parameters were used
  if (has_out_params) {
    // Output tensors were modified in-place
    return cpu_outputs;
  } else {
    // Extract new results from the stack
    return ExtractResultsFromStack(stack, cpu_outputs, op_name);
  }
}

c10::optional<c10::OperatorHandle> CPUFallbackExecutor::GetCpuOperator(
    const std::string& op_name) const {
  try {
    // Parse the operation name to get the operator
    std::string aten_name = op_name;
    if (aten_name.find("aten::") != 0) {
      // If it doesn't start with "aten::", try to find the full name
      aten_name = "aten::" + op_name;
    }

    // Try to find the operator with different overload names
    std::vector<std::string> overload_attempts = {"", "out", "Tensor", "Scalar", "_"};

    for (const auto& overload : overload_attempts) {
      try {
        auto op_handle =
            c10::Dispatcher::singleton().findOp(c10::OperatorName(aten_name, overload));
        if (op_handle.has_value()) {
          TORCH_NEURON_DEBUG("Found CPU operator", "operation=", op_name, "aten_name=", aten_name,
                             "overload=", overload);
          return op_handle;
        }
      } catch (...) {
      }
    }

    TORCH_NEURON_DEBUG("Could not find CPU operator after trying all overloads",
                       "operation=", op_name, "aten_name=", aten_name);
    return c10::nullopt;

  } catch (const std::exception& e) {
    TORCH_NEURON_DEBUG("Could not find CPU operator", "operation=", op_name, "error=", e.what());
    return c10::nullopt;
  }
}

torch::Tensor CPUFallbackExecutor::CreateCpuTensorLike(const torch::Tensor& neuron_tensor) {
  // Create a CPU tensor with the same properties as the Neuron tensor
  auto options = torch::TensorOptions()
                     .dtype(neuron_tensor.scalar_type())
                     .device(torch::kCPU)
                     .requires_grad(neuron_tensor.requires_grad());

  return torch::empty(neuron_tensor.sizes(), options);
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
      std::getenv("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS");
  if (disable_cpu_fallback_recovery) {
    enabled_ = !(std::string(disable_cpu_fallback_recovery) == "1");
  }

  TORCH_NEURON_DEBUG("CPUFallbackExecutor configuration loaded", "enabled=", enabled_);
}

void CPUFallbackExecutor::ExecuteCpuFallbackImpl(OperationContext* op) {
  const auto cpu_inputs = TransferTensorsToCpu(op->kernel_execution->GetInputs());

  // Create CPU output tensors
  const auto& output_tensors = op->kernel_execution->GetOutputs();
  std::vector<torch::Tensor> cpu_outputs;
  cpu_outputs.reserve(output_tensors.size());
  for (const auto& tensor : output_tensors) {
    cpu_outputs.emplace_back(CreateCpuTensorLike(tensor));
  }

  // Execute on CPU with native PyTorch out parameter handling
  const auto cpu_results = ExecuteOperationOnCpu(op->GetOpName(), cpu_inputs, cpu_outputs);
  HandleNeuronOutputTensorReuse(cpu_results, op);
}

void CPUFallbackExecutor::IncrementStatsCounter(uint64_t Stats::* counter) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  (stats_.*counter)++;
}

std::vector<torch::Tensor> CPUFallbackExecutor::ExtractResultsFromStack(
    const std::vector<c10::IValue>& stack, const std::vector<torch::Tensor>& expected_outputs,
    const std::string& op_name) {
  if (stack.empty()) {
    throw std::runtime_error("CPU operation '" + op_name + "' returned no results");
  }

  std::vector<torch::Tensor> results;
  results.reserve(std::min(stack.size(), expected_outputs.size()));

  if (stack.size() == 1 && stack[0].isTensor()) {
    // Single tensor result
    results.emplace_back(stack[0].toTensor());
  } else {
    // Multiple results
    auto output_it = expected_outputs.begin();
    for (size_t i = 0; i < stack.size() && output_it != expected_outputs.end(); ++i, ++output_it) {
      if (stack[i].isTensor()) {
        results.emplace_back(stack[i].toTensor());
      }
    }
  }

  TORCH_NEURON_DEBUG("Extracted CPU results", "operation=", op_name, "count=", results.size());
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
    const c10::FunctionSchema& schema, bool has_out_params) const {
  std::vector<c10::IValue> stack;
  stack.reserve(schema.arguments().size());

  size_t input_idx = 0;
  size_t output_idx = 0;

  // Add arguments in schema order
  for (const auto& arg : schema.arguments()) {
    const std::string& arg_name = arg.name();
    TORCH_NEURON_DEBUG("Processing schema argument", "name=", arg_name, "is_out=", arg.is_out());

    if (arg.is_out()) {
      if (output_idx < cpu_outputs.size()) {
        stack.emplace_back(cpu_outputs[output_idx++]);
      } else {
        throw std::runtime_error("Missing output argument: " + arg_name);
      }
    } else {
      if (input_idx < cpu_inputs.size()) {
        stack.emplace_back(cpu_inputs[input_idx++]);
      } else {
        if (arg.default_value().has_value()) {
          stack.emplace_back(arg.default_value().value());
        } else {
          throw std::runtime_error("Missing required input argument: " + arg_name);
        }
      }
    }
  }

  TORCH_NEURON_DEBUG("Constructed stack", "size=", stack.size(),
                     "expected=", schema.arguments().size());
  return stack;
}

void CPUFallbackExecutor::HandleNeuronOutputTensorReuse(
    const std::vector<torch::Tensor>& cpu_results, OperationContext* op) {
  // Get the original Neuron output tensors that we want to reuse
  const auto& original_neuron_outputs = op->kernel_execution->GetOutputs();

  TORCH_CHECK(original_neuron_outputs.size() == cpu_results.size(),
              "Mismatch output tensor sizes (", original_neuron_outputs.size(), ") and (",
              cpu_results.size(), ") for operation ", op->GetOpName());

  for (size_t i = 0; i < original_neuron_outputs.size(); ++i) {
    const auto& original_neuron_tensor = original_neuron_outputs[i];
    const auto& cpu_result = cpu_results[i];

    try {
      // Get mutable reference to original Neuron tensor
      torch::Tensor& mutable_original = const_cast<torch::Tensor&>(original_neuron_tensor);

      // Resize original Neuron tensor if needed
      if (mutable_original.sizes() != cpu_result.sizes()) {
        TORCH_NEURON_DEBUG("Resizing original Neuron output tensor", "operation=", op->GetOpName(),
                           "index=", i, "current_shape=", mutable_original.sizes(),
                           "result_shape=", cpu_result.sizes());

        mutable_original.resize_(cpu_result.sizes());
      }

      torch_neuron::utils::copy_cpu_to_neuron(cpu_result, mutable_original, /*non_blocking=*/true);

      TORCH_NEURON_DEBUG("Reused original Neuron output tensor", "operation=", op->GetOpName(),
                         "index=", i, "final_shape=", mutable_original.sizes());

    } catch (const std::exception& e) {
      TORCH_NEURON_ERROR("Failed to reuse original Neuron output tensor",
                         "operation=", op->GetOpName(), "index=", i, "error=", e.what());
      throw std::runtime_error("Neuron output tensor reuse failed for index " + std::to_string(i) +
                               ": " + std::string(e.what()));
    }
  }
}

}  // namespace at::neuron
