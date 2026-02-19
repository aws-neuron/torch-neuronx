#include "IrNode.h"

#include <stdexcept>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/concatenation/OpConcatUtils.h"

namespace torch_neuronx {

namespace {

std::string moduleOpToString_(mlir::ModuleOp module) {
  TORCH_NEURONX_TRACE_FUNCTION();

  std::string result;
  llvm::raw_string_ostream stream(result);

  // Use OpPrintingFlags for clean, parseable output
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);     // Remove debug info for cleaner output
  flags.printGenericOpForm(false);  // Use specific op syntax

  module.print(stream, flags);
  TORCH_NEURONX_DEBUG("Successfully converted MLIR module to string representation",
                      "output_string_size:", result.size(), "debug_info_enabled:", false,
                      "generic_form:", false);
  return result;
}

}  // namespace

IrNode::IrNode(const std::string& op_name, const std::string& cache_key,
               const std::vector<uint8_t>& ir_serialized, const IrNodeType ir_type,
               std::vector<void*>&& inputs, std::vector<void*>&& outputs,
               const bool has_collectives)
    : op_name(op_name),
      cache_key(cache_key),
      ir_serialized(ir_serialized),
      ir_type(ir_type),
      inputs(std::move(inputs)),
      outputs(std::move(outputs)),
      status(IrNodeStatus::CREATED),
      has_collectives(has_collectives) {
  TORCH_NEURONX_DEBUG("Created IrNode with complete configuration", "op_name:", op_name,
                      "cache_key:", cache_key, "ir_type:", static_cast<int>(ir_type),
                      "inputs_count:", this->inputs.size(), "outputs_count:", this->outputs.size(),
                      "has_collectives:", has_collectives,
                      "ir_serialized_size:", ir_serialized.size(), "status:", "CREATED");
}

IrNode::IrNode(const std::string& op_name, const std::string&& cache_key,
               std::vector<uint8_t>& ir_serialized, const IrNodeType ir_type,
               std::vector<void*>&& inputs, std::vector<void*>&& outputs,
               const bool has_collectives)
    : op_name(op_name),
      cache_key(std::move(cache_key)),
      ir_serialized(ir_serialized),
      ir_type(ir_type),
      inputs(std::move(inputs)),
      outputs(std::move(outputs)),
      status(IrNodeStatus::CREATED),
      has_collectives(has_collectives) {
  TORCH_NEURONX_DEBUG("Created IrNode with complete configuration", "op_name:", op_name,
                      "cache_key:", cache_key, "ir_type:", static_cast<int>(ir_type),
                      "inputs_count:", this->inputs.size(), "outputs_count:", this->outputs.size(),
                      "has_collectives:", has_collectives,
                      "ir_serialized_size:", ir_serialized.size(), "status:", "CREATED");
}

StableHloNode::StableHloNode(const std::string& op_name, const std::string& cache_key,
                             std::vector<uint8_t>&& ir_serialized, std::vector<void*>&& inputs,
                             std::vector<void*>&& outputs, const bool has_collectives,
                             mlir::MLIRContext* context)
    : IrNode(op_name, cache_key, std::move(ir_serialized), IrNodeType::STABLEHLO, std::move(inputs),
             std::move(outputs), has_collectives) {
  TORCH_NEURONX_DEBUG("Creating StableHloNode with MLIR context and dialects", "op_name:", op_name,
                      "cache_key:", cache_key, "inputs_count:", this->inputs.size(),
                      "outputs_count:", this->outputs.size());

  std::string ir_string{this->ir_serialized.begin(), this->ir_serialized.end()};
  auto tmp_module = mlir::parseSourceString<mlir::ModuleOp>(ir_string, context);
  if (!tmp_module) {
    throw std::runtime_error(std::string("MLIR module parsing failed for StableHloNode op_name: ") +
                             op_name + " cache_key: " + cache_key +
                             " ir: " + std::string(ir_string.begin(), ir_string.end()));
  }

  module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(tmp_module));
}

StableHloNode::StableHloNode(const std::string& op_name, const std::string&& cache_key,
                             std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>& module,
                             std::vector<uint8_t>& ir_serialized, std::vector<void*>&& inputs,
                             std::vector<void*>&& outputs, const bool has_collectives)
    : IrNode(op_name, std::move(cache_key), ir_serialized, IrNodeType::STABLEHLO, std::move(inputs),
             std::move(outputs), has_collectives),
      module(module) {}

void StableHloNode::SetMergeMapping(std::unique_ptr<MergeMapping>&& mapping) {
  merge_mapping = std::move(mapping);
}

const MergeMapping* StableHloNode::GetMergeMapping() const { return merge_mapping.get(); }

StableHloNode::StableHloNode(const std::string& op_name, const std::string& cache_key,
                             std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>& module,
                             std::vector<void*>&& inputs, std::vector<void*>&& outputs,
                             const bool has_collectives)
    : module(module) {
  TORCH_NEURONX_DEBUG("Creating StableHloNode from existing MLIR module and context",
                      "op_name:", op_name, "cache_key:", cache_key, "inputs_count:", inputs.size(),
                      "outputs_count:", outputs.size(), "has_collectives:", has_collectives);
  this->op_name = op_name;
  this->cache_key = cache_key;
  this->ir_type = IrNodeType::STABLEHLO;
  this->inputs = std::move(inputs);
  this->outputs = std::move(outputs);
  this->status = IrNodeStatus::CREATED;
  this->has_collectives = has_collectives;

  std::string mlir_string = moduleOpToString_(*(*(this->module)));
  this->ir_serialized = std::vector<uint8_t>(mlir_string.begin(), mlir_string.end());
}

StableHloNode::StableHloNode(const at::neuron::OperationContext& operation_context,
                             mlir::MLIRContext* context) {
  auto& kernel_execution = operation_context.kernel_execution;

  // Cast to XLACompilableKernelExecution to access XLA-specific methods
  auto* xla_kernel =
      dynamic_cast<at::neuron::XLACompilableKernelExecution*>(kernel_execution.get());
  if (!xla_kernel) {
    TORCH_NEURONX_ERROR("Kernel execution is not XLACompilableKernelExecution",
                        "op_name:", kernel_execution->GetOpName());
    throw std::runtime_error("StableHloNode constructor requires XLACompilableKernelExecution");
  }

  TORCH_NEURONX_TRACE_FUNCTION_WITH_ARGS("op_name:", xla_kernel->GetOpName());

  this->op_name = xla_kernel->GetOpName();
  this->cache_key = xla_kernel->GetCacheKey();
  this->ir_type = IrNodeType::STABLEHLO;
  this->status = IrNodeStatus::CREATED;
  this->ir_serialized = xla_kernel->GetHloBytes();
  this->has_collectives = xla_kernel->HasCollectives();

  TORCH_NEURONX_DEBUG("Creating StableHloNode from OperationContext with extracted data",
                      "op_name:", this->op_name, "cache_key:", this->cache_key,
                      "ir_serialized_size:", this->ir_serialized.size(),
                      "has_collectives:", this->has_collectives, "status:", "CREATED");

  std::string ir_string{ir_serialized.begin(), ir_serialized.end()};
  auto tmp_module = mlir::parseSourceString<mlir::ModuleOp>(ir_string, context);
  if (!tmp_module) {
    throw std::runtime_error(std::string("MLIR module parsing failed for StableHloNode op_name: ") +
                             this->op_name + " cache_key: " + this->cache_key +
                             " ir: " + std::string(ir_string.begin(), ir_string.end()));
  }

  this->module = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(tmp_module));

  std::vector<void*> input_data_ptr;
  for (const auto& input : xla_kernel->GetSrcPtrs()) {
    input_data_ptr.push_back(input);
  }

  std::vector<void*> output_data_ptr;
  for (const auto& output : xla_kernel->GetDstPtrs()) {
    output_data_ptr.push_back(output);
  }

  this->inputs = input_data_ptr;
  this->outputs = output_data_ptr;
}

}  // namespace torch_neuronx
