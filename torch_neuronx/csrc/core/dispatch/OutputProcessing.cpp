#include "torch_neuronx/csrc/core/dispatch/OutputProcessing.h"

#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/DtypeUtils.h"

namespace torch_neuronx {
namespace dispatch {

// ═══════════════════════════════════════════════════════════════
//  ExtractTensorInputs
// ═══════════════════════════════════════════════════════════════

std::vector<torch::Tensor> ExtractTensorInputs(const std::vector<c10::IValue>& args) {
  std::vector<torch::Tensor> tensors;
  ForEachTensor(args, [&](const at::Tensor& t) { tensors.push_back(t); });
  return tensors;
}

// ═══════════════════════════════════════════════════════════════
//  AllocateExecutionOutputs
// ═══════════════════════════════════════════════════════════════

std::vector<torch::Tensor> AllocateExecutionOutputs(
    const DispatchCacheEntry& entry, const std::vector<c10::IValue>& args,
    const std::vector<torch::Tensor>& filtered_inputs, const torch::Device& device,
    const c10::FunctionSchema& schema) {
  size_t num_outputs = entry.output_specs.size();
  std::vector<torch::Tensor> outputs(num_outputs);

  // ── Step 1: Apply input_output_aliases — reuse IR input buffers as outputs ──
  for (const auto& [ir_param_idx, output_idx] : entry.input_output_aliases) {
    if (ir_param_idx >= 0 && static_cast<size_t>(ir_param_idx) < filtered_inputs.size() &&
        output_idx >= 0 && static_cast<size_t>(output_idx) < num_outputs) {
      const auto& input_tensor = filtered_inputs[ir_param_idx];
      const auto& ospec = entry.output_specs[output_idx];
      if (input_tensor.defined() && input_tensor.scalar_type() == ospec.dtype &&
          input_tensor.is_contiguous()) {
        outputs[output_idx] = input_tensor;
        TORCH_NEURONX_DISPATCH_DEBUG("[AllocateExecutionOutputs] aliasing IR input[", ir_param_idx,
                                     "] → output[", output_idx, "]",
                                     " shape=", input_tensor.sizes(),
                                     " dtype=", input_tensor.scalar_type());
      }
    }
  }

  // ── Step 2: For .out ops, use user-provided out tensors directly ──
  if (!entry.is_inplace) {
    const auto& schema_args = schema.arguments();
    const auto& returns = schema.returns();

    // 2a: Single-tensor out args matched via return alias sets
    for (size_t ret_idx = 0; ret_idx < returns.size() && ret_idx < num_outputs; ++ret_idx) {
      const auto& ret = returns[ret_idx];
      if (!ret.alias_info() || !ret.alias_info()->isWrite()) {
        continue;
      }

      for (size_t arg_idx = 0; arg_idx < schema_args.size() && arg_idx < args.size(); ++arg_idx) {
        const auto& arg_alias = schema_args[arg_idx].alias_info();
        if (!arg_alias || !arg_alias->isWrite()) continue;
        if (ret.alias_info()->beforeSets() != arg_alias->beforeSets()) continue;
        if (!args[arg_idx].isTensor()) continue;

        auto out_tensor = args[arg_idx].toTensor();
        if (!out_tensor.defined()) continue;

        const auto& ospec = entry.output_specs[ret_idx];

        if (out_tensor.sizes().vec() != ospec.shape) {
          TORCH_NEURONX_DISPATCH_DEBUG("[AllocateExecutionOutputs] resizing out tensor",
                                       " arg_idx=", arg_idx);
          ResizeOutputWithWarning(out_tensor, ospec.shape);
        }

        if (out_tensor.scalar_type() == ospec.dtype && out_tensor.is_contiguous()) {
          outputs[ret_idx] = out_tensor;
          TORCH_NEURONX_DISPATCH_DEBUG("[AllocateExecutionOutputs] using out tensor from arg[",
                                       arg_idx, "] for output[", ret_idx, "]");
        }
        break;
      }
    }

    // 2b: TensorList out args (e.g. split_with_sizes_copy.out where out is Tensor[])
    // These have void return but the arg is aliased as write.
    bool any_out_filled = false;
    for (size_t i = 0; i < num_outputs; ++i) {
      if (outputs[i].defined()) {
        any_out_filled = true;
        break;
      }
    }
    if (!any_out_filled) {
      for (size_t arg_idx = 0; arg_idx < schema_args.size() && arg_idx < args.size(); ++arg_idx) {
        const auto& arg_alias = schema_args[arg_idx].alias_info();
        if (!arg_alias || !arg_alias->isWrite()) continue;
        if (!args[arg_idx].isTensorList()) continue;

        auto tensor_list = args[arg_idx].toTensorList();
        for (size_t i = 0; i < tensor_list.size() && i < num_outputs; ++i) {
          at::Tensor out_tensor = tensor_list[i];
          if (!out_tensor.defined()) continue;

          const auto& ospec = entry.output_specs[i];
          if (out_tensor.sizes().vec() != ospec.shape) {
            ResizeOutputWithWarning(out_tensor, ospec.shape);
          }
          if (out_tensor.scalar_type() == ospec.dtype && out_tensor.is_contiguous()) {
            outputs[i] = out_tensor;
            TORCH_NEURONX_DISPATCH_DEBUG("[AllocateExecutionOutputs] using TensorList out[", i,
                                         "] from arg[", arg_idx, "]");
          }
        }
        break;
      }
    }
  }

  // ── Step 3: Allocate fresh tensors only for remaining unfilled slots ──
  for (size_t i = 0; i < num_outputs; ++i) {
    if (!outputs[i].defined()) {
      const auto& spec = entry.output_specs[i];
      auto opts = torch::TensorOptions().dtype(spec.dtype).device(device);
      outputs[i] = torch::empty(spec.shape, opts);
    }
  }

  return outputs;
}

// ═══════════════════════════════════════════════════════════════
//  UpcastExecutionOutputs
// ═══════════════════════════════════════════════════════════════

// Internal helper: copy-back all out/inplace tensors from result to originals.
// Used when no dtype upcast is needed but PreprocessArgs may have replaced tensors.
static void CopyBackOutTensors(
    std::vector<torch::Tensor>& result,
    const std::vector<std::pair<size_t, torch::Tensor>>& original_out_tensors) {
  for (const auto& [ret_idx, orig_tensor] : original_out_tensors) {
    if (ret_idx >= result.size() || !result[ret_idx].defined()) continue;
    // CopyBackToOriginal handles data_ptr identity check, resize, dtype cast
    auto orig_copy = orig_tensor;  // non-const copy for CopyBackToOriginal
    CopyBackToOriginal(orig_copy, result[ret_idx],
                       "UpcastExecutionOutputs ret_idx=" + std::to_string(ret_idx));
    if (orig_copy.data_ptr() != result[ret_idx].data_ptr()) {
      result[ret_idx] = orig_copy;
    }
  }
}

std::vector<torch::Tensor> UpcastExecutionOutputs(
    const std::vector<torch::Tensor>& exec_outputs, const DispatchCacheEntry& entry,
    const std::vector<std::pair<size_t, torch::Tensor>>& original_out_tensors) {
  // Use inferred_output_dtypes from the cache entry (populated by lower_func's
  // dtype inference on original-precision args) for exact per-output upcasting.
  const auto& inferred = entry.inferred_output_dtypes;

  if (inferred.empty()) {
    // No dtype inference was performed — just handle out-tensor copy-back.
    if (!original_out_tensors.empty()) {
      auto result = exec_outputs;
      CopyBackOutTensors(result, original_out_tensors);
      return result;
    }
    return exec_outputs;
  }

  // ── Build out_tensor_map for quick lookup ──
  std::unordered_map<size_t, torch::Tensor> out_tensor_map;
  for (const auto& [ret_idx, t] : original_out_tensors) {
    out_tensor_map[ret_idx] = t;
  }

  // ── Check if any upcast is needed ──
  bool needs_upcast = false;
  for (size_t i = 0; i < exec_outputs.size() && i < inferred.size(); ++i) {
    if (inferred[i] == c10::ScalarType::Undefined) continue;
    if (exec_outputs[i].defined() && exec_outputs[i].scalar_type() != inferred[i]) {
      needs_upcast = true;
      break;
    }
  }

  if (!needs_upcast && original_out_tensors.empty()) {
    return exec_outputs;
  }

  if (!needs_upcast) {
    auto result = exec_outputs;
    CopyBackOutTensors(result, original_out_tensors);
    return result;
  }

  // ── Perform upcast using exact inferred dtypes ──
  std::vector<torch::Tensor> upcasted;
  upcasted.reserve(exec_outputs.size());
  for (size_t i = 0; i < exec_outputs.size(); ++i) {
    auto target_dtype = (i < inferred.size()) ? inferred[i] : c10::ScalarType::Undefined;
    bool dtype_mismatch = target_dtype != c10::ScalarType::Undefined && exec_outputs[i].defined() &&
                          exec_outputs[i].scalar_type() != target_dtype;

    if (dtype_mismatch) {
      const auto& t = exec_outputs[i];

      // Check if this output corresponds to an out/inplace tensor
      auto out_it = out_tensor_map.find(i);
      if (out_it != out_tensor_map.end()) {
        auto orig_tensor = out_it->second;
        TORCH_NEURONX_DISPATCH_DEBUG("[UpcastExecutionOutputs] upcast + copy-back for out/inplace",
                                     " idx=", i, " exec_dtype=", t.scalar_type(),
                                     " target_dtype=", target_dtype);
        auto casted = NeuronSafeCast(t, target_dtype);
        CopyBackToOriginal(orig_tensor, casted,
                           "UpcastExecutionOutputs upcast ret_idx=" + std::to_string(i));
        upcasted.push_back(orig_tensor);
        continue;
      }

      upcasted.push_back(NeuronSafeCast(t, target_dtype));
      continue;
    }

    // No upcast needed for this output — check out/inplace copy-back
    auto out_it = out_tensor_map.find(i);
    if (out_it != out_tensor_map.end()) {
      auto orig_tensor = out_it->second;
      CopyBackToOriginal(orig_tensor, exec_outputs[i],
                         "UpcastExecutionOutputs no-upcast ret_idx=" + std::to_string(i));
      if (orig_tensor.data_ptr() != exec_outputs[i].data_ptr()) {
        upcasted.push_back(orig_tensor);
        continue;
      }
    }

    upcasted.push_back(exec_outputs[i]);
  }
  return upcasted;
}

// ═══════════════════════════════════════════════════════════════
//  ReconstructOutputs
// ═══════════════════════════════════════════════════════════════

std::vector<torch::Tensor> ReconstructOutputs(const std::vector<torch::Tensor>& exec_outputs,
                                              const DispatchCacheEntry& entry,
                                              const std::vector<c10::IValue>& args,
                                              const torch::Device& device) {
  size_t num_original_outputs = entry.original_output_specs.size();

  // Fast path: no reconstruction needed
  if (entry.kept_output_indices.empty() && entry.passthrough_mappings.empty() &&
      exec_outputs.size() == num_original_outputs) {
    return exec_outputs;
  }

  if (num_original_outputs == 1 && exec_outputs.size() == 1) {
    return exec_outputs;
  }

  // Build full output vector
  std::vector<torch::Tensor> full_outputs(num_original_outputs);

  // Step 1: Place execution outputs at their original positions
  for (size_t exec_idx = 0; exec_idx < exec_outputs.size(); ++exec_idx) {
    if (exec_idx < entry.kept_output_indices.size()) {
      int orig_idx = entry.kept_output_indices[exec_idx];
      if (orig_idx >= 0 && static_cast<size_t>(orig_idx) < num_original_outputs) {
        full_outputs[orig_idx] = exec_outputs[exec_idx];
      }
    }
  }

  // Step 2: Fill passthrough slots from input tensors
  auto flat_inputs = ExtractTensorInputs(args);
  for (const auto& [out_idx, in_idx] : entry.passthrough_mappings) {
    if (out_idx >= 0 && static_cast<size_t>(out_idx) < num_original_outputs &&
        !full_outputs[out_idx].defined()) {
      if (in_idx >= 0 && static_cast<size_t>(in_idx) < flat_inputs.size()) {
        auto input_tensor = flat_inputs[in_idx];
        if (input_tensor.device() != device) {
          input_tensor = input_tensor.to(device);
        }
        full_outputs[out_idx] = input_tensor.numel() == 0 ? input_tensor.clone() : input_tensor;
      }
    }
  }

  // Step 3: Fill remaining undefined slots with empty tensors.
  // Slots with ScalarType::Undefined are intentionally None outputs.
  for (size_t i = 0; i < num_original_outputs; ++i) {
    if (!full_outputs[i].defined()) {
      const auto& spec = entry.original_output_specs[i];
      if (spec.dtype == c10::ScalarType::Undefined) {
        continue;  // None output — leave undefined
      }
      auto opts = torch::TensorOptions().dtype(spec.dtype).device(device);
      full_outputs[i] = torch::empty(spec.shape, opts);
    }
  }

  return full_outputs;
}

// ═══════════════════════════════════════════════════════════════
//  PushOutputsToStack
// ═══════════════════════════════════════════════════════════════

void PushOutputsToStack(torch::jit::Stack* stack, const std::vector<torch::Tensor>& outputs,
                        const c10::FunctionSchema& schema) {
  auto returns = schema.returns();
  if (returns.size() == 1) {
    auto ret_type = returns[0].type();
    if (ret_type->isSubtypeOf(*c10::ListType::ofTensors())) {
      c10::List<at::Tensor> tensor_list;
      tensor_list.reserve(outputs.size());
      for (const auto& t : outputs) {
        tensor_list.push_back(t);
      }
      torch::jit::push(*stack, std::move(tensor_list));
    } else if (outputs.size() == 1) {
      torch::jit::push(*stack, outputs[0]);
    } else if (outputs.empty()) {
      // Void return — push nothing
    } else {
      torch::jit::push(*stack, outputs[0]);
    }
  } else if (returns.size() > 1) {
    for (size_t i = 0; i < returns.size() && i < outputs.size(); ++i) {
      torch::jit::push(*stack, outputs[i]);
    }
  }
  // returns.size() == 0 → push nothing
}

}  // namespace dispatch
}  // namespace torch_neuronx
