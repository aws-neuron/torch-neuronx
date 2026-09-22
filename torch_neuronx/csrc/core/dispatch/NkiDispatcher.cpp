#include "torch_neuronx/csrc/core/dispatch/NkiDispatcher.h"

#include <torch/extension.h>

#include <sstream>
#include <string>

#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/NeuronOpTracking.h"
#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"
#include "torch_neuronx/csrc/core/dispatch/OutputProcessing.h"
#include "torch_neuronx/csrc/core/dispatch/ParseLoweringResult.h"

namespace torch_neuronx {
namespace dispatch {

namespace {

// Build cache key: "nki_call:<kernel_idx>:<constant_args_key>:g<grid>|<tensor_specs>"
std::string BuildNkiCacheKey(const c10::List<at::Tensor>& tensor_args, int64_t kernel_idx,
                             const std::string& constant_args_key, const c10::List<int64_t>& grid) {
  std::ostringstream key;
  key << "nki_call:" << kernel_idx << ":" << constant_args_key << ":g";
  for (size_t i = 0; i < grid.size(); ++i) {
    if (i > 0) key << ",";
    key << grid[i];
  }
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    key << "|";
    at::neuron::CacheKeyGenerator::AppendTensor(key, tensor_args[i]);
  }
  return key.str();
}

// Call Python nki_call_lower_func on cache miss.
py::object CallPythonLowerFunc(const c10::List<at::Tensor>& tensor_args, int64_t kernel_idx,
                               const std::string& constant_args_key, const c10::List<int64_t>& grid,
                               const std::string& operand_output_aliases, bool has_collectives) {
  // Use a leaked pointer to avoid py::object destructor running during
  // interpreter finalization.
  static py::object* lower_func =
      new py::object(py::module_::import("torch_neuronx.python_ops.neuron_nki_call_lowering")
                         .attr("nki_call_lower_func"));

  py::list py_tensors;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    py_tensors.append(py::cast(tensor_args[i]));
  }

  py::list py_grid;
  for (size_t i = 0; i < grid.size(); ++i) {
    py_grid.append(grid[i]);
  }

  return (*lower_func)(py_tensors, kernel_idx, constant_args_key, py_grid, operand_output_aliases,
                       has_collectives);
}

}  // namespace

// ═══════════════════════════════════════════════════════════════
//  NkiDispatchCore — shared pipeline for NKI dispatch
// ═══════════════════════════════════════════════════════════════

std::vector<torch::Tensor> NkiDispatchCore(const c10::List<at::Tensor>& tensor_args,
                                           int64_t kernel_idx, const std::string& constant_args_key,
                                           const c10::List<int64_t>& grid,
                                           const std::string& operand_output_aliases,
                                           bool has_collectives, const c10::FunctionSchema& schema,
                                           std::shared_ptr<DispatchCacheEntry>* out_entry) {
  auto cache_key = BuildNkiCacheKey(tensor_args, kernel_idx, constant_args_key, grid);

  // Reject complex-dtype tensors — NKI ops don't support them.
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    TORCH_CHECK(!tensor_args.get(i).is_complex(),
                "[NkiDispatch] neuron::nki_call does not support complex tensors");
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[NkiDispatch] cache_key=", cache_key);

  auto& dispatcher = NeuronDispatcher::Instance();
  auto entry = dispatcher.CacheLookup(cache_key);

  if (!entry) {
    py::gil_scoped_acquire gil;

    entry = dispatcher.CacheLookup(cache_key);
    if (!entry) {
      TORCH_NEURONX_DISPATCH_DEBUG("[NkiDispatch] CACHE MISS, calling Python lower_func",
                                   " cache_key=", cache_key);

      py::object result;
      try {
        result = CallPythonLowerFunc(tensor_args, kernel_idx, constant_args_key, grid,
                                     operand_output_aliases, has_collectives);
      } catch (py::error_already_set& e) {
        TORCH_CHECK(false, "[NkiDispatch] Python lower_func failed for cache_key=", cache_key, ": ",
                    e.what());
      }

      TORCH_CHECK(!result.is_none(),
                  "[NkiDispatch] nki_call_lower_func returned None for key=", cache_key);

      entry = ParseLoweringResult(result);
      TORCH_CHECK(entry != nullptr,
                  "[NkiDispatch] ParseLoweringResult returned null for key=", cache_key);

      dispatcher.CacheInsert(cache_key, entry);
    }
  }

  TORCH_NEURONX_DISPATCH_DEBUG("[NkiDispatch] entry obtained", " cache_key=", cache_key,
                               " op_name=", entry->op_name);

  // ── Determine device from first tensor ──
  torch::Device device(c10::DeviceType::PrivateUse1,
                       static_cast<c10::DeviceIndex>(c10_neuron::current_device()));
  if (tensor_args.size() > 0 && tensor_args.get(0).is_privateuseone()) {
    device = tensor_args.get(0).device();
  }

  // ── Handle empty tensor case ──
  if (entry->is_empty_tensor) {
    std::vector<torch::Tensor> empty_outputs;
    for (const auto& spec : entry->original_output_specs) {
      if (spec.dtype == c10::ScalarType::Undefined) {
        empty_outputs.push_back(torch::Tensor());
      } else {
        empty_outputs.push_back(
            torch::empty(spec.shape, torch::TensorOptions().dtype(spec.dtype).device(device)));
      }
    }
    if (out_entry) *out_entry = entry;
    logOperationExecuted("neuron::nki_call");
    return empty_outputs;
  }

  // ── Execution pipeline ──
  std::vector<torch::Tensor> all_inputs(tensor_args.begin(), tensor_args.end());
  std::vector<torch::Tensor> filtered_inputs;
  if (entry->kept_input_indices.empty()) {
    filtered_inputs = all_inputs;
  } else {
    filtered_inputs.reserve(entry->kept_input_indices.size());
    for (int idx : entry->kept_input_indices) {
      TORCH_CHECK(idx >= 0 && static_cast<size_t>(idx) < all_inputs.size(),
                  "[NkiDispatch] kept_input_index ", idx, " out of range for ", all_inputs.size(),
                  " inputs");
      filtered_inputs.push_back(all_inputs[idx]);
    }
  }

  std::vector<c10::IValue> ivalue_args;
  ivalue_args.reserve(all_inputs.size());
  for (auto& t : all_inputs) {
    ivalue_args.emplace_back(t);
  }

  for (const auto& [out_idx, in_idx] : entry->passthrough_mappings) {
    TORCH_CHECK(static_cast<size_t>(in_idx) < ivalue_args.size(),
                "[NkiDispatch] passthrough mapping references input ", in_idx, " but only ",
                ivalue_args.size(), " tensor args available");
  }

  auto exec_outputs =
      AllocateExecutionOutputs(*entry, ivalue_args, filtered_inputs, device, schema);

  int device_id = c10_neuron::GetTargetDeviceId(filtered_inputs, exec_outputs);

  auto fallback_ctx = BuildCPUFallbackContext(ivalue_args, schema);
  NeuronDispatcher::EnqueueXlaOperation(entry->op_name, filtered_inputs, exec_outputs,
                                        entry->stablehlo_bytes, entry->neff_cache_key,
                                        entry->has_collectives, device_id, std::move(fallback_ctx));

  auto final_outputs = ReconstructOutputs(exec_outputs, *entry, ivalue_args, device);

  if (out_entry) *out_entry = entry;
  logOperationExecuted("neuron::nki_call");
  return final_outputs;
}

// ═══════════════════════════════════════════════════════════════
//  NkiDispatchEager — pybind entry point (eager bypass)
// ═══════════════════════════════════════════════════════════════

py::object NkiDispatchEager(std::vector<at::Tensor> tensor_args_vec, int64_t kernel_idx,
                            const std::string& constant_args_key, std::vector<int64_t> grid,
                            const std::string& operand_output_aliases, bool has_collectives) {
  TORCH_NEURONX_DISPATCH_DEBUG("[NkiDispatchEager] ENTERED: kernel_idx=", kernel_idx,
                               " num_tensors=", tensor_args_vec.size());

  // Static schema — no dependency on neuron::nki_call op registration.
  static const auto schema = torch::jit::parseSchema(
      "neuron::nki_call(Tensor[] tensor_args, int kernel_idx, str constant_args_key, "
      "int[] grid, str operand_output_aliases, bool has_collectives) -> Tensor[]");

  // Convert to c10::List for NkiDispatchCore
  c10::List<at::Tensor> tensor_args;
  tensor_args.reserve(tensor_args_vec.size());
  for (auto& t : tensor_args_vec) {
    tensor_args.push_back(std::move(t));
  }

  std::shared_ptr<DispatchCacheEntry> entry;
  auto results =
      NkiDispatchCore(tensor_args, kernel_idx, constant_args_key, c10::List<int64_t>(grid),
                      operand_output_aliases, has_collectives, schema, &entry);

  // Post-processing using cached DumpedConfig (P1 optimization).
  // On cache hit, nki_output_specs and nki_operand_output_aliases are already
  // in the entry — no need to call Python get_dumped_config().
  if (entry && entry->has_nki_config) {
    TORCH_NEURONX_DISPATCH_DEBUG("[NkiDispatchEager] CACHE HIT with DumpedConfig: ",
                                 "num_output_specs=", entry->nki_output_specs.size(),
                                 " num_aliases=", entry->nki_operand_output_aliases.size());
    // Alias copy-back: copy aliased outputs back to corresponding input tensors.
    // Skipped when input_output_aliases already made the NEFF write in place.
    if (entry->input_output_aliases.empty()) {
      for (const auto& [input_id, output_id] : entry->nki_operand_output_aliases) {
        TORCH_CHECK(input_id >= 0 && static_cast<size_t>(input_id) < tensor_args.size(),
                    "[NkiDispatchEager] alias input_id ", input_id, " out of range for ",
                    tensor_args.size(), " tensor args");
        TORCH_CHECK(output_id >= 0 && static_cast<size_t>(output_id) < results.size(),
                    "[NkiDispatchEager] alias output_id ", output_id, " out of range for ",
                    results.size(), " results");
        tensor_args.get(input_id).copy_(results[output_id]);
      }
    }

    // Return only actual outputs (first nki_output_specs.size() tensors)
    size_t num_outputs = entry->nki_output_specs.size();
    if (num_outputs == 1 && results.size() >= 1) {
      return py::cast(results[0]);
    }
    py::tuple out(num_outputs);
    for (size_t i = 0; i < num_outputs && i < results.size(); ++i) {
      out[i] = py::cast(results[i]);
    }
    return out;
  }

  // Fallback: return raw results as list (caller handles post-processing)
  py::list out;
  for (auto& t : results) {
    out.append(py::cast(std::move(t)));
  }
  return out;
}

}  // namespace dispatch
}  // namespace torch_neuronx
