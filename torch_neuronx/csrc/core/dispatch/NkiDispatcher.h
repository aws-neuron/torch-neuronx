#pragma once

#include <torch/extension.h>
#include <torch/library.h>

#include <memory>

namespace torch_neuronx {
namespace dispatch {

struct DispatchCacheEntry;  // forward declaration — defined in NeuronDispatcher.h

// ═══════════════════════════════════════════════════════════════
//  NKI Dispatch
// ═══════════════════════════════════════════════════════════════
//
//  Custom NKI kernels (@nki_op / wrap_nki) dispatch through
//  NkiDispatchEager (pybind11) → NkiDispatchCore in eager mode.
//  torch.compile uses the HOP + torch-mlir path.
//
//  ATen NKI overrides (gather, scatter_add) use @neuron_op →
//  C++ DispatchImpl, the same path as all other ATen ops.
//
//  SDPA has dedicated C++ handlers (DispatchSdpaFwdImpl/BwdImpl).
//
//  Onboarding a new custom NKI kernel requires zero C++ changes.
// ═══════════════════════════════════════════════════════════════

// Shared core: cache-key → cache-lookup → [miss: lower] → execute → reconstruct.
// Called by NkiDispatchEager (pybind11 entry point).
// If out_entry is non-null, receives the DispatchCacheEntry used for execution.
std::vector<torch::Tensor> NkiDispatchCore(
    const c10::List<at::Tensor>& tensor_args, int64_t kernel_idx,
    const std::string& constant_args_key, const c10::List<int64_t>& grid,
    const std::string& operand_output_aliases, bool has_collectives,
    const c10::FunctionSchema& schema, std::shared_ptr<DispatchCacheEntry>* out_entry = nullptr);

// pybind11 entry point — bypasses HOP + boxed dispatch for eager mode.
py::object NkiDispatchEager(std::vector<at::Tensor> tensor_args, int64_t kernel_idx,
                            const std::string& constant_args_key, std::vector<int64_t> grid,
                            const std::string& operand_output_aliases, bool has_collectives);

}  // namespace dispatch
}  // namespace torch_neuronx
