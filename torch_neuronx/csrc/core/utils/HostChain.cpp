#include "torch_neuronx/csrc/core/utils/HostChain.h"

#include <vector>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/OperationContext.h"
#include "torch_neuronx/csrc/core/utils/CopyUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"

namespace at::neuron {

bool SubmitHostChain(const std::string& op_name, c10::ArrayRef<at::Tensor> inputs,
                     c10::ArrayRef<at::Tensor> outputs, HostKernelExecution::HostFn fn) {
  EXECUTION_RUNTIME_CHECK(static_cast<bool>(fn), "SubmitHostChain: null host fn");

  if (outputs.empty()) return false;

  const at::Tensor& anchor = inputs.empty() ? outputs.front() : inputs.front();
  const int device_idx = static_cast<int>(anchor.device().index());

  // Allocate pinned CPU mirrors and submit one kRead per input.
  std::vector<at::Tensor> pinned_inputs;
  std::vector<TensorDataRef> src_refs;
  pinned_inputs.reserve(inputs.size());
  src_refs.reserve(inputs.size());
  for (const at::Tensor& in : inputs) {
    if (in.numel() == 0) return false;
    EXECUTION_RUNTIME_CHECK(in.is_privateuseone() && in.is_contiguous(),
                            "SubmitHostChain: inputs must be contiguous Neuron tensors");
    EXECUTION_RUNTIME_CHECK(static_cast<int>(in.device().index()) == device_idx,
                            "SubmitHostChain: all tensors must be on the same device");
    auto pinned =
        at::empty(in.sizes(),
                  at::TensorOptions().dtype(in.scalar_type()).device(at::kCPU).pinned_memory(true));
    torch_neuronx::utils::copy_neuron_to_cpu(in, pinned, /*non_blocking=*/true);
    src_refs.push_back(TensorDataRef{pinned.data_ptr()});
    pinned_inputs.push_back(std::move(pinned));
  }

  // Allocate pinned CPU mirrors for outputs.
  std::vector<at::Tensor> pinned_outputs;
  std::vector<TensorDataRef> dst_refs;
  pinned_outputs.reserve(outputs.size());
  dst_refs.reserve(outputs.size());
  for (const at::Tensor& out : outputs) {
    if (out.numel() == 0) return false;
    EXECUTION_RUNTIME_CHECK(out.is_privateuseone() && out.is_contiguous(),
                            "SubmitHostChain: outputs must be contiguous Neuron tensors");
    EXECUTION_RUNTIME_CHECK(static_cast<int>(out.device().index()) == device_idx,
                            "SubmitHostChain: all tensors must be on the same device");
    auto pinned = at::empty(
        out.sizes(),
        at::TensorOptions().dtype(out.scalar_type()).device(at::kCPU).pinned_memory(true));
    dst_refs.push_back(TensorDataRef{pinned.data_ptr()});
    pinned_outputs.push_back(std::move(pinned));
  }

  // kHost leg: run fn on the pinned buffers.
  std::vector<at::Tensor> keepalives;
  keepalives.reserve(pinned_inputs.size() + pinned_outputs.size());
  for (auto& t : pinned_inputs) keepalives.push_back(t);
  for (auto& t : pinned_outputs) keepalives.push_back(t);

  auto kernel =
      std::make_unique<HostKernelExecution>(op_name, std::move(src_refs), std::move(dst_refs),
                                            std::move(fn), device_idx, std::move(keepalives));
  auto ctx = std::make_unique<OperationContext>(std::move(kernel));
  auto stream = getCurrentNeuronStream(device_idx);
  SubmitOperationContext(stream, std::move(ctx));

  // kWrite legs: pinned_out → device out, async.
  for (size_t i = 0; i < outputs.size(); ++i) {
    at::Tensor out = outputs[i];
    torch_neuronx::utils::copy_cpu_to_neuron(pinned_outputs[i], out, /*non_blocking=*/true);
  }

  return true;
}

}  // namespace at::neuron
