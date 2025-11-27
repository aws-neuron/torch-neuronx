// Backend selection for scaled_dot_product_attention on Neuron devices
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <c10/core/DeviceType.h>

#include "torch_neuronx/csrc/core/NeuronOpTracking.h"

namespace torch_neuronx {

// This function determines which backend to use for scaled_dot_product_attention
// on Neuron (PrivateUse1) devices
int64_t _fused_sdp_choice_neuron(const at::Tensor& query, const at::Tensor& key,
                                 const at::Tensor& value,
                                 const std::optional<at::Tensor>& attn_mask, double dropout_p,
                                 bool is_causal, std::optional<double> scale, bool enable_gqa) {
  // SDPBackend enum values:
  // math = 0
  // flash_attention = 1
  // efficient_attention = 2
  // cudnn_attention = 3
  // overrideable = 4

  constexpr int64_t SDPBackend_math = 0;
  constexpr int64_t SDPBackend_overrideable = 4;

  // Check if inputs are on Neuron device
  if (query.device().type() != c10::DeviceType::PrivateUse1) {
    // Not on Neuron device
    return SDPBackend_math;
  }

  // Check tensor shapes - must be 4D (batch, heads, seq_len, embed_dim) for our implementation
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    // Wrong tensor dimensions
    return SDPBackend_math;
  }

  // For now, use the overrideable backend whenever we're on Neuron device
  // and have the right tensor shapes. Our Python implementation will
  // handle the actual execution and can raise errors if it encounters
  // unsupported configurations.
  //
  // The overrideable backend is designed for custom device implementations
  // and properly handles both forward and backward passes through the
  // _scaled_dot_product_fused_attention_overrideable ops.
  // Using overrideable backend
  return SDPBackend_overrideable;
}

}  // namespace torch_neuronx

// Register using the dispatch stub system
namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &torch_neuronx::_fused_sdp_choice_neuron);
}  // namespace at::native
