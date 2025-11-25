#pragma once
#include <nrt/nrt.h>
#include <torch/torch.h>

#include "torch_neuronx/csrc/core/utils/TensorContext.h"

namespace torch_neuronx {
namespace utils {

void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void nrt_copy_neuron_to_cpu(nrt_tensor_t* src_data_ptr, void* dst_data_ptr, size_t src_offset,
                            size_t size);
void nrt_copy_cpu_to_neuron(void* src_data_ptr, nrt_tensor_t* dst_data_ptr, size_t dst_offset,
                            size_t size);
void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking = false);

}  // namespace utils
}  // namespace torch_neuronx
