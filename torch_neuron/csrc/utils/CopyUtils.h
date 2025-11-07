#pragma once
#include <nrt/nrt.h>
#include <torch/torch.h>

namespace torch_neuron {
namespace utils {

void copy_cpu_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void copy_neuron_to_cpu(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
void copy_neuron_to_neuron(const at::Tensor& src, at::Tensor& dst, bool non_blocking = false);
nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor, bool non_blocking = false);

}  // namespace utils
}  // namespace torch_neuron
