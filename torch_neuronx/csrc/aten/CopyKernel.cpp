#include <torch/torch.h>

namespace torch_neuronx {

at::Tensor _copy_from_and_resize_neuron(const at::Tensor& self, const at::Tensor& dst) {
  // Validate that both tensors are defined
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");

  // Check that we're copying from CPU to Neuron
  TORCH_CHECK(self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1,
              "_copy_from_and_resize now only support copy from cpu tensor to neuron tensor, but "
              "got src tensor device is ",
              self.device(), " and dst device is ", dst.device());

  // If dst is empty (numel() == 0), resize it to match self
  if (dst.numel() == 0) {
    // Cast away const to resize - this is safe as we're modifying dst which is the output
    const_cast<at::Tensor&>(dst).resize_as_(self);
  }

  // After potential resize, check that sizes match
  TORCH_CHECK(self.sizes() == dst.sizes(),
              "_copy_from_and_resize now only support copy with same size, or dst.numel() == 0!");

  // Perform the copy using the existing copy_ operation
  const_cast<at::Tensor&>(dst).copy_(self);

  return dst;
}

}  // namespace torch_neuronx
