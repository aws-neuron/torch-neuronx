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

  // Resize dst to match self if shapes differ. This implements PyTorch's
  // out= contract: if the out tensor has a different shape than the result
  // of the computation, it is resized.
  if (self.sizes() != dst.sizes()) {
    // Cast away const to resize - this is safe as we're modifying dst which is the output
    const_cast<at::Tensor&>(dst).resize_as_(self);
  }

  // Perform the copy using the existing copy_ operation
  const_cast<at::Tensor&>(dst).copy_(self);

  return dst;
}

}  // namespace torch_neuronx
