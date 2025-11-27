#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace at::autocast {

namespace {
Tensor binary_cross_entropy_banned(const Tensor&, const Tensor&, const std::optional<Tensor>&,
                                   int64_t) {
  TORCH_CHECK(
      false,
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
      "In this case, combine the two layers using "
      "torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
      "safe to autocast.");
}
}  // namespace

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
#define _KERNEL_NEURON_LOW_PRECISION_FP(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_NEURON_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_NEURON_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_NEURON_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_NEURON_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_NEURON_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)

  // promote
#define _KERNEL_NEURON_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_NEURON_PROMOTE)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

}  // namespace at::autocast
