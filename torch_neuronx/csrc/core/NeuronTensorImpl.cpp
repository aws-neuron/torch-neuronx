#include "torch_neuronx/csrc/core/NeuronTensorImpl.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10_neuron {

NeuronTensorImpl::NeuronTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                      data_type) {
  // Set neuron-specific dispatch keys
  SetNeuronDispatchKeys();
}

void NeuronTensorImpl::SetNeuronDispatchKeys() {
  // Set the dispatch keys for Neuron device
  key_set_ = key_set_.add(c10::DispatchKey::PrivateUse1);
  key_set_ = key_set_.add(c10::DispatchKey::AutogradPrivateUse1);
}

void NeuronTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TensorImpl::shallow_copy_from(impl);
  if (impl->device().type() != c10::DeviceType::CPU) {
    SetNeuronDispatchKeys();
  }
}

c10::intrusive_ptr<c10::TensorImpl> NeuronTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter, bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<NeuronTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> NeuronTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter, bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<NeuronTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

}  // namespace c10_neuron
