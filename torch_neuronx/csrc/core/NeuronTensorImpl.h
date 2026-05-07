#pragma once

#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

// Forward declarations
namespace at::neuron {
class NeuronStream;
}

namespace c10_neuron {

class NeuronTensorImpl : public c10::TensorImpl {
 public:
  explicit NeuronTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type);

  // Shallow copy operations
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter, bool allow_tensor_metadata_change) const override;

 private:
  // Helper to set device-specific dispatch keys
  void SetNeuronDispatchKeys();
};

}  // namespace c10_neuron
