#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>

namespace torch_neuronx {

// True if another backend already owns PrivateUse1 (snapshotted at first call).
inline bool privateuse1_owned_by_other() {
  static const bool owned =
      c10::Dispatcher::singleton().hasBackendFallbackForDispatchKey(c10::DispatchKey::PrivateUse1);
  return owned;
}

}  // namespace torch_neuronx
