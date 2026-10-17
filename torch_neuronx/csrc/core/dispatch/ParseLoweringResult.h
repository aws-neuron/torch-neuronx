#pragma once

#include <torch/extension.h>

#include <memory>

#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"

namespace torch_neuronx {
namespace dispatch {

/**
 * Parse a Python LoweringResult object into a DispatchCacheEntry.
 *
 * Must be called under the GIL. Extracts stablehlo_bytes, cache_key,
 * kept indices, output specs, aliases, and metadata.
 *
 * Returns nullptr if the LoweringResult has an error field set,
 * signalling that the caller should fall back to Python.
 */
std::shared_ptr<DispatchCacheEntry> ParseLoweringResult(const py::object& result);

}  // namespace dispatch
}  // namespace torch_neuronx
