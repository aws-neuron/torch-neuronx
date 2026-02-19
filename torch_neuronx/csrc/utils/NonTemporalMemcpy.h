#pragma once

#include <cstddef>

namespace torch_neuronx {
namespace utils {

// Copy memory using non-temporal stores to bypass CPU cache.
void non_temporal_memcpy(void* dst, const void* src, size_t size);

// Store fence for non-temporal stores.
void non_temporal_sfence();

// CPU feature detection - returns true if AVX512F is supported
bool TestCPUFeatureAVX512F();

}  // namespace utils
}  // namespace torch_neuronx
