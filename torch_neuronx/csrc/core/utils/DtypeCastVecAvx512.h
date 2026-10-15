#pragma once

#include <cstddef>
#include <cstdint>

namespace torch_neuronx {
namespace utils {
namespace detail {

// AVX-512 backend for RunDtypeCast's four casts (impl in DtypeCastVecAvx512.cpp,
// built with the AVX-512 ISA flags + CPU_CAPABILITY macros so at::vec resolves
// to its AVX-512 path). bf16/fp16 storage is uint16_t.
//
// CALLER CONTRACT: must pre-check AVX-512 support — calling these on a non-AVX
// host is an illegal-instruction fault. The DtypeCastVec.cpp dispatcher's CPU
// probe is the only guard.
void cast_bfloat16_to_float_avx512(const uint16_t* src, float* dst, size_t numel);
void cast_float_to_bfloat16_avx512(const float* src, uint16_t* dst, size_t numel);
void cast_half_to_float_avx512(const uint16_t* src, float* dst, size_t numel);
void cast_float_to_half_avx512(const float* src, uint16_t* dst, size_t numel);

}  // namespace detail
}  // namespace utils
}  // namespace torch_neuronx
