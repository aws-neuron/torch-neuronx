#pragma once

#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>

namespace torch_neuronx {
namespace utils {

// Vectorized dtype-cast primitives for the cross-dtype CPU bounce on the FSDP
// mixed-precision path. bf16/fp16 are uint16_t (bf16 = high 16 bits of fp32,
// round-to-nearest-even). Each dispatches at runtime between an AVX-512 backend
// (DtypeCastVecAvx512.cpp, isolating the ISA dep to one leaf TU via at::vec) and
// a scalar fallback, on TestCPUFeatureAVX512F(). Any numel is safe (AVX-512
// strip-mines in 32-element chunks + scalar tail).
//
// NaN note: cast_float_to_bfloat16 emits 0xFFFF (AVX-512, matches at::vec) vs
// 0x7FC0 (scalar c10::BFloat16 ctor) for NaN inputs. Both are valid bf16 NaN;
// consumers must not depend on the exact bit-pattern.

void cast_bfloat16_to_float(const uint16_t* src, float* dst, size_t numel);
void cast_float_to_bfloat16(const float* src, uint16_t* dst, size_t numel);
void cast_half_to_float(const uint16_t* src, float* dst, size_t numel);
void cast_float_to_half(const float* src, uint16_t* dst, size_t numel);

// Cast `numel` elements from `src` (src_dtype) to `dst` (dst_dtype), raw
// contiguous CPU buffers, non-null. Identity dtype → memcpy. Supported pairs:
// bf16<->fp32, fp16<->fp32; any other raises EXECUTION_RUNTIME_CHECK. This is
// the host-compute body of the kHost cross-dtype chain.
void RunDtypeCast(const void* src, void* dst, int64_t numel, c10::ScalarType src_dtype,
                  c10::ScalarType dst_dtype);

}  // namespace utils
}  // namespace torch_neuronx
