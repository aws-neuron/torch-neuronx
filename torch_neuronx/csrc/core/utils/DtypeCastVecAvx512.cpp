// AVX-512 cast backend for RunDtypeCast via at::vec (CPU_CAPABILITY=AVX512
// specializations: bf16/fp16 vector = 32 elems, fp32 = 16). Built with AVX-512
// flags + CPU_CAPABILITY_AVX512 in a SEPARATE leaf TU from the dispatcher so
// those macros (and the AVX-512-only codegen) don't leak into other TUs or
// disable the dispatcher's scalar fallback. Caller MUST pre-check AVX-512 — this
// TU emits AVX-512 unconditionally; the DtypeCastVec.cpp probe is the only guard.
#include "DtypeCastVecAvx512.h"

#include <cstring>

// Body gated on __x86_64__ (the only ISA where these macros/specializations make
// sense); other arches get link-time stubs the dispatcher never reaches. Mirrors
// NonTemporalMemcpy.cpp.
#ifdef __x86_64__

#include <ATen/cpu/vec/vec.h>

namespace torch_neuronx {
namespace utils {
namespace detail {

namespace {
// One bf16/fp16 vector (32) = two fp32 vectors (16). Strip-mine in 32-element
// chunks; the scalar tail is bit-equivalent to at::vec's narrow lanes.
constexpr int64_t kReducedVecSize = 32;  // size of Vectorized<BFloat16/Half>
constexpr int64_t kFloatVecSize = 16;    // size of Vectorized<float>
static_assert(kReducedVecSize == 2 * kFloatVecSize,
              "AVX-512 layout assumption: bf16/fp16 vector is 2× the fp32 vector");
}  // namespace

void cast_bfloat16_to_float_avx512(const uint16_t* src, float* dst, size_t numel) {
  using Vec_bf = at::vec::Vectorized<at::BFloat16>;
  using Vec_f = at::vec::Vectorized<float>;
  static_assert(Vec_bf::size() == kReducedVecSize, "AVX-512 BFloat16 vector size mismatch");
  static_assert(Vec_f::size() == kFloatVecSize, "AVX-512 float vector size mismatch");
  const at::BFloat16* src_bf = reinterpret_cast<const at::BFloat16*>(src);
  size_t i = 0;
  for (; i + kReducedVecSize <= numel; i += kReducedVecSize) {
    Vec_bf in = Vec_bf::loadu(src_bf + i);
    auto [lo, hi] = at::vec::convert_bfloat16_float(in);
    lo.store(dst + i);
    hi.store(dst + i + kFloatVecSize);
  }
  // Scalar tail: zero-extend uint16 → uint32, shift left 16, bitcast to float.
  for (; i < numel; ++i) {
    uint32_t u = static_cast<uint32_t>(src[i]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    dst[i] = f;
  }
}

void cast_float_to_bfloat16_avx512(const float* src, uint16_t* dst, size_t numel) {
  using Vec_bf = at::vec::Vectorized<at::BFloat16>;
  using Vec_f = at::vec::Vectorized<float>;
  static_assert(Vec_bf::size() == kReducedVecSize, "AVX-512 BFloat16 vector size mismatch");
  static_assert(Vec_f::size() == kFloatVecSize, "AVX-512 float vector size mismatch");
  at::BFloat16* dst_bf = reinterpret_cast<at::BFloat16*>(dst);
  size_t i = 0;
  for (; i + kReducedVecSize <= numel; i += kReducedVecSize) {
    Vec_f lo = Vec_f::loadu(src + i);
    Vec_f hi = Vec_f::loadu(src + i + kFloatVecSize);
    Vec_bf out = at::vec::convert_float_bfloat16(lo, hi);
    out.store(dst_bf + i);
  }
  for (; i < numel; ++i) {
    at::BFloat16 b(src[i]);
    dst[i] = b.x;
  }
}

void cast_half_to_float_avx512(const uint16_t* src, float* dst, size_t numel) {
  using Vec_h = at::vec::Vectorized<at::Half>;
  using Vec_f = at::vec::Vectorized<float>;
  static_assert(Vec_h::size() == kReducedVecSize, "AVX-512 Half vector size mismatch");
  static_assert(Vec_f::size() == kFloatVecSize, "AVX-512 float vector size mismatch");
  const at::Half* src_h = reinterpret_cast<const at::Half*>(src);
  size_t i = 0;
  for (; i + kReducedVecSize <= numel; i += kReducedVecSize) {
    Vec_h in = Vec_h::loadu(src_h + i);
    auto [lo, hi] = at::vec::convert_half_float(in);
    lo.store(dst + i);
    hi.store(dst + i + kFloatVecSize);
  }
  for (; i < numel; ++i) {
    at::Half h;
    h.x = src[i];
    dst[i] = static_cast<float>(h);
  }
}

void cast_float_to_half_avx512(const float* src, uint16_t* dst, size_t numel) {
  using Vec_h = at::vec::Vectorized<at::Half>;
  using Vec_f = at::vec::Vectorized<float>;
  static_assert(Vec_h::size() == kReducedVecSize, "AVX-512 Half vector size mismatch");
  static_assert(Vec_f::size() == kFloatVecSize, "AVX-512 float vector size mismatch");
  at::Half* dst_h = reinterpret_cast<at::Half*>(dst);
  size_t i = 0;
  for (; i + kReducedVecSize <= numel; i += kReducedVecSize) {
    Vec_f lo = Vec_f::loadu(src + i);
    Vec_f hi = Vec_f::loadu(src + i + kFloatVecSize);
    Vec_h out = at::vec::convert_float_half(lo, hi);
    out.store(dst_h + i);
  }
  for (; i < numel; ++i) {
    at::Half h(src[i]);
    dst[i] = h.x;
  }
}

}  // namespace detail
}  // namespace utils
}  // namespace torch_neuronx

#else  // __x86_64__

// Non-x86-64 stubs — link-time placeholders that the dispatcher will never
// reach (its `#ifdef __x86_64__` ensures it). If something actually invokes
// these, abort loudly rather than silently producing zeros.
#include <cstdlib>
namespace torch_neuronx {
namespace utils {
namespace detail {

[[noreturn]] static void stub_unreachable() {
  std::abort();  // dispatcher contract violated — should be impossible.
}

void cast_bfloat16_to_float_avx512(const uint16_t*, float*, size_t) { stub_unreachable(); }
void cast_float_to_bfloat16_avx512(const float*, uint16_t*, size_t) { stub_unreachable(); }
void cast_half_to_float_avx512(const uint16_t*, float*, size_t) { stub_unreachable(); }
void cast_float_to_half_avx512(const float*, uint16_t*, size_t) { stub_unreachable(); }

}  // namespace detail
}  // namespace utils
}  // namespace torch_neuronx

#endif  // __x86_64__
