#include "DtypeCastVec.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cstring>

#include "DtypeCastVecAvx512.h"  // AVX-512 backend (header is ISA-agnostic).
#include "NonTemporalMemcpy.h"   // For TestCPUFeatureAVX512F (shared probe).

namespace {

// One-shot AVX-512 probe (magic-static, thread-safe).
bool HostHasAvx512() {
  static const bool kHave = torch_neuronx::utils::TestCPUFeatureAVX512F();
  return kHave;
}

// ── Scalar fallbacks (modulo the documented fp32→bf16 NaN divergence) ─────────

void cast_bf16_to_fp32_scalar(const uint16_t* src, float* dst, size_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    uint32_t u = static_cast<uint32_t>(src[i]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    dst[i] = f;
  }
}

void cast_fp32_to_bf16_scalar(const float* src, uint16_t* dst, size_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    at::BFloat16 b(src[i]);
    dst[i] = b.x;
  }
}

void cast_fp16_to_fp32_scalar(const uint16_t* src, float* dst, size_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    at::Half h;
    h.x = src[i];
    dst[i] = static_cast<float>(h);
  }
}

void cast_fp32_to_fp16_scalar(const float* src, uint16_t* dst, size_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    at::Half h(src[i]);
    dst[i] = h.x;
  }
}

}  // namespace

namespace torch_neuronx {
namespace utils {

void cast_bfloat16_to_float(const uint16_t* src, float* dst, size_t numel) {
#ifdef __x86_64__
  if (HostHasAvx512()) {
    detail::cast_bfloat16_to_float_avx512(src, dst, numel);
    return;
  }
#endif
  cast_bf16_to_fp32_scalar(src, dst, numel);
}

void cast_float_to_bfloat16(const float* src, uint16_t* dst, size_t numel) {
#ifdef __x86_64__
  if (HostHasAvx512()) {
    detail::cast_float_to_bfloat16_avx512(src, dst, numel);
    return;
  }
#endif
  cast_fp32_to_bf16_scalar(src, dst, numel);
}

void cast_half_to_float(const uint16_t* src, float* dst, size_t numel) {
#ifdef __x86_64__
  if (HostHasAvx512()) {
    detail::cast_half_to_float_avx512(src, dst, numel);
    return;
  }
#endif
  cast_fp16_to_fp32_scalar(src, dst, numel);
}

void cast_float_to_half(const float* src, uint16_t* dst, size_t numel) {
#ifdef __x86_64__
  if (HostHasAvx512()) {
    detail::cast_float_to_half_avx512(src, dst, numel);
    return;
  }
#endif
  cast_fp32_to_fp16_scalar(src, dst, numel);
}

void RunDtypeCast(const void* src, void* dst, int64_t numel, c10::ScalarType src_dtype,
                  c10::ScalarType dst_dtype) {
  TORCH_CHECK(src != nullptr && dst != nullptr, "RunDtypeCast: null src or dst pointer");
  using ST = c10::ScalarType;
  const size_t n = static_cast<size_t>(numel);

  // Identity: same dtype → memcpy.
  if (src_dtype == dst_dtype) {
    std::memcpy(dst, src, n * c10::elementSize(src_dtype));
    return;
  }

  if (src_dtype == ST::BFloat16 && dst_dtype == ST::Float) {
    cast_bfloat16_to_float(static_cast<const uint16_t*>(src), static_cast<float*>(dst), n);
  } else if (src_dtype == ST::Float && dst_dtype == ST::BFloat16) {
    cast_float_to_bfloat16(static_cast<const float*>(src), static_cast<uint16_t*>(dst), n);
  } else if (src_dtype == ST::Half && dst_dtype == ST::Float) {
    cast_half_to_float(static_cast<const uint16_t*>(src), static_cast<float*>(dst), n);
  } else if (src_dtype == ST::Float && dst_dtype == ST::Half) {
    cast_float_to_half(static_cast<const float*>(src), static_cast<uint16_t*>(dst), n);
  } else {
    TORCH_CHECK(false, "RunDtypeCast: unsupported cast ", static_cast<int>(src_dtype), " -> ",
                static_cast<int>(dst_dtype));
  }
}

}  // namespace utils
}  // namespace torch_neuronx
