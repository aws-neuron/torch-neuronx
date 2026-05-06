#include "NonTemporalMemcpy.h"

#include <cstdint>
#include <cstring>

#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>

namespace {

void __attribute__((target("avx512f"))) non_temporal_memcpy_avx512(uint8_t* dst, const uint8_t* src,
                                                                   size_t size) {
  __m512i val = {};
  for (; size >= 64; size -= 64, src += 64, dst += 64) {
    val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), val);
  }
  if (size > 0) {
    std::memcpy(&val, src, size);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), val);
  }
}

void __attribute__((target("avx2"))) non_temporal_memcpy_avx2(uint8_t* dst, const uint8_t* src,
                                                              size_t size) {
  __m256i vals[2] = {};
  for (; size >= 64; size -= 64, src += 64, dst += 64) {
    vals[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    vals[1] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), vals[0]);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), vals[1]);
  }
  if (size > 0) {
    std::memcpy(vals, src, size);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst), vals[0]);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), vals[1]);
  }
}

}  // namespace

#endif  // __x86_64__

namespace torch_neuronx {
namespace utils {

bool TestCPUFeatureAVX512F() {
#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;

  if (__get_cpuid_max(0, nullptr) < 7) {
    return false;
  }

  // Check if OS supports XSAVE
  __cpuid_count(1, 0, eax, ebx, ecx, edx);
  const bool os_uses_xsave = (ecx >> 27) & 1;
  if (!os_uses_xsave) {
    return false;
  }

  // Check XCR0 for AVX-512 state support (bits 1,2,5,6,7)
  uint32_t xcr0_eax, xcr0_edx;
  __asm__ __volatile__("xgetbv" : "=a"(xcr0_eax), "=d"(xcr0_edx) : "c"(0));
  const uint64_t xcr0 = (static_cast<uint64_t>(xcr0_edx) << 32) | xcr0_eax;
  if ((xcr0 & 0xE6) != 0xE6) {
    return false;
  }

  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1 << 16)) != 0;
#else
  return false;
#endif
}

void non_temporal_memcpy(void* dst, const void* src, size_t size) {
  auto d = static_cast<uint8_t*>(dst);
  auto s = static_cast<const uint8_t*>(src);

#ifdef __x86_64__
  static const bool have_avx512 = TestCPUFeatureAVX512F();
  if (have_avx512) {
    non_temporal_memcpy_avx512(d, s, size);
  } else {
    non_temporal_memcpy_avx2(d, s, size);
  }
#else
  std::memcpy(dst, src, size);
#endif
}

void non_temporal_sfence() {
#ifdef __x86_64__
  _mm_sfence();
#endif
}

}  // namespace utils
}  // namespace torch_neuronx
