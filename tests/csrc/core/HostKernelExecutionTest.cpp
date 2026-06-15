#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tests/csrc/utils/TestUtils.h"
#include "torch_neuronx/csrc/core/KernelExecution.h"
#include "torch_neuronx/csrc/core/utils/DtypeCastVec.h"

using namespace at::neuron;
using torch_neuronx::utils::RunDtypeCast;

namespace {

template <typename T>
::testing::AssertionResult VectorsEqual(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "size mismatch: " << a.size() << " vs " << b.size();
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return ::testing::AssertionFailure() << "mismatch at index " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

// ── RunDtypeCast: bf16 → fp32 ────────────────────────────────────────────────
// bf16 widening to fp32 is loss-less (low 16 mantissa bits filled with zero).
TEST(RunDtypeCastTest, BF16ToFP32Correctness) {
  constexpr int64_t numel = 8;
  std::vector<at::BFloat16> src(numel);
  std::vector<float> dst(numel, std::nanf(""));
  for (int64_t i = 0; i < numel; ++i) src[i] = at::BFloat16(static_cast<float>(i) * 0.25f);

  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::BFloat16,
                               c10::ScalarType::Float));
  for (int64_t i = 0; i < numel; ++i)
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(src[i])) << "i=" << i;
}

// ── RunDtypeCast: fp32 → bf16 ────────────────────────────────────────────────
TEST(RunDtypeCastTest, FP32ToBF16Correctness) {
  constexpr int64_t numel = 8;
  std::vector<float> src(numel);
  std::vector<at::BFloat16> dst(numel);
  for (int64_t i = 0; i < numel; ++i) src[i] = static_cast<float>(i) * 0.25f;

  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float,
                               c10::ScalarType::BFloat16));
  for (int64_t i = 0; i < numel; ++i) EXPECT_EQ(static_cast<float>(dst[i]), src[i]) << "i=" << i;
}

// ── RunDtypeCast: fp16 → fp32 ────────────────────────────────────────────────
TEST(RunDtypeCastTest, FP16ToFP32Correctness) {
  constexpr int64_t numel = 6;
  std::vector<at::Half> src(numel);
  std::vector<float> dst(numel, std::nanf(""));
  for (int64_t i = 0; i < numel; ++i) src[i] = at::Half(static_cast<float>(i) * 0.5f);

  EXPECT_NO_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Half, c10::ScalarType::Float));
  for (int64_t i = 0; i < numel; ++i)
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(src[i])) << "i=" << i;
}

// ── RunDtypeCast: fp32 → fp16 ────────────────────────────────────────────────
TEST(RunDtypeCastTest, FP32ToFP16Correctness) {
  constexpr int64_t numel = 6;
  std::vector<float> src(numel);
  std::vector<at::Half> dst(numel);
  for (int64_t i = 0; i < numel; ++i) src[i] = static_cast<float>(i) * 0.5f;

  EXPECT_NO_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float, c10::ScalarType::Half));
  for (int64_t i = 0; i < numel; ++i) EXPECT_EQ(static_cast<float>(dst[i]), src[i]) << "i=" << i;
}

// ── RunDtypeCast: identity (same dtype) routes to memcpy ─────────────────────
TEST(RunDtypeCastTest, IdentityBF16Memcpy) {
  constexpr int64_t numel = 16;
  std::vector<at::BFloat16> src(numel);
  std::vector<at::BFloat16> dst(numel);
  for (int64_t i = 0; i < numel; ++i) src[i] = at::BFloat16(static_cast<float>(i + 1));

  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::BFloat16,
                               c10::ScalarType::BFloat16));
  EXPECT_TRUE(VectorsEqual(src, dst));
}

TEST(RunDtypeCastTest, IdentityFP32Memcpy) {
  constexpr int64_t numel = 16;
  std::vector<float> src(numel);
  std::vector<float> dst(numel, 0.0f);
  for (int64_t i = 0; i < numel; ++i) src[i] = static_cast<float>(i) - 7.5f;

  EXPECT_NO_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float, c10::ScalarType::Float));
  EXPECT_TRUE(VectorsEqual(src, dst));
}

// ── RunDtypeCast: unsupported pair throws ────────────────────────────────────
// fp16 ↔ bf16 is intentionally not supported in v1. Verify a hard error.
TEST(RunDtypeCastTest, UnsupportedPairBF16ToFP16Throws) {
  constexpr int64_t numel = 4;
  std::vector<at::BFloat16> src(numel, at::BFloat16(1.0f));
  std::vector<at::Half> dst(numel);
  EXPECT_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::BFloat16, c10::ScalarType::Half),
      std::exception);
}

// ── RunDtypeCast: null pointer is rejected ───────────────────────────────────
TEST(RunDtypeCastTest, NullPointersThrow) {
  EXPECT_THROW(RunDtypeCast(nullptr, nullptr, 4, c10::ScalarType::Float, c10::ScalarType::BFloat16),
               std::exception);
}

// ── RunDtypeCast: AVX-512 chunk + tail coverage ──────────────────────────────
// The vec path strip-mines in 32-element chunks; numel=33 crosses one chunk
// boundary with a 1-element tail, exercising both the AVX-512 loop (on capable
// hosts) and the scalar tail / scalar fallback.
TEST(RunDtypeCastTest, BF16ToFP32ChunkPlusTail) {
  constexpr int64_t numel = 33;
  std::vector<at::BFloat16> src(numel);
  std::vector<float> dst(numel, std::nanf(""));
  for (int64_t i = 0; i < numel; ++i) src[i] = at::BFloat16(static_cast<float>(i) * 0.25f - 4.0f);
  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::BFloat16,
                               c10::ScalarType::Float));
  for (int64_t i = 0; i < numel; ++i)
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(src[i])) << "i=" << i;
}

TEST(RunDtypeCastTest, FP32ToBF16ChunkPlusTail) {
  constexpr int64_t numel = 33;
  std::vector<float> src(numel);
  std::vector<at::BFloat16> dst(numel);
  for (int64_t i = 0; i < numel; ++i) src[i] = static_cast<float>(i) * 0.25f - 4.0f;
  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float,
                               c10::ScalarType::BFloat16));
  for (int64_t i = 0; i < numel; ++i) EXPECT_EQ(static_cast<float>(dst[i]), src[i]) << "i=" << i;
}

TEST(RunDtypeCastTest, FP16ToFP32ChunkPlusTail) {
  constexpr int64_t numel = 33;
  std::vector<at::Half> src(numel);
  std::vector<float> dst(numel, std::nanf(""));
  for (int64_t i = 0; i < numel; ++i) src[i] = at::Half(static_cast<float>(i) * 0.5f - 8.0f);
  EXPECT_NO_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Half, c10::ScalarType::Float));
  for (int64_t i = 0; i < numel; ++i)
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(src[i])) << "i=" << i;
}

TEST(RunDtypeCastTest, FP32ToFP16ChunkPlusTail) {
  constexpr int64_t numel = 33;
  std::vector<float> src(numel);
  std::vector<at::Half> dst(numel);
  for (int64_t i = 0; i < numel; ++i) src[i] = static_cast<float>(i) * 0.5f - 8.0f;
  EXPECT_NO_THROW(
      RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float, c10::ScalarType::Half));
  for (int64_t i = 0; i < numel; ++i) EXPECT_EQ(static_cast<float>(dst[i]), src[i]) << "i=" << i;
}

TEST(RunDtypeCastTest, BF16ToFP32Large) {
  constexpr int64_t numel = 1024;
  std::vector<at::BFloat16> src(numel);
  std::vector<float> dst(numel, std::nanf(""));
  for (int64_t i = 0; i < numel; ++i)
    src[i] = at::BFloat16((static_cast<float>(i) - 512.0f) * 0.0078125f);
  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::BFloat16,
                               c10::ScalarType::Float));
  for (int64_t i = 0; i < numel; ++i)
    EXPECT_FLOAT_EQ(dst[i], static_cast<float>(src[i])) << "i=" << i;
}

// NaN canonicalisation: fp32 → bf16 must map NaN to a bf16 NaN (all-ones
// exponent, non-zero mantissa). vec emits 0xFFFF, scalar emits 0x7FC0 — both
// valid; verify is_nan + exponent, not the exact bits.
TEST(RunDtypeCastTest, FP32ToBF16NaNStaysNaN) {
  constexpr int64_t numel = 16;
  std::vector<float> src(numel, std::nanf(""));
  std::vector<at::BFloat16> dst(numel);
  EXPECT_NO_THROW(RunDtypeCast(src.data(), dst.data(), numel, c10::ScalarType::Float,
                               c10::ScalarType::BFloat16));
  for (int64_t i = 0; i < numel; ++i) {
    const uint16_t bits = dst[i].x;
    const uint16_t exp = (bits >> 7) & 0xff;
    const uint16_t mant = bits & 0x7f;
    EXPECT_EQ(exp, 0xffu) << "i=" << i << " bits=0x" << std::hex << bits;
    EXPECT_NE(mant, 0u) << "i=" << i << " bits=0x" << std::hex << bits;
    EXPECT_TRUE(std::isnan(static_cast<float>(dst[i]))) << "i=" << i;
  }
}

// ── HostKernelExecution: closure runs and sees the data refs ─────────────────
TEST(HostKernelExecutionTest, ClosureReceivesSrcAndDstRefs) {
  std::vector<int> src_storage = {1, 2, 3};
  std::vector<int> dst_storage(3, 0);

  std::vector<TensorDataRef> srcs;
  srcs.emplace_back(static_cast<void*>(src_storage.data()));
  std::vector<TensorDataRef> dsts;
  dsts.emplace_back(static_cast<void*>(dst_storage.data()));

  bool fn_called = false;
  HostKernelExecution::HostFn fn = [&fn_called, &src_storage, &dst_storage](
                                       const std::vector<TensorDataRef>& s,
                                       const std::vector<TensorDataRef>& d) {
    fn_called = true;
    ASSERT_EQ(s.size(), 1u);
    ASSERT_EQ(d.size(), 1u);
    ASSERT_EQ(s[0].ptr, src_storage.data());
    ASSERT_EQ(d[0].ptr, dst_storage.data());
    std::memcpy(d[0].ptr, s[0].ptr, sizeof(int) * 3);
  };

  HostKernelExecution kernel("host", std::move(srcs), std::move(dsts), std::move(fn), /*dev=*/0);
  ASSERT_TRUE(kernel.IsValid());
  EXPECT_NO_THROW(kernel.RunHost());
  EXPECT_TRUE(fn_called);
  EXPECT_EQ(dst_storage[0], 1);
  EXPECT_EQ(dst_storage[1], 2);
  EXPECT_EQ(dst_storage[2], 3);
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHost);
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_FALSE(IsDeviceKernelType(kernel.GetKernelType()));
}

// ── HostKernelExecution: keepalives hold tensor refcounts for the op lifetime ─
// The keepalive vector must keep its at::Tensor storage alive as long as the
// kernel object exists (the OperationContext owns the kernel until completion).
TEST(HostKernelExecutionTest, KeepalivesHoldStorage) {
  auto t = at::empty({64}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  void* data = t.data_ptr();
  const auto use_count_before = t.use_count();

  HostKernelExecution kernel(
      "host_keep", {TensorDataRef{data}}, {TensorDataRef{data}},
      [](const std::vector<TensorDataRef>&, const std::vector<TensorDataRef>&) {}, /*dev=*/0,
      /*keepalives=*/std::vector<at::Tensor>{t});

  // The kernel holds an extra reference to t's storage.
  EXPECT_GT(t.use_count(), use_count_before);
  EXPECT_TRUE(kernel.IsValid());
  EXPECT_NO_THROW(kernel.RunHost());
}

// ── HostKernelExecution: empty functor is invalid ───────────────────────────
TEST(HostKernelExecutionTest, EmptyFunctorIsInvalid) {
  HostKernelExecution kernel("host_empty", {}, {}, HostKernelExecution::HostFn{}, /*dev=*/0);
  EXPECT_FALSE(kernel.IsValid());
}

// ── HostKernelExecution: no-op closure smoke test ───────────────────────────
TEST(HostKernelExecutionTest, NoOpClosureSucceeds) {
  HostKernelExecution kernel(
      "noop", {}, {}, [](const std::vector<TensorDataRef>&, const std::vector<TensorDataRef>&) {},
      /*dev=*/0);
  EXPECT_TRUE(kernel.IsValid());
  EXPECT_EQ(kernel.GetKernelType(), KernelTypeEnum::kHost);
  EXPECT_FALSE(kernel.RequiresCompilation());
  EXPECT_NO_THROW(kernel.RunHost());
}

// ── HostKernelExecution::Schedule is unreachable ────────────────────────────
// Belt-and-suspenders against future regressions that route a kHost op through
// the device-XU scheduling path.
TEST(HostKernelExecutionTest, ScheduleIsAssertUnreachable) {
  HostKernelExecution kernel(
      "noop", {}, {}, [](const std::vector<TensorDataRef>&, const std::vector<TensorDataRef>&) {},
      /*dev=*/0);
  NRT_STATUS status = NRT_SUCCESS;
  nrt::SequenceId seq = 0;
  EXPECT_THROW(kernel.Schedule(&status, &seq, /*stream_id=*/0), std::exception);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
