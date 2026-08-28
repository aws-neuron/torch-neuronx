/**
 * Unit tests for DispatchUtils free functions.
 *
 * Covers: HasEmptyTensorInput, HasEmptyOutputTensor, HasComplexInput,
 * HasComplexNonOutputInput, IsSupportedDtype, DowncastArgs,
 * SnapshotOutTensors, HasExplicitDtypeKwarg, ExtractBaseOpName,
 * GetReductionConfig, GetIndexOpConfig, IsIndexOp, TensorToMeta,
 * IValueToMeta.
 *
 * No Python or hardware required — all tests use CPU tensors.
 */

#include <gtest/gtest.h>

#include <cstdlib>

#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"

namespace torch_neuronx {
namespace dispatch {
namespace {

// ═══════════════════════════════════════════════════════════════
//  Schema helpers — reuse well-known ATen schemas for testing
// ═══════════════════════════════════════════════════════════════

static c10::FunctionSchema AddSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
}

// aten::sum.dim_IntList has a dtype kwarg
static c10::FunctionSchema SumSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum", "dim_IntList").schema();
}

// aten::copy_ has a write alias (Tensor(a!) self)
static c10::FunctionSchema CopySchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_", "").schema();
}

// aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
// Non-write inputs + write-aliased out = good schema for out-param tests.
static c10::FunctionSchema ComplexOutSchema() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::complex", "out").schema();
}

// ═══════════════════════════════════════════════════════════════
//  HasEmptyTensorInput
// ═══════════════════════════════════════════════════════════════

TEST(HasEmptyTensorInputTest, NoEmpty) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({4})};
  EXPECT_FALSE(HasEmptyTensorInput(args));
}

TEST(HasEmptyTensorInputTest, HasEmpty) {
  std::vector<c10::IValue> args = {torch::zeros({0, 3}), torch::ones({4})};
  EXPECT_TRUE(HasEmptyTensorInput(args));
}

TEST(HasEmptyTensorInputTest, ZeroDimNotEmpty) {
  std::vector<c10::IValue> args = {torch::tensor(5.0)};
  EXPECT_FALSE(HasEmptyTensorInput(args));
}

TEST(HasEmptyTensorInputTest, NoTensors) {
  std::vector<c10::IValue> args = {c10::IValue(42)};
  EXPECT_FALSE(HasEmptyTensorInput(args));
}

TEST(HasEmptyTensorInputTest, EmptyInTensorList) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::zeros({0, 3})});
  std::vector<c10::IValue> args = {c10::IValue(tl)};
  EXPECT_TRUE(HasEmptyTensorInput(args));
}

TEST(HasEmptyTensorInputTest, EmptyArgs) { EXPECT_FALSE(HasEmptyTensorInput({})); }

// ═══════════════════════════════════════════════════════════════
//  HasEmptyOutputTensor
// ═══════════════════════════════════════════════════════════════

TEST(HasEmptyOutputTensorTest, CopyWithNonEmptySelf) {
  auto self = torch::zeros({2, 3});
  auto src = torch::ones({2, 3});
  std::vector<c10::IValue> args = {self, src, false};
  EXPECT_FALSE(HasEmptyOutputTensor(args, CopySchema()));
}

TEST(HasEmptyOutputTensorTest, CopyWithEmptySelf) {
  auto self = torch::zeros({0, 3});
  auto src = torch::ones({2, 3});
  std::vector<c10::IValue> args = {self, src, false};
  EXPECT_TRUE(HasEmptyOutputTensor(args, CopySchema()));
}

TEST(HasEmptyOutputTensorTest, NonInplaceSchemaIgnoresArgs) {
  // aten::add has no write alias — even if a tensor is empty, it's not an "output"
  std::vector<c10::IValue> args = {torch::zeros({0, 3}), torch::ones({2}), 1};
  EXPECT_FALSE(HasEmptyOutputTensor(args, AddSchema()));
}

// ShouldHandleEmpty was removed — its logic is now consolidated into
// NeuronDispatcher::TryHandleEmptyTensors, tested via Python integration tests.

// ═══════════════════════════════════════════════════════════════
//  HasComplexNonOutputInput — skips out= params via schema
// ═══════════════════════════════════════════════════════════════

TEST(HasComplexNonOutputInputTest, NoComplexAnywhere) {
  std::vector<c10::IValue> args = {torch::ones({2, 3}), torch::zeros({2, 3}), torch::zeros({2, 3})};
  EXPECT_FALSE(HasComplexNonOutputInput(args, ComplexOutSchema()));
}

TEST(HasComplexNonOutputInputTest, ComplexOutputOnly) {
  // aten::complex.out: args are (real, imag, out=). Only `out` is complex.
  auto real = torch::ones({2, 3});
  auto imag = torch::zeros({2, 3});
  auto out = torch::zeros({2, 3}, torch::kComplexFloat);
  std::vector<c10::IValue> args = {real, imag, out};
  // HasComplexInput sees all tensors and flags the complex out.
  EXPECT_TRUE(HasComplexInput(args));
  // HasComplexNonOutputInput skips the write-aliased out param.
  EXPECT_FALSE(HasComplexNonOutputInput(args, ComplexOutSchema()));
}

TEST(HasComplexNonOutputInputTest, ComplexInput) {
  // Complex in the real input position — should be caught.
  auto real = torch::ones({2, 3}, torch::kComplexFloat);
  auto imag = torch::zeros({2, 3});
  auto out = torch::zeros({2, 3}, torch::kComplexFloat);
  std::vector<c10::IValue> args = {real, imag, out};
  EXPECT_TRUE(HasComplexNonOutputInput(args, ComplexOutSchema()));
}

TEST(HasComplexNonOutputInputTest, NoOutParamFallsBackToHasComplex) {
  // aten::add has no out= param in this schema variant — every tensor
  // is an input. Complex anywhere should be caught.
  auto a = torch::ones({2, 3}, torch::kComplexFloat);
  auto b = torch::zeros({2, 3});
  std::vector<c10::IValue> args = {a, b, 1};
  EXPECT_TRUE(HasComplexNonOutputInput(args, AddSchema()));
}

TEST(HasComplexNonOutputInputTest, EmptyArgs) {
  EXPECT_FALSE(HasComplexNonOutputInput({}, AddSchema()));
}

// ═══════════════════════════════════════════════════════════════
//  RaiseComplexFallbackDisabled — unconditional noreturn helper
// ═══════════════════════════════════════════════════════════════

TEST(RaiseComplexFallbackDisabledTest, ThrowsWithExpectedMessage) {
  try {
    RaiseComplexFallbackDisabled("aten::some_op");
    FAIL() << "Expected RaiseComplexFallbackDisabled to throw";
  } catch (const c10::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("CPU fallback disabled"), std::string::npos);
    EXPECT_NE(msg.find("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1"), std::string::npos);
    EXPECT_NE(msg.find("complex-dtype input"), std::string::npos);
    EXPECT_NE(msg.find("aten::some_op"), std::string::npos);
  }
}

// ═══════════════════════════════════════════════════════════════
//  IsSupportedDtype
// ═══════════════════════════════════════════════════════════════

TEST(IsSupportedDtypeTest, Supported) {
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Float));
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Half));
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::BFloat16));
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Int));
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Bool));
}

TEST(IsSupportedDtypeTest, Unsupported) { EXPECT_FALSE(IsSupportedDtype(c10::ScalarType::Double)); }

TEST(IsSupportedDtypeTest, LongDependsOnInt64Mode) {
  // Long is "supported" (kept, not downcast in DowncastArgs) when it is handled
  // on-device: S64 to the compiler (native / native_strict) OR the packed view
  // (packed keeps logical int64 and binds its int32[..., 2] physical view at the
  // graph boundary). cpu_bounce is a CPU-roundtrip cast, so Long is unsupported.
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);

  SetInt64ModeForTesting(Int64Mode::kNative);
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Long));

  SetInt64ModeForTesting(Int64Mode::kNativeStrict);
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Long));

  SetInt64ModeForTesting(Int64Mode::kPacked);
  EXPECT_TRUE(IsSupportedDtype(c10::ScalarType::Long));

  SetInt64ModeForTesting(Int64Mode::kCpuBounce);
  EXPECT_FALSE(IsSupportedDtype(c10::ScalarType::Long));

  SetInt64ModeForTesting(Int64Mode::kNative);  // restore default
}

// ═══════════════════════════════════════════════════════════════
//  DowncastArgs — tests with CPU tensors (NeuronSafeCast on CPU
//  uses direct .to(), no device round-trip needed)
// ═══════════════════════════════════════════════════════════════

TEST(DowncastArgsTest, Float64ToFloat32) {
  auto t = torch::ones({2, 3}, torch::kFloat64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kFloat64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Float);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Float);
}

TEST(DowncastArgsTest, Int64KeptUnderPackedMode) {
  // In packed mode, int64 operands are kept Long (packed on-device via the
  // graph-boundary physical view), not downcast to int32. (Default is now
  // native, so packed must be selected explicitly.)
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);
  SetInt64ModeForTesting(Int64Mode::kPacked);
  auto t = torch::ones({2, 3}, torch::kInt64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kInt64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Long);
  SetInt64ModeForTesting(Int64Mode::kNative);
}

TEST(DowncastArgsTest, Int64ToInt32_CpuBounce) {
  // In cpu_bounce mode, int64 is downcast to int32 at the DowncastArgs layer
  // (via a CPU roundtrip in NeuronSafeCast on device).
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);
  SetInt64ModeForTesting(Int64Mode::kCpuBounce);

  auto t = torch::ones({2, 3}, torch::kInt64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kInt64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Int);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Int);
  SetInt64ModeForTesting(Int64Mode::kNative);
}

TEST(DowncastArgsTest, Int64Preserved_NativeMode) {
  // In native mode, int64 to a NATIVE-listed op reaches the compiler unchanged
  // (S64 passthrough). DowncastArgs is op-aware, so add must be registered.
  std::unordered_set<std::string> native = {"aten::add"};
  std::unordered_set<std::string> empty;
  InitInt64Policy(native, empty);
  SetInt64ModeForTesting(Int64Mode::kNative);

  auto t = torch::ones({2, 3}, torch::kInt64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kInt64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Long);
}

TEST(DowncastArgsTest, Int64KeptInNativeMode_EvenForNonNativeOp) {
  // DowncastArgs is op-unaware: in native mode int64 is kept (IsSupportedDtype
  // (Long) is true) whether or not the op is NATIVE. The per-op choice of S64
  // vs the packed int32[..., 2] view is made downstream in FlattenAndFilterInputs
  // (UsePackedInt64ViewForOp), not here.
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);  // add not registered → non-native
  SetInt64ModeForTesting(Int64Mode::kNative);

  auto t = torch::ones({2, 3}, torch::kInt64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kInt64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Long);
}

TEST(DowncastArgsTest, Int64Preserved_NativeStrictMode) {
  // In native_strict mode, int64 also reaches the compiler unchanged for
  // ops that are NATIVE; for non-native ops the dispatcher's CPU-fallback
  // gate fires before DowncastArgs, so this layer never has to differ from
  // native mode. DowncastArgs is op-aware, so add must be registered.
  std::unordered_set<std::string> native = {"aten::add"};
  std::unordered_set<std::string> empty;
  InitInt64Policy(native, empty);
  SetInt64ModeForTesting(Int64Mode::kNativeStrict);

  auto t = torch::ones({2, 3}, torch::kInt64);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kInt64), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Long);
}

TEST(DowncastArgsTest, Float64DowncastIndependentOfInt64Mode) {
  // Float64 → Float32 must happen regardless of int64 mode (ensures the
  // refactor didn't accidentally couple Float64 handling to int64 state).
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);
  for (auto m :
       {Int64Mode::kNative, Int64Mode::kNativeStrict, Int64Mode::kPacked, Int64Mode::kCpuBounce}) {
    SetInt64ModeForTesting(m);
    auto t = torch::ones({2, 3}, torch::kFloat64);
    std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kFloat64), 1};
    DowncastArgs(args, AddSchema());
    EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Float)
        << "Float64 should always downcast; mode=" << static_cast<int>(m);
    EXPECT_EQ(args[1].toTensor().scalar_type(), c10::ScalarType::Float)
        << "Float64 should always downcast; mode=" << static_cast<int>(m);
  }
  SetInt64ModeForTesting(Int64Mode::kNative);
}

TEST(DowncastArgsTest, Int64TensorListDowncast_CpuBounce) {
  // _foreach_add(Tensor[] self, Tensor[] other, ...) — the TensorList path
  // must also honor the mode for int64.
  auto schema =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::_foreach_add", "List").schema();

  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);
  SetInt64ModeForTesting(Int64Mode::kCpuBounce);

  c10::List<at::Tensor> self_list(
      {torch::ones({2}, torch::kInt64), torch::ones({2}, torch::kInt64)});
  c10::List<at::Tensor> other_list(
      {torch::ones({2}, torch::kInt64), torch::ones({2}, torch::kInt64)});
  std::vector<c10::IValue> args = {c10::IValue(self_list), c10::IValue(other_list)};
  DowncastArgs(args, schema);

  auto self_after = args[0].toTensorList();
  for (size_t j = 0; j < self_after.size(); ++j) {
    EXPECT_EQ(self_after.get(j).scalar_type(), c10::ScalarType::Int);
  }
  auto other_after = args[1].toTensorList();
  for (size_t j = 0; j < other_after.size(); ++j) {
    EXPECT_EQ(other_after.get(j).scalar_type(), c10::ScalarType::Int);
  }
}

TEST(DowncastArgsTest, Int64TensorListPreserved_Native) {
  auto schema =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::_foreach_add", "List").schema();

  std::unordered_set<std::string> native = {"aten::_foreach_add"};
  std::unordered_set<std::string> empty;
  InitInt64Policy(native, empty);
  SetInt64ModeForTesting(Int64Mode::kNative);

  c10::List<at::Tensor> self_list({torch::ones({2}, torch::kInt64)});
  c10::List<at::Tensor> other_list({torch::ones({2}, torch::kInt64)});
  std::vector<c10::IValue> args = {c10::IValue(self_list), c10::IValue(other_list)};
  DowncastArgs(args, schema);

  EXPECT_EQ(args[0].toTensorList().get(0).scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(args[1].toTensorList().get(0).scalar_type(), c10::ScalarType::Long);
}

TEST(DowncastArgsTest, Int64DtypeKwarg_CpuBounce) {
  // aten::sum.dim_IntList has a `ScalarType? dtype` kwarg. In cpu_bounce
  // mode, an int64 dtype request should be rewritten to int32 alongside the
  // tensor inputs.
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);
  SetInt64ModeForTesting(Int64Mode::kCpuBounce);

  auto t = torch::ones({2, 3}, torch::kFloat32);
  // sum.dim_IntList: (Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None)
  std::vector<c10::IValue> args = {
      t,
      c10::IValue(std::vector<int64_t>{0}),
      false,
      c10::IValue(static_cast<int64_t>(c10::ScalarType::Long)),
  };
  DowncastArgs(args, SumSchema());
  EXPECT_EQ(static_cast<c10::ScalarType>(args[3].toInt()), c10::ScalarType::Int);
}

TEST(DowncastArgsTest, Int64DtypeKwargPreserved_Native) {
  std::unordered_set<std::string> native = {"aten::sum"};
  std::unordered_set<std::string> empty;
  InitInt64Policy(native, empty);
  SetInt64ModeForTesting(Int64Mode::kNative);

  auto t = torch::ones({2, 3}, torch::kFloat32);
  std::vector<c10::IValue> args = {
      t,
      c10::IValue(std::vector<int64_t>{0}),
      false,
      c10::IValue(static_cast<int64_t>(c10::ScalarType::Long)),
  };
  DowncastArgs(args, SumSchema());
  EXPECT_EQ(static_cast<c10::ScalarType>(args[3].toInt()), c10::ScalarType::Long);
}

TEST(DowncastArgsTest, WriteAliasedInt64_CpuBounceDowncasts_PackedKeeps) {
  // copy_(Tensor(a!) self, Tensor src, bool non_blocking) — `self` is
  // write-aliased. DowncastArgs keys only off IsSupportedDtype:
  //   - cpu_bounce: int64 is unsupported → demoted to int32 here (the alias
  //     contract is preserved via the original_out_tensors snapshot +
  //     UpcastExecutionOutputs/CopyBackToOriginal path).
  //   - packed: int64 is "supported" (kept logical; its int32[..., 2] physical
  //     view is bound at the graph boundary), so it is NOT downcast here.
  std::unordered_set<std::string> empty;
  InitInt64Policy(empty, empty);

  SetInt64ModeForTesting(Int64Mode::kCpuBounce);
  {
    auto self = torch::ones({2, 3}, torch::kInt64);
    auto src = torch::ones({2, 3}, torch::kInt32);
    std::vector<c10::IValue> args = {self, src, false};
    DowncastArgs(args, CopySchema());
    EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Int)
        << "cpu_bounce: write-aliased int64 is demoted to int32";
  }

  SetInt64ModeForTesting(Int64Mode::kPacked);
  {
    auto self = torch::ones({2, 3}, torch::kInt64);
    auto src = torch::ones({2, 3}, torch::kInt32);
    std::vector<c10::IValue> args = {self, src, false};
    DowncastArgs(args, CopySchema());
    EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Long)
        << "packed: int64 is kept (physical view bound at the boundary)";
  }
  SetInt64ModeForTesting(Int64Mode::kNative);
}

TEST(DowncastArgsTest, SupportedDtypeUnchanged) {
  auto t = torch::ones({2, 3}, torch::kFloat32);
  std::vector<c10::IValue> args = {t, torch::ones({2, 3}, torch::kFloat32), 1};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[0].toTensor().scalar_type(), c10::ScalarType::Float);
}

TEST(DowncastArgsTest, UndefinedTensorSkipped) {
  std::vector<c10::IValue> args = {c10::IValue(torch::Tensor()), torch::zeros({2}), 1};
  EXPECT_NO_THROW(DowncastArgs(args, AddSchema()));
}

TEST(DowncastArgsTest, NonTensorArgsUnchanged) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::zeros({2}), 42};
  DowncastArgs(args, AddSchema());
  EXPECT_EQ(args[2].toInt(), 42);
}

TEST(DowncastArgsTest, PreservesValues) {
  auto t = torch::tensor({1.5, 2.5, 3.5}, torch::kFloat64);
  std::vector<c10::IValue> args = {t, torch::zeros({3}, torch::kFloat64), 1};
  DowncastArgs(args, AddSchema());
  auto result = args[0].toTensor();
  EXPECT_EQ(result.scalar_type(), c10::ScalarType::Float);
  EXPECT_NEAR(result[0].item<float>(), 1.5f, 1e-6);
  EXPECT_NEAR(result[2].item<float>(), 3.5f, 1e-6);
}

// ═══════════════════════════════════════════════════════════════
//  SnapshotOutTensors
// ═══════════════════════════════════════════════════════════════

TEST(SnapshotOutTensorsTest, InplaceOpCapturesSelf) {
  // copy_ schema: Tensor(a!) self, Tensor src, bool non_blocking → Tensor(a!)
  auto self = torch::zeros({2, 3});
  auto src = torch::ones({2, 3});
  std::vector<c10::IValue> args = {self, src, false};
  auto result = SnapshotOutTensors(CopySchema(), args);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].first, 0u);  // return index 0
  EXPECT_EQ(result[0].second.data_ptr(), self.data_ptr());
}

TEST(SnapshotOutTensorsTest, NonInplaceReturnsEmpty) {
  // add schema has no write aliases
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({2}), 1};
  auto result = SnapshotOutTensors(AddSchema(), args);
  EXPECT_TRUE(result.empty());
}

TEST(SnapshotOutTensorsTest, UndefinedTensorSkipped) {
  std::vector<c10::IValue> args = {c10::IValue(torch::Tensor()), torch::ones({2}), false};
  auto result = SnapshotOutTensors(CopySchema(), args);
  EXPECT_TRUE(result.empty());
}

// ═══════════════════════════════════════════════════════════════
//  HasExplicitDtypeKwarg
// ═══════════════════════════════════════════════════════════════

TEST(HasExplicitDtypeKwargTest, SumWithDtype) {
  auto schema = SumSchema();
  // sum.dim_IntList(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)
  auto self = torch::zeros({2, 3});
  auto dim = c10::IValue(c10::List<int64_t>({1}));
  auto keepdim = c10::IValue(false);
  auto dtype = c10::IValue(static_cast<int64_t>(c10::ScalarType::Float));

  std::vector<c10::IValue> args = {self, dim, keepdim, dtype};
  EXPECT_TRUE(HasExplicitDtypeKwarg(schema, args));
}

TEST(HasExplicitDtypeKwargTest, SumWithoutDtype) {
  auto schema = SumSchema();
  auto self = torch::zeros({2, 3});
  auto dim = c10::IValue(c10::List<int64_t>({1}));
  auto keepdim = c10::IValue(false);
  auto dtype = c10::IValue();  // None

  std::vector<c10::IValue> args = {self, dim, keepdim, dtype};
  EXPECT_FALSE(HasExplicitDtypeKwarg(schema, args));
}

TEST(HasExplicitDtypeKwargTest, NoDtypeKwargInSchema) {
  // aten::add has no dtype kwarg
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::zeros({2}), 1};
  EXPECT_FALSE(HasExplicitDtypeKwarg(AddSchema(), args));
}

// ═══════════════════════════════════════════════════════════════
//  HasComplexInput
// ═══════════════════════════════════════════════════════════════

TEST(HasComplexInputTest, NoComplex) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({4})};
  EXPECT_FALSE(HasComplexInput(args));
}

TEST(HasComplexInputTest, HasComplex64) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}, torch::kComplexFloat)};
  EXPECT_TRUE(HasComplexInput(args));
}

TEST(HasComplexInputTest, HasComplex128) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}, torch::kComplexDouble)};
  EXPECT_TRUE(HasComplexInput(args));
}

TEST(HasComplexInputTest, MixedRealAndComplex) {
  std::vector<c10::IValue> args = {torch::zeros({2}, torch::kFloat32),
                                   torch::zeros({2}, torch::kComplexFloat)};
  EXPECT_TRUE(HasComplexInput(args));
}

TEST(HasComplexInputTest, EmptyArgs) { EXPECT_FALSE(HasComplexInput({})); }

TEST(HasComplexInputTest, NonTensorArgsOnly) {
  std::vector<c10::IValue> args = {c10::IValue(42), c10::IValue(3.14)};
  EXPECT_FALSE(HasComplexInput(args));
}

TEST(HasComplexInputTest, ComplexInTensorList) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::zeros({3}, torch::kComplexFloat)});
  std::vector<c10::IValue> args = {c10::IValue(tl)};
  EXPECT_TRUE(HasComplexInput(args));
}

// ═══════════════════════════════════════════════════════════════
//  ExtractBaseOpName
// ═══════════════════════════════════════════════════════════════

TEST(ExtractBaseOpNameTest, FullyQualified) {
  EXPECT_EQ(ExtractBaseOpName("aten::sum.dim_IntList"), "sum");
}

TEST(ExtractBaseOpNameTest, WithoutNamespace) { EXPECT_EQ(ExtractBaseOpName("add.Tensor"), "add"); }

TEST(ExtractBaseOpNameTest, PlainName) { EXPECT_EQ(ExtractBaseOpName("relu"), "relu"); }

TEST(ExtractBaseOpNameTest, InplaceOp) { EXPECT_EQ(ExtractBaseOpName("aten::copy_"), "copy_"); }

TEST(ExtractBaseOpNameTest, EmptyString) { EXPECT_EQ(ExtractBaseOpName(""), ""); }

TEST(ExtractBaseOpNameTest, CaseNormalization) {
  // ExtractBaseOpName lowercases
  EXPECT_EQ(ExtractBaseOpName("aten::SUM.dim_IntList"), "sum");
}

// ═══════════════════════════════════════════════════════════════
//  GetReductionConfig
// ═══════════════════════════════════════════════════════════════

TEST(GetReductionConfigTest, SumIsIdentityZero) {
  auto* cfg = GetReductionConfig("aten::sum.dim_IntList");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->fill_type, ReductionConfig::FillType::kIdentity);
  EXPECT_DOUBLE_EQ(cfg->identity_value, 0.0);
}

TEST(GetReductionConfigTest, ProdIsIdentityOne) {
  auto* cfg = GetReductionConfig("prod");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->fill_type, ReductionConfig::FillType::kIdentity);
  EXPECT_DOUBLE_EQ(cfg->identity_value, 1.0);
}

TEST(GetReductionConfigTest, MeanIsNaN) {
  auto* cfg = GetReductionConfig("aten::mean");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->fill_type, ReductionConfig::FillType::kNaN);
}

TEST(GetReductionConfigTest, MaxIsError) {
  auto* cfg = GetReductionConfig("aten::max");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->fill_type, ReductionConfig::FillType::kError);
}

TEST(GetReductionConfigTest, UnknownOpReturnsNull) {
  EXPECT_EQ(GetReductionConfig("aten::matmul"), nullptr);
}

TEST(GetReductionConfigTest, AllKnownReductions) {
  // Verify all documented reductions are present
  EXPECT_NE(GetReductionConfig("sum"), nullptr);
  EXPECT_NE(GetReductionConfig("prod"), nullptr);
  EXPECT_NE(GetReductionConfig("all"), nullptr);
  EXPECT_NE(GetReductionConfig("any"), nullptr);
  EXPECT_NE(GetReductionConfig("norm"), nullptr);
  EXPECT_NE(GetReductionConfig("mean"), nullptr);
  EXPECT_NE(GetReductionConfig("std"), nullptr);
  EXPECT_NE(GetReductionConfig("var"), nullptr);
  EXPECT_NE(GetReductionConfig("max"), nullptr);
  EXPECT_NE(GetReductionConfig("min"), nullptr);
  EXPECT_NE(GetReductionConfig("amax"), nullptr);
  EXPECT_NE(GetReductionConfig("amin"), nullptr);
  EXPECT_NE(GetReductionConfig("argmax"), nullptr);
  EXPECT_NE(GetReductionConfig("argmin"), nullptr);
}

// ═══════════════════════════════════════════════════════════════
//  GetIndexOpConfig
// ═══════════════════════════════════════════════════════════════

TEST(GetIndexOpConfigTest, ScatterConfig) {
  auto* cfg = GetIndexOpConfig("scatter");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->index_argnum, 2);
  EXPECT_EQ(cfg->input_argnum, 0);
  EXPECT_EQ(cfg->source_argnum, 3);
}

TEST(GetIndexOpConfigTest, ScatterAddConfig) {
  auto* cfg = GetIndexOpConfig("scatter_add");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->index_argnum, 2);
  EXPECT_EQ(cfg->input_argnum, 0);
  EXPECT_EQ(cfg->source_argnum, 3);
}

TEST(GetIndexOpConfigTest, IndexPutImplConfig) {
  auto* cfg = GetIndexOpConfig("aten::_index_put_impl_");
  ASSERT_NE(cfg, nullptr);
  EXPECT_EQ(cfg->index_argnum, 1);
  EXPECT_EQ(cfg->input_argnum, 0);
  EXPECT_EQ(cfg->source_argnum, 2);
}

// The overload is stripped before lookup, so `scatter.value_out` shares
// `scatter`'s config. Its source argument is a Scalar rather than a tensor,
// which is what keeps an empty `out=` from being read as an empty source.
TEST(GetIndexOpConfigTest, ScatterOverloadsShareTheBaseConfig) {
  for (const char* name : {"scatter.value_out", "scatter.src_out", "scatter.value_reduce_out"}) {
    auto* cfg = GetIndexOpConfig(name);
    ASSERT_NE(cfg, nullptr) << name;
    EXPECT_EQ(cfg->source_argnum, 3) << name;
  }
}

TEST(GetIndexOpConfigTest, UnknownOpReturnsNull) { EXPECT_EQ(GetIndexOpConfig("matmul"), nullptr); }

// ═══════════════════════════════════════════════════════════════
//  IsIndexOp
// ═══════════════════════════════════════════════════════════════

TEST(IsIndexOpTest, KnownIndexOps) {
  EXPECT_TRUE(IsIndexOp("aten::index_copy"));
  EXPECT_TRUE(IsIndexOp("index_copy_"));
  EXPECT_TRUE(IsIndexOp("aten::index_add"));
  EXPECT_TRUE(IsIndexOp("aten::index_select"));
  EXPECT_TRUE(IsIndexOp("aten::gather"));
  EXPECT_TRUE(IsIndexOp("aten::put"));
  EXPECT_TRUE(IsIndexOp("put_"));
}

TEST(IsIndexOpTest, NonIndexOps) {
  EXPECT_FALSE(IsIndexOp("aten::add"));
  EXPECT_FALSE(IsIndexOp("aten::sum"));
  EXPECT_FALSE(IsIndexOp("matmul"));
}

TEST(IsIndexOpTest, IndexOpConfigOpsAreNotIsIndexOp) {
  // scatter, scatter_add, _index_put_impl_ have their own handling
  EXPECT_FALSE(IsIndexOp("scatter"));
  EXPECT_FALSE(IsIndexOp("scatter_add"));
  EXPECT_FALSE(IsIndexOp("_index_put_impl_"));
}

// ═══════════════════════════════════════════════════════════════
//  TensorToMeta
// ═══════════════════════════════════════════════════════════════

TEST(TensorToMetaTest, PreservesShapeAndDtype) {
  auto t = torch::zeros({3, 4, 5}, torch::kFloat32);
  auto meta = TensorToMeta(t);
  EXPECT_EQ(meta.sizes(), t.sizes());
  EXPECT_EQ(meta.scalar_type(), c10::ScalarType::Float);
  EXPECT_TRUE(meta.device().is_meta());
}

TEST(TensorToMetaTest, PreservesStrides) {
  auto t = torch::zeros({3, 4}).t();  // transposed → non-contiguous
  auto meta = TensorToMeta(t);
  EXPECT_EQ(meta.strides(), t.strides());
}

TEST(TensorToMetaTest, UndefinedReturnsUndefined) {
  auto t = torch::Tensor();
  auto meta = TensorToMeta(t);
  EXPECT_FALSE(meta.defined());
}

TEST(TensorToMetaTest, ZeroDim) {
  auto t = torch::tensor(5.0);
  auto meta = TensorToMeta(t);
  EXPECT_EQ(meta.dim(), 0);
  EXPECT_TRUE(meta.device().is_meta());
}

TEST(TensorToMetaTest, DifferentDtypes) {
  EXPECT_EQ(TensorToMeta(torch::zeros({2}, torch::kInt64)).scalar_type(), c10::ScalarType::Long);
  EXPECT_EQ(TensorToMeta(torch::zeros({2}, torch::kBool)).scalar_type(), c10::ScalarType::Bool);
  EXPECT_EQ(TensorToMeta(torch::zeros({2}, torch::kHalf)).scalar_type(), c10::ScalarType::Half);
}

// ═══════════════════════════════════════════════════════════════
//  IValueToMeta
// ═══════════════════════════════════════════════════════════════

TEST(IValueToMetaTest, TensorArg) {
  auto val = c10::IValue(torch::zeros({2, 3}));
  auto meta_val = IValueToMeta(val);
  ASSERT_TRUE(meta_val.isTensor());
  EXPECT_TRUE(meta_val.toTensor().device().is_meta());
  EXPECT_EQ(meta_val.toTensor().sizes(), (std::vector<int64_t>{2, 3}));
}

TEST(IValueToMetaTest, TensorListArg) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::ones({3})});
  auto meta_val = IValueToMeta(c10::IValue(tl));
  ASSERT_TRUE(meta_val.isTensorList());
  auto meta_tl = meta_val.toTensorList();
  EXPECT_EQ(meta_tl.size(), 2u);
  at::Tensor t0 = meta_tl[0];
  EXPECT_TRUE(t0.device().is_meta());
}

TEST(IValueToMetaTest, NonTensorPassthrough) {
  auto int_val = c10::IValue(42);
  auto result = IValueToMeta(int_val);
  EXPECT_TRUE(result.isInt());
  EXPECT_EQ(result.toInt(), 42);
}

TEST(IValueToMetaTest, NonePassthrough) {
  auto none_val = c10::IValue();
  auto result = IValueToMeta(none_val);
  EXPECT_TRUE(result.isNone());
}

TEST(IValueToMetaTest, GenericListWithTensors) {
  c10::List<c10::IValue> gl(c10::AnyType::get());
  gl.push_back(c10::IValue(torch::zeros({2})));
  gl.push_back(c10::IValue(42));
  auto result = IValueToMeta(c10::IValue(gl));
  ASSERT_TRUE(result.isList());
  auto list = result.toListRef();
  EXPECT_TRUE(list[0].toTensor().device().is_meta());
  EXPECT_EQ(list[1].toInt(), 42);
}

// ═══════════════════════════════════════════════════════════════
//  RawNrtCopyOverlaps — byte-range overlap test for shared storage
// ═══════════════════════════════════════════════════════════════

TEST(RawNrtCopyOverlapsTest, DifferentStorageNeverOverlaps) {
  auto a = torch::zeros({100}, torch::kFloat32);
  auto b = torch::zeros({100}, torch::kFloat32);
  EXPECT_FALSE(RawNrtCopyOverlaps(a, b));
}

TEST(RawNrtCopyOverlapsTest, IdenticalView) {
  auto base = torch::zeros({100}, torch::kFloat32);
  EXPECT_TRUE(RawNrtCopyOverlaps(base, base));
}

TEST(RawNrtCopyOverlapsTest, AdjacentViewsDoNotOverlap) {
  auto base = torch::zeros({100}, torch::kFloat32);
  auto a = base.slice(0, 0, 50);    // [0, 50)
  auto b = base.slice(0, 50, 100);  // [50, 100)
  EXPECT_FALSE(RawNrtCopyOverlaps(a, b));
  EXPECT_FALSE(RawNrtCopyOverlaps(b, a));
}

TEST(RawNrtCopyOverlapsTest, PartialOverlap) {
  auto base = torch::zeros({100}, torch::kFloat32);
  auto a = base.slice(0, 10, 50);  // [10, 50)
  auto b = base.slice(0, 30, 80);  // [30, 80)
  EXPECT_TRUE(RawNrtCopyOverlaps(a, b));
  EXPECT_TRUE(RawNrtCopyOverlaps(b, a));
}

TEST(RawNrtCopyOverlapsTest, ContainedView) {
  auto base = torch::zeros({100}, torch::kFloat32);
  auto outer = base.slice(0, 10, 80);  // [10, 80)
  auto inner = base.slice(0, 30, 50);  // [30, 50)
  EXPECT_TRUE(RawNrtCopyOverlaps(outer, inner));
  EXPECT_TRUE(RawNrtCopyOverlaps(inner, outer));
}

TEST(RawNrtCopyOverlapsTest, ExactBoundaryDoesNotOverlap) {
  // Half-open intervals: [0, 50) and [50, 100) touch but do not overlap.
  auto base = torch::zeros({100}, torch::kFloat32);
  auto a = base.slice(0, 0, 50);
  auto b = base.slice(0, 50, 100);
  EXPECT_FALSE(RawNrtCopyOverlaps(a, b));
}

TEST(RawNrtCopyOverlapsTest, DifferentDtypesSameStorage) {
  // Bool view of fp32 storage via storage-sharing.
  auto fp32 = torch::zeros({10}, torch::kFloat32);  // 40 bytes
  auto i8_view = torch::empty({40}, torch::kInt8);
  i8_view.set_(fp32.storage(), 0, {40}, {1});
  // fp32[0:5] occupies bytes [0, 20); i8_view[10:30] occupies bytes [10, 30) —
  // overlap on [10, 20).
  auto fp32_slice = fp32.slice(0, 0, 5);
  auto i8_slice = i8_view.slice(0, 10, 30);
  EXPECT_TRUE(RawNrtCopyOverlaps(fp32_slice, i8_slice));
}

TEST(RawNrtCopyOverlapsTest, EmptyTensorsDoNotOverlap) {
  auto base = torch::zeros({100}, torch::kFloat32);
  auto empty = base.slice(0, 50, 50);  // numel == 0
  auto nonempty = base.slice(0, 0, 100);
  EXPECT_FALSE(RawNrtCopyOverlaps(empty, nonempty));
  EXPECT_FALSE(RawNrtCopyOverlaps(nonempty, empty));
}

TEST(RawNrtCopyOverlapsTest, NonZeroStorageOffsets) {
  auto base = torch::zeros({200}, torch::kFloat32);
  auto a = base.slice(0, 50, 100);  // storage_offset=50
  auto b = base.slice(0, 75, 125);  // storage_offset=75; overlap [75, 100)
  EXPECT_TRUE(RawNrtCopyOverlaps(a, b));

  auto c = base.slice(0, 100, 150);  // storage_offset=100; no overlap with a
  EXPECT_FALSE(RawNrtCopyOverlaps(a, c));
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
