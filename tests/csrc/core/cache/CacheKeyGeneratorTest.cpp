/**
 * Unit tests for CacheKeyGenerator.
 *
 * Tests cover: Generate() (IValue path), kwarg filtering, dtype auto-static
 * detection, static_argnames, all IValue types, list encoding, and edge cases.
 *
 * GenerateFromPython() parity tests are in a separate section and require
 * Python interpreter initialization (provided by :neuron_bindings dep).
 */

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "torch_neuronx/csrc/core/cache/CacheKeyGenerator.h"

using namespace at::neuron;

// ═══════════════════════════════════════════════════════════════
//  Suite 1: Direct Generate() tests (IValue path)
// ═══════════════════════════════════════════════════════════════

class CacheKeyGeneratorTest : public ::testing::Test {
 protected:
  // Use aten::add.Tensor as a representative schema:
  //   add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  c10::FunctionSchema schema() {
    return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
  }

  std::string gen(const std::vector<c10::IValue>& args,
                  const std::unordered_set<size_t>& static_argnums = {},
                  const std::unordered_set<std::string>& static_argnames = {}) {
    return CacheKeyGenerator::Generate("add.Tensor", schema(), args, static_argnums,
                                       static_argnames);
  }
};

// --- Tensor encoding ---

TEST_F(CacheKeyGeneratorTest, TensorShapeInKey) {
  auto k = gen({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  EXPECT_NE(k.find("[2,3]"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, TensorDtypeUsesToString) {
  auto k = gen({torch::zeros({2}, torch::kFloat32), torch::zeros({2}), 1});
  EXPECT_NE(k.find("Float"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, TensorStridesInKey) {
  auto k = gen({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  EXPECT_NE(k.find("S(C)"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, DifferentShapesDifferentKeys) {
  auto a = gen({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  auto b = gen({torch::zeros({4, 5}), torch::zeros({4, 5}), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, SameShapesSameKey) {
  auto a = gen({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  auto b = gen({torch::ones({2, 3}), torch::ones({2, 3}), 1});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyGeneratorTest, DifferentDtypesDifferentKeys) {
  auto a = gen({torch::zeros({2}, torch::kFloat32), torch::zeros({2}, torch::kFloat32), 1});
  auto b = gen({torch::zeros({2}, torch::kFloat64), torch::zeros({2}, torch::kFloat64), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, DifferentStridesDifferentKeys) {
  auto a = gen({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  auto b = gen({torch::zeros({3, 2}).t(), torch::zeros({3, 2}).t(), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, ZeroDimTensor) {
  auto k = gen({torch::tensor(5.0), torch::zeros({2}), 1});
  EXPECT_NE(k.find("[]"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, EmptyTensor) {
  auto k = gen({torch::zeros({0, 3}), torch::zeros({2}), 1});
  EXPECT_NE(k.find("[0,3]"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, UndefinedTensor) {
  auto k = gen({torch::zeros({2}), c10::IValue(torch::Tensor()), 1});
  EXPECT_NE(k.find("undef"), std::string::npos);
}

// --- Scalar encoding (static vs non-static) ---
// Note: IValue(c10::Scalar(1.0)) decomposes to Tag::Double, so these tests
// exercise the isInt()/isDouble() branches, not the isScalar() branch.
// The isScalar() branch is only reachable for symbolic scalars (SymInt/SymFloat).

TEST_F(CacheKeyGeneratorTest, NonStaticIntSameKey) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), 1});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), 2});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyGeneratorTest, StaticIntDifferentKeys) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), 1}, {2});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), 2}, {2});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, NonStaticDoubleSameKey) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), 1.5});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), 2.5});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyGeneratorTest, StaticDoubleDifferentKeys) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), 1.5}, {2});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), 2.5}, {2});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, BoolValueAware) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), true});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), false});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, StringValueAware) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(std::string("abc"))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(std::string("xyz"))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, NoneArg) {
  auto k = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue()});
  EXPECT_NE(k.find("None"), std::string::npos);
}

// --- Scalar type-kind distinction (via decomposed Int/Double tags) ---

TEST_F(CacheKeyGeneratorTest, NonStaticFloatScalarSameValueSameKey) {
  // Both are Double IValues (Scalar(1.0) decomposes to Tag::Double)
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(1.0))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyGeneratorTest, NonStaticDifferentScalarTypesDifferentKey) {
  // Scalar(2.0) -> Tag::Double, Scalar(2) -> Tag::Int
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, StaticScalarDifferentKeys) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(1.0))}, {2});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))}, {2});
  EXPECT_NE(a, b);
}

// --- ComplexDouble ---

TEST_F(CacheKeyGeneratorTest, ComplexDoubleStaticIncludesValue) {
  auto a =
      gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::complex<double>(1.0, 2.0))}, {2});
  auto b =
      gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::complex<double>(3.0, 4.0))}, {2});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, ComplexDoubleNonStaticSameKey) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::complex<double>(1.0, 2.0))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::complex<double>(3.0, 4.0))});
  EXPECT_EQ(a, b);
}

// --- Lists ---

TEST_F(CacheKeyGeneratorTest, IntListValueAware) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<int64_t>({1, 2}))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<int64_t>({3, 4}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, BoolListValueAware) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<bool>({true, false}))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<bool>({false, true}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, DoubleListValueAware) {
  auto a = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<double>({1.0, 2.0}))});
  auto b = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<double>({3.0, 4.0}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, TensorListDifferentShapes) {
  c10::List<at::Tensor> tl1({torch::zeros({2})});
  c10::List<at::Tensor> tl2({torch::zeros({3})});
  auto a = gen({c10::IValue(tl1), c10::IValue(tl1), 1});
  auto b = gen({c10::IValue(tl2), c10::IValue(tl2), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, TensorListWithUndefined) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::Tensor()});
  auto k = gen({c10::IValue(tl), c10::IValue(tl), 1});
  EXPECT_NE(k.find("undef"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, EmptyTensorList) {
  c10::List<at::Tensor> tl;
  auto k = gen({c10::IValue(tl), torch::zeros({2}), 1});
  EXPECT_NE(k.find("TL[]"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, GenericListEncoding) {
  c10::List<c10::IValue> gl(c10::AnyType::get());
  gl.push_back(c10::IValue(42));
  gl.push_back(c10::IValue(std::string("hello")));
  auto k = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(gl)});
  EXPECT_NE(k.find("GL["), std::string::npos);
}

// --- Special types ---

TEST_F(CacheKeyGeneratorTest, DeviceArg) {
  auto k = gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(torch::Device(torch::kCPU))});
  EXPECT_NE(k.find("dev:cpu"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, DifferentDevicesDifferentKeys) {
  auto a = gen({torch::zeros({2}), c10::IValue(torch::Device(torch::kCPU))});
  auto b = gen({torch::zeros({2}), c10::IValue(torch::Device(c10::DeviceType::PrivateUse1, 0))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, GeneratorArg) {
  auto k =
      gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(at::detail::createCPUGenerator())});
  EXPECT_NE(k.find("gen"), std::string::npos);
}

// --- Op name and stability ---

TEST_F(CacheKeyGeneratorTest, OpNameInKey) {
  auto k = gen({torch::zeros({2}), torch::zeros({2}), 1});
  EXPECT_NE(k.find("add.Tensor"), std::string::npos);
}

TEST_F(CacheKeyGeneratorTest, DifferentOpsProduceDifferentKeys) {
  auto s = schema();
  auto a = CacheKeyGenerator::Generate("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1});
  auto b = CacheKeyGenerator::Generate("sub.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, StabilityAcrossCalls) {
  auto a = gen({torch::zeros({4, 4}), torch::zeros({4, 4}), 1});
  auto b = gen({torch::zeros({4, 4}), torch::zeros({4, 4}), 1});
  EXPECT_EQ(a, b);
}

// --- Fallback ---

TEST_F(CacheKeyGeneratorTest, FallbackDoesNotCrash) {
  // Generator has its own handler, but this verifies the path doesn't crash
  auto k =
      gen({torch::zeros({2}), torch::zeros({2}), c10::IValue(at::detail::createCPUGenerator())});
  EXPECT_FALSE(k.empty());
}

// ═══════════════════════════════════════════════════════════════
//  Suite 2: Kwarg filtering and static detection
// ═══════════════════════════════════════════════════════════════

TEST(CacheKeyFilteredKwargsTest, FilteredKwargsContainsExpected) {
  const auto& filtered = CacheKeyGenerator::FilteredKwargs();
  EXPECT_TRUE(filtered.count("device"));
  EXPECT_TRUE(filtered.count("pin_memory"));
  EXPECT_TRUE(filtered.count("memory_format"));
  EXPECT_TRUE(filtered.count("layout"));
  EXPECT_TRUE(filtered.count("requires_grad"));
  EXPECT_TRUE(filtered.count("out"));
}

class CacheKeyDtypeAutoStaticTest : public ::testing::Test {
 protected:
  // aten::sum.dim_IntList:
  //   sum(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  c10::FunctionSchema schema() {
    return c10::Dispatcher::singleton().findSchemaOrThrow("aten::sum", "dim_IntList").schema();
  }
};

TEST_F(CacheKeyDtypeAutoStaticTest, DifferentDtypesDifferentKeys) {
  auto s = schema();
  auto args_f32 = std::vector<c10::IValue>{
      torch::zeros({2, 3}),
      c10::IValue(c10::List<int64_t>({1})),
      false,
      c10::IValue(static_cast<int64_t>(c10::ScalarType::Float)),
  };
  auto args_f16 = std::vector<c10::IValue>{
      torch::zeros({2, 3}),
      c10::IValue(c10::List<int64_t>({1})),
      false,
      c10::IValue(static_cast<int64_t>(c10::ScalarType::Half)),
  };
  auto k1 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args_f32);
  auto k2 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args_f16);
  EXPECT_NE(k1, k2);
}

TEST_F(CacheKeyDtypeAutoStaticTest, NoneDtypeSameKey) {
  auto s = schema();
  // Both have dtype=None — should produce same key regardless of static detection
  auto args1 = std::vector<c10::IValue>{
      torch::zeros({2, 3}), c10::IValue(c10::List<int64_t>({1})), false,
      c10::IValue(),  // None
  };
  auto args2 = std::vector<c10::IValue>{
      torch::zeros({2, 3}), c10::IValue(c10::List<int64_t>({1})), false,
      c10::IValue(),  // None
  };
  auto k1 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args1);
  auto k2 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args2);
  EXPECT_EQ(k1, k2);
}

TEST_F(CacheKeyDtypeAutoStaticTest, KeepdimBoolDiffers) {
  auto s = schema();
  auto args_true = std::vector<c10::IValue>{
      torch::zeros({2, 3}),
      c10::IValue(c10::List<int64_t>({0})),
      true,
      c10::IValue(),
  };
  auto args_false = std::vector<c10::IValue>{
      torch::zeros({2, 3}),
      c10::IValue(c10::List<int64_t>({0})),
      false,
      c10::IValue(),
  };
  auto k1 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args_true);
  auto k2 = CacheKeyGenerator::Generate("sum.dim_IntList", s, args_false);
  EXPECT_NE(k1, k2);
}

// ═══════════════════════════════════════════════════════════════
//  Suite 3: GenerateFromPython() tests
//
//  TODO: Moved to pytest (tests/cache/test_cache_key_generator.py) because
//  embedded Python in Bazel C++ tests requires environment-specific
//  PYTHONHOME/PYTHONPATH configuration that is fragile across CI envs.
//  The Python pybind path (GenerateFromPython, AppendPyObject) is tested
//  via _C._generate_cache_key() from Python pytest instead.
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
//  Suite: GenerateDynamicCollectiveKey (dynamic multi-tier collective cache key)
// ═══════════════════════════════════════════════════════════════

TEST_F(CacheKeyGeneratorTest, DynamicCollectiveKeyEncodesIdentityDims) {
  // Signature: (op_type, trace_numel, large, medium, small, replica_groups, dtype, unroll,
  // [reduce_op]).
  auto k = CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 1024, 512,
                                                           {{0, 1, 2, 3}}, c10::kBFloat16,
                                                           /*unroll=*/2);
  EXPECT_NE(k.find("allgather"), std::string::npos);
  EXPECT_NE(k.find("dynamic"), std::string::npos);
  EXPECT_NE(k.find("trace_numel=2048"), std::string::npos);
  EXPECT_NE(k.find("tiles=2048,1024,512"), std::string::npos);
  EXPECT_NE(k.find("|u=2"), std::string::npos);
  // Distinct across each identity dim (op_type / trace_numel / tiles / dtype / partition / unroll)
  auto rg = std::vector<std::vector<int>>{{0, 1, 2, 3}};
  EXPECT_NE(k, CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 128, 128, 64, 32, rg,
                                                               c10::kBFloat16, 2));
  EXPECT_NE(k, CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 1024, 512,
                                                               rg, c10::kFloat, 2));
  EXPECT_NE(k, CacheKeyGenerator::GenerateDynamicCollectiveKey(
                   "allgather", 2048, 2048, 1024, 512, {{0, 1}, {2, 3}}, c10::kBFloat16, 2));
  // op_type disambiguates the collective family.
  EXPECT_NE(k, CacheKeyGenerator::GenerateDynamicCollectiveKey("reduce_scatter", 2048, 2048, 1024,
                                                               512, rg, c10::kBFloat16, 2));
  // unroll is part of the NEFF identity (u1/u2/u4 are distinct NEFFs).
  EXPECT_NE(k, CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 1024, 512,
                                                               rg, c10::kBFloat16, /*unroll=*/4));
}

TEST_F(CacheKeyGeneratorTest, DynamicCollectiveKeyDistinctTilesSameTraceNumel) {
  // Two regimes can emit the same trace_numel (2048) with DIFFERENT tiles. Keying on the tiles
  // (alongside trace_numel) keeps the NEFFs distinct, so no cross-regime aliasing.
  auto rg = std::vector<std::vector<int>>{{0, 1, 2, 3}};
  auto a = CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 1024, 512, rg,
                                                           c10::kBFloat16, 2);
  auto b = CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 512, 256, rg,
                                                           c10::kBFloat16, 2);
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyGeneratorTest, DynamicCollectiveKeyReduceOpDistinctAndOmittedForAllgather) {
  auto rg = std::vector<std::vector<int>>{{0, 1, 2, 3}};
  // reduce_op omitted (default -1) for allgather => no "red=" token.
  auto ag = CacheKeyGenerator::GenerateDynamicCollectiveKey("allgather", 2048, 2048, 1024, 512, rg,
                                                            c10::kBFloat16, /*unroll=*/2);
  EXPECT_EQ(ag.find("red="), std::string::npos);
  // A reducing collective encodes reduce_op, and distinct reduce_ops => distinct keys.
  auto rs0 = CacheKeyGenerator::GenerateDynamicCollectiveKey(
      "reduce_scatter", 2048, 2048, 1024, 512, rg, c10::kBFloat16, 2, /*reduce_op=*/0);
  auto rs1 = CacheKeyGenerator::GenerateDynamicCollectiveKey(
      "reduce_scatter", 2048, 2048, 1024, 512, rg, c10::kBFloat16, 2, /*reduce_op=*/1);
  EXPECT_NE(rs0.find("red=0"), std::string::npos);
  EXPECT_NE(rs0, rs1);
}

TEST_F(CacheKeyGeneratorTest, DynamicCollectiveKeyDisjointFromNativeAllgather) {
  // The "dynamic" tag keeps the bucketed key from colliding with a native allgather key.
  auto dynamic_key = CacheKeyGenerator::GenerateDynamicCollectiveKey(
      "allgather", 2048, 2048, 1024, 512, {{0, 1, 2, 3}}, c10::kBFloat16, /*unroll=*/2);
  auto native = CacheKeyGenerator::GenerateCollective(
      "allgather", {torch::zeros({2048, 4096}, torch::kBFloat16)}, {{0, 1, 2, 3}}, 0, false);
  EXPECT_NE(dynamic_key, native);
}
