/**
 * Unit tests for NeuronDispatcher.
 *
 * Tests cover: singleton, cache, GenerateCacheKey, FlattenAndFilterInputs,
 * ExtractDevice, Cleanup, BoxedDispatchFunctor, ForEachTensor, and struct
 * defaults.
 *
 * No Python or hardware required.
 */

#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"
#include "torch_neuronx/csrc/core/dispatch/NeuronDispatcher.h"

namespace torch_neuronx {
namespace dispatch {

// Friend class declared in NeuronDispatcher.h — exposes private methods for testing.
class NeuronDispatcherTestAccess {
 public:
  static std::shared_ptr<DispatchCacheEntry> CacheLookup(NeuronDispatcher& d,
                                                         const std::string& key) {
    return d.CacheLookup(key);
  }
  static void CacheInsert(NeuronDispatcher& d, const std::string& key,
                          std::shared_ptr<DispatchCacheEntry> entry) {
    d.CacheInsert(key, std::move(entry));
  }
  static torch::Device ExtractDevice(const std::vector<c10::IValue>& args) {
    return NeuronDispatcher::ExtractDevice(args);
  }
  static std::string GenerateCacheKey(const std::string& op_name, const c10::FunctionSchema& schema,
                                      const std::vector<c10::IValue>& args,
                                      const OpRegistration& reg = OpRegistration{
                                          py::none(), py::none(), {}, {}, false}) {
    return NeuronDispatcher::GenerateCacheKey(op_name, schema, args, reg);
  }
  static std::vector<torch::Tensor> FlattenAndFilterInputs(const std::vector<c10::IValue>& args,
                                                           const c10::FunctionSchema& schema,
                                                           const std::vector<int>& kept,
                                                           const std::string& op_name,
                                                           int64_t ir_input_count = -1) {
    return NeuronDispatcher::FlattenAndFilterInputs(args, schema, kept, op_name, {}, {},
                                                    ir_input_count);
  }
  static const auto& GetOpRegistrations(NeuronDispatcher& d) { return d.op_registrations_; }
};

namespace {

using TA = NeuronDispatcherTestAccess;

// ═══════════════════════════════════════════════════════════════
//  Fixture
// ═══════════════════════════════════════════════════════════════

class NeuronDispatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    d_ = &NeuronDispatcher::Instance();
    d_->Cleanup();  // Start each test with clean state
  }
  NeuronDispatcher* d_;
};

// ═══════════════════════════════════════════════════════════════
//  Singleton tests
// ═══════════════════════════════════════════════════════════════

TEST_F(NeuronDispatcherTest, SingletonIsNotNull) { EXPECT_NE(d_, nullptr); }

TEST_F(NeuronDispatcherTest, SingletonIsSameInstance) {
  EXPECT_EQ(d_, &NeuronDispatcher::Instance());
}

TEST_F(NeuronDispatcherTest, SingletonFromMultipleThreads) {
  std::vector<NeuronDispatcher*> ptrs(4, nullptr);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([&ptrs, i]() { ptrs[i] = &NeuronDispatcher::Instance(); });
  for (auto& t : threads) t.join();
  for (auto* p : ptrs) EXPECT_EQ(p, d_);
}

// ═══════════════════════════════════════════════════════════════
//  Cleanup
// ═══════════════════════════════════════════════════════════════

TEST_F(NeuronDispatcherTest, CleanupDoesNotCrash) { EXPECT_NO_THROW(d_->Cleanup()); }

TEST_F(NeuronDispatcherTest, CleanupClearsRegistrations) {
  EXPECT_TRUE(d_->GetRegisteredOps().empty());
}

TEST_F(NeuronDispatcherTest, CleanupIdempotent) {
  d_->Cleanup();
  EXPECT_NO_THROW(d_->Cleanup());
}

TEST_F(NeuronDispatcherTest, CleanupClearsCache) {
  TA::CacheInsert(*d_, "dispatch_key", std::make_shared<DispatchCacheEntry>());
  d_->Cleanup();
  EXPECT_EQ(TA::CacheLookup(*d_, "dispatch_key"), nullptr);
}

// ═══════════════════════════════════════════════════════════════
//  Registration storage uses shared_ptr
// ═══════════════════════════════════════════════════════════════

TEST_F(NeuronDispatcherTest, RegistrationMapUsesSharedPtr) {
  // Verify the map stores shared_ptr (compile-time check via type access)
  const auto& regs = TA::GetOpRegistrations(*d_);
  static_assert(std::is_same_v<std::decay_t<decltype(regs)>,
                               std::unordered_map<std::string, std::shared_ptr<OpRegistration>>>,
                "op_registrations_ must store shared_ptr<OpRegistration>");
  EXPECT_TRUE(regs.empty());
}

// ═══════════════════════════════════════════════════════════════
//  Dispatch cache operations
// ═══════════════════════════════════════════════════════════════

TEST_F(NeuronDispatcherTest, CacheLookupMissReturnsNull) {
  EXPECT_EQ(TA::CacheLookup(*d_, "nonexistent_key"), nullptr);
}

TEST_F(NeuronDispatcherTest, CacheInsertAndLookup) {
  auto entry = std::make_shared<DispatchCacheEntry>();
  entry->op_name = "test_op";
  entry->neff_cache_key = "neff_123";
  TA::CacheInsert(*d_, "key1", entry);

  auto found = TA::CacheLookup(*d_, "key1");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->op_name, "test_op");
  EXPECT_EQ(found->neff_cache_key, "neff_123");
}

TEST_F(NeuronDispatcherTest, CacheInsertOverwrites) {
  auto e1 = std::make_shared<DispatchCacheEntry>();
  e1->op_name = "first";
  TA::CacheInsert(*d_, "key", e1);

  auto e2 = std::make_shared<DispatchCacheEntry>();
  e2->op_name = "second";
  TA::CacheInsert(*d_, "key", e2);

  EXPECT_EQ(TA::CacheLookup(*d_, "key")->op_name, "second");
}

TEST_F(NeuronDispatcherTest, CacheReturnsSameSharedPtr) {
  auto entry = std::make_shared<DispatchCacheEntry>();
  entry->op_name = "shared_test";
  TA::CacheInsert(*d_, "ptr_key", entry);

  auto found = TA::CacheLookup(*d_, "ptr_key");
  EXPECT_EQ(found.get(), entry.get());
}

// ═══════════════════════════════════════════════════════════════
//  ExtractDevice
// ═══════════════════════════════════════════════════════════════

TEST_F(NeuronDispatcherTest, ExtractDeviceFromCPUTensors) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3})};
  EXPECT_EQ(TA::ExtractDevice(args).type(), c10::DeviceType::PrivateUse1);
}

TEST_F(NeuronDispatcherTest, ExtractDeviceEmptyArgs) {
  EXPECT_EQ(TA::ExtractDevice({}).type(), c10::DeviceType::PrivateUse1);
}

TEST_F(NeuronDispatcherTest, ExtractDeviceNonTensorArgs) {
  std::vector<c10::IValue> args = {c10::IValue(42), c10::IValue(3.14), c10::IValue(true)};
  EXPECT_EQ(TA::ExtractDevice(args).type(), c10::DeviceType::PrivateUse1);
}

TEST_F(NeuronDispatcherTest, ExtractDeviceUndefinedTensor) {
  std::vector<c10::IValue> args = {c10::IValue(torch::Tensor())};
  EXPECT_EQ(TA::ExtractDevice(args).type(), c10::DeviceType::PrivateUse1);
}

// ═══════════════════════════════════════════════════════════════
//  GenerateCacheKey
// ═══════════════════════════════════════════════════════════════

class CacheKeyTest : public ::testing::Test {
 protected:
  c10::FunctionSchema schema() {
    return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
  }
  std::string key(const std::vector<c10::IValue>& args) {
    return TA::GenerateCacheKey("add.Tensor", schema(), args);
  }
};

TEST_F(CacheKeyTest, DifferentShapes) {
  auto s = schema();
  EXPECT_NE(TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2, 3}), torch::zeros({2, 3}), 1}),
            TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({4, 5}), torch::zeros({4, 5}), 1}));
}

TEST_F(CacheKeyTest, SameShapesSameKey) {
  auto a = key({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  auto b = key({torch::ones({2, 3}), torch::ones({2, 3}), 1});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyTest, DifferentDtypes) {
  auto a = key({torch::zeros({2}, torch::kFloat32), torch::zeros({2}, torch::kFloat32), 1});
  auto b = key({torch::zeros({2}, torch::kFloat64), torch::zeros({2}, torch::kFloat64), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, DifferentStrides) {
  auto a = key({torch::zeros({2, 3}), torch::zeros({2, 3}), 1});
  auto b = key({torch::zeros({3, 2}).t(), torch::zeros({3, 2}).t(), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, ContainsOpName) {
  EXPECT_NE(key({torch::zeros({2}), torch::zeros({2}), 1}).find("add.Tensor"), std::string::npos);
}

TEST_F(CacheKeyTest, NonStaticIntArgSameKey) {
  // Non-static int kwargs produce type-only keys ("int"), not values.
  auto a = key({torch::zeros({2}), torch::zeros({2}), 1});
  auto b = key({torch::zeros({2}), torch::zeros({2}), 2});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyTest, StaticIntArgDiffers) {
  // When the arg is marked static, its value is included in the key.
  auto s = schema();
  OpRegistration reg{py::none(), py::none(), {2}, {}, false};
  auto a = TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1}, reg);
  auto b = TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 2}, reg);
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, UndefinedTensor) {
  auto k = key({torch::zeros({2}), c10::IValue(torch::Tensor()), 1});
  EXPECT_NE(k.find("undef"), std::string::npos);
}

TEST_F(CacheKeyTest, NoneArg) {
  auto k = key({torch::zeros({2}), torch::zeros({2}), c10::IValue()});
  EXPECT_NE(k.find("None"), std::string::npos);
}

TEST_F(CacheKeyTest, BoolArg) {
  auto a = key({torch::zeros({2}), torch::zeros({2}), true});
  auto b = key({torch::zeros({2}), torch::zeros({2}), false});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, NonStaticDoubleArgSameKey) {
  // Non-static double kwargs produce type-only keys ("float"), not values.
  auto a = key({torch::zeros({2}), torch::zeros({2}), 1.5});
  auto b = key({torch::zeros({2}), torch::zeros({2}), 2.5});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyTest, StaticDoubleArgDiffers) {
  auto s = schema();
  OpRegistration reg{py::none(), py::none(), {2}, {}, false};
  auto a = TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1.5}, reg);
  auto b = TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 2.5}, reg);
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, StringArg) {
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(std::string("abc"))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(std::string("xyz"))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, IntListArg) {
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<int64_t>({1, 2}))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<int64_t>({3, 4}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, BoolListArg) {
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<bool>({true, false}))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<bool>({false, true}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, DoubleListArg) {
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<double>({1.0, 2.0}))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::List<double>({3.0, 4.0}))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, TensorListArg) {
  c10::List<at::Tensor> tl1({torch::zeros({2})});
  c10::List<at::Tensor> tl2({torch::zeros({3})});
  auto a = key({c10::IValue(tl1), c10::IValue(tl1), 1});
  auto b = key({c10::IValue(tl2), c10::IValue(tl2), 1});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, TensorListWithUndef) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::Tensor()});
  auto k = key({c10::IValue(tl), c10::IValue(tl), 1});
  EXPECT_NE(k.find("undef"), std::string::npos);
}

TEST_F(CacheKeyTest, NonStaticScalarArgSameValueSameKey) {
  // Non-static Scalar kwargs with same type produce same keys (value excluded).
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(1.0))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))});
  EXPECT_EQ(a, b);
}

TEST_F(CacheKeyTest, NonStaticScalarArgDifferentTypeDifferentKey) {
  // Non-static Scalar kwargs with different type kinds (float vs int) produce
  // different keys because PreprocessArgs creates 0-dim tensors with different
  // dtypes, leading to different HLO input signatures.
  auto a = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))});
  auto b = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2))});
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, StaticScalarArgDiffers) {
  auto s = schema();
  OpRegistration reg{py::none(), py::none(), {2}, {}, false};
  auto a = TA::GenerateCacheKey(
      "add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(1.0))}, reg);
  auto b = TA::GenerateCacheKey(
      "add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), c10::IValue(c10::Scalar(2.0))}, reg);
  EXPECT_NE(a, b);
}

TEST_F(CacheKeyTest, ZeroDimTensor) {
  auto k = key({torch::tensor(5.0), torch::zeros({2}), 1});
  EXPECT_NE(k.find("[]"), std::string::npos);
}

TEST_F(CacheKeyTest, EmptyTensor) {
  auto k = key({torch::zeros({0, 3}), torch::zeros({2}), 1});
  EXPECT_NE(k.find("[0,3]"), std::string::npos);
}

TEST_F(CacheKeyTest, DeviceArg) {
  auto k = key({torch::zeros({2}), torch::zeros({2}), c10::IValue(torch::Device(torch::kCPU))});
  EXPECT_NE(k.find("dev:cpu"), std::string::npos);
}

TEST_F(CacheKeyTest, DifferentDeviceArgsDifferentKeys) {
  auto k1 = key({torch::zeros({2}), c10::IValue(torch::Device(torch::kCPU))});
  auto k2 = key({torch::zeros({2}), c10::IValue(torch::Device(c10::DeviceType::PrivateUse1, 0))});
  EXPECT_NE(k1, k2);
}

TEST_F(CacheKeyTest, GeneratorArg) {
  auto k =
      key({torch::zeros({2}), torch::zeros({2}), c10::IValue(at::detail::createCPUGenerator())});
  EXPECT_NE(k.find("gen"), std::string::npos);
}

TEST_F(CacheKeyTest, DifferentOpsProduceDifferentKeys) {
  auto s = schema();
  auto k1 = TA::GenerateCacheKey("add.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1});
  auto k2 = TA::GenerateCacheKey("sub.Tensor", s, {torch::zeros({2}), torch::zeros({2}), 1});
  EXPECT_NE(k1, k2);
}

TEST_F(CacheKeyTest, StabilityAcrossCalls) {
  auto k1 = key({torch::zeros({4, 4}), torch::zeros({4, 4}), 1});
  auto k2 = key({torch::zeros({4, 4}), torch::zeros({4, 4}), 1});
  EXPECT_EQ(k1, k2);
}

// ═══════════════════════════════════════════════════════════════
//  FlattenAndFilterInputs
// ═══════════════════════════════════════════════════════════════

class FlattenFilterTest : public ::testing::Test {
 protected:
  c10::FunctionSchema schema() {
    return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor").schema();
  }
};

TEST_F(FlattenFilterTest, BasicTensorArgs) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({2, 3}), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {}, "t").size(), 2u);
}

TEST_F(FlattenFilterTest, FilterByKeptIndices) {
  auto t1 = torch::zeros({2});
  std::vector<c10::IValue> args = {t1, torch::ones({2}), 1};
  auto r = TA::FlattenAndFilterInputs(args, schema(), {0}, "t");
  EXPECT_EQ(r.size(), 1u);
  EXPECT_TRUE(r[0].equal(t1));
}

TEST_F(FlattenFilterTest, FilterSelectsSecond) {
  auto t2 = torch::ones({3});
  std::vector<c10::IValue> args = {torch::zeros({2}), t2, 1};
  auto r = TA::FlattenAndFilterInputs(args, schema(), {1}, "t");
  EXPECT_EQ(r.size(), 1u);
  EXPECT_TRUE(r[0].equal(t2));
}

TEST_F(FlattenFilterTest, FilterMultipleIndices) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({3}), 1};
  auto r = TA::FlattenAndFilterInputs(args, schema(), {0, 1}, "t");
  EXPECT_EQ(r.size(), 2u);
}

TEST_F(FlattenFilterTest, EmptyKeptReturnsAll) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({2}), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {}, "t").size(), 2u);
}

TEST_F(FlattenFilterTest, ZeroIrInputsReturnsNone) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({2}), 1};
  EXPECT_TRUE(TA::FlattenAndFilterInputs(args, schema(), {}, "t", 0).empty());
}

TEST_F(FlattenFilterTest, ZeroIrInputsRejectsKeptIndices) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({2}), 1};
  EXPECT_THROW(TA::FlattenAndFilterInputs(args, schema(), {0}, "t", 0), c10::Error);
}

TEST_F(FlattenFilterTest, UndefinedSkipped) {
  std::vector<c10::IValue> args = {torch::zeros({2}), c10::IValue(torch::Tensor()), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {}, "t").size(), 1u);
}

TEST_F(FlattenFilterTest, TensorListFlattened) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::ones({3})});
  std::vector<c10::IValue> args = {c10::IValue(tl), torch::zeros({4}), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {}, "t").size(), 3u);
}

TEST_F(FlattenFilterTest, OutOfBoundsKeptIgnored) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::zeros({2}), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {0, 99}, "t").size(), 1u);
}

TEST_F(FlattenFilterTest, NegativeKeptIgnored) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::zeros({2}), 1};
  EXPECT_EQ(TA::FlattenAndFilterInputs(args, schema(), {-1, 0}, "t").size(), 1u);
}

TEST_F(FlattenFilterTest, NoTensorsEmpty) {
  std::vector<c10::IValue> args = {c10::IValue(1), c10::IValue(2), c10::IValue(3)};
  EXPECT_TRUE(TA::FlattenAndFilterInputs(args, schema(), {}, "t").empty());
}

TEST_F(FlattenFilterTest, EmptyArgsEmpty) {
  EXPECT_TRUE(TA::FlattenAndFilterInputs({}, schema(), {}, "t").empty());
}

// A zero-input factory NEFF (arange with out=) keeps no inputs. The write-aliased
// out= must not become a phantom input, which would trip an NRT arity check.
TEST_F(FlattenFilterTest, FactoryOutIsDroppedFromEmptyKept) {
  auto sch = c10::Dispatcher::singleton().findSchemaOrThrow("aten::arange", "start_out").schema();
  std::vector<c10::IValue> args = {c10::IValue(0), c10::IValue(4), c10::IValue(1),
                                   torch::empty({4})};
  EXPECT_TRUE(TA::FlattenAndFilterInputs(args, sch, {}, "aten::arange.start_out").empty());
}

// A write-only in-place self (zero_) is an output binding, not an input.
TEST_F(FlattenFilterTest, WriteOnlyInplaceSelfDroppedFromEmptyKept) {
  auto sch = c10::Dispatcher::singleton().findSchemaOrThrow("aten::zero_", "").schema();
  std::vector<c10::IValue> args = {torch::ones({4})};
  EXPECT_TRUE(TA::FlattenAndFilterInputs(args, sch, {}, "aten::zero_").empty());
}

// A write-aliased self stays an input when the NEFF references it (non-empty
// kept). The drop happens only in the empty-kept factory branch.
TEST_F(FlattenFilterTest, ReadModifyWriteInplaceSelfKeptWhenReferenced) {
  auto sch = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add_", "Tensor").schema();
  auto self = torch::ones({4});
  std::vector<c10::IValue> args = {self, torch::ones({4}), 1};
  auto r = TA::FlattenAndFilterInputs(args, sch, {0}, "aten::add_.Tensor");
  EXPECT_EQ(r.size(), 1u);
  EXPECT_TRUE(r[0].equal(self));
}

// A mix of read-only args and a write-aliased out= keeps only the read-only
// inputs under empty kept.
TEST_F(FlattenFilterTest, MixedReadOnlyAndWriteAliasedKeepsOnlyReadOnly) {
  auto sch = c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "out").schema();
  auto self = torch::zeros({4});
  auto other = torch::ones({4});
  std::vector<c10::IValue> args = {self, other, 1, torch::empty({4})};
  auto r = TA::FlattenAndFilterInputs(args, sch, {}, "aten::add.out");
  EXPECT_EQ(r.size(), 2u);
  EXPECT_TRUE(r[0].equal(self));
  EXPECT_TRUE(r[1].equal(other));
}

// ═══════════════════════════════════════════════════════════════
//  ForEachTensor (DispatchUtils.h)
// ═══════════════════════════════════════════════════════════════

TEST(ForEachTensorTest, PlainTensors) {
  std::vector<c10::IValue> args = {torch::zeros({2}), torch::ones({3}), c10::IValue(42)};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 2);
}

TEST(ForEachTensorTest, UndefinedTensorsSkipped) {
  std::vector<c10::IValue> args = {torch::zeros({2}), c10::IValue(torch::Tensor())};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(ForEachTensorTest, TensorList) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::ones({3})});
  std::vector<c10::IValue> args = {c10::IValue(tl)};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 2);
}

TEST(ForEachTensorTest, TensorListWithUndefined) {
  c10::List<at::Tensor> tl({torch::zeros({2}), torch::Tensor()});
  std::vector<c10::IValue> args = {c10::IValue(tl)};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(ForEachTensorTest, GenericListWithOptionalTensors) {
  c10::List<c10::IValue> gl(c10::AnyType::get());
  gl.push_back(c10::IValue(torch::zeros({2})));
  gl.push_back(c10::IValue());  // None
  gl.push_back(c10::IValue(torch::ones({3})));
  std::vector<c10::IValue> args = {c10::IValue(gl)};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 2);
}

TEST(ForEachTensorTest, EmptyArgs) {
  int count = 0;
  ForEachTensor({}, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 0);
}

TEST(ForEachTensorTest, NonTensorArgsIgnored) {
  std::vector<c10::IValue> args = {c10::IValue(1), c10::IValue(3.14), c10::IValue(true)};
  int count = 0;
  ForEachTensor(args, [&](const at::Tensor&) { count++; });
  EXPECT_EQ(count, 0);
}

TEST(ForEachTensorTest, CollectsShapes) {
  std::vector<c10::IValue> args = {torch::zeros({2, 3}), torch::ones({4, 5})};
  std::vector<int64_t> total_elems;
  ForEachTensor(args, [&](const at::Tensor& t) { total_elems.push_back(t.numel()); });
  ASSERT_EQ(total_elems.size(), 2u);
  EXPECT_EQ(total_elems[0], 6);
  EXPECT_EQ(total_elems[1], 20);
}

// HasEmptyTensorInput and IsSupportedDtype tests are in DispatchUtilsTest.cpp.

// ═══════════════════════════════════════════════════════════════
//  BoxedDispatchFunctor
// ═══════════════════════════════════════════════════════════════

TEST(BoxedDispatchFunctorTest, CapturesRegistration) {
  auto reg =
      std::make_shared<OpRegistration>(OpRegistration{py::none(), py::none(), {}, {}, false});

  bool called = false;
  const OpRegistration* received_reg = nullptr;
  auto handler = [](const c10::OperatorHandle&, c10::DispatchKeySet, torch::jit::Stack*,
                    const OpRegistration& reg) {
    // Can't capture in a function pointer, so we test construction only
  };

  // Verify construction doesn't crash and holds the shared_ptr
  BoxedDispatchFunctor functor(reg, handler);
  EXPECT_EQ(reg.use_count(), 2);  // shared by reg and functor
}

TEST(BoxedDispatchFunctorTest, SharedPtrKeptAlive) {
  std::weak_ptr<OpRegistration> weak;
  {
    auto reg =
        std::make_shared<OpRegistration>(OpRegistration{py::none(), py::none(), {}, {}, false});
    weak = reg;

    auto handler = [](const c10::OperatorHandle&, c10::DispatchKeySet, torch::jit::Stack*,
                      const OpRegistration&) {};
    auto functor = std::make_unique<BoxedDispatchFunctor>(reg, handler);

    reg.reset();                   // Release original shared_ptr
    EXPECT_FALSE(weak.expired());  // Functor keeps it alive
  }
  EXPECT_TRUE(weak.expired());  // Functor destroyed, reg freed
}

// ═══════════════════════════════════════════════════════════════
//  Struct tests
// ═══════════════════════════════════════════════════════════════

TEST(DispatchCacheEntryTest, DefaultValues) {
  DispatchCacheEntry e;
  EXPECT_TRUE(e.stablehlo_bytes.empty());
  EXPECT_TRUE(e.neff_cache_key.empty());
  EXPECT_TRUE(e.kept_input_indices.empty());
  EXPECT_EQ(e.ir_input_count, -1);
  EXPECT_TRUE(e.kept_output_indices.empty());
  EXPECT_TRUE(e.output_specs.empty());
  EXPECT_TRUE(e.original_output_specs.empty());
  EXPECT_TRUE(e.passthrough_mappings.empty());
  EXPECT_TRUE(e.input_output_aliases.empty());
  EXPECT_TRUE(e.op_name.empty());
  EXPECT_FALSE(e.is_inplace);
  EXPECT_FALSE(e.has_collectives);
  EXPECT_FALSE(e.is_empty_tensor);
}

TEST(DispatchCacheEntryTest, OutputSpecFields) {
  DispatchCacheEntry::OutputSpec spec;
  spec.shape = {2, 3, 4};
  spec.dtype = c10::ScalarType::Float;
  EXPECT_EQ(spec.shape.size(), 3u);
  EXPECT_EQ(spec.dtype, c10::ScalarType::Float);
}

TEST(DispatchCacheEntryTest, InputOutputAliases) {
  DispatchCacheEntry e;
  e.input_output_aliases = {{0, 0}, {1, 1}};
  ASSERT_EQ(e.input_output_aliases.size(), 2u);
  EXPECT_EQ(e.input_output_aliases[0].first, 0);
  EXPECT_EQ(e.input_output_aliases[0].second, 0);
}

TEST(DispatchCacheEntryTest, PassthroughMappings) {
  DispatchCacheEntry e;
  e.passthrough_mappings[0] = 2;
  e.passthrough_mappings[1] = 3;
  EXPECT_EQ(e.passthrough_mappings.size(), 2u);
  EXPECT_EQ(e.passthrough_mappings.at(0), 2);
}

TEST(OpRegistrationTest, DefaultConstruction) {
  OpRegistration reg{py::none(), py::none(), {}, {}, false};
  EXPECT_TRUE(reg.python_wrapper.is_none());
  EXPECT_TRUE(reg.lower_func.is_none());
  EXPECT_TRUE(reg.static_argnums.empty());
  EXPECT_TRUE(reg.static_argnames.empty());
  EXPECT_FALSE(reg.skip_cache);
  EXPECT_FALSE(reg.has_lower_func);
}

TEST(OpRegistrationTest, SkipCacheFlag) {
  OpRegistration reg{py::none(), py::none(), {}, {}, true};
  EXPECT_TRUE(reg.skip_cache);
}

TEST(OpRegistrationTest, HasLowerFuncFlag) {
  // With a non-none lower_func, has_lower_func should be set by RegisterOp.
  // Direct construction test:
  OpRegistration reg{py::none(), py::none(), {}, {}, false, true};
  EXPECT_TRUE(reg.has_lower_func);
}

TEST(OpRegistrationTest, StaticArgnums) {
  OpRegistration reg{py::none(), py::none(), {0, 2, 5}, {}, false};
  EXPECT_EQ(reg.static_argnums.size(), 3u);
  EXPECT_TRUE(reg.static_argnums.count(0));
  EXPECT_TRUE(reg.static_argnums.count(2));
  EXPECT_TRUE(reg.static_argnums.count(5));
  EXPECT_FALSE(reg.static_argnums.count(1));
}

TEST(OpRegistrationTest, StaticArgnames) {
  OpRegistration reg{py::none(), py::none(), {}, {"dim", "keepdim"}, false};
  EXPECT_EQ(reg.static_argnames.size(), 2u);
  EXPECT_TRUE(reg.static_argnames.count("dim"));
  EXPECT_TRUE(reg.static_argnames.count("keepdim"));
}

}  // namespace
}  // namespace dispatch
}  // namespace torch_neuronx
