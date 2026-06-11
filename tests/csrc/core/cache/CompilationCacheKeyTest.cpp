#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"

using namespace at::neuron;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

class CompilationCacheKeyTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  // Existing tests assert the new (HLO-content) keying contract, so default
  // the flag to on for this fixture. The CompilationCacheKeyLegacyKeyTest
  // suite below exercises the off (default-prod) path.
  std::string saved_flag_value_;
  bool had_flag_value_ = false;

  void SetUp() override {
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    const char* existing = std::getenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
    saved_flag_value_ = existing ? std::string(existing) : "";
    had_flag_value_ = existing != nullptr;
    ::setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1", /*overwrite=*/1);
  }

  void TearDown() override {
    if (had_flag_value_) {
      ::setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", saved_flag_value_.c_str(),
               /*overwrite=*/1);
    } else {
      ::unsetenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
    }
  }
};

TEST_F(CompilationCacheKeyTest, DefaultConstructorIsEmpty) {
  CompilationCacheKey key;
  EXPECT_TRUE(key.IsEmpty());
  EXPECT_EQ(key.GetKey(), "");
}

TEST_F(CompilationCacheKeyTest, CreateProducesKeyFromBaseKey) {
  auto key = CompilationCacheKey("my_base_key", "--auto-cast=none", "2");
  EXPECT_FALSE(key.IsEmpty());
  EXPECT_EQ(key.GetKey(), "my_base_key");
}

TEST_F(CompilationCacheKeyTest, CreateWithEmptyBaseKeyIsInvalid) {
  auto key = CompilationCacheKey("", "--flag", "1");
  EXPECT_TRUE(key.IsEmpty());
}

TEST_F(CompilationCacheKeyTest, SameBaseKeySameConfigSameKey) {
  auto key1 = CompilationCacheKey("my_op", "--flag=1", "2", nullptr, CacheKeySource::EAGER);
  auto key2 = CompilationCacheKey("my_op", "--flag=1", "2", nullptr, CacheKeySource::TORCH_COMPILE);
  EXPECT_EQ(key1, key2);
  EXPECT_EQ(key1.GetKey(), key2.GetKey());
  EXPECT_EQ(key1.GetSource(), CacheKeySource::EAGER);
  EXPECT_EQ(key2.GetSource(), CacheKeySource::TORCH_COMPILE);
}

TEST_F(CompilationCacheKeyTest, DifferentBaseKeyDifferentKey) {
  auto key1 = CompilationCacheKey("op_add", "", "");
  auto key2 = CompilationCacheKey("op_mul", "", "");
  EXPECT_NE(key1, key2);
}

TEST_F(CompilationCacheKeyTest, SameBaseKeyDifferentCompilerArgsSameInMemoryKey) {
  auto key1 = CompilationCacheKey("same_op", "--auto-cast=none", "2");
  auto key2 = CompilationCacheKey("same_op", "--auto-cast=matmul", "2");
  EXPECT_EQ(key1, key2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyNullWithEmptyBaseKey) {
  CompilationCacheKey key;
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyNullWithoutHloBytes) {
  // New contract: persistent key requires HLO bytes. Without them,
  // GetPersistentKey() returns nullptr and the caller skips persistent caching.
  auto key = CompilationCacheKey("my_op", "", "");
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyAvailableWithHloBytes) {
  std::vector<uint8_t> hlo = {0x01, 0x02, 0x03, 0x04};
  auto key = CompilationCacheKey("my_op", "", "", &hlo);
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyIsCached) {
  std::vector<uint8_t> hlo = {0xAA, 0xBB};
  auto key = CompilationCacheKey("my_op", "", "", &hlo);
  const std::string* pk1 = key.GetPersistentKey();
  const std::string* pk2 = key.GetPersistentKey();
  EXPECT_EQ(pk1, pk2);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentCompilerArgs) {
  std::vector<uint8_t> hlo = {0x10, 0x20};
  auto key1 = CompilationCacheKey("op", "--auto-cast=none", "-O2", &hlo);
  auto key2 = CompilationCacheKey("op", "--auto-cast=matmul", "-O2", &hlo);
  EXPECT_EQ(key1, key2);  // Same in-memory key
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);  // Different persistent keys (compiler args in config)
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentOptLevel) {
  std::vector<uint8_t> hlo = {0x10, 0x20};
  auto key1 = CompilationCacheKey("op", "--flag", "-O1", &hlo);
  auto key2 = CompilationCacheKey("op", "--flag", "-O2", &hlo);
  EXPECT_EQ(key1, key2);
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, BuildCompilerExtensionDeterministic) {
  auto key1 = CompilationCacheKey("op", "--flag", "-O2");
  auto key2 = CompilationCacheKey("op", "--flag", "-O2");
  EXPECT_EQ(key1.BuildCompilerExtension(), key2.BuildCompilerExtension());
}

TEST_F(CompilationCacheKeyTest, BuildCompilerExtensionIncludesAllFields) {
  auto key = CompilationCacheKey("op", "--auto-cast=none", "-O2");
  std::string ext = key.BuildCompilerExtension();
  EXPECT_NE(ext.find("-O2"), std::string::npos);
  EXPECT_NE(ext.find("--auto-cast=none"), std::string::npos);
}

TEST_F(CompilationCacheKeyTest, UsableInUnorderedMap) {
  std::unordered_map<CompilationCacheKey, std::string> cache;
  auto key1 = CompilationCacheKey("key1", "", "");
  auto key2 = CompilationCacheKey("key2", "", "");
  cache[key1] = "neff_1";
  cache[key2] = "neff_2";
  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(cache[key1], "neff_1");
  auto key1_again = CompilationCacheKey("key1", "", "");
  EXPECT_EQ(cache[key1_again], "neff_1");
}

TEST_F(CompilationCacheKeyTest, ToStringContainsKey) {
  auto key = CompilationCacheKey("test_op", "--flag", "2", nullptr, CacheKeySource::EAGER);
  const std::string& str = key.ToString();
  EXPECT_NE(str.find("test_op"), std::string::npos);
  EXPECT_NE(str.find("EAGER"), std::string::npos);
}

TEST_F(CompilationCacheKeyTest, ToStringIsCached) {
  auto key = CompilationCacheKey("test", "", "");
  const std::string& str1 = key.ToString();
  const std::string& str2 = key.ToString();
  EXPECT_EQ(&str1, &str2);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringPreservesKeyAndSource) {
  auto key = CompilationCacheKey("my_base_key", CacheKeySource::EAGER);
  EXPECT_FALSE(key.IsEmpty());
  EXPECT_EQ(key.GetKey(), "my_base_key");
  EXPECT_EQ(key.GetSource(), CacheKeySource::EAGER);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringPersistentKeyUnavailable) {
  // The (key, source) constructor is for execution-only paths
  // (e.g., NeffDirectKernelExecution) that have no HLO bytes. With the new
  // contract, persistent key requires HLO bytes, so this returns nullptr
  // and callers gracefully skip the persistent cache.
  auto key = CompilationCacheKey("full_key_string", CacheKeySource::UNKNOWN);
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringUsableInMap) {
  std::unordered_map<CompilationCacheKey, int> cache;
  auto full_key = CompilationCacheKey("op_add", "--flag", "-O2");
  auto simple_key = CompilationCacheKey("op_add", CacheKeySource::EAGER);
  cache[full_key] = 1;
  EXPECT_EQ(cache.count(simple_key), 1);
  EXPECT_EQ(cache[simple_key], 1);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyDeterministicAcrossInstances) {
  std::vector<uint8_t> hlo = {0xDE, 0xAD, 0xBE, 0xEF};
  auto key1 = CompilationCacheKey("op_add_shapes_4x8_f32", "--flag", "-O2", &hlo);
  auto key2 = CompilationCacheKey("op_add_shapes_4x8_f32", "--flag", "-O2", &hlo);
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, ToStringUpdatesAfterPersistentKey) {
  std::vector<uint8_t> hlo = {0x01};
  auto key = CompilationCacheKey("test", "--flag", "-O2", &hlo);
  key.GetPersistentKey();
  const std::string& str = key.ToString();
  EXPECT_NE(str.find("persistent_key"), std::string::npos);
}

// ============================================================================
// New contract: persistent key derives from HLO bytes + compiler config.
// These tests exercise the cross-commit cache reuse scenario.
// ============================================================================

TEST_F(CompilationCacheKeyTest, CrossCommitReuse_BenignSourceEditPreservesPersistentKey) {
  // The headline scenario: a Python source edit changes base_key (because
  // global_version_hash is in there) but the lowering output is identical.
  // Persistent key must NOT change, so the on-disk NEFF cache hits across
  // commits.
  std::vector<uint8_t> hlo = {0x10, 0x20, 0x30, 0x40};
  auto before_edit = CompilationCacheKey("op:hash_v1", "--flag", "-O2", &hlo);
  auto after_edit = CompilationCacheKey("op:hash_v2_after_comment_edit", "--flag", "-O2", &hlo);

  // In-memory keys differ (Python source changed)
  EXPECT_NE(before_edit, after_edit);

  // Persistent keys match (HLO bytes and config unchanged)
  const std::string* pk_before = before_edit.GetPersistentKey();
  const std::string* pk_after = after_edit.GetPersistentKey();
  ASSERT_NE(pk_before, nullptr);
  ASSERT_NE(pk_after, nullptr);
  EXPECT_EQ(*pk_before, *pk_after);
}

TEST_F(CompilationCacheKeyTest, CrossCommitReuse_RealLoweringChangeBustsPersistentKey) {
  // Inverse of the above: a real lowering change produces different HLO bytes
  // and must invalidate the persistent NEFF cache.
  std::vector<uint8_t> hlo_v1 = {0x10, 0x20};
  std::vector<uint8_t> hlo_v2 = {0x10, 0x21};  // one byte different
  auto key1 = CompilationCacheKey("op", "--flag", "-O2", &hlo_v1);
  auto key2 = CompilationCacheKey("op", "--flag", "-O2", &hlo_v2);

  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, EmptyHloBytesProducesNullPersistentKey) {
  // Pointer non-null but vector empty — should still skip persistent caching.
  std::vector<uint8_t> empty_hlo;
  auto key = CompilationCacheKey("op", "--flag", "-O2", &empty_hlo);
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, NullHloBytesPtrProducesNullPersistentKey) {
  auto key = CompilationCacheKey("op", "--flag", "-O2", nullptr);
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, SetHloBytesPtrEnablesPersistentKey) {
  // Construct without HLO bytes, then set them later — persistent key
  // becomes available. Mirrors the MockNeuronKernelExecution pattern.
  auto key = CompilationCacheKey("op", "--flag", "-O2");
  EXPECT_EQ(key.GetPersistentKey(), nullptr);

  std::vector<uint8_t> hlo = {0xCA, 0xFE};
  key.SetHloBytesPtr(&hlo);
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyTest, SetHloBytesPtrInvalidatesCachedPersistentKey) {
  // After a persistent key is computed and cached, swapping to different
  // HLO bytes must produce a fresh key (no stale cached value).
  std::vector<uint8_t> hlo_v1 = {0x01, 0x02};
  auto key = CompilationCacheKey("op", "--flag", "-O2", &hlo_v1);
  std::string pk_v1 = *key.GetPersistentKey();

  std::vector<uint8_t> hlo_v2 = {0x03, 0x04};
  key.SetHloBytesPtr(&hlo_v2);
  std::string pk_v2 = *key.GetPersistentKey();

  EXPECT_NE(pk_v1, pk_v2);
}

TEST_F(CompilationCacheKeyTest, BuildCompilerExtensionIncludesCompilerVersion) {
  // Compiler version is part of the config string, so a compiler upgrade
  // changes BuildCompilerExtension output and (transitively) the persistent
  // key. Test depends on whatever version is set in the test environment;
  // we just verify the field is appended after a "_v" marker.
  auto key = CompilationCacheKey("op", "--flag", "-O2");
  std::string ext = key.BuildCompilerExtension();
  EXPECT_NE(ext.find("_v"), std::string::npos);
}

// ============================================================================
// Legacy key path: TORCH_NEURONX_NEFF_KEY_FROM_HLO unset/!="1".
// Persistent key derives from base_key + compiler_config (production default).
// HLO bytes are not required, and changing them does not affect the key.
// ============================================================================

class CompilationCacheKeyLegacyKeyTest : public ::testing::Test {
 protected:
  torch_neuronx::testing::MockNRTSession mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();

  std::string saved_flag_value_;
  bool had_flag_value_ = false;

  void SetUp() override {
    nrt_instance_info_t info{};
    info.family = NRT_INSTANCE_TRN3;
    EXPECT_CALL(*mock_nrt_, nrt_get_instance_info(_, _))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(DoAll(SetArgPointee<0>(info), Return(NRT_SUCCESS)));

    const char* existing = std::getenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
    saved_flag_value_ = existing ? std::string(existing) : "";
    had_flag_value_ = existing != nullptr;
    ::unsetenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
  }

  void TearDown() override {
    if (had_flag_value_) {
      ::setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", saved_flag_value_.c_str(),
               /*overwrite=*/1);
    } else {
      ::unsetenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO");
    }
  }
};

TEST_F(CompilationCacheKeyLegacyKeyTest, PersistentKeyAvailableWithoutHloBytes) {
  // The legacy contract: HLO bytes are not required for persistent caching.
  // base_key alone (with compiler config) produces the persistent key.
  auto key = CompilationCacheKey("my_op", "--flag", "-O2");
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyLegacyKeyTest, FromKeyStringPersistentKeyAvailable) {
  // The (key, source) constructor (used by NeffDirectKernelExecution) still
  // produces a persistent key in legacy mode — no HLO bytes needed.
  auto key = CompilationCacheKey("full_key_string", CacheKeySource::UNKNOWN);
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyLegacyKeyTest, DifferentBaseKeyDifferentPersistentKey) {
  // Inverse of HLO-content path: base_key drives the persistent key.
  auto key1 = CompilationCacheKey("op_add", "--flag", "-O2");
  auto key2 = CompilationCacheKey("op_mul", "--flag", "-O2");
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyLegacyKeyTest, HloBytesIgnoredWhenFlagOff) {
  // Critical guarantee: when the flag is off, swapping HLO bytes does NOT
  // change the persistent key. This is what makes the flag a safe rollback.
  std::vector<uint8_t> hlo_v1 = {0x01, 0x02};
  std::vector<uint8_t> hlo_v2 = {0xFF, 0xEE, 0xDD};
  auto key1 = CompilationCacheKey("op", "--flag", "-O2", &hlo_v1);
  auto key2 = CompilationCacheKey("op", "--flag", "-O2", &hlo_v2);
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);  // Same base_key + config → same persistent key.
}

TEST_F(CompilationCacheKeyLegacyKeyTest, FlagSwitchProducesDifferentPersistentKey) {
  // Sanity-check: flipping the flag changes the persistent key for the same
  // input. This documents that cached entries from one mode are NOT reused
  // by the other (which is correct — they hash different inputs).
  std::vector<uint8_t> hlo = {0xCA, 0xFE};
  auto key_off = CompilationCacheKey("op", "--flag", "-O2", &hlo);
  std::string pk_off = *key_off.GetPersistentKey();

  ::setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1", /*overwrite=*/1);
  auto key_on = CompilationCacheKey("op", "--flag", "-O2", &hlo);
  std::string pk_on = *key_on.GetPersistentKey();

  EXPECT_NE(pk_off, pk_on);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
