#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"

using namespace at::neuron;

class CompilationCacheKeyTest : public ::testing::Test {
 protected:
  std::vector<uint8_t> MakeHloBytes(const std::string& content) {
    return std::vector<uint8_t>(content.begin(), content.end());
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

  // Same base_key + same config → identical in-memory key regardless of source
  EXPECT_EQ(key1, key2);
  EXPECT_EQ(key1.GetKey(), key2.GetKey());

  // Source metadata differs
  EXPECT_EQ(key1.GetSource(), CacheKeySource::EAGER);
  EXPECT_EQ(key2.GetSource(), CacheKeySource::TORCH_COMPILE);
}

TEST_F(CompilationCacheKeyTest, DifferentBaseKeyDifferentKey) {
  auto key1 = CompilationCacheKey("op_add", "", "");
  auto key2 = CompilationCacheKey("op_mul", "", "");

  EXPECT_NE(key1, key2);
  EXPECT_NE(key1.GetKey(), key2.GetKey());
}

TEST_F(CompilationCacheKeyTest, SameBaseKeyDifferentCompilerArgsSameInMemoryKey) {
  auto key1 = CompilationCacheKey("same_op", "--auto-cast=none", "2");
  auto key2 = CompilationCacheKey("same_op", "--auto-cast=matmul", "2");

  EXPECT_EQ(key1, key2);
}

TEST_F(CompilationCacheKeyTest, SameBaseKeyDifferentOptLevelSameInMemoryKey) {
  auto key1 = CompilationCacheKey("same_op", "", "1");
  auto key2 = CompilationCacheKey("same_op", "", "2");

  EXPECT_EQ(key1, key2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyNullWithoutHloBytes) {
  auto key = CompilationCacheKey("my_op", "", "");
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyComputedWithHloBytes) {
  auto hlo = MakeHloBytes("module @test { func.func @main() {} }");
  auto key = CompilationCacheKey("my_op", "", "", &hlo);

  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  // Persistent key should be a 32-char hex string
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyIsCached) {
  auto hlo = MakeHloBytes("test hlo content");
  auto key = CompilationCacheKey("my_op", "", "", &hlo);

  const std::string* pk1 = key.GetPersistentKey();
  const std::string* pk2 = key.GetPersistentKey();

  // Should return the same pointer (cached)
  EXPECT_EQ(pk1, pk2);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, SameHloDifferentBaseKeySamePersistentKey) {
  auto hlo = MakeHloBytes("identical hlo bytes");
  auto key1 = CompilationCacheKey("eager_op_add", "", "", &hlo);
  auto key2 = CompilationCacheKey("torch_compile_segment_0", "", "", &hlo);

  // Different in-memory keys
  EXPECT_NE(key1.GetKey(), key2.GetKey());

  // Same persistent key (same HLO + same compiler config)
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, DifferentHloDifferentPersistentKey) {
  auto hlo1 = MakeHloBytes("hlo for add operation");
  auto hlo2 = MakeHloBytes("hlo for mul operation");
  auto key1 = CompilationCacheKey("op", "", "", &hlo1);
  auto key2 = CompilationCacheKey("op", "", "", &hlo2);

  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, ComputePersistentKeyFromExplicitHlo) {
  auto hlo = MakeHloBytes("test hlo");
  auto key = CompilationCacheKey("my_op", "", "");  // no hlo_bytes_ptr

  std::string pk = key.ComputePersistentKey(hlo);
  EXPECT_EQ(pk.size(), 32);
}

TEST_F(CompilationCacheKeyTest, BuildCompilerExtensionIncludesAllFields) {
  auto key = CompilationCacheKey("op", "--auto-cast=none", "-O2");
  std::string ext = key.BuildCompilerExtension();
  EXPECT_NE(ext.find("-O2"), std::string::npos);
  EXPECT_NE(ext.find("--auto-cast=none"), std::string::npos);
}

TEST_F(CompilationCacheKeyTest, BuildCompilerExtensionDeterministic) {
  auto key1 = CompilationCacheKey("op", "--flag", "-O2");
  auto key2 = CompilationCacheKey("op", "--flag", "-O2");
  EXPECT_EQ(key1.BuildCompilerExtension(), key2.BuildCompilerExtension());
}

TEST_F(CompilationCacheKeyTest, UsableInUnorderedMap) {
  std::unordered_map<CompilationCacheKey, std::string> cache;

  auto key1 = CompilationCacheKey("key1", "", "");
  auto key2 = CompilationCacheKey("key2", "", "");

  cache[key1] = "neff_1";
  cache[key2] = "neff_2";

  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(cache[key1], "neff_1");
  EXPECT_EQ(cache[key2], "neff_2");

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
  // Same address = cached
  EXPECT_EQ(&str1, &str2);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringPreservesKeyAndSource) {
  std::string full_key = "my_base_key";

  auto key_eager = CompilationCacheKey(full_key, CacheKeySource::EAGER);
  EXPECT_FALSE(key_eager.IsEmpty());
  EXPECT_EQ(key_eager.GetKey(), full_key);
  EXPECT_EQ(key_eager.GetSource(), CacheKeySource::EAGER);

  auto key_compile = CompilationCacheKey(full_key, CacheKeySource::TORCH_COMPILE);
  EXPECT_EQ(key_compile.GetKey(), full_key);
  EXPECT_EQ(key_compile.GetSource(), CacheKeySource::TORCH_COMPILE);

  EXPECT_EQ(key_eager, key_compile);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringNoPersistentKey) {
  auto key = CompilationCacheKey("full_key_string", CacheKeySource::UNKNOWN);
  EXPECT_EQ(key.GetPersistentKey(), nullptr);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringToStringIncludesSource) {
  auto key = CompilationCacheKey("neff_graph_0", CacheKeySource::TORCH_COMPILE);
  const std::string& str = key.ToString();
  EXPECT_NE(str.find("neff_graph_0"), std::string::npos);
  EXPECT_NE(str.find("TORCH_COMPILE"), std::string::npos);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringUsableInMap) {
  // FromKeyString keys should work as map keys alongside full keys.
  std::unordered_map<CompilationCacheKey, int> cache;

  auto full_key = CompilationCacheKey("op_add", "--flag", "-O2");
  auto simple_key = CompilationCacheKey("op_add", CacheKeySource::EAGER);

  cache[full_key] = 1;
  // Same base_key should find the same entry
  EXPECT_EQ(cache.count(simple_key), 1);
  EXPECT_EQ(cache[simple_key], 1);
}

TEST_F(CompilationCacheKeyTest, FromKeyStringSetHloBytesThenGetPersistentKey) {
  auto key = CompilationCacheKey("graph_name_0", CacheKeySource::TORCH_COMPILE);
  EXPECT_EQ(key.GetPersistentKey(), nullptr);

  // After setting HLO bytes, persistent key should be available
  auto hlo = MakeHloBytes("module @main { func.func @entry() {} }");
  key.SetHloBytesPtr(&hlo);

  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentCompilerArgs) {
  auto hlo = MakeHloBytes("identical hlo bytes for both keys");
  auto key1 = CompilationCacheKey("op", "--auto-cast=none", "-O2", &hlo);
  auto key2 = CompilationCacheKey("op", "--auto-cast=matmul", "-O2", &hlo);

  // Same in-memory key
  EXPECT_EQ(key1, key2);

  // Different persistent keys (compiler args differ)
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentOptLevel) {
  auto hlo = MakeHloBytes("same hlo for opt level test");
  auto key1 = CompilationCacheKey("op", "--flag", "-O1", &hlo);
  auto key2 = CompilationCacheKey("op", "--flag", "-O2", &hlo);

  // Same in-memory key
  EXPECT_EQ(key1, key2);

  // Different persistent keys (optimization level differs)
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeySameWithIdenticalConfig) {
  auto hlo = MakeHloBytes("same hlo content");
  auto key1 = CompilationCacheKey("op_A", "--flag", "-O2", &hlo);
  auto key2 = CompilationCacheKey("op_B", "--flag", "-O2", &hlo);

  // Different in-memory keys
  EXPECT_NE(key1, key2);

  // Same persistent keys (same HLO + same compiler config)
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, InvalidateCachedStringsRecomputesPersistentKey) {
  auto hlo = MakeHloBytes("original hlo bytes");
  auto key = CompilationCacheKey("op", "--flag", "-O2", &hlo);

  // Compute and cache persistent key
  const std::string* pk1 = key.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  std::string pk1_copy = *pk1;

  // Modify the underlying HLO bytes (simulates UpdateHloBytes)
  hlo = MakeHloBytes("modified hlo bytes");

  // Invalidate and recompute
  key.InvalidateCachedStrings();

  const std::string* pk2 = key.GetPersistentKey();
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(pk1_copy, *pk2);
}

TEST_F(CompilationCacheKeyTest, ToStringUpdatesAfterPersistentKey) {
  auto hlo = MakeHloBytes("test hlo");
  auto key = CompilationCacheKey("test", "", "", &hlo);

  // Before persistent key computed
  const std::string& str1 = key.ToString();
  EXPECT_EQ(str1.find("persistent_key"), std::string::npos);

  // Compute persistent key
  key.GetPersistentKey();

  // ToString should now include persistent key
  const std::string& str2 = key.ToString();
  EXPECT_NE(str2.find("persistent_key"), std::string::npos);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
