#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "torch_neuronx/csrc/core/cache/CompilationCacheKey.h"

using namespace at::neuron;

class CompilationCacheKeyTest : public ::testing::Test {};

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

TEST_F(CompilationCacheKeyTest, PersistentKeyAvailableWithBaseKey) {
  auto key = CompilationCacheKey("my_op", "", "");
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyIsCached) {
  auto key = CompilationCacheKey("my_op", "", "");
  const std::string* pk1 = key.GetPersistentKey();
  const std::string* pk2 = key.GetPersistentKey();
  EXPECT_EQ(pk1, pk2);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, DifferentBaseKeyDifferentPersistentKey) {
  auto key1 = CompilationCacheKey("op_add", "", "");
  auto key2 = CompilationCacheKey("op_mul", "", "");
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, SameBaseKeySameConfigSamePersistentKey) {
  auto key1 = CompilationCacheKey("op", "--flag", "-O2");
  auto key2 = CompilationCacheKey("op", "--flag", "-O2");
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentCompilerArgs) {
  auto key1 = CompilationCacheKey("op", "--auto-cast=none", "-O2");
  auto key2 = CompilationCacheKey("op", "--auto-cast=matmul", "-O2");
  EXPECT_EQ(key1, key2);  // Same in-memory key
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_NE(*pk1, *pk2);  // Different persistent keys
}

TEST_F(CompilationCacheKeyTest, PersistentKeyChangesWithDifferentOptLevel) {
  auto key1 = CompilationCacheKey("op", "--flag", "-O1");
  auto key2 = CompilationCacheKey("op", "--flag", "-O2");
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

TEST_F(CompilationCacheKeyTest, FromKeyStringPersistentKeyAvailable) {
  auto key = CompilationCacheKey("full_key_string", CacheKeySource::UNKNOWN);
  const std::string* pk = key.GetPersistentKey();
  ASSERT_NE(pk, nullptr);
  EXPECT_EQ(pk->size(), 32);
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
  auto key1 = CompilationCacheKey("op_add_shapes_4x8_f32", "--flag", "-O2");
  auto key2 = CompilationCacheKey("op_add_shapes_4x8_f32", "--flag", "-O2");
  const std::string* pk1 = key1.GetPersistentKey();
  const std::string* pk2 = key2.GetPersistentKey();
  ASSERT_NE(pk1, nullptr);
  ASSERT_NE(pk2, nullptr);
  EXPECT_EQ(*pk1, *pk2);
}

TEST_F(CompilationCacheKeyTest, ToStringUpdatesAfterPersistentKey) {
  auto key = CompilationCacheKey("test", "--flag", "-O2");
  key.GetPersistentKey();
  const std::string& str = key.ToString();
  EXPECT_NE(str.find("persistent_key"), std::string::npos);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
