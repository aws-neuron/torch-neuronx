#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "torch_neuronx/csrc/profiler/NConfigParser.h"

namespace torch_neuronx {
namespace profiler {
namespace {

TEST(NConfigParserTest, Trim) {
  using detail::trim;
  EXPECT_EQ(trim("  hello  "), "hello");
  EXPECT_EQ(trim("hello"), "hello");
  EXPECT_EQ(trim("   "), "");
  EXPECT_EQ(trim(""), "");
  EXPECT_EQ(trim("\t\n hello \r\n"), "hello");
}

TEST(NConfigParserTest, SplitAndTrim) {
  using detail::splitAndTrim;
  std::vector<std::string> expected = {"a", "b", "c"};
  EXPECT_EQ(splitAndTrim("a;b;c", ';'), expected);
  EXPECT_EQ(splitAndTrim(" a ; b ; c ", ';'), expected);
  EXPECT_EQ(splitAndTrim("a", ';'), std::vector<std::string>{"a"});
}

TEST(NConfigParserTest, ParseIndexOrRange_Single) {
  using detail::parseIndexOrRange;
  std::vector<uint32_t> expected = {5};
  EXPECT_EQ(parseIndexOrRange("5"), expected);
}

TEST(NConfigParserTest, ParseIndexOrRange_Range) {
  using detail::parseIndexOrRange;
  std::vector<uint32_t> expected = {0, 1, 2};
  EXPECT_EQ(parseIndexOrRange("0-2"), expected);
}

TEST(NConfigParserTest, ParseIndexOrRange_InvalidRange) {
  using detail::parseIndexOrRange;
  // start > end returns empty
  EXPECT_TRUE(parseIndexOrRange("5-2").empty());
}

TEST(NConfigParserTest, ParseIndexOrRange_NegativeEndInRange) {
  using detail::parseIndexOrRange;
  // "2--1" splits at first dash: startStr="2", endStr="-1", end < 0 -> empty
  EXPECT_TRUE(parseIndexOrRange("2--1").empty());
}

TEST(NConfigParserTest, ParseIndexOrRange_LeadingDashThrows) {
  using detail::parseIndexOrRange;
  // "-1" has dash at pos 0: startStr="" which makes std::stol throw.
  // This bubbles up as an exception (caught by parseAndApply in production).
  EXPECT_THROW(parseIndexOrRange("-1"), std::invalid_argument);
}

TEST(NConfigParserTest, ParseIndexOrRange_Zero) {
  using detail::parseIndexOrRange;
  std::vector<uint32_t> expected = {0};
  EXPECT_EQ(parseIndexOrRange("0"), expected);
}

TEST(NConfigParserTest, ParseBool_True) {
  EXPECT_EQ(std::get<bool>(NConfigParser::parseBool("true")), true);
  EXPECT_EQ(std::get<bool>(NConfigParser::parseBool("TRUE")), true);
  EXPECT_EQ(std::get<bool>(NConfigParser::parseBool("True")), true);
}

TEST(NConfigParserTest, ParseBool_False) {
  EXPECT_EQ(std::get<bool>(NConfigParser::parseBool("false")), false);
  EXPECT_EQ(std::get<bool>(NConfigParser::parseBool("FALSE")), false);
}

TEST(NConfigParserTest, ParseBool_Invalid) {
  EXPECT_THROW(NConfigParser::parseBool("yes"), std::invalid_argument);
  EXPECT_THROW(NConfigParser::parseBool("1"), std::invalid_argument);
  EXPECT_THROW(NConfigParser::parseBool("garbage"), std::invalid_argument);
}

TEST(NConfigParserTest, ParseInt64) {
  EXPECT_EQ(std::get<int64_t>(NConfigParser::parseInt64("12345")), 12345);
  EXPECT_EQ(std::get<int64_t>(NConfigParser::parseInt64("-12345")), -12345);
  EXPECT_EQ(std::get<int64_t>(NConfigParser::parseInt64("0")), 0);
}

TEST(NConfigParserTest, ParseUint64) {
  EXPECT_EQ(std::get<uint64_t>(NConfigParser::parseUint64("100000")), 100000u);
  EXPECT_EQ(std::get<uint64_t>(NConfigParser::parseUint64("0")), 0u);
}

TEST(NConfigParserTest, ParseString) {
  EXPECT_EQ(std::get<std::string>(NConfigParser::parseString("/tmp/test")), "/tmp/test");
  EXPECT_EQ(std::get<std::string>(NConfigParser::parseString("")), "");
}

TEST(NConfigParserTest, ParseNcIndexList_SingleIndices) {
  std::vector<uint32_t> expected = {0, 1, 2};
  EXPECT_EQ(std::get<std::vector<uint32_t>>(NConfigParser::parseNcIndexList("0,1,2")), expected);
}

TEST(NConfigParserTest, ParseNcIndexList_Range) {
  std::vector<uint32_t> expected = {0, 1, 2};
  EXPECT_EQ(std::get<std::vector<uint32_t>>(NConfigParser::parseNcIndexList("0-2")), expected);
}

TEST(NConfigParserTest, ParseNcIndexList_MixedRangeAndSingle) {
  std::vector<uint32_t> expected = {0, 1, 2, 7, 9};
  EXPECT_EQ(std::get<std::vector<uint32_t>>(NConfigParser::parseNcIndexList("0-2,7,9")), expected);
}

TEST(NConfigParserTest, ParseNcIndexList_InvalidRange) {
  // Invalid range (start > end) returns empty for that range
  auto result = std::get<std::vector<uint32_t>>(NConfigParser::parseNcIndexList("5-2"));
  EXPECT_TRUE(result.empty());
}

TEST(NConfigParserTest, ParseNcIndexList_InvalidRangeWithValid) {
  // Invalid range skipped, valid parts kept
  std::vector<uint32_t> expected = {8, 9};
  EXPECT_EQ(std::get<std::vector<uint32_t>>(NConfigParser::parseNcIndexList("5-2,8,9")), expected);
}

// Test context to capture applied values
struct TestContext {
  bool boolValue = false;
  int64_t int64Value = 0;
  uint64_t uint64Value = 0;
  std::string stringValue;
  std::vector<uint32_t> ncIndices;
  int appliedCount = 0;
};

// Create a comprehensive test registry
ConfigRegistry createTestRegistry(TestContext& ctx) {
  return {
      {"max_events_per_nc", NConfigParser::parseUint64,
       [&ctx](void*, const ConfigValue& v) {
         ctx.uint64Value = std::get<uint64_t>(v);
         ctx.appliedCount++;
         return true;
       }},

      {"capture_enabled_for_nc", NConfigParser::parseNcIndexList,
       [&ctx](void*, const ConfigValue& v) {
         ctx.ncIndices = std::get<std::vector<uint32_t>>(v);
         ctx.appliedCount++;
         return true;
       }},

      {"host_memory", NConfigParser::parseBool,
       [&ctx](void*, const ConfigValue& v) {
         ctx.boolValue = std::get<bool>(v);
         ctx.appliedCount++;
         return true;
       }},

      {"profile_output_dir", NConfigParser::parseString,
       [&ctx](void*, const ConfigValue& v) {
         ctx.stringValue = std::get<std::string>(v);
         ctx.appliedCount++;
         return true;
       }},

      {"int64_test", NConfigParser::parseInt64,
       [&ctx](void*, const ConfigValue& v) {
         ctx.int64Value = std::get<int64_t>(v);
         ctx.appliedCount++;
         return true;
       }},
  };
}

TEST(NConfigParserTest, ParseAndApply_EmptyConfig) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(NConfigParser::parseAndApply("", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);
}

TEST(NConfigParserTest, ParseAndApply_MultipleOptions) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(
      NConfigParser::parseAndApply("max_events_per_nc:100;host_memory:false", registry, nullptr));
  EXPECT_EQ(ctx.uint64Value, 100u);
  EXPECT_FALSE(ctx.boolValue);
  EXPECT_EQ(ctx.appliedCount, 2);
}

TEST(NConfigParserTest, ParseAndApply_TrailingSemicolon) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(NConfigParser::parseAndApply("max_events_per_nc:100;", registry, nullptr));
  EXPECT_EQ(ctx.uint64Value, 100u);
  EXPECT_EQ(ctx.appliedCount, 1);
}

TEST(NConfigParserTest, ParseAndApply_WhitespaceHandling) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(NConfigParser::parseAndApply("  max_events_per_nc : 100 ; host_memory : true  ",
                                           registry, nullptr));
  EXPECT_EQ(ctx.uint64Value, 100u);
  EXPECT_TRUE(ctx.boolValue);
  EXPECT_EQ(ctx.appliedCount, 2);
}

TEST(NConfigParserTest, ParseAndApply_EmptyPairs) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  // Double semicolon creates empty pair - should be handled gracefully
  EXPECT_TRUE(
      NConfigParser::parseAndApply("max_events_per_nc:100;;host_memory:true", registry, nullptr));
  EXPECT_EQ(ctx.uint64Value, 100u);
  EXPECT_TRUE(ctx.boolValue);
  EXPECT_EQ(ctx.appliedCount, 2);
}

TEST(NConfigParserTest, ParseAndApply_MalformedNoColon) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  // Missing colon - should warn, return true, no values set
  EXPECT_TRUE(NConfigParser::parseAndApply("max_events_per_nc100", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);
}

TEST(NConfigParserTest, ParseAndApply_UnrecognizedOption) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  // Unrecognized option - should warn, return true
  EXPECT_TRUE(NConfigParser::parseAndApply("unknown_key:value", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);
}

TEST(NConfigParserTest, ParseAndApply_UnrecognizedWithValid) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  // Mix of unrecognized and valid options - continues after unknown
  EXPECT_TRUE(
      NConfigParser::parseAndApply("unknown_key:value;max_events_per_nc:200", registry, nullptr));
  EXPECT_EQ(ctx.uint64Value, 200u);
  EXPECT_EQ(ctx.appliedCount, 1);
}

TEST(NConfigParserTest, ParseAndApply_EmptyRegistry) {
  ConfigRegistry emptyRegistry = {};
  // Should not crash with empty registry, just warn about unknown options
  EXPECT_TRUE(NConfigParser::parseAndApply("any_config:value", emptyRegistry, nullptr));
}

TEST(NConfigParserTest, ParseAndApply_ApplierFailure) {
  int callCount = 0;
  ConfigRegistry registry = {
      {"failing_config", NConfigParser::parseString,
       [&callCount](void*, const ConfigValue&) {
         callCount++;
         return false;  // Applier returns failure
       }},
  };
  // Should still return true (parsing continues), but applier failed
  EXPECT_TRUE(NConfigParser::parseAndApply("failing_config:value", registry, nullptr));
  EXPECT_EQ(callCount, 1);
}

TEST(NConfigParserTest, ParseAndApply_ContextPassed) {
  int* capturedContext = nullptr;
  int testValue = 42;

  ConfigRegistry registry = {
      {"test_config", NConfigParser::parseString,
       [&capturedContext](void* ctx, const ConfigValue&) {
         capturedContext = static_cast<int*>(ctx);
         return true;
       }},
  };

  EXPECT_TRUE(NConfigParser::parseAndApply("test_config:value", registry, &testValue));
  EXPECT_EQ(capturedContext, &testValue);
}

TEST(NConfigParserTest, ParseAndApply_ParserException) {
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  // Invalid integer value - should warn, but still return true (continues parsing)
  EXPECT_TRUE(NConfigParser::parseAndApply("max_events_per_nc:abc", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);  // Failed to apply due to parse error
}

TEST(NConfigParserTest, ParseAndApply_ValueContainingColons) {
  // Values like URLs or paths may contain colons.
  // The parser splits on the first colon only, so the value should include
  // everything after the first colon.
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(
      NConfigParser::parseAndApply("profile_output_dir:s3://bucket/traces:v2", registry, nullptr));
  EXPECT_EQ(ctx.stringValue, "s3://bucket/traces:v2");
  EXPECT_EQ(ctx.appliedCount, 1);
}

TEST(NConfigParserTest, ParseAndApply_EmptyKey) {
  // ":value" has colon at position 0, resulting in empty key.
  // Should be handled gracefully (unrecognized empty key).
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(NConfigParser::parseAndApply(":value", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);
}

TEST(NConfigParserTest, ParseAndApply_EmptyValue) {
  // "host_memory:" has empty value. parseBool("") throws invalid_argument,
  // which is caught and logged.
  TestContext ctx;
  auto registry = createTestRegistry(ctx);
  EXPECT_TRUE(NConfigParser::parseAndApply("host_memory:", registry, nullptr));
  EXPECT_EQ(ctx.appliedCount, 0);  // Parse error for empty value
}

}  // namespace
}  // namespace profiler
}  // namespace torch_neuronx
