#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace torch_neuronx {
namespace profiler {

// Value types supported by config parser
using ConfigValue = std::variant<bool, int64_t, uint64_t, std::string, std::vector<uint32_t>>;

// Parser: string -> typed value (throws on error)
using ValueParser = std::function<ConfigValue(const std::string&)>;

// Applier: apply parsed value to opaque context (returns success)
using ConfigApplier = std::function<bool(void* context, const ConfigValue&)>;

struct ConfigDef {
  std::string name;
  ValueParser parser;
  ConfigApplier applier;
};

// Registry is a vector of config definitions
using ConfigRegistry = std::vector<ConfigDef>;

/**
 * Config format: semicolon-delimited key:value pairs
 *   max_events_per_nc:100000;capture_enabled_for_nc:0,1,2-5;host_memory:true
 *
 * Supported value types (via built-in parsers):
 *   - bool: "true", "false"
 *   - int64: Signed 64-bit integer
 *   - uint64: Unsigned 64-bit integer
 *   - string: Raw string value
 *   - NC index list: Comma-separated indices or ranges (e.g., "0,1,2" or "0-2,7")
 */
class NConfigParser {
 public:
  static ConfigValue parseBool(const std::string& val);
  static ConfigValue parseInt64(const std::string& val);
  static ConfigValue parseUint64(const std::string& val);
  static ConfigValue parseString(const std::string& val);
  static ConfigValue parseNcIndexList(const std::string& val);  // "0,1,2" or "0-2,7"

  // Parse config string using provided registry, apply to context
  // Returns true if parsing completed.
  // Individual config errors are logged as warning but don't fail.
  static bool parseAndApply(const std::string& customConfig, const ConfigRegistry& registry,
                            void* context);
};

// Internal helpers exposed for testing
namespace detail {
std::string trim(const std::string& s);
std::vector<std::string> splitAndTrim(const std::string& s, char delim);
std::vector<uint32_t> parseIndexOrRange(const std::string& s);
}  // namespace detail

}  // namespace profiler
}  // namespace torch_neuronx
