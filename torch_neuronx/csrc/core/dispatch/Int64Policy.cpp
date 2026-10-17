#include "torch_neuronx/csrc/core/dispatch/Int64Policy.h"

#include <atomic>

namespace torch_neuronx {
namespace dispatch {

namespace {

// Immutable after InitInt64Policy(). No mutex needed for reads.
std::unordered_set<std::string> g_native_ops;
std::unordered_set<std::string> g_unsupported_ops;
std::atomic<bool> g_initialized{false};

/// Strip overload: "aten::add.Tensor" → "aten::add"
std::string StripOverload(const std::string& op_name) {
  auto colon_pos = op_name.find("::");
  if (colon_pos == std::string::npos) return op_name;
  auto after_ns = op_name.substr(colon_pos + 2);
  auto dot_pos = after_ns.find('.');
  if (dot_pos == std::string::npos) return op_name;
  return op_name.substr(0, colon_pos + 2) + after_ns.substr(0, dot_pos);
}

}  // namespace

void InitInt64Policy(const std::unordered_set<std::string>& native_ops,
                     const std::unordered_set<std::string>& unsupported_ops) {
  g_native_ops = native_ops;
  g_unsupported_ops = unsupported_ops;
  g_initialized.store(true, std::memory_order_release);
}

Int64Decision DecideInt64(const std::string& op_name) {
  if (ShouldCpuFallback(op_name)) return Int64Decision::UNSUPPORTED;

  std::string base = StripOverload(op_name);

  // Op tables are populated by InitInt64Policy. Before initialization they're
  // empty — treat all ops as "non-native" until Python wires us up.
  if (IsS64Passthrough() && g_native_ops.count(base)) {
    return Int64Decision::NATIVE;
  }
  return Int64Decision::DOWNCAST;
}

bool ShouldCpuFallback(const std::string& op_name) {
  auto mode = GetInt64Mode();
  // cpu_fallback: every int64 op runs entirely on CPU. (The dispatcher gates
  // this on the op actually having an int64 input, so float ops stay on device.)
  if (mode == Int64Mode::kCpuFallback) return true;

  // Only the native dispatch modes (native / native_strict) route int64 ops to
  // CPU. packed and cpu_bounce keep their existing on-device downcast handling,
  // so we never force a whole-op CPU fallback in those modes.
  if (mode != Int64Mode::kNative && mode != Int64Mode::kNativeStrict) {
    return false;
  }

  std::string base = StripOverload(op_name);
  // matmul/conv etc. would produce garbage in S64 — always CPU in native modes.
  if (g_unsupported_ops.count(base)) return true;

  // native_strict: anything not in the native op set runs on CPU (no precision loss).
  if (mode == Int64Mode::kNativeStrict && !g_native_ops.count(base)) {
    return true;
  }
  return false;
}

}  // namespace dispatch
}  // namespace torch_neuronx
