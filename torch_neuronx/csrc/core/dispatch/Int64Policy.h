#pragma once

#include <string>
#include <unordered_set>

// Int64Mode, GetInt64Mode(), IsS64Passthrough() are the canonical definitions
// (unified packed-view + native dispatch policy). This file only adds the
// per-op decision layer on top of them.
#include "torch_neuronx/csrc/core/dispatch/DispatchUtils.h"

namespace torch_neuronx {
namespace dispatch {

/// Per-op int64 routing decision (mirrors Python Int64Decision enum).
enum class Int64Decision { NATIVE, DOWNCAST, UNSUPPORTED };

/// Initialize the per-op int64 policy tables from Python. Called once during
/// module init (with GIL held); after this, all lookups are lock-free.
///
/// The active mode is NOT passed here — it is read from the canonical
/// GetInt64Mode() in DispatchUtils.h so there is a single source of truth.
void InitInt64Policy(const std::unordered_set<std::string>& native_ops,
                     const std::unordered_set<std::string>& unsupported_ops);

/// Per-op int64 decision. Lock-free after InitInt64Policy().
/// @param op_name  Full op name (e.g., "aten::add.Tensor") — overload stripped.
Int64Decision DecideInt64(const std::string& op_name);

/// True if the op passes int64 through as S64 (natively supported AND the
/// active mode lets int64 reach the compiler).
inline bool SupportsNativeInt64(const std::string& op_name) {
  return DecideInt64(op_name) == Int64Decision::NATIVE;
}

/// True when this op's int64 tensors execute on the packed int32[...,2] physical
/// view rather than as S64: explicit packed mode (every int64 op), or the native
/// cascade's non-native (DOWNCAST) ops. NATIVE ops return false and keep S64 to
/// the compiler. This is the single per-op gate for the input-side physical-view
/// binding (FlattenAndFilterInputs); the output side keys off the compiled
/// specs (int32 physical / int64 logical) under UsePackedInt64().
inline bool UsePackedInt64ViewForOp(const std::string& op_name) {
  return IsPackedInt64Enabled() ||
         (IsS64Passthrough() && DecideInt64(op_name) == Int64Decision::DOWNCAST);
}

/// True when int64 inputs to this op should trigger a whole-op CPU fallback.
/// Only the native / native_strict modes route int64 ops to CPU; packed and
/// cpu_bounce keep their existing on-device handling. Callers should still
/// verify the op actually has an int64 input before falling back.
bool ShouldCpuFallback(const std::string& op_name);

}  // namespace dispatch
}  // namespace torch_neuronx
