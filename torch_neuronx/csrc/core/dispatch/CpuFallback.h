#pragma once

#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace torch_neuronx {
namespace dispatch {

namespace py = pybind11;

/// Run PyTorch's ``at::native::cpu_fallback`` for the named op, using the
/// supplied Python positional and keyword arguments.
///
/// Honors every alias contract the schema declares: ``Tensor(a!) self`` is
/// mutated in place and returned, ``Tensor(a!) out`` is written and returned,
/// multi-return tuples are correctly placed back on the original device, and
/// kwargs are filtered against the schema rather than blindly forwarded.
///
/// This is the same machinery the C++ dispatcher uses for its own fallback
/// path. Exposed to Python so the python-dispatch CPU-fallback in
/// ``op_impl.py`` doesn't have to re-implement (and inevitably get wrong)
/// the alias-handling logic.
///
/// @param op_name  Full op name with overload, e.g. ``"aten::mm.out"``.
///                 Overload-less ops use just ``"aten::name"``.
/// @param args     Positional arguments (whatever the schema expects).
/// @param kwargs   Keyword arguments.
/// @return The op's return value(s) — a single Python object for
///         single-return ops, a Python tuple for multi-return ops, ``None``
///         for void ops.
py::object RunCpuFallback(const std::string& op_name, py::args args, py::kwargs kwargs);

namespace detail {

/// Split ``"ns::name.overload"`` into ``("ns::name", "overload")``;
/// ``"ns::name"`` (no overload) returns ``("ns::name", "")``.
/// Exposed only for unit tests.
std::pair<std::string, std::string> SplitOpName(const std::string& op_name);

}  // namespace detail
}  // namespace dispatch
}  // namespace torch_neuronx
