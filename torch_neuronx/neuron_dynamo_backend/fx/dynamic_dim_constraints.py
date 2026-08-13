"""Dynamic dimension size constraint analysis for FX graphs.

Extracts, from a post-AOTAutograd FX graph, the mapping between dynamic tensor
dimensions and the SymInt scalar arguments that carry their runtime sizes, so
the information can be attached as StableHLO metadata for the downstream
compiler.
"""

from dataclasses import dataclass

import sympy
import torch
import torch.fx


@dataclass
class DynamicDimConstraints:
    """Dynamic dimension size constraint metadata extracted from an FX graph.

    Captures two pieces of information for propagation to the downstream compiler:

    1. ``symbol_to_arg_index``: maps each sympy symbol name (e.g. "s0") to its
       positional argument index in the function signature. This tells the compiler
       which runtime scalar argument provides the value for that symbol.

    2. ``tensor_dim_exprs``: maps each tensor argument index to a dict of
       ``{dim_index: expr_string}`` for its dynamic dimensions. The expression
       string is a sympy expression (e.g. "s0", "s0 + 1", "2*s1") that the
       compiler is responsible for parsing and lowering to arithmetic.
    """

    symbol_to_arg_index: dict[str, int]
    tensor_dim_exprs: dict[int, dict[int, str]]


def build_dynamic_dim_constraints(g: torch.fx.Graph) -> DynamicDimConstraints | None:
    """Extract dynamic dimension constraint metadata from an FX graph.

    Scans placeholder nodes to identify SymInt arguments and the symbolic
    expressions on each tensor argument's dynamic dimensions.

    Returns:
        DynamicDimConstraints if any dynamic dimensions exist, else None.
    """
    placeholders = [node for node in g.nodes if node.op == "placeholder"]

    # Map sympy symbol name -> placeholder arg index for SymInt inputs
    symbol_to_arg_index: dict[str, int] = {}
    for idx, node in enumerate(placeholders):
        val = node.meta.get("val")
        if isinstance(val, torch.SymInt):
            expr: sympy.Expr = val.node.expr
            if expr.is_Symbol:
                symbol_to_arg_index[str(expr)] = idx
            else:
                assert expr.is_Integer, (
                    f"SymInt input placeholder {node.name!r} has non-symbol, non-constant "
                    f"expression {expr!r}; expected a bare symbol (e.g. 's0') or a constant "
                    f"int. Compound-expression SymInt inputs are not supported for dynamic "
                    f"dimension constraints."
                )

    if not symbol_to_arg_index:
        return None

    # For each tensor placeholder with symbolic dims, record dimension expressions
    tensor_dim_exprs: dict[int, dict[int, str]] = {}
    for idx, node in enumerate(placeholders):
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if not hasattr(val, "_has_symbolic_sizes_strides") or not val._has_symbolic_sizes_strides:
            continue

        dim_exprs: dict[int, str] = {}
        for dim_idx, size in enumerate(val.size()):
            if isinstance(size, torch.SymInt):
                dim_expr: sympy.Expr = size.node.expr
                unresolved = {str(s) for s in dim_expr.free_symbols} - symbol_to_arg_index.keys()
                assert not unresolved, (
                    f"Tensor arg {idx} dim {dim_idx} expression {dim_expr!r} references "
                    f"symbol(s) {sorted(unresolved)} with no scalar input arg; "
                    f"known symbols: {sorted(symbol_to_arg_index)}."
                )
                dim_exprs[dim_idx] = str(dim_expr)

        if dim_exprs:
            tensor_dim_exprs[idx] = dim_exprs

    if not tensor_dim_exprs:
        return None

    return DynamicDimConstraints(symbol_to_arg_index, tensor_dim_exprs)


__all__ = ["DynamicDimConstraints", "build_dynamic_dim_constraints"]
