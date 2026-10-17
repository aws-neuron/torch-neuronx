"""
Rewrite a causal attn_mask on scaled_dot_product_attention to is_causal=True.

A model can pass a dense mask even when the mask only expresses causality. A mask
operand makes the SDPA decomposition reject the NKI flash kernel (see
``_can_use_nki_flash_attention``), so the call materializes the L x S scores.
Passing is_causal=True instead expresses the same mask and keeps the kernel.

The pass accepts a bool keep-mask and keeps every other mask, including an
additive float mask. A float mask drops a position with a large negative
sentinel, and reading a sentinel back depends on the float width, so the pass
does not read one.

The pass folds the mask's producers on the host and matches the folded value. It
never matches on the producer ops: a sliding-window mask and a key-offset mask
build the same comparison ops as a plain causal mask.

The pass must run before AOTAutograd, because the decomposition consumes the
mask and no SDPA node survives into the post-AOT graph.

The pass keeps the mask when the query length and the key length differ. The
nkilib backward kernel asserts that the two lengths match under a causal mask.

The fold runs producers that the pass does not vet, which is safe for two
reasons. The producers read no placeholder and no module attribute, so every
value comes from a literal in the trace and nothing can read host or device
state. The folded value must then equal tril over every cell. An op that the
fold cannot run raises and the pass keeps the mask, so a new producer op costs
coverage and never correctness.

Keep that comparison exact. A tolerance, a sampled subset, or an approximate
match removes the second reason above, and the fold would then need a vetted set
of producer ops to stay sound.
"""

import inspect
import logging
import operator
import types

import torch
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.aliasing_analysis import AliasingAnalysis
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass

logger = logging.getLogger(__name__)

_SDPA_TARGETS = {
    torch._C._nn.scaled_dot_product_attention,
    torch.ops.aten.scaled_dot_product_attention,
    torch.ops.aten.scaled_dot_product_attention.default,
}

# The schema gives the parameter order, so a rebuilt call keeps every argument.
_SDPA_POSITIONAL = tuple(
    arg.name for arg in torch.ops.aten.scaled_dot_product_attention.default._schema.arguments
)

# A folded mask and the cone that produced it, so the caller can erase the cone.
_Folded = tuple[torch.Tensor, frozenset[torch.fx.Node]]


def _example_value(node: torch.fx.Node):
    """Return a node's traced tensor, or None. Dynamo names it ``example_value``."""
    meta = getattr(node, "meta", None)
    return meta.get("example_value") if meta else None


def _cpu_if_device(value):
    """Return CPU for a device specifier, otherwise the value unchanged.

    A string counts as a device only if torch.device parses it. An unrelated
    string argument, such as rounding_mode="floor", must reach the op unchanged.
    """
    if isinstance(value, torch.device):
        return torch.device("cpu")
    if isinstance(value, str):
        try:
            torch.device(value)
        except (RuntimeError, ValueError):
            return value
        return torch.device("cpu")
    return value


def _normalized_sdpa_kwargs(node: torch.fx.Node) -> dict:
    """Return every SDPA argument by name, whether it was passed positionally."""
    merged = dict(node.kwargs)
    for name, value in zip(_SDPA_POSITIONAL, node.args, strict=False):
        merged[name] = value
    return merged


def _writes_through_inplace_flag(user: torch.fx.Node) -> bool:
    """Return True if ``user`` passes a truthy inplace flag.

    A Python functional such as F.hardtanh writes through inplace=True. It carries
    no schema and no trailing underscore, so the flag is the only evidence. The
    flag can arrive positionally, so bind it against the signature.
    """
    if "inplace" not in user.kwargs and not isinstance(user.target, types.FunctionType):
        return False
    try:
        parameters = inspect.signature(user.target).parameters
    except (TypeError, ValueError):
        return bool(user.kwargs.get("inplace"))
    if "inplace" not in parameters:
        return False
    try:
        bound = inspect.signature(user.target).bind_partial(*user.args, **user.kwargs)
    except TypeError:
        # The flag is reachable but unreadable, so treat the call as a write.
        return True
    return bound.arguments.get("inplace", parameters["inplace"].default) is not False


def _mutates(user: torch.fx.Node) -> bool:
    """Return True if ``user`` can write to one of its arguments in place."""
    target = user.target
    if target is operator.setitem or target in AliasingAnalysis.INPLACE_OPERATORS:
        return True
    if user.kwargs.get("out") is not None:
        return True
    if _writes_through_inplace_flag(user):
        return True
    # Every remaining operator builtin is pure, and the underscore rule below
    # would read the ones that spell a keyword, such as operator.and_, as a write.
    if getattr(target, "__module__", None) == operator.add.__module__:
        return False
    # A torch builtin declares no schema, so a trailing underscore is the only
    # remaining evidence of a write. Every dunder ends in one, so a dunder needs
    # the in-place and setter prefixes instead. That over-reads __invert__ as a
    # write, which costs coverage and keeps the mask.
    name = target if isinstance(target, str) else getattr(target, "__name__", "")
    if name.startswith("__") and name.endswith("__"):
        return name.startswith(("__i", "__set", "__del"))
    if name.endswith("_"):
        return True
    schema = getattr(target, "_schema", None)
    return bool(schema is not None and schema.is_mutable)


def _storage_id(node: torch.fx.Node) -> int | None:
    """Return the identity of a node's traced storage, or None if it has none.

    Dynamo traces with fake tensors, which model storage sharing faithfully: a
    view reports its base's storage and a copy reports a new one. Two nodes that
    report one storage therefore share their contents.
    """
    value = _example_value(node)
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return value.untyped_storage()._cdata
    except Exception:
        # A tensor without a storage, such as a nested one, has no identity to
        # compare. The callers treat None as unknown and fail closed.
        return None


class SdpaCausalMaskRewrite(NeuronFXPass):
    """Replace a provably causal attn_mask with is_causal=True.

    Args:
        state (NeuronFXPassState, optional): Shared FX-pass state used for artifact dumping.
            Defaults to ``None``.
    """

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        """Replace provably causal SDPA masks with the causal flag.

        Args:
            gm (torch.fx.GraphModule): Graph module to transform in place.

        Returns:
            PassResult: The graph module and whether any mask was replaced.

        Note:
            Unexpected rewrite failures keep the original mask.
        """
        # The fold memo is scoped to one graph.
        self._folded: dict[torch.fx.Node, _Folded | None] = {}
        orphaned: set[torch.fx.Node] = set()
        modified = False

        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target not in _SDPA_TARGETS:
                continue
            try:
                modified |= self._rewrite(node, orphaned)
            except Exception:
                # The pass is an optimization and the caller re-raises, so an
                # unexpected value must not fail a compile that worked before.
                logger.debug("%s: the rewrite failed. The pass keeps the mask.", node.name)

        if modified:
            self._erase_orphaned(gm.graph, orphaned)
            gm.graph.lint()
            gm.recompile()

        return PassResult(gm, modified)

    def _rewrite(self, node: torch.fx.Node, orphaned: set[torch.fx.Node]) -> bool:
        """Trade a causal mask for the flag on one SDPA node. Return True if it fired."""
        args = _normalized_sdpa_kwargs(node)
        mask = args.get("attn_mask")
        if not isinstance(mask, torch.fx.Node):
            return False
        if args.get("dropout_p", 0.0):
            logger.debug("%s: dropout_p is not zero. The pass keeps the mask.", node.name)
            return False

        if mask not in self._folded:
            self._folded[mask] = self._fold(mask)
        if self._folded[mask] is None:
            return False
        value, cone = self._folded[mask]

        # The extents are this call's, so two calls sharing a mask can differ.
        if not self._is_causal(mask.name, value, self._operand_extents(args)):
            return False

        # Rebuild args and kwargs: a positional mask must go, a positional dropout_p
        # stays. Build both, then assign, so no raise can leave the node half-written.
        args.pop("attn_mask")
        args["is_causal"] = True
        operands = (args.pop("query"), args.pop("key"), args.pop("value"))
        node.args = operands
        node.kwargs = args
        orphaned |= cone
        logger.debug("%s: the mask is causal. The pass sets is_causal=True.", node.name)
        return True

    def _is_causal(
        self, name: str, value: torch.Tensor, extents: tuple[int, int, int, int] | None
    ) -> bool:
        """Return True if is_causal=True is equivalent to this mask value.

        Every gate fails closed. SDPA broadcasts the mask, so a rank from 2 to 4
        is legal and a leading extent above the operands' widens the output. The
        code does not show these reasons:

        - A float mask drops a position with a sentinel. A read of that sentinel
          depends on the float width.
        - tril of an (L, 1) mask is all-true. The extent gate is what rejects an
          all-true mask.
        - The nkilib backward kernel asserts that the query length and the key
          length match under a causal mask.
        - An empty mask holds no plane to compare.

        The retained plane must then equal tril over every cell. Index the leading
        axes to reach the plane, because reshape copies a mask that an expand made
        wider.
        """
        if value.dtype != torch.bool:
            logger.debug("%s: the mask dtype is %s, not bool", name, value.dtype)
            return False

        if not 2 <= value.ndim <= 4:
            logger.debug("%s: mask rank %d is not broadcastable", name, value.ndim)
            return False

        if extents is None or tuple(value.shape[-2:]) != extents[-2:]:
            logger.debug("%s: mask extent %s is not (Lq, Skv)=%s", name, value.shape, extents)
            return False

        if extents[-2] != extents[-1]:
            logger.debug("%s: the extents %s are not square", name, extents)
            return False

        if not value.numel():
            logger.debug("%s: the mask %s is empty", name, value.shape)
            return False

        leading = zip(reversed(value.shape[:-2]), reversed(extents[:2]), strict=False)
        if any(mask_dim > operand_dim for mask_dim, operand_dim in leading):
            logger.debug("%s: the mask %s widens (B, H)", name, value.shape)
            return False

        keep = value[(0,) * (value.ndim - 2)]
        if not torch.equal(value, keep.expand_as(value)):
            logger.debug("%s: mask varies across batch or head", name)
            return False

        q_len, k_len = keep.shape
        with torch.device("cpu"):
            causal = torch.ones(q_len, k_len, dtype=torch.bool).tril_()
        return bool(torch.equal(keep, causal))

    @staticmethod
    def _operand_extents(args: dict) -> tuple[int, int, int, int] | None:
        """Return (batch, heads, query length, key length) from the SDPA operands."""
        query, key = args.get("query"), args.get("key")
        if not isinstance(query, torch.fx.Node) or not isinstance(key, torch.fx.Node):
            return None
        qv, kv = _example_value(query), _example_value(key)
        if not isinstance(qv, torch.Tensor) or not isinstance(kv, torch.Tensor):
            return None
        if qv.ndim != 4 or kv.ndim != 4:
            return None
        extents = (*qv.shape[:2], qv.shape[-2], kv.shape[-2])
        if any(not isinstance(dim, int) for dim in extents):
            return None
        return extents

    def _fold(self, mask: torch.fx.Node) -> _Folded | None:
        """Evaluate the mask's producer subgraph on the host.

        Returns the folded value and the cone that produced it, so the caller can
        erase exactly those nodes. Returns None when the fold is refused.
        """
        cone = self._cone(mask)
        if cone is None:
            return None

        for node in cone:
            value = _example_value(node)
            if not isinstance(value, torch.Tensor):
                logger.debug("%s: %s has no traced tensor", mask.name, node.name)
                return None
            if any(not isinstance(dim, int) for dim in value.shape):
                logger.debug("%s: an extent is symbolic. The fold needs static extents.", node.name)
                return None

        env: dict[torch.fx.Node, object] = {}
        try:
            # A producer without a device argument takes the ambient default device,
            # so pin CPU for the whole replay.
            with torch.no_grad(), torch.device("cpu"):
                for node in cone:
                    args = torch.fx.node.map_arg(node.args, env.__getitem__)
                    kwargs = torch.fx.node.map_arg(node.kwargs, env.__getitem__)
                    # A trace bakes in its device, and the fold runs on the host.
                    args = tuple(_cpu_if_device(a) for a in args)
                    kwargs = {k: _cpu_if_device(v) for k, v in kwargs.items()}
                    if node.op == "call_function":
                        env[node] = node.target(*args, **kwargs)
                    else:
                        env[node] = getattr(args[0], node.target)(*args[1:], **kwargs)
        except Exception:
            # The fold runs producers the pass has not vetted, so a raise is an
            # expected outcome rather than a defect. Keeping the mask is correct.
            logger.debug("%s: the host fold failed. The pass keeps the mask.", mask.name)
            return None

        folded = env.get(mask)
        if not isinstance(folded, torch.Tensor):
            return None
        return folded, frozenset(cone)

    @staticmethod
    def _cone(mask: torch.fx.Node) -> list[torch.fx.Node] | None:
        """Return the mask's producers in graph order, or None if unfoldable.

        Returns None when the value is not fixed at compile time. A placeholder or
        a module attribute can hold different contents on a later call, and the
        compiled artifact carries no guard on tensor contents.
        """
        graph = mask.graph
        keep: set[torch.fx.Node] = set()
        frontier = [mask]
        while frontier:
            node = frontier.pop()
            if node in keep:
                continue
            if node.op not in ("call_function", "call_method"):
                logger.debug("%s: cone reaches %s %r", mask.name, node.op, node.name)
                return None
            keep.add(node)
            frontier.extend(node.all_input_nodes)

        # A view hop is not a mutation, but a write through the view changes the
        # folded value, so follow every user that shares a producer's storage.
        storages = {_storage_id(node) for node in keep} - {None}
        seen = set(keep)
        frontier = list(keep)
        while frontier:
            for user in frontier.pop().users:
                if _mutates(user):
                    logger.debug(
                        "%s: %s writes to a mask producer. The pass keeps the mask.",
                        mask.name,
                        user.name,
                    )
                    return None
                if user in seen:
                    continue
                # A node that holds no tensor can still carry one, as a tuple of
                # views does, so follow it. The walk stops at the first copy.
                storage = _storage_id(user)
                if storage is None or storage in storages:
                    seen.add(user)
                    frontier.append(user)

        return [n for n in graph.nodes if n in keep]

    @staticmethod
    def _erase_orphaned(graph: torch.fx.Graph, orphaned: set[torch.fx.Node]) -> None:
        """Erase the mask producers the rewrite left without users.

        Scoped to the folded cones. Whole-graph DCE would also drop a dead
        in-place call, which ``Node.is_impure`` does not report as impure.
        """
        for node in reversed(list(graph.nodes)):
            if node in orphaned and not node.users:
                graph.erase_node(node)
