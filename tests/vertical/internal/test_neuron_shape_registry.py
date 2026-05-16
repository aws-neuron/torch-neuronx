"""Tests for NeuronShapeRegistry and NeuronOpInfo."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from .neuron_shape_registry import NeuronShapeRegistry, OpShapeEntry

# ── NeuronShapeRegistry ─────────────────────────────────────────────────────


class TestNeuronShapeRegistry:
    def test_register_and_get(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="test")
        entries = reg.get("aten::mm")
        assert len(entries) == 1
        assert entries[0].arg_shapes == {"self": (32, 64), "mat2": (64, 32)}
        assert entries[0].sources == {"test": 1}

    def test_dedup_same_shapes(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="llama", count=10
        )
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="mistral", count=5
        )
        entries = reg.get("aten::mm")
        assert len(entries) == 1
        assert entries[0].total_count == 15
        assert entries[0].sources == {"llama": 10, "mistral": 5}

    def test_dedup_accumulates_same_source(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="llama", count=10
        )
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="llama", count=3
        )
        entries = reg.get("aten::mm")
        assert len(entries) == 1
        assert entries[0].sources["llama"] == 13

    def test_different_shapes_not_deduped(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="test")
        reg.register("aten::mm", arg_shapes={"self": (128, 256), "mat2": (256, 128)}, source="test")
        assert len(reg.get("aten::mm")) == 2

    def test_sorted_by_count(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 32), "mat2": (32, 32)}, source="a", count=5
        )
        reg.register(
            "aten::mm", arg_shapes={"self": (64, 64), "mat2": (64, 64)}, source="a", count=50
        )
        entries = reg.get("aten::mm")
        assert entries[0].arg_shapes["self"] == (64, 64)  # higher count first

    def test_none_values_preserved(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::linear",
            arg_shapes={"input": (32, 64), "weight": (128, 64), "bias": None},
            source="test",
        )
        entry = reg.get("aten::linear")[0]
        assert entry.arg_shapes["bias"] is None

    def test_list_shapes_normalized_to_tuples(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": [32, 64], "mat2": [64, 32]}, source="test")
        entry = reg.get("aten::mm")[0]
        assert entry.arg_shapes["self"] == (32, 64)
        assert isinstance(entry.arg_shapes["self"], tuple)

    def test_has(self):
        reg = NeuronShapeRegistry()
        assert not reg.has("aten::mm")
        reg.register("aten::mm", arg_shapes={"self": (32, 32), "mat2": (32, 32)}, source="test")
        assert reg.has("aten::mm")

    def test_all_ops(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": (32, 32), "mat2": (32, 32)}, source="test")
        reg.register(
            "aten::add.Tensor", arg_shapes={"self": (4, 4), "other": (4, 4)}, source="test"
        )
        assert reg.all_ops() == ["aten::add.Tensor", "aten::mm"]

    def test_len(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": (32, 32), "mat2": (32, 32)}, source="test")
        reg.register("aten::mm", arg_shapes={"self": (64, 64), "mat2": (64, 64)}, source="test")
        reg.register(
            "aten::add.Tensor", arg_shapes={"self": (4, 4), "other": (4, 4)}, source="test"
        )
        assert len(reg) == 3

    def test_get_empty(self):
        reg = NeuronShapeRegistry()
        assert reg.get("aten::nonexistent") == []

    # ── CSV round-trip ───────────────────────────────────────────────────────

    def test_save_and_load_csv(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="llama", count=10
        )
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="mistral", count=5
        )
        reg.register(
            "aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (11008, 4096), "bias": None},
            source="llama",
            count=47,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        reg.save_csv(path)

        # Load into fresh registry
        reg2 = NeuronShapeRegistry()
        loaded = reg2.load_csv(path)
        assert loaded == 3  # 2 source rows for mm + 1 for linear

        mm_entries = reg2.get("aten::mm")
        assert len(mm_entries) == 1
        assert mm_entries[0].total_count == 15

        linear_entries = reg2.get("aten::linear")
        assert len(linear_entries) == 1
        assert linear_entries[0].arg_shapes["bias"] is None
        assert linear_entries[0].total_count == 47

        Path(path).unlink()

    def test_csv_content_format(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::mm", arg_shapes={"self": (32, 64), "mat2": (64, 32)}, source="test", count=7
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        reg.save_csv(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["op_name"] == "aten::mm"
        assert rows[0]["source_model"] == "test"
        assert rows[0]["count"] == "7"
        shapes = json.loads(rows[0]["arg_shapes"])
        assert shapes["self"] == [32, 64]

        Path(path).unlink()

    # ── JSON loading ─────────────────────────────────────────────────────────

    def test_load_json(self):
        data = [
            {
                "op_name": "aten::mm",
                "arg_shapes": {"self": [256, 4096], "mat2": [4096, 4096]},
                "count": 32,
            },
            {
                "op_name": "aten::linear",
                "arg_shapes": {"input": [1, 2048, 4096], "weight": [11008, 4096], "bias": None},
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        reg = NeuronShapeRegistry()
        loaded = reg.load_json(path, source="llama-7b")
        assert loaded == 2

        mm = reg.get("aten::mm")
        assert len(mm) == 1
        assert mm[0].arg_shapes["self"] == (256, 4096)
        assert mm[0].sources["llama-7b"] == 32

        linear = reg.get("aten::linear")
        assert linear[0].arg_shapes["bias"] is None
        assert linear[0].sources["llama-7b"] == 1  # default count

        Path(path).unlink()

    def test_load_json_default_source_from_filename(self):
        data = [{"op_name": "aten::relu", "arg_shapes": {"self": [4, 4]}}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="mistral_shapes_"
        ) as f:
            json.dump(data, f)
            path = f.name

        reg = NeuronShapeRegistry()
        reg.load_json(path)
        entry = reg.get("aten::relu")[0]
        assert Path(path).stem in next(iter(entry.sources.keys()))

        Path(path).unlink()

    # ── Multi-model consolidation ────────────────────────────────────────────

    def test_multi_model_consolidation(self):
        """Simulate loading traces from multiple models and consolidating."""
        reg = NeuronShapeRegistry()

        # Model 1: llama-7b
        reg.register(
            "aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (11008, 4096), "bias": None},
            source="llama-7b",
            count=47,
        )
        reg.register(
            "aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (4096, 4096), "bias": (4096,)},
            source="llama-7b",
            count=32,
        )

        # Model 2: mistral-7b (same gate_proj shape as llama)
        reg.register(
            "aten::linear",
            arg_shapes={"input": (1, 2048, 4096), "weight": (11008, 4096), "bias": None},
            source="mistral-7b",
            count=47,
        )

        entries = reg.get("aten::linear")
        assert len(entries) == 2  # two unique shape combos

        # The shared shape should have highest count
        top = entries[0]
        assert top.total_count == 94  # 47 + 47
        assert top.sources == {"llama-7b": 47, "mistral-7b": 47}

    def test_stats(self):
        reg = NeuronShapeRegistry()
        reg.register("aten::mm", arg_shapes={"self": (32, 32), "mat2": (32, 32)}, source="test")
        reg.register("aten::mm", arg_shapes={"self": (64, 64), "mat2": (64, 64)}, source="test")
        reg.register("aten::relu", arg_shapes={"self": (4, 4)}, source="test")
        s = reg.stats()
        assert s["total_ops"] == 2
        assert s["total_shapes"] == 3


# ── Propagation to compatible overloads ─────────────────────────────────────


class TestPropagation:
    """Pre-propagation of base-op entries onto schema-compatible overloads.

    These tests exercise the runtime-free replacement for the legacy
    ``.out`` / ``.default`` fallbacks in ``NeuronOpInfo._registry_entries``.
    """

    def test_propagates_to_out_variant(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::amax",
            {"self": (4, 4), "dim": [0], "keepdim": False},
            source="builtin",
        )
        n = reg.propagate_to_compatible_overloads({"aten::amax", "aten::amax.out"})
        assert n == 1
        assert reg.has("aten::amax.out")
        # source attribution preserved
        assert "builtin" in reg.get("aten::amax.out")[0].sources

    def test_skips_already_registered(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::amax",
            {"self": (4, 4), "dim": [0], "keepdim": False},
            source="builtin",
        )
        reg.register(
            "aten::amax.out",
            {"self": (256, 256), "dim": [0], "keepdim": False, "out": (1, 256)},
            source="manual",
        )
        n = reg.propagate_to_compatible_overloads({"aten::amax", "aten::amax.out"})
        assert n == 0
        # existing entry untouched; no extra shape appended
        entries = reg.get("aten::amax.out")
        assert len(entries) == 1
        assert entries[0].arg_shapes["self"] == (256, 256)

    def test_skips_schema_incompatible(self):
        reg = NeuronShapeRegistry()
        # scatter.src entry has `src:` positional; scatter.value_out expects
        # `value: Scalar` (required, no default). Copying the entry would
        # leave `value` missing in the propagated sample.
        reg.register(
            "aten::scatter.src",
            {
                "self": (8, 8),
                "dim": 0,
                "index": {"shape": (8, 8), "dtype": "long"},
                "src": (8, 8),
            },
            source="builtin",
        )
        n = reg.propagate_to_compatible_overloads({"aten::scatter.src", "aten::scatter.value_out"})
        assert n == 0
        assert not reg.has("aten::scatter.value_out")

    def test_idempotent(self):
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::amax",
            {"self": (4, 4), "dim": [0], "keepdim": False},
            source="builtin",
        )
        reg.propagate_to_compatible_overloads({"aten::amax", "aten::amax.out"})
        # Second call is a no-op because tgt is now registered (has(tgt) skip).
        n = reg.propagate_to_compatible_overloads({"aten::amax", "aten::amax.out"})
        assert n == 0
        assert len(reg.get("aten::amax.out")) == 1

    def test_does_not_propagate_from_writeout_to_non_writeout(self):
        """Propagation is src-is-non-writeout → tgt-is-writeout (or .default)."""
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::amax.out",
            {"self": (8, 8), "dim": [0], "keepdim": False, "out": (1, 8)},
            source="manual",
        )
        reg.propagate_to_compatible_overloads({"aten::amax", "aten::amax.out"})
        # aten::amax (base, non-writeout) is NOT a write-out variant, and
        # amax.out's extra kwarg is write-only, so the schema pair is still
        # compatible; but amax's schema lacks `out` so entries CAN in theory
        # propagate backward. The implementation specifically gates targets on
        # being write-out (or .default), so the non-writeout base is untouched.
        assert not reg.has("aten::amax")

    def test_skips_required_kwarg_without_default(self):
        """A target with a required NON-write-alias kwarg (no default) is incompatible.

        Regression: aten::scatter.reduce_out has ``str reduce`` as a required
        kwarg. An entry from aten::scatter.src lacks it — propagating would
        silently produce an invalid sample at test time.
        """
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::scatter.src",
            {
                "self": (8, 8),
                "dim": 0,
                "index": {"shape": (8, 8), "dtype": "long"},
                "src": (8, 8),
            },
            source="builtin",
        )
        n = reg.propagate_to_compatible_overloads({"aten::scatter.src", "aten::scatter.reduce_out"})
        assert n == 0
        assert not reg.has("aten::scatter.reduce_out")

    def test_skips_dimname_variant_from_int_dim_entry(self):
        """Type-aware compat: an entry with ``dim: int`` must NOT propagate to
        a sibling schema whose ``dim`` is ``str`` / ``List[str]`` (dimname
        variants). Previously only arg names were checked; value-type
        incompatibilities silently produced invalid propagated entries."""
        reg = NeuronShapeRegistry()
        # aten::all.dim takes `dim: int`, .dimname_out takes `dim: str`.
        reg.register(
            "aten::all.dim",
            {"self": (256, 4096), "dim": 1, "keepdim": False},
            source="builtin",
        )
        n = reg.propagate_to_compatible_overloads({"aten::all.dim", "aten::all.dimname_out"})
        assert n == 0
        assert not reg.has("aten::all.dimname_out")

    def test_propagates_to_default_sibling(self):
        """Base dotless → .default overload coverage (replaces legacy fallback)."""
        reg = NeuronShapeRegistry()
        reg.register(
            "aten::linear", {"input": (4, 8), "weight": (16, 8), "bias": None}, source="builtin"
        )
        n = reg.propagate_to_compatible_overloads({"aten::linear", "aten::linear.default"})
        assert n == 1
        assert reg.has("aten::linear.default")


# ── max_tensor_elems ─────────────────────────────────────────────────────────


class TestMaxTensorElems:
    def test_nested_dict_shape(self):
        # Model-traced format: {"self": {"dtype": "...", "shape": [...]}}
        from .neuron_shape_registry import max_tensor_elems

        assert (
            max_tensor_elems({"self": {"dtype": "bfloat16", "shape": [16, 5120]}}, "aten::log")
            == 16 * 5120
        )

    def test_bare_tuple_under_tensor_key(self):
        # Most common form in hand-registered entries. Schema says ``self``
        # is a tensor, so the bare tuple must be treated as its shape.
        from .neuron_shape_registry import max_tensor_elems

        assert max_tensor_elems({"self": (256, 4096)}, "aten::log") == 256 * 4096

    def test_bare_tuple_multi_tensor_args(self):
        # linear has tensor args ``input``, ``weight``, ``bias`` per schema.
        # Result is the max over the three — not the sum, not just one.
        from .neuron_shape_registry import max_tensor_elems

        assert (
            max_tensor_elems(
                {"input": (32, 64), "weight": (10, 64), "bias": (10,)},
                "aten::linear",
            )
            == 32 * 64
        )  # input has 2048 elems, weight has 640, bias has 10

    def test_factory_size_arg(self):
        # For ops like aten::zeros(int[] size, ...), ``size`` is a non-tensor
        # IntList arg per schema — so max_tensor_elems returns 0 for it.
        # (The tool is shape-input ranker, not output-size ranker.)
        from .neuron_shape_registry import max_tensor_elems

        assert max_tensor_elems({"size": (8192, 2880)}, "aten::zeros") == 0

    def test_ignores_parameter_lists(self):
        # layer_norm: ``input`` / ``weight`` are tensors; ``normalized_shape``
        # is int[] — MUST NOT inflate the ranking.
        from .neuron_shape_registry import max_tensor_elems

        args = {
            "input": {"dtype": "bfloat16", "shape": [1, 2048, 4096]},
            "normalized_shape": [4096],
            "weight": (4096,),
            "bias": (4096,),
            "eps": 1e-6,
        }
        assert max_tensor_elems(args, "aten::layer_norm") == 1 * 2048 * 4096

    def test_ignores_split_sizes(self):
        # split_with_sizes.out has ``split_sizes`` as int[] — must not be
        # counted despite its large values.
        from .neuron_shape_registry import max_tensor_elems

        args = {
            "self": (1024,),
            "split_sizes": [262144, 262144, 262144, 64],
            "dim": -1,
        }
        assert max_tensor_elems(args, "aten::split_with_sizes") == 1024

    def test_foreach_list_of_specs(self):
        # _foreach_* ops take Tensor[] — value is list of per-tensor specs.
        from .neuron_shape_registry import max_tensor_elems

        args = {"self": [{"shape": (4, 4)}, {"shape": (8, 8)}, {"shape": (2,)}]}
        assert max_tensor_elems(args, "aten::_foreach_neg") == 64  # max over the three

    def test_foreach_list_of_bare_tuples(self):
        # Alternate list-of-tuples form accepted by _tensor_list_from_spec.
        from .neuron_shape_registry import max_tensor_elems

        args = {"tensors": [(4, 4), (8, 8), (2,)]}
        assert max_tensor_elems(args, "aten::cat") == 64

    def test_unresolvable_op_returns_zero(self):
        # Unknown op name → schema missing → degrades to 0 (sort falls back
        # to registration-order, safer than over-counting).
        from .neuron_shape_registry import max_tensor_elems

        assert max_tensor_elems({"self": (256, 4096)}, "aten::not_a_real_op") == 0

    def test_empty_and_none(self):
        from .neuron_shape_registry import max_tensor_elems

        assert max_tensor_elems({}, "aten::log") == 0
        assert max_tensor_elems({"self": None}, "aten::log") == 0
        assert max_tensor_elems(None, "aten::log") == 0


# ── NeuronOpInfo ─────────────────────────────────────────────────────────────


class TestNeuronOpInfo:
    @pytest.fixture(autouse=True)
    def _import_op_info(self):
        try:
            from .neuron_op_info import NeuronOpInfo

            self.NeuronOpInfo = NeuronOpInfo
        except ImportError as e:
            pytest.skip(f"NeuronOpInfo not importable: {e}")

    def test_from_schema_resolves_op(self):
        info = self.NeuronOpInfo.from_schema("aten::relu")
        assert info.op is not None
        assert info.op_name == "aten::relu"

    def test_make_sample_from_registry(self):
        import torch

        info = self.NeuronOpInfo.from_schema("aten::relu")
        sample = info.make_sample({"self": (4, 4)}, "cpu", torch.float32)
        assert sample is not None
        assert sample.input.shape == (4, 4)

    def test_shape_sweep_uses_registry(self):
        import torch

        from .neuron_shape_registry import SHAPE_REGISTRY

        SHAPE_REGISTRY.register("aten::relu", arg_shapes={"self": (7, 7)}, source="test_sweep")

        info = self.NeuronOpInfo.from_schema("aten::relu")
        samples = list(info.shape_sweep_inputs("cpu", torch.float32))

        shapes = [s.input.shape for s in samples]
        assert (7, 7) in shapes

    def test_sample_inputs_includes_registry(self):
        import torch

        from .neuron_shape_registry import SHAPE_REGISTRY

        SHAPE_REGISTRY.register("aten::neg", arg_shapes={"self": (13, 13)}, source="test_sample")

        info = self.NeuronOpInfo.from_schema("aten::neg")
        samples = list(info.sample_inputs("cpu", torch.float32, requires_grad=False))

        shapes = [s.input.shape for s in samples]
        assert (13, 13) in shapes

    def test_schema_make_sample_handles_none(self):
        import torch

        info = self.NeuronOpInfo.from_schema("aten::linear")
        sample = info.make_sample(
            {"input": (4, 8), "weight": (16, 8), "bias": None},
            "cpu",
            torch.float32,
        )
        assert sample is not None

    def test_opdb_fallback_for_unregistered_op(self):
        """Base ops with no registry entry fall back to pytorch op_db samples."""
        import torch

        try:
            from torch.testing._internal.common_methods_invocations import op_db
        except ImportError:
            pytest.skip("op_db not available")

        target = next((o for o in op_db if o.name == "clone"), None)
        if target is None:
            pytest.skip("clone not in op_db")

        info = self.NeuronOpInfo(
            "clone",
            op_name="aten::clone",
            op_group="elementwise",
            _pytorch_sample_inputs_func=target.sample_inputs_func,
        )
        assert info.op_resolved
        samples = list(info.sample_inputs("cpu", torch.float32, requires_grad=False))
        assert len(samples) > 0

    def test_registry_entries_capped_to_top_n_by_size(self, monkeypatch):
        """_registry_entries caps at ``NEURON_MAX_SHAPES_PER_OP`` env var
        (default 10), keeping the largest tensors first."""
        from .neuron_shape_registry import SHAPE_REGISTRY, max_tensor_elems

        # Use a real aten op (valid schema) that has no pre-registered
        # entries in the defaults file so the test controls the full set.
        op_name = "aten::i0"
        assert not SHAPE_REGISTRY.get(op_name), "precondition: op has no prior entries"

        # Register shapes of strictly increasing numel.
        registered_sides = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        for side in registered_sides:
            SHAPE_REGISTRY.register(
                op_name,
                arg_shapes={"self": (side, side)},
                source=f"trace_{side}",
                count=1,
            )

        info = self.NeuronOpInfo.from_schema(op_name)

        def _sizes(entries):
            return [max_tensor_elems(e.arg_shapes, op_name) for e in entries]

        # Default cap = 10: keep the 10 largest.
        monkeypatch.setenv("NEURON_MAX_SHAPES_PER_OP", "10")
        entries = info._registry_entries()
        assert len(entries) == 10
        sizes = _sizes(entries)
        assert sizes == sorted(sizes, reverse=True), f"not descending: {sizes}"
        assert sizes[0] == 4096 * 4096
        # Two smallest sides (2, 4) must have been dropped.
        dropped_sides = {2, 4}
        kept_shapes = {e.arg_shapes["self"] for e in entries}
        assert all((s, s) not in kept_shapes for s in dropped_sides)

        # Custom cap = 3: top 3 by numel.
        monkeypatch.setenv("NEURON_MAX_SHAPES_PER_OP", "3")
        entries = info._registry_entries()
        sizes = _sizes(entries)
        assert sizes == [4096 * 4096, 2048 * 2048, 1024 * 1024]

        # Cap = 0 disables the cap.
        monkeypatch.setenv("NEURON_MAX_SHAPES_PER_OP", "0")
        entries = info._registry_entries()
        assert len(entries) == len(registered_sides)

        # Negative cap also disables.
        monkeypatch.setenv("NEURON_MAX_SHAPES_PER_OP", "-1")
        entries = info._registry_entries()
        assert len(entries) == len(registered_sides)
