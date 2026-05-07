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
