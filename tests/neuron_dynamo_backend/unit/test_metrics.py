"""Unit tests for neuron_dynamo_backend metrics module."""

import os
import re
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.utils import counters

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.metrics import (
    NeuronCompilationMetrics,
    record_compilation,
)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    torch.neuron.reset_dynamo_metrics()
    yield
    torch.neuron.reset_dynamo_metrics()


class TestMetricsUnit:
    """Unit tests for metrics recording without actual compilation."""

    def test_record_compilation_increments_counters(self):
        """Each record_compilation() call increments compiled_graphs counter."""
        record_compilation(NeuronCompilationMetrics(graph_id="g1"))
        record_compilation(NeuronCompilationMetrics(graph_id="g2"))

        assert counters["neuron"]["compiled_graphs"] == 2

    def test_collective_graphs_tracked_separately(self):
        """Graphs with collectives are tracked for distributed debugging."""
        record_compilation(NeuronCompilationMetrics(graph_id="g1", has_collectives=False))
        record_compilation(NeuronCompilationMetrics(graph_id="g2", has_collectives=True))

        assert counters["neuron"]["compiled_graphs"] == 2
        assert counters["neuron"]["graphs_with_collectives"] == 1

    def test_reset_clears_counters_and_metrics(self):
        """reset_dynamo_metrics() clears both counters and stored metrics."""
        record_compilation(NeuronCompilationMetrics(graph_id="g1"))
        torch.neuron.reset_dynamo_metrics()

        assert counters["neuron"]["compiled_graphs"] == 0
        assert len(torch.neuron.get_dynamo_metrics()) == 0

    def test_get_compilation_cache_entries_binding(self):
        """_get_compilation_cache_entries returns per-graph compilation times."""
        import torch.nn as nn

        torch_neuronx._C._clear_compilation_cache()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron")
            compiled(torch.randn(2, 10, device="neuron"))

        torch_neuronx.synchronize()
        entries = torch_neuronx._C._get_compilation_cache_entries()

        assert len(entries) >= 1
        entry = entries[0]
        assert "cache_key" in entry
        assert "compilation_time_ms" in entry
        assert entry["compilation_time_ms"] >= 0


class TestMetricsIntegration:
    """Integration tests verifying metrics with actual torch.compile."""

    def setup_method(self):
        torch._dynamo.reset()

    def test_get_dynamo_metrics_syncs_and_populates_timing(self):
        """get_dynamo_metrics() blocks until async compilation completes.

        This is the primary API for retrieving timing data. It calls
        synchronize() internally to ensure torch_neuronx_compile_us is populated.
        """
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = SimpleModel().to("neuron")
            compiled_model = torch.compile(model, backend="neuron")
            x = torch.randn(2, 10, device="neuron")
            compiled_model(x)

        # Before get_dynamo_metrics(), timing may be incomplete
        # After calling it, all timing is guaranteed populated
        metrics = torch.neuron.get_dynamo_metrics()

        assert len(metrics) == 1
        assert metrics[0].torch_neuronx_fx_passes_us >= 0, "FX passes timing should be captured"
        assert metrics[0].torch_neuronx_lower_us > 0, "Lowering timing should be captured"
        assert metrics[0].torch_neuronx_compile_us > 0, "Async timing populated after sync"
        assert metrics[0].graph_node_count > 0

    def test_explicit_synchronize_populates_timing(self):
        """Explicit torch_neuronx.synchronize() populates timing."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = SimpleModel().to("neuron")
            compiled_model = torch.compile(model, backend="neuron")
            compiled_model(torch.randn(2, 10, device="neuron"))

        # Explicit sync before accessing metrics
        torch_neuronx.synchronize()

        from torch_neuronx.neuron_dynamo_backend.metrics import _compilation_metrics

        # Access internal metrics directly (without get_dynamo_metrics sync)
        assert len(_compilation_metrics) == 1

    def test_multiple_compilations_tracked_separately(self):
        """Each torch.compile'd model gets its own metrics entry."""
        import torch.nn as nn

        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        class ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 10)

            def forward(self, x):
                return torch.relu(self.linear(x))

        with torch.inference_mode():
            compiled_a = torch.compile(ModelA().to("neuron"), backend="neuron")
            compiled_a(torch.randn(2, 10, device="neuron"))

            compiled_b = torch.compile(ModelB().to("neuron"), backend="neuron")
            compiled_b(torch.randn(2, 20, device="neuron"))

        assert counters["neuron"]["compiled_graphs"] == 2
        assert len(torch.neuron.get_dynamo_metrics()) == 2

    def test_different_shapes_cause_recompile(self):
        """Different input shapes trigger recompilation and new metrics entries.

        Without dynamic=True, each unique shape causes a new compilation.
        No dynamo.reset() needed - dynamo automatically recompiles for new shapes.
        """
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron", dynamic=False)

            # First shape
            compiled(torch.randn(2, 10, device="neuron"))
            first_count = counters["neuron"]["compiled_graphs"]

            # Different batch size - triggers recompilation without reset
            compiled(torch.randn(4, 10, device="neuron"))
            second_count = counters["neuron"]["compiled_graphs"]

        assert second_count > first_count, "Different shape should trigger new compilation"
        assert len(torch.neuron.get_dynamo_metrics()) >= 2

    def test_graph_name_includes_model_name_from_options(self):
        """graph_name combines user's model_name with AOT compile ID.

        Format: {model_name}_{aot_id}_{graph_type}_{nth}
        Example: LlamaForCausalLM_0_inference_0
        """
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = Model().to("neuron")
            compiled = torch.compile(
                model, backend="neuron", options={"model_name": "LlamaForCausalLM"}
            )
            compiled(torch.randn(2, 10, device="neuron"))

        metrics = torch.neuron.get_dynamo_metrics()
        assert len(metrics) == 1
        assert re.fullmatch(r"LlamaForCausalLM_\d+_\w+_\d+", metrics[0].graph_name)
        assert metrics[0].model_name == "LlamaForCausalLM"

    def test_graph_name_unique_across_graph_breaks(self):
        """Graph breaks produce unique graph_names with different AOT IDs."""
        import torch.nn as nn

        class ModelWithBreak(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(5, 3)

            def forward(self, x):
                x = self.linear1(x)
                torch._dynamo.graph_break()
                return self.linear2(x)

        with torch.inference_mode():
            model = ModelWithBreak().to("neuron")
            compiled = torch.compile(model, backend="neuron", options={"model_name": "MyModel"})
            compiled(torch.randn(2, 10, device="neuron"))

        metrics = torch.neuron.get_dynamo_metrics()
        assert len(metrics) == 2
        graph_names = [m.graph_name for m in metrics]
        assert all(re.fullmatch(r"MyModel_\d+_\w+_\d+", name) for name in graph_names)
        assert len(set(graph_names)) == 2, f"Expected unique names, got: {graph_names}"

    def test_graph_name_defaults_without_model_name(self):
        """Without model_name option, graph_name uses 'model_default'."""
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron")
            compiled(torch.randn(2, 10, device="neuron"))

        metrics = torch.neuron.get_dynamo_metrics()
        assert len(metrics) == 1
        assert re.fullmatch(r"model_default_\d+_\w+_\d+", metrics[0].graph_name)


class TestCSVExport:
    """Test CSV export controlled by TORCH_NEURONX_ENABLED_METRIC_TABLES env var."""

    def test_no_csv_when_env_var_unset(self, tmp_path):
        """CSV export is disabled when env var is not set."""
        env = {k: v for k, v in os.environ.items() if k != "TORCH_NEURONX_ENABLED_METRIC_TABLES"}
        env["TORCH_NEURONX_METRICS_DIR"] = str(tmp_path)
        with patch.dict(os.environ, env, clear=True):
            record_compilation(NeuronCompilationMetrics(graph_id="test123"))
            from torch_neuronx.neuron_dynamo_backend.metrics import _write_csv_at_exit

            _write_csv_at_exit()

        assert not (tmp_path / "metric_table_graph_stats_rank0.csv").exists()

    def test_env_var_enables_csv(self, tmp_path):
        """TORCH_NEURONX_ENABLED_METRIC_TABLES enables CSV export."""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_ENABLED_METRIC_TABLES": "graph_stats",
                "TORCH_NEURONX_METRICS_DIR": str(tmp_path),
            },
        ):
            from torch_neuronx.neuron_dynamo_backend.metrics import _get_enabled_tables

            _get_enabled_tables.cache_clear()

            record_compilation(NeuronCompilationMetrics(graph_id="test_env"))
            from torch_neuronx.neuron_dynamo_backend.metrics import _write_csv_at_exit

            _write_csv_at_exit()

        assert (tmp_path / "metric_table_graph_stats_rank0.csv").exists()

    def test_atexit_writes_csv_on_exit(self, tmp_path):
        """Verify _write_csv_at_exit writes CSV with timing data."""
        import torch.nn as nn

        from torch_neuronx.neuron_dynamo_backend.metrics import _write_csv_at_exit

        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_ENABLED_METRIC_TABLES": "graph_stats",
                "TORCH_NEURONX_METRICS_DIR": str(tmp_path),
            },
        ):
            from torch_neuronx.neuron_dynamo_backend.metrics import _get_enabled_tables

            _get_enabled_tables.cache_clear()

            torch._dynamo.reset()

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)

                def forward(self, x):
                    return self.linear(x)

            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron")
            with torch.inference_mode():
                compiled(torch.randn(2, 10, device="neuron"))

            # Directly call the CSV export function (registered with atexit)
            _write_csv_at_exit()

        # Verify CSV was created
        comp_csv = tmp_path / "metric_table_graph_stats_rank0.csv"
        assert comp_csv.exists(), "graph_stats CSV not created"

        # Verify content has timing data
        comp_content = comp_csv.read_text()
        assert "torch_neuronx_compile_us" in comp_content
        lines = comp_content.strip().split("\n")
        assert len(lines) >= 2  # header + data

    def test_invalid_table_name_raises_error(self):
        """Unknown table names are rejected with helpful error message."""
        with patch.dict(os.environ, {"TORCH_NEURONX_ENABLED_METRIC_TABLES": "invalid_table"}):
            from torch_neuronx.neuron_dynamo_backend.metrics import _get_enabled_tables

            _get_enabled_tables.cache_clear()

            with pytest.raises(ValueError, match="not registered"):
                _get_enabled_tables()

    def test_csv_contains_both_sync_and_async_timing(self, tmp_path):
        """CSV output includes sync (fx_passes, lowering) and async (stablehlo_to_neff) timing."""
        import torch.nn as nn

        from torch_neuronx.neuron_dynamo_backend.metrics import _write_csv_at_exit

        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_ENABLED_METRIC_TABLES": "graph_stats",
                "TORCH_NEURONX_METRICS_DIR": str(tmp_path),
            },
        ):
            from torch_neuronx.neuron_dynamo_backend.metrics import _get_enabled_tables

            _get_enabled_tables.cache_clear()

            torch._dynamo.reset()

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)

                def forward(self, x):
                    return self.linear(x)

            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron")
            with torch.inference_mode():
                compiled(torch.randn(2, 10, device="neuron"))

            _write_csv_at_exit()

        csv_path = tmp_path / "metric_table_graph_stats_rank0.csv"
        lines = csv_path.read_text().strip().split("\n")
        header = lines[0].split(",")
        data = lines[1].split(",")

        # Sync timing columns
        assert int(data[header.index("torch_neuronx_lower_us")]) > 0
        # Async timing column
        assert int(data[header.index("torch_neuronx_compile_us")]) > 0


class TestConcurrentMetrics:
    """Verify thread-safety of metrics recording."""

    def test_concurrent_record_calls_are_thread_safe(self):
        """All concurrent record_compilation() calls are captured."""
        from concurrent.futures import ThreadPoolExecutor

        def record_task(thread_id):
            record_compilation(NeuronCompilationMetrics(graph_id=f"graph_{thread_id}"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(record_task, i) for i in range(4)]
            for f in futures:
                f.result()

        assert counters["neuron"]["compiled_graphs"] == 4
        assert len(torch.neuron.get_dynamo_metrics()) == 4


class TestDynamoIntegration:
    """Test integration with PyTorch's dynamo timing infrastructure."""

    def test_neuron_phases_in_compilation_time_metrics(self):
        """Neuron phases appear in compilation_time_metrics (used by TORCH_LOGS=+dynamo)."""
        import torch.nn as nn
        from torch._dynamo.utils import compilation_time_metrics, compile_times

        torch._dynamo.reset()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            model = Model().to("neuron")
            compiled = torch.compile(model, backend="neuron")
            compiled(torch.randn(2, 10, device="neuron"))

        # Verify neuron phases are registered in metrics dict
        assert "torch_neuronx_fx_passes" in compilation_time_metrics
        assert "torch_neuronx_lower" in compilation_time_metrics
        assert "torch_neuronx_compile" in compilation_time_metrics

        # Access triggers lazy resolution
        assert len(compilation_time_metrics["torch_neuronx_compile"]) > 0
        assert compilation_time_metrics["torch_neuronx_compile"][0] > 0

        # Verify phases appear in formatted output (what dump_compile_times logs)
        output = compile_times(repr="str")
        assert "torch_neuronx_fx_passes" in output
        assert "torch_neuronx_lower" in output
        assert "torch_neuronx_compile" in output

    def test_lazy_list_re_resolves_after_new_compilation(self):
        """LazyTimingList re-resolves when new compilations occur after access.

        Scenario: compile model A -> access timing -> compile model B -> access again
        Both compilations should appear in the timing list.
        """
        import torch.nn as nn
        from torch._dynamo.utils import compilation_time_metrics

        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        class ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 10)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            # First compilation
            compiled_a = torch.compile(ModelA().to("neuron"), backend="neuron")
            compiled_a(torch.randn(2, 10, device="neuron"))

        # Access timing - triggers first resolution
        timing_list = compilation_time_metrics["torch_neuronx_compile"]
        first_access_len = len(timing_list)
        assert first_access_len == 1

        with torch.inference_mode():
            # Second compilation
            compiled_b = torch.compile(ModelB().to("neuron"), backend="neuron")
            compiled_b(torch.randn(2, 20, device="neuron"))

        # Access again - should re-resolve and include both
        second_access_len = len(timing_list)
        assert (
            second_access_len == 2
        ), f"Expected 2 entries after second compilation, got {second_access_len}"

    def test_compile_times_includes_neuron_timing(self):
        """compile_times() includes neuron phases for mid-execution timing access."""
        import torch.nn as nn
        from torch._dynamo.utils import compile_times

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            compiled = torch.compile(Model().to("neuron"), backend="neuron")
            compiled(torch.randn(2, 10, device="neuron"))

        # compile_times() returns formatted string
        timing_str = compile_times(repr="str")

        assert "torch_neuronx_fx_passes" in timing_str
        assert "torch_neuronx_lower" in timing_str
        assert "torch_neuronx_compile" in timing_str

    @pytest.mark.xfail(reason="LazyTimingList not triggered by calculate_time_spent()")
    def test_calculate_time_spent_includes_neuron_timing(self):
        """calculate_time_spent() does NOT automatically include neuron async timing.

        It reads from cumulative_time_spent_ns, which is separate from
        compilation_time_metrics where our LazyTimingList lives.

        We could fix this by patching cumulative_time_spent_ns with a custom dict
        that triggers lazy resolution on access, but this is fragile - PyTorch
        internal changes could break it. Given calculate_time_spent() is rarely
        used compared to compile_times() and TORCH_LOGS=+dynamo, we accept this
        limitation.

        Workaround: Access compilation_time_metrics["torch_neuronx_compile"] first to
        trigger resolution, which populates cumulative_time_spent_ns as a side effect.
        """
        import torch.nn as nn
        from torch._dynamo.utils import calculate_time_spent

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        with torch.inference_mode():
            compiled = torch.compile(Model().to("neuron"), backend="neuron")
            compiled(torch.randn(2, 10, device="neuron"))

        time_spent = calculate_time_spent()

        assert "torch_neuronx_compile" in time_spent
        assert time_spent["torch_neuronx_compile"] > 0
