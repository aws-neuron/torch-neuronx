"""Metrics tracking for Neuron torch.compile backend."""

import atexit
import csv
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import lru_cache

from torch._dynamo.utils import compilation_time_metrics, counters, cumulative_time_spent_ns

import torch_neuronx
from torch_neuronx.neuron_dynamo_backend.config import get_rank

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module-level constants and state
# -----------------------------------------------------------------------------

# Environment variable for enabling metric tables
ENABLED_TABLES_ENV = "TORCH_NEURONX_ENABLED_METRIC_TABLES"

# Environment variable for CSV output directory (default: /tmp)
METRICS_DIR_ENV = "TORCH_NEURONX_METRICS_DIR"
DEFAULT_METRICS_DIR = "/tmp"

# Lock for thread-safe operations
_lock = threading.Lock()

# Registry of metric tables
REGISTERED_METRIC_TABLES: dict[str, list[str]] = {}

# Storage for compilation metrics
_compilation_metrics: list["NeuronCompilationMetrics"] = []


# -----------------------------------------------------------------------------
# LazyTimingList for deferred timing resolution
# -----------------------------------------------------------------------------


class LazyTimingList(list):
    """A list that lazily resolves its values on access, re-resolving if count changes.

    Used to defer torch_neuronx_compile timing retrieval until someone actually
    reads from compilation_time_metrics. Re-resolves if new compilations occurred.
    """

    def __init__(self, resolver: Callable[[], list[float]], get_count: Callable[[], int]):
        super().__init__()
        self._resolver = resolver
        self._get_count = get_count
        self._last_count = 0
        self._resolve_lock = threading.Lock()

    def _ensure_resolved(self) -> None:
        with self._resolve_lock:
            current_count = self._get_count()
            if current_count != self._last_count:
                self.clear()
                self.extend(self._resolver())
                self._last_count = current_count

    def __iter__(self):
        self._ensure_resolved()
        return super().__iter__()

    def __getitem__(self, idx):
        self._ensure_resolved()
        return super().__getitem__(idx)

    def __len__(self):
        self._ensure_resolved()
        return super().__len__()

    def __repr__(self):
        self._ensure_resolved()
        return super().__repr__()


# -----------------------------------------------------------------------------
# Metric table registration and configuration
# -----------------------------------------------------------------------------


def _register_metric_table(name: str, columns: list[str]) -> None:
    """Register a metric table with its column names."""
    REGISTERED_METRIC_TABLES[name] = columns


@lru_cache(maxsize=1)
def _get_enabled_tables() -> set:
    """Get the set of enabled metric tables from environment variable.

    Returns:
        Set of enabled table names. Empty set if env var not set.

    Raises:
        ValueError: If an unregistered table name is specified.
    """
    tables_str = os.environ.get(ENABLED_TABLES_ENV, "")
    enabled = set()
    for name in tables_str.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in REGISTERED_METRIC_TABLES:
            raise ValueError(
                f"Metric table '{name}' is not registered. "
                f"Available tables: {list(REGISTERED_METRIC_TABLES.keys())}"
            )
        enabled.add(name)
    return enabled


def _get_last_phase_time_us(phase: str) -> int:
    """Get the last recorded time for a compilation phase in microseconds.

    Args:
        phase (str): Name of the compilation phase (e.g., 'torch_neuronx_fx_passes').

    Returns:
        int: Time in microseconds, or 0 if no timing recorded.
    """
    times = compilation_time_metrics.get(phase, [])
    if times:
        return int(times[-1] * 1_000_000)
    return 0


def _get_compilation_time_ms(cache_key: str) -> int:
    """Get the actual compilation time from the C++ cache.

    Queries the compilation cache entries to find the compilation time
    for a specific graph identified by its cache key.

    Args:
        cache_key (str): Unique identifier for the compiled graph.

    Returns:
        int: Compilation time in milliseconds, or 0 if not found.
    """
    entries = torch_neuronx._C._get_compilation_cache_entries()
    return next(
        (e["compilation_time_ms"] for e in entries if e["cache_key"] == cache_key),
        0,
    )


@dataclass
class NeuronCompilationMetrics:
    """Per-graph compilation metrics.

    Attributes:
        graph_id: Unique identifier (cache key) for the compiled graph.
        graph_name: Human-readable name in format {model_name}_{aot_id}_{graph_type}_{nth}.
        model_name: User-provided or auto-generated model name.
        has_collectives: Whether the graph contains collective operations (e.g., all-reduce).
        graph_node_count: Number of nodes in the FX graph before lowering.
        timestamp: Compilation timestamp/segment ID for ordering.
        torch_neuronx_fx_passes_us: Time for FX graph-level passes (microseconds).
        torch_neuronx_lower_us: Time for torch-mlir lowering to StableHLO (microseconds).
        torch_neuronx_compile_us: Time to compile StableHLO to NEFF (microseconds, lazy).
    """

    graph_id: str
    graph_name: str = ""
    model_name: str = ""
    has_collectives: bool = False
    graph_node_count: int = 0
    timestamp: str = ""
    torch_neuronx_fx_passes_us: int = 0
    torch_neuronx_lower_us: int = 0
    torch_neuronx_compile_us: int = 0


# Register metric tables
_register_metric_table("graph_stats", [f.name for f in fields(NeuronCompilationMetrics)])


def record_compilation(metrics: NeuronCompilationMetrics) -> None:
    """Record compilation metrics for a graph.

    torch_neuronx_fx_passes and torch_neuronx_lower timing are captured immediately (sync).
    torch_neuronx_compile timing is populated lazily on get (async operation).
    """
    # Ensure lazy list is registered (may have been cleared by torch._dynamo.reset())
    _register_lazy_timing_list()

    # Capture sync timing immediately
    metrics.torch_neuronx_fx_passes_us = _get_last_phase_time_us("torch_neuronx_fx_passes")
    metrics.torch_neuronx_lower_us = _get_last_phase_time_us("torch_neuronx_lower")

    with _lock:
        _compilation_metrics.append(metrics)
        counters["neuron"]["compiled_graphs"] += 1
        if metrics.has_collectives:
            counters["neuron"]["graphs_with_collectives"] += 1


def _populate_compile_timing(metrics: NeuronCompilationMetrics) -> None:
    """Populate the torch_neuronx_compile timing field from the C++ cache.

    Must be called after synchronization to ensure compilation has completed.
    Converts milliseconds from cache to microseconds for the metrics object.

    Args:
        metrics (NeuronCompilationMetrics): Metrics object to update in place.
    """
    compilation_time_ms = _get_compilation_time_ms(metrics.graph_id)
    metrics.torch_neuronx_compile_us = compilation_time_ms * 1000


def get_dynamo_metrics() -> list[NeuronCompilationMetrics]:
    """Get all torch.compile compilation metrics with timing.

    Syncs to ensure async compilation timing is populated.
    """
    torch_neuronx.synchronize()
    with _lock:
        for m in _compilation_metrics:
            if m.torch_neuronx_compile_us == 0:
                _populate_compile_timing(m)
        return list(_compilation_metrics)


def _get_metrics_dir() -> str:
    """Get the directory for metrics CSV output."""
    return os.environ.get(METRICS_DIR_ENV, DEFAULT_METRICS_DIR)


def _write_csv_at_exit() -> None:
    """Write CSV files at program exit if enabled."""
    enabled = _get_enabled_tables()
    if not enabled:
        return

    # Sync to ensure all compilations complete
    torch_neuronx.synchronize()

    with _lock:
        if "graph_stats" in enabled and _compilation_metrics:
            # Populate torch_neuronx_compile timing for all metrics
            for m in _compilation_metrics:
                if m.torch_neuronx_compile_us == 0:
                    _populate_compile_timing(m)

            rank = get_rank()
            metrics_dir = _get_metrics_dir()
            filename = os.path.join(metrics_dir, f"metric_table_graph_stats_rank{rank}.csv")
            header = REGISTERED_METRIC_TABLES["graph_stats"]
            try:
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for m in _compilation_metrics:
                        writer.writerow([getattr(m, col) for col in header])
            except OSError as e:
                logger.warning(f"Failed to write metrics CSV to {filename}: {e}")


# Register atexit handler for CSV export
atexit.register(_write_csv_at_exit)


def _resolve_compile_timing() -> list[float]:
    """Resolver for LazyTimingList - syncs and returns timing in seconds."""
    torch_neuronx.synchronize()

    with _lock:
        result = []
        for m in _compilation_metrics:
            if m.torch_neuronx_compile_us == 0:
                _populate_compile_timing(m)
            result.append(m.torch_neuronx_compile_us / 1_000_000)

        # Also populate cumulative_time_spent_ns for calculate_time_spent()
        cumulative_time_spent_ns["torch_neuronx_compile"] = sum(result) * 1e9

        return result


def _get_compilation_count() -> int:
    """Get current compilation count for LazyTimingList."""
    with _lock:
        return len(_compilation_metrics)


def _register_lazy_timing_list() -> None:
    """Register LazyTimingList for torch_neuronx_compile in dynamo's metrics if not present."""
    if "torch_neuronx_compile" not in compilation_time_metrics:
        compilation_time_metrics["torch_neuronx_compile"] = LazyTimingList(
            _resolve_compile_timing,
            _get_compilation_count,
        )


def reset_dynamo_metrics() -> None:
    """Reset all torch.compile compilation metrics.

    Clears recorded metrics and counters. Call this between test runs
    or when starting a new profiling session.
    """
    with _lock:
        _compilation_metrics.clear()
        counters["neuron"].clear()
