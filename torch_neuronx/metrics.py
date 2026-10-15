#!/usr/bin/env python3
"""
Neuron Metrics Python API

Provides a high-level Python interface for accessing Neuron metrics.

Usage:
    import torch_neuronx.metrics as met

    # Get all metrics report
    print(met.report())

    # Clear all metrics
    met.clear_all()

    # Get specific counter value
    value = met.get_counter_value("CompilationCache.InMemoryHits")

    # Disable metrics collection (for performance-critical code)
    met.set_enabled(False)

    # Check if metrics are enabled
    if met.is_enabled():
        print("Metrics are being collected")

Environment Variables:
    TORCH_NEURONX_METRICS_ENABLED: Set to "1", "true", or "yes" to enable metrics
                                   collection at startup. Default: disabled.
"""

from typing import Any

from torch_neuronx import _C

# =============================================================================
# Runtime metrics control
# =============================================================================


def is_enabled() -> bool:
    """Check whether metrics collection is enabled process-wide.

    Returns:
        bool: ``True`` when metrics are being collected.
    """
    return _C._neuron_metrics_enabled()


def set_enabled(enabled: bool) -> None:
    """Enable or disable process-wide metrics collection at runtime.

    Args:
        enabled (bool): Whether to collect metrics.

    Note:
        Disabled metric-recording calls are no-ops with minimal overhead.
    """
    _C._neuron_set_metrics_enabled(enabled)


# =============================================================================
# Core metric functions - directly call C++ bindings
# =============================================================================


def get_counter_names() -> list[str]:
    """Return counter names registered in the process-wide metrics arena.

    Returns:
        list[str]: Registered counter names, including counters cleared to zero.
    """
    return _C._neuron_counter_names()


def get_counter_value(name: str) -> int | None:
    """Return the value of a named counter.

    Args:
        name (str): Counter name to query.

    Returns:
        int or None: Counter value, or ``None`` when the counter does not exist.
    """
    return _C._neuron_counter_value(name)


def get_metric_names() -> list[str]:
    """Return sampled-metric names registered in the process-wide arena.

    Returns:
        list[str]: Registered metric names, including metrics whose samples were cleared.
    """
    return _C._neuron_metric_names()


def get_metric_data(name: str) -> dict[str, Any] | None:
    """Return detailed data for a sampled metric.

    Args:
        name (str): Metric name to query.

    Returns:
        dict or None: Dictionary with exactly ``accumulator`` (sum across all samples
            since the last clear), ``total_samples`` (count across all such samples),
            ``repr`` (formatted accumulator), and ``samples`` (bounded circular history
            in chronological order). Each retained sample has exactly ``timestamp_ns``
            and ``value``. Raw accumulator and value units are nanoseconds for time
            metrics, bytes for byte metrics, and metric-defined for generic metrics.
            Timestamps use a steady clock, not wall time, and are not comparable across
            processes. Returns ``None`` when the metric is not registered.
    """
    return _C._neuron_metric_data(name)


def report() -> str:
    """Generate a report of positive counters and sampled metrics with data.

    Returns:
        str: Formatted metrics report.
    """
    return _C._neuron_metrics_report()


def report_filtered(
    counter_names: list[str] | None = None, metric_names: list[str] | None = None
) -> str:
    """Generate a report for selected counters and sampled metrics.

    Args:
        counter_names (list[str], optional): Exact counter names to include. Defaults to
            ``None``, which selects no counters.
        metric_names (list[str], optional): Exact metric names to include. Defaults to
            ``None``, which selects no sampled metrics.

    Returns:
        str: Formatted report containing selected metrics with samples and selected
            counters with positive values.
    """
    return _C._neuron_metrics_report_filtered(counter_names or [], metric_names or [])


def clear_counters() -> None:
    """Clear all counter values from the process-wide metrics arena."""
    _C._neuron_clear_counters()


def clear_metrics() -> None:
    """Clear all sampled metric data from the process-wide metrics arena."""
    _C._neuron_clear_metrics()


def clear_all() -> None:
    """Clear all counters and sampled metrics from the process-wide arena."""
    _C._neuron_clear_counters()
    _C._neuron_clear_metrics()


def set_concatenation_profiling_enabled(enabled: bool) -> None:
    """Enable or disable detailed per-type concatenation profiling.

    Args:
        enabled (bool): Whether to record per-invocation concatenation details.

    Note:
        This debug feature records merge times and group sizes in unbounded vectors. Use it
        only for analysis. ``TORCH_NEURONX_CONCATENATION_PROFILING=1`` also enables it.
    """
    _C._set_concatenation_profiling_enabled(enabled)


# =============================================================================
# Structured metrics access
# =============================================================================


def get_metrics_dict(prefix: str | None = None) -> dict[str, Any]:
    """Return counters and sampled metrics as a structured dictionary.

    Args:
        prefix (str, optional): Name substring to include. Defaults to ``None`` for all names.

    Returns:
        dict[str, Any]: Counter integers and metric dictionaries containing ``accumulator``,
            ``total_samples``, and ``formatted`` fields.
    """
    result = {}
    for name in get_counter_names():
        if prefix is None or prefix in name:
            result[name] = get_counter_value(name)
    for name in get_metric_names():
        if prefix is None or prefix in name:
            data = get_metric_data(name)
            if data:
                result[name] = {
                    "accumulator": data["accumulator"],
                    "total_samples": data["total_samples"],
                    "formatted": data["repr"],
                }
    return result
