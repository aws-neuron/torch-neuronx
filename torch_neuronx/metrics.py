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
    """Check if metrics collection is currently enabled.

    Returns:
        True if metrics are being collected, False otherwise.
    """
    return _C._neuron_metrics_enabled()


def set_enabled(enabled: bool) -> None:
    """Enable or disable metrics collection at runtime.

    When disabled, metrics calls become no-ops with minimal overhead.
    This can be useful for performance-critical sections of code.

    Args:
        enabled: True to enable metrics collection, False to disable.

    Example:
        >>> import torch_neuronx.metrics as met
        >>> met.set_enabled(False)  # Disable metrics
        >>> # ... performance-critical code ...
        >>> met.set_enabled(True)   # Re-enable metrics
    """
    _C._neuron_set_metrics_enabled(enabled)


# =============================================================================
# Core metric functions - directly call C++ bindings
# =============================================================================


def get_counter_names() -> list[str]:
    """Get list of all counter names that have data."""
    return _C._neuron_counter_names()


def get_counter_value(name: str) -> int | None:
    """Get the value of a specific counter."""
    return _C._neuron_counter_value(name)


def get_metric_names() -> list[str]:
    """Get list of all metric names that have data."""
    return _C._neuron_metric_names()


def get_metric_data(name: str) -> dict[str, Any] | None:
    """Get detailed data for a specific metric."""
    return _C._neuron_metric_data(name)


def report() -> str:
    """Generate a comprehensive metrics report."""
    return _C._neuron_metrics_report()


def report_filtered(
    counter_names: list[str] | None = None, metric_names: list[str] | None = None
) -> str:
    """Generate a filtered metrics report."""
    return _C._neuron_metrics_report_filtered(counter_names or [], metric_names or [])


def clear_counters() -> None:
    """Clear all counter values."""
    _C._neuron_clear_counters()


def clear_metrics() -> None:
    """Clear all metric data."""
    _C._neuron_clear_metrics()


def clear_all() -> None:
    """Clear all counters and metrics."""
    _C._neuron_clear_counters()
    _C._neuron_clear_metrics()


# =============================================================================
# Convenience functions for common metric categories
# =============================================================================


def get_compilation_metrics() -> dict[str, Any]:
    """Get compilation-related metrics."""
    result = {}
    for name in get_counter_names():
        if "Compilation" in name:
            result[name] = get_counter_value(name)
    for name in get_metric_names():
        if "Compilation" in name:
            data = get_metric_data(name)
            if data:
                result[name] = {
                    "accumulator": data["accumulator"],
                    "total_samples": data["total_samples"],
                    "formatted": data["repr"],
                }
    return result
