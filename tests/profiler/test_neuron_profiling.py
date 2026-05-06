"""
Tests for unified Neuron profiling functionality.
"""

from unittest import mock

import pytest

from torch_neuronx.profiling import (
    NeuronConfig,
    NeuronProfiler,
    NRTProfilerError,
)


class TestNeuronProfiler:
    """Test cases for NeuronConfig.get_session_dir and NeuronProfiler.export_trace."""

    def test_get_session_dir_returns_instance_dir(self):
        """Test that NeuronConfig.get_session_dir calls the NRT API correctly."""
        config = NeuronConfig(profile_output_dir="/tmp/output")

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
            return_value="/tmp/output/i-1234_pid_5678",
        ) as mock_get_dir:
            result = config.get_session_dir()

        assert result == "/tmp/output/i-1234_pid_5678"
        mock_get_dir.assert_called_once_with("/tmp/output")

    def test_get_session_dir_raises_without_output_dir(self):
        """Test that get_session_dir raises when profile_output_dir is not set."""
        config = NeuronConfig()

        with pytest.raises(ValueError, match="profile_output_dir was not set"):
            config.get_session_dir()

    def test_constructor_raises_without_output_dir(self):
        """Test that NeuronProfiler raises when profile_output_dir is not set."""
        config = NeuronConfig()

        with pytest.raises(ValueError, match="profile_output_dir must be set"):
            NeuronProfiler(config)

    def test_export_trace_calls_export_chrome_trace(self):
        """Test that export_trace exports to the correct session subdirectory."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)
        mock_prof = mock.Mock()

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
            return_value="/tmp/output/i-1234_pid_5678",
        ):
            profiler.export_trace(mock_prof)

        mock_prof.export_chrome_trace.assert_called_once_with(
            "/tmp/output/i-1234_pid_5678/trace.json"
        )

    def test_export_trace_custom_filename(self):
        """Test that export_trace respects a custom filename."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)
        mock_prof = mock.Mock()

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
            return_value="/tmp/output/i-1234_pid_5678",
        ):
            profiler.export_trace(mock_prof, filename="my_trace.json")

        mock_prof.export_chrome_trace.assert_called_once_with(
            "/tmp/output/i-1234_pid_5678/my_trace.json"
        )

    def test_export_trace_as_on_trace_ready_callback(self):
        """Test that export_trace works when used as an on_trace_ready callback."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)
        mock_prof = mock.Mock()

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
            return_value="/tmp/output/i-1234_pid_5678",
        ):

            def callback(p):
                profiler.export_trace(p)

            callback(mock_prof)

        mock_prof.export_chrome_trace.assert_called_once_with(
            "/tmp/output/i-1234_pid_5678/trace.json"
        )

    def test_export_trace_in_custom_hook(self):
        """Test that export_trace works inside a user's custom hook function."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)
        mock_prof = mock.Mock()
        custom_actions = []

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
            return_value="/tmp/output/i-1234_pid_5678",
        ):

            def my_hook(prof):
                custom_actions.append("before")
                profiler.export_trace(prof)
                custom_actions.append("after")

            my_hook(mock_prof)

        assert custom_actions == ["before", "after"]
        mock_prof.export_chrome_trace.assert_called_once_with(
            "/tmp/output/i-1234_pid_5678/trace.json"
        )


if __name__ == "__main__":
    pytest.main([__file__])
