"""
Tests for unified Neuron profiling functionality.
"""

from unittest import mock

import pytest

from torch_neuronx.profiling import (
    ContinuousExportConfig,
    ExportStatus,
    InspectExportMask,
    NeuronConfig,
    NeuronProfiler,
    NRTProfilerError,
    ProfileMode,
    export_now,
    request_export,
)

_NRT_SUCCESS = 0
_NRT_HW_ERROR = 6


class TestContinuousExport:
    """Contract tests for the two export wrappers (C++ is mocked at the _C).

    export_now: sync NRT passthrough — None on NRT_SUCCESS, else NRTProfilerError
    whose .status carries NRT's errno verbatim.
    request_export: async submission — None on SUBMITTED, else NRTProfilerError
    whose .status carries the ExportStatus gating reason. Neither raises anything
    but NRTProfilerError (or TypeError for a non-enum mask).
    """

    def test_export_now_success_returns_none(self):
        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_export_now",
            return_value=_NRT_SUCCESS,
        ) as mock_export:
            assert export_now(InspectExportMask.SYSTEM_PROFILE) is None

        mock_export.assert_called_once_with(int(InspectExportMask.SYSTEM_PROFILE))

    def test_export_now_raises_with_verbatim_nrt_status(self):
        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_export_now",
                return_value=_NRT_HW_ERROR,
            ),
            pytest.raises(NRTProfilerError, match=r"StatusCode: 6") as exc_info,
        ):
            export_now(InspectExportMask.SYSTEM_PROFILE)

        assert exc_info.value.status == _NRT_HW_ERROR

    def test_request_export_submitted_returns_none(self):
        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_request_export",
            return_value=ExportStatus.SUBMITTED.value,
        ) as mock_request:
            assert request_export(InspectExportMask.SYSTEM_PROFILE) is None

        mock_request.assert_called_once_with(int(InspectExportMask.SYSTEM_PROFILE))

    @pytest.mark.parametrize(
        "status",
        [
            ExportStatus.NO_ACTIVE_SESSION,
            ExportStatus.EXPORT_NOT_ENABLED,
            ExportStatus.UNSUPPORTED_MASK,
        ],
    )
    def test_request_export_raises_on_gating_errors(self, status):
        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_request_export",
                return_value=status.value,
            ),
            pytest.raises(NRTProfilerError, match=status.name) as exc_info,
        ):
            request_export(InspectExportMask.SYSTEM_PROFILE)

        assert exc_info.value.status is status

    def test_request_export_unknown_status_raises_nrt_error_not_value_error(self):
        # An int the C++ enum can produce but this Python layer does not know
        # (e.g. a newer runtime paired with an older wheel) must surface as
        # NRTProfilerError, never a raw ValueError from ExportStatus(int).
        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_request_export",
                return_value=999,
            ),
            pytest.raises(NRTProfilerError, match="unknown status: 999") as exc_info,
        ):
            request_export(InspectExportMask.SYSTEM_PROFILE)

        assert exc_info.value.status == 999

    @pytest.mark.parametrize(
        ("export_fn", "binding_name"),
        [
            (export_now, "_nrt_inspect_export_now"),
            (request_export, "_nrt_inspect_request_export"),
        ],
    )
    @pytest.mark.parametrize("invalid_mask", [1, True])
    def test_export_mask_requires_enum(self, export_fn, binding_name, invalid_mask):
        with (
            mock.patch(f"torch_neuronx.profiling._C.{binding_name}") as binding,
            pytest.raises(TypeError, match="Expected InspectExportMask"),
        ):
            export_fn(invalid_mask)

        binding.assert_not_called()

    def test_export_now_forwards_mask_values_to_cpp(self):
        # export_now is a passthrough: NRT owns mask validation for the sync path.
        unnamed_mask = InspectExportMask(1 << 2)
        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_export_now",
            return_value=_NRT_SUCCESS,
        ) as binding:
            assert export_now(unnamed_mask) is None

        binding.assert_called_once_with(int(unnamed_mask))

    def test_continuous_export_config_serialization(self):
        # neuron_profile MUST stay the FIRST key: the C++ gate
        # (NeuronActivityProfiler::hasNeuronProfileKey) is a prefix check.
        config = NeuronConfig(continuous_export=ContinuousExportConfig())

        parts = config.custom_profiler_config.split(";")
        assert parts[0].startswith("neuron_profile:")
        assert "continuous_export:true" in parts

    def test_no_continuous_export_key_by_default(self):
        # Opt-in: without ContinuousExportConfig the key must be absent, so the
        # session never spawns the async worker.
        config = NeuronConfig()

        assert "continuous_export" not in config.custom_profiler_config

    def test_continuous_export_does_not_require_runtime_profile_mode(self):
        config = NeuronConfig(
            modes=[ProfileMode.DEVICE],
            continuous_export=ContinuousExportConfig(),
        )

        assert config.custom_profiler_config.startswith("neuron_profile:device;")
        assert "continuous_export:true" in config.custom_profiler_config.split(";")

    def test_continuous_export_config_type_validation(self):
        with pytest.raises(TypeError, match="continuous_export must be"):
            NeuronConfig(continuous_export=object())

    def test_top_level_reexports(self):
        import torch_neuronx

        assert torch_neuronx.export_now is export_now
        assert torch_neuronx.request_export is request_export
        assert torch_neuronx.InspectExportMask is InspectExportMask
        assert torch_neuronx.ExportStatus is ExportStatus


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
