"""
Tests for unified Neuron profiling functionality.
"""

from dataclasses import FrozenInstanceError
from enum import IntFlag
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
    TimeTrigger,
    export_now,
    request_export,
)

_NRT_SUCCESS = 0
_NRT_INVALID = 2
_NRT_HW_ERROR = 6
_EXPORT_MASKS = (
    InspectExportMask.SYSTEM_PROFILE,
    InspectExportMask.DEVICE_PROFILE,
    InspectExportMask.ALL,
)


class TestContinuousExport:
    """Contract tests for the two export wrappers (C++ is mocked at the _C).

    export_now: sync NRT passthrough — None on NRT_SUCCESS, else NRTProfilerError
    whose .status carries NRT's errno verbatim.
    request_export: async submission — None on SUBMITTED, else NRTProfilerError
    whose .status carries the ExportStatus gating reason. Neither raises anything
    but NRTProfilerError (or TypeError for a non-enum mask).
    """

    def test_mask_values_match_runtime_bits(self):
        # The public flags must preserve the NRT ABI and compose without a raw integer.
        assert int(InspectExportMask.SYSTEM_PROFILE) == 1
        assert int(InspectExportMask.DEVICE_PROFILE) == 2
        assert InspectExportMask.ALL == (
            InspectExportMask.SYSTEM_PROFILE | InspectExportMask.DEVICE_PROFILE
        )
        assert int(InspectExportMask.ALL) == 3

    @pytest.mark.parametrize("mask", _EXPORT_MASKS)
    def test_export_now_success_returns_none(self, mask):
        # Each mask must reach NRT unchanged and expose success as None.
        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_export_now",
            return_value=_NRT_SUCCESS,
        ) as mock_export:
            assert export_now(mask) is None

        mock_export.assert_called_once_with(int(mask))

    @pytest.mark.parametrize("mask", _EXPORT_MASKS)
    @pytest.mark.parametrize("status", [_NRT_INVALID, _NRT_HW_ERROR])
    def test_export_now_raises_with_verbatim_nrt_status(self, mask, status):
        # Sync errors retain the NRT status, even when its value matches an async status.
        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_export_now",
                return_value=status,
            ),
            pytest.raises(NRTProfilerError, match=f"StatusCode: {status}") as exc_info,
        ):
            export_now(mask)

        assert exc_info.value.status == status
        assert type(exc_info.value.status) is int

    @pytest.mark.parametrize("mask", _EXPORT_MASKS)
    def test_request_export_submitted_returns_none(self, mask):
        # An accepted request returns None and forwards all requested bits.
        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_request_export",
            return_value=ExportStatus.SUBMITTED.value,
        ) as mock_request:
            assert request_export(mask) is None

        mock_request.assert_called_once_with(int(mask))

    @pytest.mark.parametrize("mask", _EXPORT_MASKS)
    @pytest.mark.parametrize(
        "status",
        [
            ExportStatus.NO_ACTIVE_SESSION,
            ExportStatus.EXPORT_NOT_ENABLED,
            ExportStatus.UNSUPPORTED_MASK,
        ],
    )
    def test_request_export_raises_on_gating_errors(self, mask, status):
        # Worker gating errors use ExportStatus for every valid mask.
        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_request_export",
                return_value=status.value,
            ),
            pytest.raises(NRTProfilerError, match=status.name) as exc_info,
        ):
            request_export(mask)

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
    @pytest.mark.parametrize("invalid_mask", [1, 2, 3, True])
    def test_export_mask_requires_enum(self, export_fn, binding_name, invalid_mask):
        # Raw integers and booleans must fail before either binding runs.
        with (
            mock.patch(f"torch_neuronx.profiling._C.{binding_name}") as binding,
            pytest.raises(TypeError, match="Expected InspectExportMask"),
        ):
            export_fn(invalid_mask)

        binding.assert_not_called()

    @pytest.mark.parametrize("mask_value", [0, 1 << 2, 3 | (1 << 2)])
    def test_export_now_forwards_mask_values_to_cpp(self, mask_value):
        # export_now is a passthrough: NRT owns mask validation for the sync path.
        unnamed_mask = InspectExportMask(mask_value)
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

    def test_export_snapshot_dumps_to_session_dir(self):
        """Test that export_snapshot writes the snapshot into the session directory."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)

        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
                return_value="/tmp/output/i-1234_pid_5678",
            ),
            mock.patch("torch_neuronx.memory._dump_snapshot") as mock_dump,
        ):
            path = profiler.export_snapshot()

        expected = "/tmp/output/i-1234_pid_5678/memory_snapshot.pickle"
        mock_dump.assert_called_once_with(expected)
        assert path == expected

    def test_export_snapshot_custom_filename(self):
        """Test that export_snapshot respects a custom filename."""
        config = NeuronConfig(profile_output_dir="/tmp/output")
        profiler = NeuronProfiler(config)

        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_get_instance_output_dir",
                return_value="/tmp/output/i-1234_pid_5678",
            ),
            mock.patch("torch_neuronx.memory._dump_snapshot") as mock_dump,
        ):
            path = profiler.export_snapshot(filename="snap.pickle")

        expected = "/tmp/output/i-1234_pid_5678/snap.pickle"
        mock_dump.assert_called_once_with(expected)
        assert path == expected


class TestTimeTrigger:
    """Validate time policies without an active profiler session."""

    @pytest.mark.parametrize("interval_ms", [1, 250, (2**63 - 1) // 1_000_000])
    def test_time_policy_serialization(self, interval_ms: int) -> None:
        # Preserve the first profiler key and serialize the interval limits with an explicit mask.
        mask = InspectExportMask(sum(int(member) for member in InspectExportMask))
        config = NeuronConfig(
            continuous_export=ContinuousExportConfig(TimeTrigger(interval_ms, mask)),
            precache_node_info=False,
        )

        parts = config.custom_profiler_config.split(";")
        assert parts[0].startswith("neuron_profile:")
        assert parts[-4:] == [
            "continuous_export:true",
            "export_policy:time",
            f"export_interval_ms:{interval_ms}",
            f"export_mask:{int(mask)}",
        ]

    @pytest.mark.parametrize("interval_ms", [True, False, None, 1.0, "1", object()])
    def test_interval_rejects_invalid_types(self, interval_ms: object) -> None:
        # Reject values that could otherwise coerce to an integer interval.
        with pytest.raises(TypeError, match="interval_ms"):
            TimeTrigger(interval_ms, InspectExportMask.SYSTEM_PROFILE)

    @pytest.mark.parametrize("interval_ms", [0, -1, (2**63 - 1) // 1_000_000 + 1])
    def test_interval_rejects_invalid_values(self, interval_ms: int) -> None:
        # Reject empty intervals and values that overflow signed nanoseconds.
        with pytest.raises(ValueError, match="interval_ms"):
            TimeTrigger(interval_ms, InspectExportMask.SYSTEM_PROFILE)

    def test_mask_is_required(self) -> None:
        # Require an explicit mask so a new enum member cannot widen an existing policy.
        with pytest.raises(TypeError):
            TimeTrigger(interval_ms=1)

    @pytest.mark.parametrize("mask", [None, True, False, 1, 1.0, "1", object()])
    def test_mask_rejects_invalid_types(self, mask: object) -> None:
        # Require the enum even when a raw integer has a supported value.
        with pytest.raises(TypeError, match="InspectExportMask"):
            TimeTrigger(1, mask)

    def test_mask_rejects_empty_and_unknown_bits(self) -> None:
        # Reject unknown bits even when the mask also contains supported bits.
        supported_bits = sum(int(member) for member in InspectExportMask)
        unknown_bit = 1 << supported_bits.bit_length()
        for bits in (0, unknown_bit, supported_bits | unknown_bit):
            with pytest.raises(ValueError, match="mask"):
                TimeTrigger(1, InspectExportMask(bits))

    def test_mask_validation_accepts_new_enum_members(self) -> None:
        # A future enum extension must work for individual bits and their combination.
        extended_masks = IntFlag("ExtendedInspectExportMask", {"FIRST": 1, "ADDED": 4})
        with mock.patch("torch_neuronx.profiling.InspectExportMask", extended_masks):
            for mask in (*extended_masks, extended_masks.FIRST | extended_masks.ADDED):
                config = ContinuousExportConfig(TimeTrigger(1, mask))
                assert config._to_config_parts()[-1] == f"export_mask:{int(mask)}"

    @pytest.mark.parametrize("field", ["interval_ms", "mask"])
    def test_trigger_is_immutable(self, field: str) -> None:
        # Both fields must remain fixed when multiple sessions share a policy.
        policy = TimeTrigger(1, InspectExportMask.SYSTEM_PROFILE)
        with pytest.raises(FrozenInstanceError):
            setattr(policy, field, None)

    @pytest.mark.parametrize("policy", [True, 1, "time", object()])
    def test_config_rejects_invalid_policy_types(self, policy: object) -> None:
        # Reject unsupported policy objects before a profiler session can start.
        with pytest.raises(TypeError, match="policy"):
            ContinuousExportConfig(policy)

    def test_policy_config_can_be_reused(self) -> None:
        # Two session configs serialize the same immutable policy without changing its values.
        policy = TimeTrigger(100, InspectExportMask.SYSTEM_PROFILE)
        continuous_export = ContinuousExportConfig(policy)
        first = NeuronConfig(continuous_export=continuous_export, precache_node_info=False)
        second = NeuronConfig(continuous_export=continuous_export, precache_node_info=False)

        assert first.custom_profiler_config == second.custom_profiler_config
        assert first.continuous_export is second.continuous_export is continuous_export
        assert continuous_export.policy is policy
        assert policy.interval_ms == 100
        assert policy.mask is InspectExportMask.SYSTEM_PROFILE

    def test_manual_only_and_disabled_serialization(self) -> None:
        # Preserve manual-only and disabled configurations without automatic policy keys.
        for continuous_export in (None, ContinuousExportConfig(), ContinuousExportConfig(None)):
            config = NeuronConfig(continuous_export=continuous_export, precache_node_info=False)
            export_parts = [
                part
                for part in config.custom_profiler_config.split(";")
                if part.startswith(("continuous_export:", "export_"))
            ]
            assert export_parts == ([] if continuous_export is None else ["continuous_export:true"])

    def test_top_level_reexport(self) -> None:
        # The public package and profiling module must expose the same immutable type.
        import torch_neuronx

        assert torch_neuronx.TimeTrigger is TimeTrigger


if __name__ == "__main__":
    pytest.main([__file__])
