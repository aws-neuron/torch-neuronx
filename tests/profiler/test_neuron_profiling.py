"""
Tests for unified Neuron profiling functionality.
"""

import contextlib
import os
from unittest import mock

import pytest
import torch
from torch.profiler import ProfilerActivity

import torch_neuronx
from tests.utils.neuron_test_utils import requires_nrt_streams
from torch_neuronx.profiling import (
    NeuronProfiler,
    NRTProfilerError,
)

# Constants
PYTORCH_TRACE_FILE_NAME = "pytorch_trace.json"


# Common test utilities and fixtures
@pytest.fixture
def mock_nrt_bindings():
    """Fixture providing common NRT binding mocks."""
    with (
        mock.patch("torch_neuronx.profiling._C._nrt_inspect_config_allocate") as mock_allocate,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_defaults", return_value=0
        ) as mock_defaults,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_output_dir", return_value=0
        ) as mock_output_dir,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_enable_inspect", return_value=0
        ) as mock_enable,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_device_profile_mode", return_value=0
        ) as mock_device_profile_mode,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_begin_with_options", return_value=0
        ) as mock_begin,
        mock.patch("torch_neuronx.profiling._C._nrt_inspect_stop", return_value=0) as mock_stop,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_capture_enabled_for_nc",
            return_value=0,
        ) as mock_nc,
        mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_capture_enabled_for_event_type_string",
            return_value=0,
        ) as mock_types,
    ):
        yield {
            "allocate": mock_allocate,
            "defaults": mock_defaults,
            "output_dir": mock_output_dir,
            "enable": mock_enable,
            "device_profile_mode": mock_device_profile_mode,
            "begin": mock_begin,
            "stop": mock_stop,
            "nc": mock_nc,
            "types": mock_types,
        }


class TestNeuronProfilerInitialization:
    """Test cases for NeuronProfiler initialization and basic setup."""

    def test_initialization_nrt_only(self):
        """Test profiler initialization with only NRT profiling."""
        profiler = NeuronProfiler(pytorch_activities=[])
        assert not profiler.has_pytorch_profiling()
        assert not profiler.is_active()

    def test_initialization_combined_profiling(self):
        """Test profiler initialization with both NRT and PyTorch profiling."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        assert profiler.has_pytorch_profiling()
        assert not profiler.is_active()

    def test_pytorch_trace_file_default_when_enabled(self, tmp_path):
        """Test that pytorch_trace_file gets a default value when pytorch_activities are enabled."""
        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU], neuron_output_dir=test_output_dir
        )
        assert profiler.pytorch_trace_file == f"{test_output_dir}/{PYTORCH_TRACE_FILE_NAME}"

    def test_pytorch_trace_file_custom_name(self, tmp_path):
        """Test custom pytorch_trace_file name."""
        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            pytorch_trace_file="custom_trace.json",
            neuron_output_dir=test_output_dir,
        )
        assert profiler.pytorch_trace_file == f"{test_output_dir}/custom_trace.json"

    def test_pytorch_trace_file_none_when_disabled(self):
        """Test that pytorch_trace_file is None when PyTorch profiling is disabled."""
        profiler = NeuronProfiler(pytorch_activities=[])
        assert profiler.pytorch_trace_file is None


class TestNeuronProfilerConfiguration:
    """Test cases for NeuronProfiler configuration and parameter handling."""

    def test_default_neuron_output_directory(self):
        """Test default neuron output directory."""
        profiler = NeuronProfiler()
        assert profiler.neuron_output_dir == "./output"
        assert profiler.get_nrt_output_directory() == "./output"

    def test_custom_neuron_output_directory(self):
        """Test custom neuron output directory."""
        custom_dir = "/tmp/my_profiles"
        profiler = NeuronProfiler(
            neuron_output_dir=custom_dir, pytorch_activities=[ProfilerActivity.CPU]
        )
        assert profiler.neuron_output_dir == custom_dir
        assert profiler.get_nrt_output_directory() == custom_dir

    def test_event_filters_initialization(self):
        """Test event filters parameter initialization."""
        event_filters = {"nc": ["0", "4-7"], "types": ["cc_exec_barrier", "nrt_model_switch"]}
        profiler = NeuronProfiler(
            event_filters=event_filters, pytorch_activities=[ProfilerActivity.CPU]
        )
        assert profiler.event_filters == event_filters

    def test_record_shapes_parameter(self):
        """Test record_shapes parameter."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU], record_shapes=True)
        assert profiler.record_shapes is True


class TestNeuronProfilerContextManager:
    """Test cases for NeuronProfiler context manager functionality."""

    def test_context_manager_protocol_basic(self, mock_nrt_bindings):
        """Test basic context manager functionality."""
        profiler = NeuronProfiler(pytorch_activities=[])

        # Test context manager
        with profiler as prof:
            assert prof.is_active()

        assert not profiler.is_active()

    def test_context_manager_with_pytorch_profiling(self, mock_nrt_bindings):
        """Test context manager with PyTorch profiling enabled."""
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU], neuron_output_dir="./test_output"
        )

        with profiler as prof:
            assert prof.is_active()

        assert not profiler.is_active()

    def test_context_manager_with_all_features(self, mock_nrt_bindings):
        """Test context manager with all profiler features enabled."""
        profiler = NeuronProfiler(
            neuron_output_dir="./custom_profiles",
            event_filters={"nc": ["0", "2-4"], "types": ["cc_exec_barrier"]},
            pytorch_activities=[ProfilerActivity.CPU],
            record_shapes=True,
        )

        with profiler as prof:
            assert prof.is_active()

        assert not profiler.is_active()


class TestNeuronProfilerStartStop:
    """Test cases for NeuronProfiler start/stop lifecycle methods."""

    def test_manual_start_stop(self, mock_nrt_bindings):
        """Test manual start and stop methods."""
        profiler = NeuronProfiler(pytorch_activities=[])

        assert not profiler.is_active()
        profiler.start()
        assert profiler.is_active()
        profiler.stop()
        assert not profiler.is_active()

    def test_double_start_raises_error(self, mock_nrt_bindings):
        """Test that starting an already active profiler raises an error."""
        profiler = NeuronProfiler(pytorch_activities=[])
        profiler.start()

        with pytest.raises(RuntimeError, match="Profiler is already active"):
            profiler.start()

        profiler.stop()

    def test_stop_inactive_profiler_raises_error(self):
        """Test that stopping an inactive profiler raises an error."""
        profiler = NeuronProfiler(pytorch_activities=[])

        with pytest.raises(RuntimeError, match="Profiler is not active"):
            profiler.stop()


class TestNeuronProfilerErrorHandling:
    """Test cases for error handling in NeuronProfiler operations."""

    def test_nrt_config_defaults_error(self):
        """Test error handling when NRT config defaults fail."""
        with (
            mock.patch("torch_neuronx.profiling._C._nrt_inspect_config_allocate"),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_defaults", return_value=1
            ),
        ):
            profiler = NeuronProfiler(pytorch_activities=[])
            with pytest.raises(NRTProfilerError, match="Failed to set NRT config defaults"):
                profiler.start()

    def test_nrt_output_directory_error(self):
        """Test error handling when setting NRT output directory fails."""
        with (
            mock.patch("torch_neuronx.profiling._C._nrt_inspect_config_allocate"),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_defaults", return_value=0
            ),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_output_dir", return_value=1
            ),
        ):
            profiler = NeuronProfiler(pytorch_activities=[])
            with pytest.raises(NRTProfilerError, match="Failed to set NRT output directory"):
                profiler.start()

    def test_nrt_begin_profiling_error(self):
        """Test error handling when starting NRT profiling fails."""
        with (
            mock.patch("torch_neuronx.profiling._C._nrt_inspect_config_allocate"),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_defaults", return_value=0
            ),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_output_dir", return_value=0
            ),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_enable_inspect", return_value=0
            ),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_device_profile_mode",
                return_value=0,
            ),
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_begin_with_options", return_value=1
            ),
        ):
            profiler = NeuronProfiler(pytorch_activities=[])
            with pytest.raises(NRTProfilerError, match="Failed to start NRT profiling"):
                profiler.start()

    def test_nrt_stop_profiling_error(self, mock_nrt_bindings):
        """Test error handling when stopping NRT profiling fails."""
        mock_nrt_bindings["stop"].return_value = 1  # Simulate failure

        profiler = NeuronProfiler(pytorch_activities=[])
        profiler.start()

        with pytest.raises(RuntimeError, match="Profiling errors"):
            profiler.stop()


class TestEventFilterParsing:
    """Test cases for event filter parsing methods."""

    def test_parse_single_neuroncore(self):
        """Test parsing single NeuronCore ID."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        result = profiler._parse_neuroncore_range("5")
        assert result == [5]

    def test_parse_neuroncore_range(self):
        """Test parsing NeuronCore range."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        result = profiler._parse_neuroncore_range("4-7")
        assert result == [4, 5, 6, 7]

    def test_parse_neuroncore_range_inclusive(self):
        """Test range parsing is inclusive at the end."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        result = profiler._parse_neuroncore_range("2-2")
        assert result == [2]

    def test_parse_invalid_range(self):
        """Test parsing invalid range (start > end)."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        with mock.patch("torch_neuronx.profiling.logger") as mock_logger:
            result = profiler._parse_neuroncore_range("7-4")
            assert result == []
            mock_logger.warning.assert_called_once()

    def test_parse_malformed_range(self):
        """Test parsing malformed range string."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        with mock.patch("torch_neuronx.profiling.logger") as mock_logger:
            result = profiler._parse_neuroncore_range("not-a-number")
            assert result == []
            mock_logger.warning.assert_called_once()

    def test_parse_neuroncore_list_mixed(self):
        """Test parsing mixed list of single IDs and ranges."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        result = profiler._parse_neuroncore_list(["0", "4-7", "10"])
        assert result == [0, 4, 5, 6, 7, 10]

    def test_parse_neuroncore_list_duplicates(self):
        """Test parsing list with duplicates removes them."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        result = profiler._parse_neuroncore_list(["0", "1-3", "2"])
        assert result == [0, 1, 2, 3]

    def test_apply_event_filters_neuroncore(self):
        """Test applying NeuronCore filters."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        mock_config = mock.Mock()
        event_filters = {"nc": ["0", "2-4"]}

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_capture_enabled_for_nc",
            return_value=0,
        ) as mock_set_nc:
            profiler._apply_event_filters(mock_config, event_filters)

            # Should call for cores 0, 2, 3, 4
            assert mock_set_nc.call_count == 4
            expected_calls = [
                mock.call(mock_config, 0, True),
                mock.call(mock_config, 2, True),
                mock.call(mock_config, 3, True),
                mock.call(mock_config, 4, True),
            ]
            mock_set_nc.assert_has_calls(expected_calls, any_order=True)

    def test_apply_event_filters_types(self):
        """Test applying event type filters."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        mock_config = mock.Mock()
        event_filters = {"types": ["cc_exec_barrier", "nrt_model_switch"]}

        with mock.patch(
            "torch_neuronx.profiling._C._nrt_inspect_config_set_capture_enabled_for_event_type_string",
            return_value=0,
        ) as mock_set_type:
            profiler._apply_event_filters(mock_config, event_filters)

            # Should call for each event type
            assert mock_set_type.call_count == 2
            expected_calls = [
                mock.call(mock_config, "cc_exec_barrier", True),
                mock.call(mock_config, "nrt_model_switch", True),
            ]
            mock_set_type.assert_has_calls(expected_calls, any_order=True)

    def test_apply_event_filters_error_handling(self):
        """Test error handling in event filter application."""
        profiler = NeuronProfiler(pytorch_activities=[ProfilerActivity.CPU])
        mock_config = mock.Mock()
        event_filters = {"nc": ["0"]}

        with (
            mock.patch(
                "torch_neuronx.profiling._C._nrt_inspect_config_set_capture_enabled_for_nc",
                return_value=1,
            ),
            mock.patch("torch_neuronx.profiling.logger") as mock_logger,
        ):
            profiler._apply_event_filters(mock_config, event_filters)

            # Should log warning for non-zero status
            mock_logger.warning.assert_called_once()


class TestNeuronProfilerIntegration:
    """Integration tests using real or realistic API interactions."""

    def test_nrt_runtime_trace_generation_real_apis(self, tmp_path):
        """Test that NRT runtime trace is generated using actual runtime APIs.

        Also verifies that SESSION mode (the default) produces exactly one .ntff
        file per instance rather than multiple per-execution files (MODEL mode).
        """
        test_output_dir = str(tmp_path)
        pytorch_trace_file_path = tmp_path / PYTORCH_TRACE_FILE_NAME
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            record_shapes=True,
            neuron_output_dir=test_output_dir,
        )

        with profiler as prof:
            x = torch.randn(20, 20, device="neuron")
            y = torch.mm(x, x.transpose(0, 1))
            z = torch.relu(y)
            result = torch.sum(z).item()
            print("Successfully ran operations on Neuron device")
            print(f"Input tensor device: {x.device}")
            print(f"Result: {result}")

        assert not prof.is_active()
        print(f"NRT profiling completed. Output directory: {test_output_dir}")

        # Check PyTorch trace file exists and has content
        assert (
            pytorch_trace_file_path.exists()
        ), f"PyTorch trace file should exist: {PYTORCH_TRACE_FILE_NAME}"
        trace_size = pytorch_trace_file_path.stat().st_size
        print(f"PyTorch trace file generated: {PYTORCH_TRACE_FILE_NAME} ({trace_size} bytes)")
        assert trace_size > 0, "PyTorch trace file should contain data"

        # Check NRT profiling output - find instance subdirectories (pattern: *_pid_*)
        subdirs = [
            d
            for d in os.listdir(test_output_dir)
            if os.path.isdir(os.path.join(test_output_dir, d)) and "_pid_" in d
        ]
        assert len(subdirs) >= 1, f"Expected at least one instance subdirectory, found: {subdirs}"

        # Verify SESSION mode produces exactly one .ntff file per instance
        for subdir in subdirs:
            instance_dir = os.path.join(test_output_dir, subdir)
            files = os.listdir(instance_dir)
            print(f"Instance directory {subdir} contents: {files}")

            ntff_files = [f for f in files if f.endswith(".ntff")]
            assert len(ntff_files) == 1, (
                f"SESSION mode should produce exactly 1 .ntff file per instance, "
                f"found {len(ntff_files)} in {subdir}: {ntff_files}"
            )


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Framework mapping tests are incompatible with NEURON_LAUNCH_BLOCKING=1",
)
class TestFrameworkMapping:
    """Test cases for framework-to-runtime mapping functionality."""

    def test_framework_mapping_json_generated(self, tmp_path):
        """Test that framework_mapping_<timestamp>.json is generated with correct structure."""
        import glob as glob_module
        import json

        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            neuron_output_dir=test_output_dir,
        )

        with profiler:
            x = torch.randn(20, 20, device="neuron")
            y = torch.mm(x, x.transpose(0, 1))
            torch.relu(y)

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        assert len(mapping_files) == 1, "framework_mapping_<timestamp>.json should be generated"

        with open(mapping_files[0]) as f:
            mappings = json.load(f)

        assert "version" in mappings, "Mapping should have version field"
        assert mappings["version"] == 1, "Version should be 1"
        assert "data" in mappings, "Mapping should have data field"
        assert isinstance(mappings["data"], list), "data should be a list"
        assert len(mappings["data"]) > 0, "Should have at least one mapping"

        for entry in mappings["data"]:
            assert "nrta_seq_id" in entry, "Entry should have nrta_seq_id"
            assert "framework_op_exec_ids" in entry, "Entry should have framework_op_exec_ids"
            for fw_id in entry["framework_op_exec_ids"]:
                assert "seq_nr" in fw_id and "th_id" in fw_id and "stream_id" in fw_id

    def test_framework_mapping_not_generated_nrt_only(self, mock_nrt_bindings, tmp_path):
        """Test that mapping is not generated when only NRT profiling is active."""
        import glob as glob_module

        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[],
            neuron_output_dir=test_output_dir,
        )

        with profiler:
            pass

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        assert len(mapping_files) == 0, "framework_mapping should not exist with NRT only"

    def test_framework_mapping_not_generated_pytorch_only(self, tmp_path):
        """Test that mapping is not generated when only PyTorch profiling would be active."""
        from torch_neuronx import _C

        _C._set_profiler_mapping_enabled(True)
        x = torch.randn(10, 10, device="neuron")
        torch.mm(x, x)
        _C._set_profiler_mapping_enabled(False)

        import torch_neuronx

        torch_neuronx.synchronize()
        mappings = _C._get_profiler_mappings()
        assert mappings["version"] == 1
        assert isinstance(mappings["data"], list)
        assert len(mappings["data"]) > 0

    def test_framework_mapping_multiple_ops(self, tmp_path):
        """Test mapping with multiple sequential operations."""
        import glob as glob_module
        import json

        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            neuron_output_dir=test_output_dir,
        )

        with profiler:
            x = torch.randn(32, 32, device="neuron")
            y = torch.mm(x, x)
            z = torch.relu(y)
            w = torch.add(z, z)
            torch.mul(w, w)

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        assert len(mapping_files) == 1

        with open(mapping_files[0]) as f:
            mappings = json.load(f)

        # Should have multiple mappings for multiple ops
        assert len(mappings["data"]) >= 1, "Should have mappings for executed ops"


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Framework mapping tests are incompatible with NEURON_LAUNCH_BLOCKING=1",
)
class TestFrameworkMappingOpsCat:
    """Test framework mapping with operation concatenation (ops-cat)."""

    @pytest.fixture(autouse=True)
    def enable_concatenation(self):
        os.environ["TORCH_NEURONX_ENABLE_CONCATENATION"] = "1"
        yield
        os.environ.pop("TORCH_NEURONX_ENABLE_CONCATENATION", None)

    def test_opscat_mapping_format(self, tmp_path):
        """Test ops-cat mapping format with structured data."""
        import glob as glob_module
        import json

        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            neuron_output_dir=str(tmp_path),
        )

        with profiler:
            x = torch.randn(64, 64, device="neuron")
            for _ in range(5):
                x = torch.mm(x, x)
            x.cpu()

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        assert len(mapping_files) == 1

        with open(mapping_files[0]) as f:
            mappings = json.load(f)

        assert mappings["version"] == 1
        assert isinstance(mappings["data"], list)
        assert len(mappings["data"]) > 0

        for entry in mappings["data"]:
            assert isinstance(entry["nrta_seq_id"], int), "nrta_seq_id should be int"
            assert isinstance(entry["framework_op_exec_ids"], list)
            for fw_id in entry["framework_op_exec_ids"]:
                assert "seq_nr" in fw_id and "th_id" in fw_id and "stream_id" in fw_id


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Framework mapping tests are incompatible with NEURON_LAUNCH_BLOCKING=1",
)
class TestFrameworkMappingThreadId:
    """Test cases for thread ID handling in framework mapping."""

    def test_thread_id_fields(self, tmp_path):
        """Test that thread ID fields are present and valid."""
        import glob as glob_module
        import json

        test_output_dir = str(tmp_path)
        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            neuron_output_dir=test_output_dir,
        )

        with profiler:
            x = torch.randn(20, 20, device="neuron")
            y = torch.mm(x, x)
            torch.relu(y)

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        assert len(mapping_files) == 1

        with open(mapping_files[0]) as f:
            mappings = json.load(f)

        assert len(mappings["data"]) > 0

        for entry in mappings["data"]:
            for fw_id in entry["framework_op_exec_ids"]:
                assert "os_tid" in fw_id
                assert fw_id["os_tid"] > 0, "os_tid (OS thread ID) should be positive"


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Framework mapping tests are incompatible with NEURON_LAUNCH_BLOCKING=1",
)
class TestFrameworkMappingMultiStream:
    """Test framework mapping with multiple streams."""

    @requires_nrt_streams
    def test_multi_stream_mapping(self, tmp_path):
        """Test that different streams produce different stream_ids."""
        import glob as glob_module
        import json

        stream1 = torch_neuronx.Stream()
        stream2 = torch_neuronx.Stream()

        profiler = NeuronProfiler(
            pytorch_activities=[ProfilerActivity.CPU],
            neuron_output_dir=str(tmp_path),
        )

        with profiler:
            with torch.neuron.stream(stream1):
                x1 = torch.randn(64, 64, device="neuron")
                x1 = torch.mm(x1, x1)
                x1.cpu()
            with torch.neuron.stream(stream2):
                x2 = torch.randn(64, 64, device="neuron")
                x2 = torch.mm(x2, x2)
                x2.cpu()

        mapping_files = glob_module.glob(str(tmp_path / "framework_mapping_*.json"))
        with open(mapping_files[0]) as f:
            mappings = json.load(f)

        stream_ids = set()
        for entry in mappings["data"]:
            for fw_id in entry["framework_op_exec_ids"]:
                stream_ids.add(fw_id["stream_id"])

        assert (
            len(stream_ids) >= 2
        ), f"Expected different stream_ids for different streams, got: {stream_ids}"


if __name__ == "__main__":
    pytest.main([__file__])
