"""Unit tests for torch_neuronx.numa module.

Mocked — runs on any machine without Neuron hardware.
"""

import json
import os
from unittest import mock

import pytest

from torch_neuronx import numa

# Sample neuron-ls JSON for a trn2.48xlarge (4 of 16 devices shown)
_SAMPLE_TOPOLOGY = [
    {
        "neuron_device": 0,
        "bdf": "0000:cc:00.0",
        "cpu_affinity": "48-95,144-191",
        "numa_node": "1",
        "neuroncore_ids": [0, 1, 2, 3],
    },
    {
        "neuron_device": 1,
        "bdf": "0000:b5:00.0",
        "cpu_affinity": "48-95,144-191",
        "numa_node": "1",
        "neuroncore_ids": [4, 5, 6, 7],
    },
    {
        "neuron_device": 4,
        "bdf": "0000:6c:00.0",
        "cpu_affinity": "0-47,96-143",
        "numa_node": "0",
        "neuroncore_ids": [16, 17, 18, 19],
    },
    {
        "neuron_device": 5,
        "bdf": "0000:57:00.0",
        "cpu_affinity": "0-47,96-143",
        "numa_node": "0",
        "neuroncore_ids": [20, 21, 22, 23],
    },
]


@pytest.fixture(autouse=True)
def _reset_numa_state():
    """Reset module-level state and env vars between tests."""
    numa._state.numa_pinned = False
    numa._discover_topology.cache_clear()
    # Snapshot and restore NUMA-related env vars
    saved = {
        k: os.environ.get(k)
        for k in (
            "NEURON_NUMA_PIN",
            "NEURON_RT_VISIBLE_CORES",
            "NEURON_LOGICAL_NC_CONFIG",
        )
    }
    yield
    numa._state.numa_pinned = False
    numa._discover_topology.cache_clear()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _mock_neuron_ls(sample=_SAMPLE_TOPOLOGY):
    """Mock subprocess.run to return sample topology."""
    result = mock.Mock()
    result.returncode = 0
    result.stdout = json.dumps(sample)
    result.stderr = ""
    return mock.patch("torch_neuronx.numa.subprocess.run", return_value=result)


# ---------------------------------------------------------------------------
# _parse_range_list
# ---------------------------------------------------------------------------


class TestParseRangeList:
    def test_simple_range(self):
        assert numa._parse_range_list("0-3") == [0, 1, 2, 3]

    def test_comma_separated(self):
        assert numa._parse_range_list("0,1,5") == [0, 1, 5]

    def test_mixed(self):
        assert numa._parse_range_list("0-2,5,7-9") == [0, 1, 2, 5, 7, 8, 9]

    def test_single_value(self):
        assert numa._parse_range_list("42") == [42]

    def test_empty_string(self):
        assert numa._parse_range_list("") == []

    def test_whitespace(self):
        assert numa._parse_range_list("  0-2 , 5 ") == [0, 1, 2, 5]

    def test_numa_cpulist(self):
        result = numa._parse_range_list("48-95,144-191")
        assert len(result) == 96
        assert result[0] == 48
        assert result[-1] == 191
        assert set(result) == set(range(48, 96)) | set(range(144, 192))


# ---------------------------------------------------------------------------
# _visible_core_to_device_index
# ---------------------------------------------------------------------------


class TestVisibleCoreToDeviceIndex:
    def test_lnc2_default(self):
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        assert numa._visible_core_to_device_index(0) == 0
        assert numa._visible_core_to_device_index(3) == 0
        assert numa._visible_core_to_device_index(4) == 1
        assert numa._visible_core_to_device_index(16) == 4

    def test_lnc1(self):
        with mock.patch.dict(os.environ, {"NEURON_LOGICAL_NC_CONFIG": "1"}):
            assert numa._visible_core_to_device_index(0) == 0
            assert numa._visible_core_to_device_index(7) == 0
            assert numa._visible_core_to_device_index(8) == 1

    def test_large_visible_core(self):
        """High visible core indices map correctly."""
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        # visible_core=63 (last on trn2) → device 15
        assert numa._visible_core_to_device_index(63) == 15


# ---------------------------------------------------------------------------
# _get_visible_core
# ---------------------------------------------------------------------------


class TestGetVisibleCore:
    def test_explicit_arg(self):
        assert numa._get_visible_core(42) == 42

    def test_from_env_single(self):
        with mock.patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "7"}):
            assert numa._get_visible_core(None) == 7

    def test_from_env_range(self):
        with mock.patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "4-7"}):
            assert numa._get_visible_core(None) == 4

    def test_none_when_unset(self):
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
        assert numa._get_visible_core(None) is None

    def test_invalid_env(self):
        with mock.patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "abc"}):
            assert numa._get_visible_core(None) is None

    def test_from_env_comma_separated(self):
        with mock.patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "4,5,6,7"}):
            assert numa._get_visible_core(None) == 4


# ---------------------------------------------------------------------------
# _discover_topology
# ---------------------------------------------------------------------------


class TestDiscoverTopology:
    def test_success(self):
        with _mock_neuron_ls():
            topo = numa._discover_topology()
            assert topo is not None
            assert 0 in topo
            assert topo[0]["numa_node"] == 1
            assert topo[0]["bdf"] == "0000:cc:00.0"
            assert topo[4]["numa_node"] == 0

    def test_negative_numa_node_clamped_to_zero(self):
        """sysfs returns -1 for single-NUMA systems; we clamp to 0."""
        sample = [
            {
                "neuron_device": 0,
                "bdf": "0000:cc:00.0",
                "cpu_affinity": "0-3",
                "numa_node": "-1",
                "neuroncore_ids": [0, 1, 2, 3],
            }
        ]
        with _mock_neuron_ls(sample):
            topo = numa._discover_topology()
            assert topo[0]["numa_node"] == 0

    def test_neuron_ls_not_found(self):
        with mock.patch(
            "torch_neuronx.numa.subprocess.run",
            side_effect=FileNotFoundError("neuron-ls not found"),
        ):
            assert numa._discover_topology() is None

    def test_neuron_ls_failure(self):
        result = mock.Mock()
        result.returncode = 1
        result.stderr = "error"
        result.stdout = ""
        with mock.patch("torch_neuronx.numa.subprocess.run", return_value=result):
            assert numa._discover_topology() is None

    def test_caching(self):
        with _mock_neuron_ls() as mock_run:
            numa._discover_topology()
            numa._discover_topology()
            assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# pin_to_numa
# ---------------------------------------------------------------------------


class TestPinToNuma:
    def test_pins_to_correct_cpus(self):
        with (
            _mock_neuron_ls(),
            mock.patch.dict(
                os.environ,
                {"NEURON_RT_VISIBLE_CORES": "0"},
                clear=False,
            ),
            mock.patch("torch_neuronx.numa._pin_all_threads", return_value=5) as mock_pin,
        ):
            os.environ.pop("NEURON_NUMA_PIN", None)
            numa.pin_to_numa()
            mock_pin.assert_called_once()
            cpulist = mock_pin.call_args[0][0]
            # Device 0 → NUMA 1 → CPUs 48-95,144-191
            assert cpulist[0] == 48
            assert len(cpulist) == 96
            assert set(cpulist) == set(range(48, 96)) | set(range(144, 192))

    def test_noop_when_disabled(self):
        with (
            mock.patch.dict(os.environ, {"NEURON_NUMA_PIN": "0"}, clear=False),
            mock.patch("torch_neuronx.numa._pin_all_threads") as mock_pin,
        ):
            numa.pin_to_numa(0)
            mock_pin.assert_not_called()

    def test_idempotent(self):
        with (
            _mock_neuron_ls(),
            mock.patch.dict(
                os.environ,
                {"NEURON_RT_VISIBLE_CORES": "0"},
                clear=False,
            ),
            mock.patch("torch_neuronx.numa._pin_all_threads", return_value=1) as mock_pin,
        ):
            os.environ.pop("NEURON_NUMA_PIN", None)
            numa.pin_to_numa()
            numa.pin_to_numa()
            assert mock_pin.call_count == 1

    def test_explicit_visible_core_numa0(self):
        with (
            _mock_neuron_ls(),
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("torch_neuronx.numa._pin_all_threads", return_value=1) as mock_pin,
        ):
            os.environ.pop("NEURON_NUMA_PIN", None)
            # visible_core=16 → device 4 → NUMA 0
            numa.pin_to_numa(16)
            cpulist = mock_pin.call_args[0][0]
            assert cpulist[0] == 0  # NUMA 0 starts at CPU 0


# ---------------------------------------------------------------------------
# get_numa_status
# ---------------------------------------------------------------------------


class TestGetNumaStatus:
    def test_returns_expected_keys(self):
        with _mock_neuron_ls():
            status = numa.get_numa_status()
            expected_keys = {
                "numa_pin_enabled",
                "numa_pinned",
                "current_affinity",
                "topology_available",
                "num_devices",
            }
            assert set(status.keys()) == expected_keys

    def test_values_reflect_state(self):
        with _mock_neuron_ls():
            status = numa.get_numa_status()
            assert status["numa_pinned"] is False
            assert status["numa_pin_enabled"] is True
            assert status["topology_available"] is True
            assert status["num_devices"] == 4
            assert isinstance(status["current_affinity"], list)


# ---------------------------------------------------------------------------
# numa_launch_main
# ---------------------------------------------------------------------------


class TestNumaLaunchMain:
    def test_skips_numactl_when_no_topology(self):
        with (
            mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
            mock.patch("shutil.which", return_value="/usr/bin/numactl"),
            mock.patch("torch_neuronx.numa._get_device_info", return_value=None),
            mock.patch("torch_neuronx.numa.os.execvp") as mock_exec,
        ):
            numa.numa_launch_main()
            # Should fall back to plain python
            mock_exec.assert_called_once()
            assert mock_exec.call_args[0][0] == "python"

    def test_uses_numactl_when_topology_available(self):
        info = {"numa_node": 1, "bdf": "0000:cc:00.0", "cpu_affinity": "48-95"}
        with (
            mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
            mock.patch("shutil.which", return_value="/usr/bin/numactl"),
            mock.patch("torch_neuronx.numa._get_device_info", return_value=info),
            mock.patch("torch_neuronx.numa.os.execvp") as mock_exec,
            mock.patch("sys.argv", ["neuron-numa-launch", "train.py", "--lr=1e-4"]),
        ):
            numa.numa_launch_main()
            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert args[0] == "numactl"
            assert "--membind=1" in args[1]
            assert "--cpunodebind=1" in args[1]
            assert "train.py" in args[1]

    def test_no_numactl_falls_back(self):
        with (
            mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
            mock.patch("shutil.which", return_value=None),
            mock.patch("torch_neuronx.numa.os.execvp") as mock_exec,
        ):
            numa.numa_launch_main()
            assert mock_exec.call_args[0][0] == "python"
