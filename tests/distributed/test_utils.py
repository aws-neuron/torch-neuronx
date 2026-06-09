"""Unit tests for torch_neuronx.distributed.utils module."""

from torch_neuronx.distributed.utils import get_free_port


class TestGetFreePort:
    """Test cases for get_free_port function."""

    def test_returns_valid_port(self):
        """Test that get_free_port returns a valid port number."""
        port = get_free_port()
        assert isinstance(port, str)
        assert int(port) > 0
        assert int(port) < 65536
