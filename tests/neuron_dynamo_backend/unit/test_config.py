"""
Unit tests for neuron_dynamo_backend config module and related utilities
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from torch_neuronx.neuron_dynamo_backend import config


class TestConfigUtilities:
    """Test configuration utility functions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.original_model_name = config._current_model_name
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        config._current_model_name = self.original_model_name
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_model_name_management(self):
        """Test model name get/set functionality"""
        # Test default model name
        config._current_model_name = None
        model_name = config.get_model_name()
        assert model_name == "model_default"

        # Test setting custom model name
        config.set_model_name("test_model")
        assert config.get_model_name() == "test_model"

        # Test setting None resets to None
        config._current_model_name = None
        assert config._current_model_name is None


class TestDistributedUtilities:
    """Test distributed computing utility functions"""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_node_local_rank")
    def test_get_local_rank(self, mock_get_local_rank, mock_torch_dist_init):
        """Test get_local_rank function"""
        mock_get_local_rank.return_value = 2
        mock_torch_dist_init.return_value = True
        result = config.get_local_rank()
        assert result == 2
        mock_get_local_rank.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_get_world_size(self, mock_get_world_size, mock_torch_dist_init):
        """Test get_world_size function"""
        mock_get_world_size.return_value = 8
        mock_torch_dist_init.return_value = True
        result = config.get_world_size()
        assert result == 8
        mock_get_world_size.assert_called_once()

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_get_local_world_size_with_env_var(self):
        """Test get_local_world_size with environment variable set"""
        result = config.get_local_world_size()
        assert result == 4

    @patch.dict(os.environ, {}, clear=True)
    def test_get_local_world_size_without_env_var(self):
        """Test get_local_world_size without environment variable"""
        result = config.get_local_world_size()
        assert result == 1
