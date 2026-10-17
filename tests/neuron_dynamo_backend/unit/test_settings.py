"""
Unit tests for neuron_dynamo_backend settings module (environment variable helper functions)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from torch_neuronx.neuron_dynamo_backend import settings


class TestEnvironmentVariableHelpers:
    """Test environment variable helper functions"""

    def test_getenv_bool_true_values(self):
        """Test _getenv_bool with values that should return True"""
        true_values = ["1", "true", "True", "yes", "Yes", "YES", "TRUE"]

        for value in true_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                result = settings._getenv_bool("TEST_BOOL", False)
                assert result is True, f"Failed for value: {value}"

    def test_getenv_bool_false_values(self):
        """Test _getenv_bool with values that should return False"""
        false_values = ["0", "false", "False", "no", "No", "NO", "FALSE"]

        for value in false_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                result = settings._getenv_bool("TEST_BOOL", True)
                assert result is False, f"Failed for value: {value}"

    def test_getenv_bool_default_value(self):
        """Test _getenv_bool returns default when env var not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Test default True
            result = settings._getenv_bool("NONEXISTENT_BOOL", True)
            assert result is True

            # Test default False
            result = settings._getenv_bool("NONEXISTENT_BOOL", False)
            assert result is False

            # Test default None
            result = settings._getenv_bool("NONEXISTENT_BOOL", None)
            assert result is None

    def test_getenv_bool_invalid_values(self):
        """Test _getenv_bool with invalid values defaults to False"""
        invalid_values = ["invalid", "maybe", "2", ""]

        for value in invalid_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                result = settings._getenv_bool("TEST_BOOL", True)
                assert result is False, f"Failed for invalid value: {value}"

    def test_getenv_int_valid_values(self):
        """Test _getenv_int with valid integer values"""
        test_values = [("0", 0), ("42", 42), ("-10", -10), ("999", 999)]

        for str_value, expected_int in test_values:
            with patch.dict(os.environ, {"TEST_INT": str_value}):
                result = settings._getenv_int("TEST_INT", 100)
                assert result == expected_int, f"Failed for value: {str_value}"

    def test_getenv_int_default_value(self):
        """Test _getenv_int returns default when env var not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Test default value
            result = settings._getenv_int("NONEXISTENT_INT", 42)
            assert result == 42

            # Test default None
            result = settings._getenv_int("NONEXISTENT_INT", None)
            assert result is None

    def test_getenv_int_invalid_values(self):
        """Test _getenv_int raises AssertionError for invalid values"""
        invalid_values = ["not_a_number", "3.14", "42.0", ""]

        for value in invalid_values:
            with pytest.raises(AssertionError), patch.dict(os.environ, {"TEST_INT": value}):
                settings._getenv_int("TEST_INT", 100)

    def test_getenv_path_valid_values(self):
        """Test _getenv_path with valid path values"""
        test_paths = ["/tmp/test", "relative/path", "/home/user/docs", "."]

        for path_str in test_paths:
            with patch.dict(os.environ, {"TEST_PATH": path_str}):
                result = settings._getenv_path("TEST_PATH", Path("/default"))
                assert result == Path(path_str)
                assert isinstance(result, Path)

    def test_getenv_path_default_value(self):
        """Test _getenv_path returns default when env var not set"""
        default_path = Path("/default/path")

        with patch.dict(os.environ, {}, clear=True):
            # Test default Path
            result = settings._getenv_path("NONEXISTENT_PATH", default_path)
            assert result == default_path

            # Test default None
            result = settings._getenv_path("NONEXISTENT_PATH", None)
            assert result is None

    def test_getenv_flags_with_flags(self):
        """Test _getenv_flags with space-separated flags"""
        test_cases = [
            ("--verbose --debug", ["--verbose", "--debug"]),
            ("--flag1 --flag2 --flag3", ["--flag1", "--flag2", "--flag3"]),
            ("single_flag", ["single_flag"]),
            ("", []),
        ]

        for flags_str, expected_list in test_cases:
            with patch.dict(os.environ, {"TEST_FLAGS": flags_str}):
                result = settings._getenv_flags("TEST_FLAGS", ["--default"])
                assert result == expected_list

    def test_getenv_flags_default_value(self):
        """Test _getenv_flags returns default when env var not set"""
        default_flags = ["--default", "--flags"]

        with patch.dict(os.environ, {}, clear=True):
            result = settings._getenv_flags("NONEXISTENT_FLAGS", default_flags)
            assert result == default_flags

    def test_getenv_flags_complex_flags(self):
        """Test _getenv_flags with complex flag combinations"""
        complex_flags = "--target trn1 --verbose --output=/tmp/file"
        expected = ["--target", "trn1", "--verbose", "--output=/tmp/file"]

        with patch.dict(os.environ, {"TEST_FLAGS": complex_flags}):
            result = settings._getenv_flags("TEST_FLAGS", [])
            assert result == expected

    def test_getenv_string_valid_values(self):
        """Test _getenv_string with various string values"""
        test_strings = ["simple", "with spaces", "/path/to/file", "123", ""]

        for test_str in test_strings:
            with patch.dict(os.environ, {"TEST_STRING": test_str}):
                result = settings._getenv_string("TEST_STRING", "default")
                assert result == test_str

    def test_getenv_string_default_value(self):
        """Test _getenv_string returns default when env var not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Test default string
            result = settings._getenv_string("NONEXISTENT_STRING", "default_value")
            assert result == "default_value"

            # Test default None
            result = settings._getenv_string("NONEXISTENT_STRING", None)
            assert result is None
