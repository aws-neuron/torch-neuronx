"""
Unit tests for neuron_dynamo_backend.exceptions module
"""

import pytest

from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError


class TestNEFFExceptions:
    """Test custom exception classes"""

    def test_neff_execution_error_creation(self):
        """Test NEFFExecutionError can be created and raised"""
        with pytest.raises(NEFFExecutionError):
            raise NEFFExecutionError("Test execution error")

    def test_neff_execution_error_inheritance(self):
        """Test NEFFExecutionError inherits from Exception"""
        assert issubclass(NEFFExecutionError, Exception)

    def test_exception_message_handling(self):
        """Test exception messages are properly handled"""
        message = "Custom error message"

        try:
            raise NEFFExecutionError(message)
        except NEFFExecutionError as e:
            assert str(e) == message
