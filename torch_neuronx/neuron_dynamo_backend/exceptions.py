"""
Exceptions for Neuron backend
"""


class NEFFCompilationError(Exception):
    """Exception raised when NEFF compilation fails"""

    pass


class NEFFExecutionError(Exception):
    """Exception raised when NEFF execution fails"""

    pass
