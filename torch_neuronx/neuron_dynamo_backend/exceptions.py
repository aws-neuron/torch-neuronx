"""
Exceptions for Neuron backend
"""


class NEFFCompilationError(Exception):
    """Signal a failure while compiling a NEFF artifact."""

    pass


class NEFFExecutionError(Exception):
    """Signal a failure while executing a NEFF artifact."""

    pass
