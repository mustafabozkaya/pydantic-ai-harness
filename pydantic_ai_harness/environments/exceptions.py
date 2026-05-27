"""Exceptions for the execution environments."""


class ExecutionEnvironmentError(Exception):
    """Base class for all execution environment errors."""


class PathEscapeError(ExecutionEnvironmentError):
    """Path escape error."""
