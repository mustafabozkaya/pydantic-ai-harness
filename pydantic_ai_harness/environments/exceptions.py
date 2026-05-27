"""Exceptions for the execution environments."""


class ExecutionEnvironmentError(Exception):
    """Base class for all execution environment errors."""


class PathEscapeError(ExecutionEnvironmentError):
    """Path escape error."""


class EnvFileNotFoundError(ExecutionEnvironmentError):
    """File not found in the environment."""


class EnvFilePermissionError(ExecutionEnvironmentError):
    """File permission error."""


class EnvFileIsADirectoryError(ExecutionEnvironmentError):
    """File is a directory."""


class EnvFileNotADirectoryError(ExecutionEnvironmentError):
    """File is not a directory."""


class EnvFileTooLargeError(ExecutionEnvironmentError):
    """File too large."""


class EnvFileReadError(ExecutionEnvironmentError):
    """File read error."""
