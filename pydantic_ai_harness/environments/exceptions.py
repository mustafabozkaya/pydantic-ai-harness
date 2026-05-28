"""Exceptions for the execution environments."""


class ExecutionEnvironmentError(Exception):
    """Base class for all execution environment errors."""


class PathEscapeError(ExecutionEnvironmentError):
    """Path escape error."""


class EnvNotFoundError(ExecutionEnvironmentError):
    """Nothing exists at the path in the environment."""


class EnvPermissionError(ExecutionEnvironmentError):
    """The backend may not access the path."""


class EnvIsADirectoryError(ExecutionEnvironmentError):
    """The path is a directory, but the operation requires a file."""


class EnvNotADirectoryError(ExecutionEnvironmentError):
    """A component of the path is not a directory."""


class EnvTooLargeError(ExecutionEnvironmentError):
    """The file exceeds the size the operation allows."""


class EnvReadError(ExecutionEnvironmentError):
    """Unexpected I/O failure during a non-mutating operation (e.g. `read_file`, `ls`).

    The catch-all for any OS error a read-shaped operation raises that is not one of the
    specific subclasses above. Nothing changed on disk.
    """


class EnvWriteError(ExecutionEnvironmentError):
    """Unexpected I/O failure during a mutating operation (e.g. `write_file`).

    The catch-all for any OS error a write-shaped operation raises that is not one of the
    specific subclasses above. State may have been partially changed.
    """
