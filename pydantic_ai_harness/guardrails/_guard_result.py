"""GuardResult — outcome of a guard check."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GuardResult:
    """Result returned by a guard callable.

    Use the classmethods to create results:
    - ``GuardResult.allow()`` — let the request/output through
    - ``GuardResult.block(message=None)`` — block the request/output
    - ``GuardResult.replace(value)`` — substitute a different value
    - ``GuardResult.retry(message)`` — retry (OutputGuard only)
    """

    _outcome: str = field(repr=False)
    _value: Any = field(default=None, repr=False)
    _message: str | None = field(default=None, repr=False)

    @classmethod
    def allow(cls) -> GuardResult:
        """Allow the request/output to proceed."""
        return cls(_outcome='allow')

    @classmethod
    def block(cls, message: str | None = None) -> GuardResult:
        """Block the request/output."""
        return cls(_outcome='block', _message=message)

    @classmethod
    def replace(cls, value: Any) -> GuardResult:
        """Replace the request/output with a different value."""
        return cls(_outcome='replace', _value=value)

    @classmethod
    def retry(cls, message: str | None = None) -> GuardResult:
        """Retry (OutputGuard only)."""
        return cls(_outcome='retry', _message=message)

    @property
    def is_allow(self) -> bool:
        return self._outcome == 'allow'

    @property
    def is_block(self) -> bool:
        return self._outcome == 'block'

    @property
    def is_replace(self) -> bool:
        return self._outcome == 'replace'

    @property
    def is_retry(self) -> bool:
        return self._outcome == 'retry'
