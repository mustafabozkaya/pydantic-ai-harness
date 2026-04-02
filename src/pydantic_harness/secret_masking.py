"""Secret masking capability that redacts secrets from tool outputs and model responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

# --- Built-in pattern categories ---

_API_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    'openai_key': re.compile(r'sk-[A-Za-z0-9_-]{20,}'),
    'anthropic_key': re.compile(r'sk-ant-[A-Za-z0-9_-]{20,}'),
    'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
    'github_token': re.compile(r'gh[psorat]_[A-Za-z0-9_]{36,}'),
    'slack_token': re.compile(r'xox[bpas]-[A-Za-z0-9-]+'),
    'google_api_key': re.compile(r'AIza[A-Za-z0-9_-]{35}'),
    'generic_api_key': re.compile(
        r"""(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?key)\s*[:=]\s*['"]?[A-Za-z0-9_\-/+=]{16,}['"]?"""
    ),
}

_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    'bearer_token': re.compile(r'Bearer\s+[A-Za-z0-9_\-./+=]{20,}'),
    'jwt': re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_\-+=]+'),
}

_CONNECTION_STRING_PATTERNS: dict[str, re.Pattern[str]] = {
    'password_in_url': re.compile(r'://[^:/?#\s]+:[^@/?#\s]+@[^/?#\s]+'),
    'database_connection': re.compile(r'(?i)(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s]+'),
}

_PRIVATE_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

_BUILTIN_CATEGORIES: dict[str, dict[str, re.Pattern[str]]] = {
    'api_keys': _API_KEY_PATTERNS,
    'tokens': _TOKEN_PATTERNS,
    'connection_strings': _CONNECTION_STRING_PATTERNS,
    'private_keys': _PRIVATE_KEY_PATTERNS,
}

_ALL_BUILTIN_PATTERNS: dict[str, re.Pattern[str]] = {}
for _patterns in _BUILTIN_CATEGORIES.values():
    _ALL_BUILTIN_PATTERNS.update(_patterns)


def _mask_text(text: str, patterns: dict[str, re.Pattern[str]], replacement: str) -> str:
    """Apply all patterns to replace matched secrets in `text`."""
    for pattern in patterns.values():
        text = pattern.sub(replacement, text)
    return text


@dataclass
class SecretMasking(AbstractCapability[Any]):
    """Redacts secrets, API keys, and sensitive data from tool outputs and model responses.

    Uses `after_tool_execute` to scrub tool return values and `after_model_request`
    to scrub model response text before they enter the conversation history.

    By default all built-in pattern categories are enabled: `api_keys`, `tokens`,
    `connection_strings`, and `private_keys`.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import SecretMasking

        agent = Agent('openai:gpt-5', capabilities=[SecretMasking()])
        ```
    """

    categories: list[str] | None = None
    """Built-in pattern categories to enable.

    Choose from `'api_keys'`, `'tokens'`, `'connection_strings'`, `'private_keys'`.
    When `None` (default), all categories are enabled.
    """

    custom_patterns: dict[str, str] | None = None
    """Additional regex patterns as `{name: pattern}` pairs.

    These are compiled once at init time and applied alongside the built-in patterns.
    """

    replacement: str = '[REDACTED]'
    """The string that replaces matched secrets."""

    _compiled: dict[str, re.Pattern[str]] = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile built-in and custom patterns."""
        if self.categories is not None:
            for category in self.categories:
                if category not in _BUILTIN_CATEGORIES:
                    raise ValueError(
                        f'Unknown secret pattern category {category!r}, expected one of {sorted(_BUILTIN_CATEGORIES)}'
                    )
                self._compiled.update(_BUILTIN_CATEGORIES[category])
        else:
            self._compiled.update(_ALL_BUILTIN_PATTERNS)

        if self.custom_patterns:
            for name, pattern in self.custom_patterns.items():
                self._compiled[name] = re.compile(pattern)

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Scrub secrets from tool return values."""
        if isinstance(result, str):
            return _mask_text(result, self._compiled, self.replacement)
        # For non-string results, convert to string to check, but only replace if secrets found.
        text = str(result)
        masked = _mask_text(text, self._compiled, self.replacement)
        if masked != text:
            return masked
        return result

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: Any,
        response: ModelResponse,
    ) -> ModelResponse:
        """Scrub secrets from model response text parts."""
        for part in response.parts:
            if isinstance(part, TextPart):
                part.content = _mask_text(part.content, self._compiled, self.replacement)
        return response
