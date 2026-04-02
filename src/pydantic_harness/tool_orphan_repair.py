"""Capability that sanitizes message history to fix orphaned tool calls and results.

Multi-turn conversations with tools can accumulate structurally invalid message
history -- tool calls without matching results, or results referencing calls that
no longer exist.  Providers (especially Anthropic) reject such history with a 400,
and once a conversation is "poisoned" it stays broken for every subsequent run.

This capability hooks into ``before_model_request`` to repair the history before
each model call, so conversations self-heal instead of permanently breaking.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelRequestPart, ModelResponsePart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

logger = logging.getLogger(__name__)


_ORPHAN_CALL_CONTENT = 'Tool call was not completed.'
"""Synthetic content injected for orphaned tool calls."""


@dataclass
class ToolOrphanRepair(AbstractCapability[Any]):
    """Sanitizes message history to fix orphaned tool calls and results.

    Repairs three classes of structural defects:

    1. **Orphaned tool calls** -- a ``ToolCallPart`` in a ``ModelResponse``
       with no matching ``ToolReturnPart`` or ``RetryPromptPart`` in the
       following ``ModelRequest``.  A synthetic return is injected.
    2. **Orphaned builtin tool calls** -- a ``BuiltinToolCallPart`` in a
       ``ModelResponse`` with no matching ``BuiltinToolReturnPart`` in the
       same response.  A synthetic return is appended to the response.
    3. **Orphaned tool returns** -- a ``ToolReturnPart`` or ``RetryPromptPart``
       in a ``ModelRequest`` whose ``tool_call_id`` does not match any call
       in the preceding ``ModelResponse``.  The orphaned part is stripped.

    Additionally, trailing ``ModelResponse`` messages whose *only* actionable
    parts are unmatched tool calls (no text, no builtin results) are removed
    entirely, since there is no following request to receive synthetic returns.

    When stripping parts empties a ``ModelRequest``, a placeholder
    ``UserPromptPart`` is inserted to maintain user/assistant message
    alternation.

    Usage::

        from pydantic_harness import ToolOrphanRepair

        agent = Agent('anthropic:claude-sonnet', capabilities=[ToolOrphanRepair()])
    """

    orphan_call_content: str = _ORPHAN_CALL_CONTENT
    """Content used for synthetic tool return parts injected for orphaned calls."""

    warn: bool = field(default=True, kw_only=True)
    """Whether to emit a warning when orphans are detected and repaired."""

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Sanitize ``request_context.messages`` before each model request."""
        request_context.messages[:] = _repair_messages(
            request_context.messages,
            orphan_call_content=self.orphan_call_content,
            warn=self.warn,
        )
        return request_context


def _repair_messages(
    messages: list[ModelMessage],
    *,
    orphan_call_content: str = _ORPHAN_CALL_CONTENT,
    warn: bool = True,
) -> list[ModelMessage]:
    """Return a repaired copy of *messages* with orphaned tool calls/results fixed.

    The algorithm makes a single forward pass pairing each ``ModelResponse``
    with the ``ModelRequest`` that follows it.  Within each pair it:

    * collects the set of ``tool_call_id`` values from regular ``ToolCallPart``
      parts in the response,
    * strips any ``ToolReturnPart`` / ``RetryPromptPart`` in the request whose
      ``tool_call_id`` is not in that set,
    * injects synthetic ``ToolReturnPart`` for any call id that has no matching
      return or retry in the request,
    * collects ``BuiltinToolCallPart`` ids from the response and injects
      synthetic ``BuiltinToolReturnPart`` for any that lack a matching
      ``BuiltinToolReturnPart`` in the same response.

    A trailing ``ModelResponse`` (no following request) that contains
    unmatched regular tool calls is stripped.  If stripping empties a
    ``ModelRequest`` of meaningful parts, a placeholder ``UserPromptPart``
    is inserted.
    """
    if not messages:
        return messages

    repaired: list[ModelMessage] = []
    repairs_made = 0

    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, ModelResponse):
            next_request: ModelRequest | None = None
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, ModelRequest):
                    next_request = next_msg

            repaired_response, repaired_request, n_repairs = _repair_response_request_pair(
                msg,
                next_request,
                orphan_call_content=orphan_call_content,
            )
            repairs_made += n_repairs

            if repaired_response is not None:
                repaired.append(repaired_response)
            if repaired_request is not None:
                repaired.append(repaired_request)
                # Skip the original request since we already processed it.
                i += 2
                continue

            i += 1
        else:
            # ModelRequest not preceded by a ModelResponse -- pass through.
            repaired.append(msg)
            i += 1

    if warn and repairs_made:
        warnings.warn(
            f'ToolOrphanRepair: repaired {repairs_made} orphaned tool call/result part(s) in message history.',
            UserWarning,
            stacklevel=2,
        )

    return repaired


def _repair_response_request_pair(
    response: ModelResponse,
    request: ModelRequest | None,
    *,
    orphan_call_content: str,
) -> tuple[ModelResponse | None, ModelRequest | None, int]:
    """Repair a (response, request) pair, returning the repaired versions.

    Returns ``(repaired_response, repaired_request, repair_count)``.
    Either element may be ``None`` if the message was dropped entirely.
    """
    repairs = 0

    # --- Phase 1: Repair orphaned builtin tool calls within the response ---
    response, builtin_repairs = _repair_builtin_tool_calls(response, orphan_call_content)
    repairs += builtin_repairs

    # --- Phase 2: Collect regular tool call ids from the response ---
    call_ids: set[str] = set()
    call_id_to_name: dict[str, str] = {}
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            call_ids.add(part.tool_call_id)
            call_id_to_name[part.tool_call_id] = part.tool_name

    # If no regular tool calls, nothing else to repair.
    if not call_ids:
        return response, request, repairs

    # --- Phase 3: Handle trailing response with no following request ---
    if request is None:
        has_non_call_content = any(not isinstance(p, ToolCallPart) for p in response.parts)
        if has_non_call_content:
            # Keep the response but strip the dangling tool call parts.
            new_resp_parts: list[ModelResponsePart] = [p for p in response.parts if not isinstance(p, ToolCallPart)]
            for cid in sorted(call_ids):
                logger.debug('Stripped orphaned tool call %r from trailing response (text content kept)', cid)
            repairs += len(call_ids)
            return replace(response, parts=new_resp_parts), None, repairs
        else:
            # Response is nothing but unmatched tool calls -- drop it entirely.
            logger.debug(
                'Dropped trailing response containing only orphaned tool calls: %s',
                ', '.join(sorted(call_ids)),
            )
            repairs += len(call_ids)
            return None, None, repairs

    # --- Phase 4: Strip orphaned returns from the request ---
    matched_ids: set[str] = set()
    kept_parts: list[ModelRequestPart] = []

    for part in request.parts:
        if isinstance(part, ToolReturnPart | RetryPromptPart):
            if part.tool_call_id in call_ids:
                matched_ids.add(part.tool_call_id)
                kept_parts.append(part)
            else:
                part_type = 'RetryPromptPart' if isinstance(part, RetryPromptPart) else 'ToolReturnPart'
                logger.debug(
                    'Stripped orphaned %s for tool_call_id %r (no matching call in preceding response)',
                    part_type,
                    part.tool_call_id,
                )
                repairs += 1
        else:
            kept_parts.append(part)

    # --- Phase 5: Inject synthetic returns for orphaned calls ---
    orphaned_call_ids = call_ids - matched_ids
    for call_id in sorted(orphaned_call_ids):
        logger.debug(
            'Injected synthetic ToolReturnPart for orphaned call %r (tool %r)',
            call_id,
            call_id_to_name[call_id],
        )
        kept_parts.append(
            ToolReturnPart(
                tool_name=call_id_to_name[call_id],
                content=orphan_call_content,
                tool_call_id=call_id,
            )
        )
        repairs += 1

    # --- Phase 6: Ensure the request has non-system parts ---
    non_system_parts = [p for p in kept_parts if not isinstance(p, SystemPromptPart)]
    if not non_system_parts:  # pragma: no cover – defensive; Phase 5 always injects non-system parts
        logger.debug('Inserted placeholder UserPromptPart to maintain message alternation')
        kept_parts.append(UserPromptPart(content='Continue.'))
        repairs += 1

    return response, replace(request, parts=kept_parts), repairs


def _repair_builtin_tool_calls(
    response: ModelResponse,
    orphan_call_content: str,
) -> tuple[ModelResponse, int]:
    """Inject synthetic ``BuiltinToolReturnPart`` for orphaned ``BuiltinToolCallPart`` parts.

    Builtin tool calls and returns both live inside the same ``ModelResponse``.
    """
    builtin_call_ids: dict[str, str] = {}  # call_id -> tool_name
    builtin_return_ids: set[str] = set()

    for part in response.parts:
        if isinstance(part, BuiltinToolCallPart):
            builtin_call_ids[part.tool_call_id] = part.tool_name
        elif isinstance(part, BuiltinToolReturnPart):
            builtin_return_ids.add(part.tool_call_id)

    orphaned = set(builtin_call_ids) - builtin_return_ids
    if not orphaned:
        return response, 0

    new_parts: list[ModelResponsePart] = list(response.parts)
    for call_id in sorted(orphaned):
        logger.debug(
            'Injected synthetic BuiltinToolReturnPart for orphaned builtin call %r (tool %r)',
            call_id,
            builtin_call_ids[call_id],
        )
        new_parts.append(
            BuiltinToolReturnPart(
                tool_name=builtin_call_ids[call_id],
                content=orphan_call_content,
                tool_call_id=call_id,
            )
        )

    return replace(response, parts=new_parts), len(orphaned)
