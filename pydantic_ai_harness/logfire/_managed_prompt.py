"""Back an agent's instructions with a Logfire-managed prompt."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import logfire
from logfire.variables.variable import Variable
from pydantic_ai import TemplateStr
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from logfire import Logfire
    from logfire.variables.abstract import ResolvedVariable
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult


# Logfire exposes a managed prompt with slug `<slug>` as a variable named `prompt__<slug>`,
# with hyphens replaced by underscores (see the Logfire prompt-management docs). `prompt__`
# is reserved for these system-managed prompts.
_PROMPT_VARIABLE_PREFIX = 'prompt__'


def _new_resolved_var() -> ContextVar[ResolvedVariable[str] | None]:
    # `None` means nothing has been resolved for the active run.
    return ContextVar('managed_prompt_resolved', default=None)


@dataclass
class ManagedPrompt(AbstractCapability[AgentDepsT]):
    """Back an agent's instructions with a Logfire-managed prompt.

    Pass a prompt slug and a code default and the capability declares the backing
    [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    for you -- a slug of `support_agent` resolves the variable `prompt__support_agent`, matching
    the naming Logfire's [Prompt management](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/)
    uses. You can iterate on the prompt from the Logfire UI -- versioned, labelled, and rolled
    out -- without redeploying, while the code default keeps the agent working when no remote
    value is available.

    ```python {test="skip"}
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedPrompt

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            ManagedPrompt(
                'support_agent',
                default='You are a helpful customer support agent. Be friendly and concise.',
                label='production',
            )
        ],
    )
    result = agent.run_sync('My order never arrived.')
    ```

    The prompt value is resolved **once per run**, inside the run's
    [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] hook, using the
    [`ResolvedVariable`][logfire.ResolvedVariable] as a context manager that stays open for the
    whole run -- so the selected label and version are attached as baggage to every child span
    of the agent run.

    Declaring the same slug more than once is fine -- each `ManagedPrompt` constructs its own
    backing variable, so sharing a prompt across several agents just works. Pass an existing
    [`logfire.Variable`][logfire.Variable] as `prompt` instead of a slug when you want to use a
    variable you defined yourself (for example a `template_var`, or one registered for
    [`variables_push`][logfire.Logfire.variables_push]).
    """

    prompt: str | Variable[str]
    """The prompt slug (declared as the variable `prompt__<slug>`), or a pre-built `logfire.Variable`."""

    default: str | None = None
    """Code-default prompt text. Required when `prompt` is a slug; ignored when `prompt` is a `Variable`."""

    label: str | None = None
    """Explicit label to resolve (e.g. `'production'`). When `None`, the variable's
    rollout and targeting rules select the label."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = None
    """Key for deterministic label selection, or a callable that derives it from the
    [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire falls back to its
    own targeting context and then the active trace id."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = None
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    render_template: bool = False
    """When `True`, render the resolved prompt as a Handlebars template against the agent's
    `deps` (the same mechanism as [`TemplateStr`][pydantic_ai.TemplateStr]); `{{field}}` is
    filled from `deps`. Requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`).
    Defaults to `False`, so the resolved prompt is used verbatim."""

    logfire_instance: Logfire | None = None
    """Logfire instance to resolve the variable on. When `None`, the global default instance
    (the one backing the module-level [`logfire.var`][logfire.var]) is used. Ignored when
    `prompt` is a `Variable`."""

    _variable: Variable[str] = field(init=False, repr=False, compare=False)
    """The managed variable backing the prompt (declared from the slug, or the one passed in)."""

    _resolved: ContextVar[ResolvedVariable[str] | None] = field(
        default_factory=_new_resolved_var, init=False, repr=False, compare=False
    )
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            self._variable = self.prompt
            return

        if self.default is None:
            raise TypeError('`default` is required when `prompt` is a slug.')

        slug = self.prompt
        if slug.startswith(_PROMPT_VARIABLE_PREFIX):
            warnings.warn(
                f'The {_PROMPT_VARIABLE_PREFIX!r} prefix is added automatically; '
                f'pass the bare prompt slug rather than {slug!r}.',
                stacklevel=2,
            )
            slug = slug[len(_PROMPT_VARIABLE_PREFIX) :]

        name = f'{_PROMPT_VARIABLE_PREFIX}{slug.replace("-", "_")}'
        if not name.isidentifier():
            raise ValueError(
                f'Prompt slug {self.prompt!r} produces an invalid variable name {name!r}; '
                'slugs may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so re-declaring the
        # same slug is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one prompt across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        self._variable = Variable(name, type=str, default=self.default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[str] | None:
        """The prompt resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which prompt version is in play.
        """
        return self._resolved.get()

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the prompt's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """Provide the resolved prompt to the agent's system prompt."""

        def instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            resolved = self.resolved
            if resolved is None:
                # No active run -- contribute no instructions.
                return None
            if self.render_template:
                return TemplateStr[AgentDepsT](resolved.value).render(ctx.deps)
            return resolved.value

        return instructions

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Resolve the prompt once and keep its baggage active for the duration of the run."""
        if callable(self.targeting_key):
            targeting_key = self.targeting_key(ctx)
        else:
            targeting_key = self.targeting_key

        if callable(self.attributes):
            attributes = self.attributes(ctx)
        else:
            attributes = self.attributes

        resolved = self._variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)
