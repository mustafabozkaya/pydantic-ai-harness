# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can iterate on it from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying.

Install the extra:

```bash
pip install 'pydantic-ai-harness[logfire]'
```

## `ManagedPrompt`

Back an agent's instructions with a Logfire-managed
[Prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/).

### The problem

Prompts are critical to agent behavior, but iterating on them through the normal
edit → review → deploy loop is slow, and you can't easily A/B test a change or roll it
back the moment it misbehaves in production.

### The solution

`ManagedPrompt` declares the backing managed variable for you and resolves it **once per
run**, feeding the value into the agent's instructions. The resolution happens inside the
run's `wrap_run` hook using the
[`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as a context manager that stays open for the whole run -- so the selected label and version
are attached as baggage to every child span of the agent run. You get a direct correlation
between a run's behavior and the exact prompt version that produced it, plus instant
iteration and rollback from the Logfire UI.

### Usage

Pass the prompt slug and a code default. The slug `support_agent` is declared as the managed
variable `prompt__support_agent` -- the naming Logfire's Prompt management uses (hyphens in a
slug become underscores). The default keeps the agent working until a remote value is published.

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
print(result.output)
```

### Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) -- handy
when the key lives in your agent's `deps`:

```python {test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    user_id: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent.',
            targeting_key=lambda ctx: ctx.deps.user_id,
        ),
    ],
)
```

Pass `attributes` (or a callable returning them) for condition-based targeting rules.
When `label` is omitted, the variable's rollout and targeting rules pick the label;
when both `targeting_key` and `attributes` are omitted, Logfire falls back to its own
targeting context and then to the active trace id.

### Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps` — the same mechanism as
[`TemplateStr`](https://ai.pydantic.dev/api/#pydantic_ai.TemplateStr) — so `{{field}}` is filled
from `deps`:

```python {test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt


@dataclass
class Deps:
    customer_name: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are helping {{customer_name}}. Be friendly and concise.',
            render_template=True,
        ),
    ],
)
```

Rendering requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`). It is off by default.

### Using your own variable

Declaring the same slug more than once is fine -- each `ManagedPrompt` builds its own backing
variable, so sharing a prompt across several agents just works. Pass an existing
[`logfire.Variable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as the first argument instead of a slug when you want to declare the variable yourself --
for example a `template_var`, or one registered for `variables_push`:

```python {test="skip"}
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import ManagedPrompt

logfire.configure()

support_prompt = logfire.var(
    name='prompt__support_agent',
    type=str,
    default='You are a helpful customer support agent. Be friendly and concise.',
)

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `prompt` is a slug, pass `logfire_instance=` to declare the variable on a specific
Logfire instance instead of the module-level default.

### Notes

- The prompt resolves to a `str`. By default it's used verbatim; set `render_template=True`
  to render `{{...}}` against `deps` (see [Templating with deps](#templating-with-deps)).
- Resolution is isolated per run via a context variable, so a single capability instance
  is safe to share across concurrent runs.
- `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable` (value, label, version,
  reason) for inspection -- e.g. from inside a tool.
- The capability runs outermost (wrapping `Instrumentation`) so the prompt's baggage covers
  the agent run span as well as its children.
