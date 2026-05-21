# Guardrails

Intercept unsafe user prompts before they reach the model, and unsafe model outputs before they reach the caller.

## The problem

Agents take unstructured input from users and return unstructured output to callers. Without a validation layer, a prompt injection attempt, PII-laden message, or off-topic question goes to the model as-is, and any output the model produces is returned verbatim. The framework does not reason about "this is unsafe to send" or "this is unsafe to show".

## The solution

Two capabilities, each backed by a callable you supply.

| Capability | Checks | When a guard blocks | When a guard raises |
|---|---|---|---|
| `InputGuard` | The user prompt before the first model request | `SkipModelRequest` — the model call is skipped and a refusal message becomes the response for that step | The exception propagates out of the run |
| `OutputGuard` | The final run output | `OutputBlocked` is raised | The exception propagates out of the run |

The asymmetry is intentional. Blocking the input means no tokens are spent, so a graceful refusal is almost always what you want. Blocking the output means the model already generated a response you do not want exposed — raising forces the caller to decide what to do next.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuard, OutputGuard


def no_secrets(prompt: str) -> bool:
    return 'api_key' not in prompt.lower()


def no_pii(output: object) -> bool:
    return 'SSN' not in str(output)


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuard(guard=no_secrets),
        OutputGuard(guard=no_pii),
    ],
)
```

`OutputGuard` receives `result.output` unchanged — no automatic stringification. For a string output the guard reads it directly; for a typed (Pydantic model) output the guard gets the model instance, so pick the serialization that fits the check:

```python
from pydantic import BaseModel
from pydantic_ai_harness import OutputGuard


class Answer(BaseModel):
    reply: str
    sources: list[str]


def no_internal_urls(output: object) -> bool:
    if isinstance(output, Answer):
        return not any('internal.example.com' in url for url in output.sources)
    return 'internal.example.com' not in str(output)


OutputGuard(guard=no_internal_urls)
```

This avoids the trap of `str(MyModel(...))` producing a `MyModel(field=...)` repr that hides field contents from regex-based checks. If you want JSON text, call `output.model_dump_json()` inside the guard.

Both guards accept async callables too:

```python
async def check_with_moderation_api(prompt: str) -> bool:
    response = await client.moderations.create(input=prompt)
    return not response.results[0].flagged


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[InputGuard(guard=check_with_moderation_api)],
)
```

## Parallel input guards

When a guard is slow (an LLM-based classifier or a network call), running it in sequence before every model request adds latency to every turn. Set `parallel=True` to race the guard against the model call. The model call is cancelled immediately if the guard reports a violation.

```python
InputGuard(guard=slow_async_classifier, parallel=True)
```

For fast local checks (regex, keyword lookup, a small classifier) sequential is usually fine — the overhead is measured in microseconds and the wiring is simpler.

## Refusal messages

A guard returns either a bare `bool` (`True` = safe) or a `GuardResult`. Return a `GuardResult` to attach a message describing what tripped the guard — built at the moment the guard decides, so it can carry the guard's own reasoning instead of a string frozen at construction time:

```python
from pydantic_ai_harness import GuardResult, InputGuard


def no_secrets(prompt: str) -> GuardResult:
    if 'api_key' in prompt.lower():
        return GuardResult(safe=False, message='Your message looks like it contains an API key — please remove it.')
    return GuardResult(safe=True)


InputGuard(guard=no_secrets)
```

For `InputGuard` the message is returned as the model response for that step, so the caller sees a normal completion rather than an exception — multi-turn agents can continue from there. For `OutputGuard` the message is attached to the `OutputBlocked` exception. A bare `False`, or a `GuardResult` with `message=None`, falls back to a default message.

## Accessing run context

A guard may take a `RunContext` as its first parameter when it needs run state — `deps` for tenant- or role-aware policy, message history for conversation-aware checks. The parameter is detected from the signature, so prompt-only guards need not declare it:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import InputGuard


def tenant_policy(ctx: RunContext[MyDeps], prompt: str) -> bool:
    return ctx.deps.tier == 'pro' or 'advanced-feature' not in prompt


InputGuard(guard=tenant_policy)
```

## Hard-fail path

Reporting a value unsafe — returning `False` or a blocking `GuardResult` — is the graceful path. If you want the caller to see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is — you can use `InputBlocked` / `OutputBlocked` from this module or your own exception types.

## API

```python
@dataclass
class GuardResult:
    safe: bool
    message: str | None = None


InputGuard(
    guard: Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]],
    parallel: bool = False,
)

OutputGuard(
    guard: Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]],
)
```

The guard callable takes the inspected value — the prompt for `InputGuard`, `result.output` for `OutputGuard` — optionally preceded by a `RunContext`.

## Relationship to `pydantic-ai-shields`

`pydantic-ai-shields` provides opinionated implementations on top of these primitives (prompt-injection detectors, PII scrubbers, keyword blocklists, etc.). Use the guardrails here when you want to plug in your own validation logic; reach for shields when you need a batteries-included detector.
