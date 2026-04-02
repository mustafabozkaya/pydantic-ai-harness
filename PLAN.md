# SecretMasking Capability

## Problem

Without credential masking, secrets (API keys, tokens, connection strings, private keys) can leak through:
- Tool outputs (e.g., `git push` echoing credentials)
- Model responses (LLM reproducing secrets from context)
- Conversation history, logs, and serialized state

Closes #78.

## Design

`SecretMasking` is an `AbstractCapability` that uses two hooks:

1. **`after_tool_execute`** -- scrubs tool return values before they enter message history
2. **`after_model_request`** -- scrubs `TextPart` content in model responses

### Built-in pattern categories

| Category | Patterns |
|---|---|
| `api_keys` | OpenAI (`sk-*`), Anthropic (`sk-ant-*`), AWS (`AKIA*`), GitHub (`gh[psorat]_*`), Slack (`xox[bpas]-*`), Google (`AIza*`), generic `api_key=` |
| `tokens` | Bearer tokens, JWTs |
| `connection_strings` | Passwords in URLs (`://user:pass@host`), database connection strings (postgres, mongo, mysql, redis, amqp) |
| `private_keys` | RSA, EC, OpenSSH private key headers |

All patterns are compiled at module level as constants.

### Configuration

- `categories`: select which built-in categories to enable (default: all)
- `custom_patterns`: additional `{name: regex}` pairs
- `replacement`: the replacement string (default: `"[REDACTED]"`)

### Non-string tool results

For string results, masking is applied directly. For non-string results, we convert to string to check for matches; if any secret is found, the masked string is returned instead of the original object (safe default -- the model sees the sanitized representation).

## Scope

This PR implements regex-based secret _masking_ (redaction). The broader SecretRegistry / encrypted storage / env-var blocking described in #78 are left for follow-up work, as they are infrastructure concerns rather than capability hooks.

## Files

- `src/pydantic_harness/secret_masking.py` -- capability implementation
- `src/pydantic_harness/__init__.py` -- public export
- `tests/test_secret_masking.py` -- 45 tests, 100% coverage
