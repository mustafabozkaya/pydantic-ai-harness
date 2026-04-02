# Skills Capability

Closes #22. Partially addresses #40 (deferred tool loading via search + load pattern).

## Summary

Implements a `Skills` capability that enables progressive tool loading: agents discover
skills via `search_skills(query)` and activate them via `load_skill(name)`, keeping
unloaded tools hidden from the model's context window.

## Design

### `Skill` dataclass

A skill bundles a name, description, optional instructions, and a set of tools:

```python
Skill(
    name='math',              # lowercase, hyphens allowed
    description='Arithmetic', # shown in search results
    tools=[add, subtract],    # callables or a FunctionToolset
    instructions='...',       # included when skill is loaded
)
```

Tools can be plain callables (registered as `tool_plain`) or a pre-built
`FunctionToolset` for more control.

### `Skills` capability

An `AbstractCapability` subclass providing:

| Method | Purpose |
|--------|---------|
| `get_instructions()` | Tells the agent about the skill catalog |
| `get_toolset()` | Registers `search_skills`, `load_skill`, and all skill tools |
| `prepare_tools()` | Hides tools from unloaded skills each model request |
| `for_run()` | Isolates per-run loaded-skills state |
| `from_spec(dirs=[...])` | Loads markdown skills from directories (Tier S serializable) |

### Markdown skill files

Skills can be defined as `.md` files with YAML frontmatter:

```markdown
---
name: my-skill
description: Does something useful
---
Detailed instructions for the agent...
```

Loaded via `load_skills_from_directory(path)` or `Skills.from_spec(dirs=[path])`.
Markdown skills are pure knowledge (no tools) -- pair with Python skills as needed.

### Progressive disclosure flow

1. Agent sees `search_skills` and `load_skill` tools (always visible)
2. Agent calls `search_skills("math")` -- gets name + description matches
3. Agent calls `load_skill("math")` -- gets instructions + tool names, tools become visible
4. Agent can now call `add(1, 2)` etc.

## Files changed

- `src/pydantic_harness/skills.py` -- `Skill`, `Skills`, `load_skills_from_directory`
- `src/pydantic_harness/__init__.py` -- exports
- `tests/test_skills.py` -- 47 tests covering all code paths
- `pyproject.toml` -- pyright test override for private usage

## Prior art considered

- vstorm-co/pydantic-ai-skills (SkillsCapability + SkillsToolset)
- OpenAI Agents SDK ToolSearchTool
- Anthropic Tool Search
- Microsoft Agent Framework SKILL.md spec
