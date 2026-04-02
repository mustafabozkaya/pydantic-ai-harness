"""Skills capability for progressive tool loading.

Skills are reusable knowledge-and-tool packages that agents discover and load
on demand, preserving the context window by deferring full definitions until
the agent actually needs them.

A :class:`Skill` bundles a name, description, optional instructions, and a
set of tools (as callables or a :class:`~pydantic_ai.FunctionToolset`).  The
:class:`Skills` capability exposes two meta-tools to the agent:

* ``search_skills(query)`` -- returns matching skill names and descriptions.
* ``load_skill(name)`` -- activates a skill's tools for the current run.

Tools belonging to unloaded skills are hidden from the model via
:meth:`~pydantic_ai.capabilities.AbstractCapability.prepare_tools`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.function import FunctionToolset


@dataclass
class Skill:
    """A self-contained skill that an agent can discover and load on demand.

    Args:
        name: Short, unique identifier (lowercase, hyphens allowed).
        description: One-line summary shown in search results.
        tools: Callables (or :class:`~pydantic_ai.FunctionToolset`) whose
            tools become available when the skill is loaded.
        instructions: Optional long-form guidance included in the system
            prompt when the skill is loaded.
    """

    name: str
    description: str
    tools: Sequence[Callable[..., Any]] | FunctionToolset[Any] = field(
        default_factory=lambda: list[Callable[..., Any]]()
    )
    instructions: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if not re.fullmatch(r'[a-z0-9]([a-z0-9-]*[a-z0-9])?', self.name):
            raise ValueError(f'Skill name must be lowercase alphanumeric with optional hyphens, got {self.name!r}')

    def tool_names(self) -> list[str]:
        """Return the names of all tools provided by this skill."""
        if isinstance(self.tools, FunctionToolset):
            return list(self.tools.tools.keys())
        return [_func_name(fn) for fn in self.tools]


def _func_name(fn: Callable[..., Any]) -> str:
    """Best-effort name extraction from a callable."""
    return getattr(fn, '__name__', None) or getattr(fn, '__qualname__', str(fn))


def load_skills_from_directory(directory: str | Path) -> list[Skill]:
    """Load skills from markdown files in *directory*.

    Each ``.md`` file is parsed as a skill definition: YAML frontmatter
    provides ``name`` and ``description``, and the body becomes the
    ``instructions``.  Frontmatter is delimited by ``---`` lines.

    Example file ``my-skill.md``::

        ---
        name: my-skill
        description: Does something useful
        ---
        Detailed instructions for the agent...

    Skills loaded from markdown carry no tools -- they are pure
    knowledge packages.  Pair them with Python-defined skills or
    attach tools separately.

    Args:
        directory: Path to scan for ``.md`` files (non-recursive).

    Returns:
        List of :class:`Skill` instances, one per file.

    Raises:
        ValueError: If a file has invalid or missing frontmatter.
    """
    dirpath = Path(directory)
    skills: list[Skill] = []
    for md_file in sorted(dirpath.glob('*.md')):
        text = md_file.read_text(encoding='utf-8')
        skill = _parse_skill_markdown(text, source=str(md_file))
        skills.append(skill)
    return skills


def _parse_skill_markdown(text: str, *, source: str = '<string>') -> Skill:
    """Parse a markdown string with YAML frontmatter into a :class:`Skill`.

    Raises:
        ValueError: If frontmatter is missing or incomplete.
    """
    stripped = text.strip()
    if not stripped.startswith('---'):
        raise ValueError(f'Missing YAML frontmatter in {source}')

    # Find closing delimiter
    end = stripped.find('---', 3)
    if end == -1:
        raise ValueError(f'Unclosed YAML frontmatter in {source}')

    frontmatter_text = stripped[3:end].strip()
    body = stripped[end + 3 :].strip() or None

    # Minimal YAML-like parsing (key: value lines) to avoid a hard
    # dependency on PyYAML for this simple case.
    fm: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        fm[key.strip()] = value.strip()

    name = fm.get('name')
    description = fm.get('description')
    if not name:
        raise ValueError(f'Frontmatter missing required "name" field in {source}')
    if not description:
        raise ValueError(f'Frontmatter missing required "description" field in {source}')

    return Skill(name=name, description=description, instructions=body)


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """Capability for progressive skill discovery and loading.

    Provides ``search_skills`` and ``load_skill`` meta-tools.  Tools
    belonging to registered skills are hidden until the agent explicitly
    loads the skill that owns them.

    Per-run state (which skills are loaded) is isolated via
    :meth:`for_run`.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness.skills import Skill, Skills

        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        math_skill = Skill(
            name='math',
            description='Basic arithmetic operations',
            tools=[add],
        )
        agent = Agent('openai:gpt-4o', capabilities=[Skills(skills=[math_skill])])
    """

    skills: list[Skill] = field(default_factory=lambda: list[Skill]())
    """Registered skills."""

    _loaded_skill_names: set[str] = field(default_factory=lambda: set[str](), init=False, repr=False)
    """Names of skills that have been loaded in the current run (per-run state)."""

    @classmethod
    def get_serialization_name(cls) -> str | None:  # noqa: D102
        return 'Skills'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Skills[Any]:
        """Create from spec arguments.

        Accepts ``dirs`` (list of directory paths) to load markdown skills.
        """
        dirs: list[str] = kwargs.pop('dirs', []) or list(args)
        all_skills: list[Skill] = []
        for d in dirs:
            all_skills.extend(load_skills_from_directory(d))
        return cls(skills=all_skills, **kwargs)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Skills[AgentDepsT]:
        """Return a fresh copy with empty loaded-skills state."""
        clone: Skills[AgentDepsT] = Skills(skills=self.skills)
        return clone

    def get_instructions(self) -> str | None:
        """Provide baseline instructions for skill discovery."""
        if not self.skills:
            return None
        return (
            'You have access to a skill catalog. '
            'Use `search_skills` to find relevant skills by keyword, '
            'then `load_skill` to activate a skill and make its tools available. '
            "Only loaded skills' tools appear in your tool list."
        )

    def get_toolset(self) -> FunctionToolset[AgentDepsT] | None:
        """Build the toolset containing meta-tools and all skill tools."""
        if not self.skills:
            return None

        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        # Register meta-tools
        toolset.add_function(self._search_skills, takes_ctx=False, name='search_skills')
        toolset.add_function(self._load_skill, takes_ctx=False, name='load_skill')

        # Register each skill's tools (they will be hidden until loaded)
        for skill in self.skills:
            if isinstance(skill.tools, FunctionToolset):
                for tool in skill.tools.tools.values():
                    toolset.add_tool(tool)
            else:
                for fn in skill.tools:
                    toolset.add_function(fn, takes_ctx=False)

        return toolset

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide tools belonging to skills that have not been loaded yet."""
        # Build set of tool names that should be hidden
        hidden: set[str] = set()
        for skill in self.skills:
            if skill.name not in self._loaded_skill_names:
                hidden.update(skill.tool_names())

        # Always keep meta-tools visible
        meta_tools = {'search_skills', 'load_skill'}

        return [td for td in tool_defs if td.name in meta_tools or td.name not in hidden]

    # -- Meta-tool implementations --

    def _search_skills(self, query: str) -> list[dict[str, str]]:
        """Search available skills by keyword.

        Returns a list of matching skills with their name, description,
        and whether they are currently loaded.

        Args:
            query: A keyword or phrase to search for in skill names and descriptions.
        """
        query_lower = query.lower()
        results: list[dict[str, str]] = []
        for skill in self.skills:
            if query_lower in skill.name.lower() or query_lower in skill.description.lower():
                results.append(
                    {
                        'name': skill.name,
                        'description': skill.description,
                        'loaded': 'yes' if skill.name in self._loaded_skill_names else 'no',
                    }
                )
        return results

    def _load_skill(self, name: str) -> str:
        """Load a skill by name, making its tools available.

        After loading, the skill's tools will appear in subsequent tool
        lists and any associated instructions will be included.

        Args:
            name: The exact name of the skill to load (as returned by ``search_skills``).
        """
        skill = self._find_skill(name)
        if skill is None:
            available = ', '.join(s.name for s in self.skills)
            return f'Skill {name!r} not found. Available skills: {available}'

        self._loaded_skill_names.add(name)

        parts = [f'Skill {name!r} loaded.']
        tool_names = skill.tool_names()
        if tool_names:
            parts.append(f'Available tools: {", ".join(tool_names)}')
        if skill.instructions:
            parts.append(f'Instructions:\n{skill.instructions}')
        return '\n'.join(parts)

    # -- Helpers --

    def _find_skill(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None
