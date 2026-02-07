"""Tests for the @agent decorator."""

from __future__ import annotations

import pytest

from openagentflow import agent, tool, get_agent, get_all_agents
from openagentflow.core.types import AgentStatus, ReasoningStrategy


class TestAgentDecorator:
    """Tests for the @agent decorator."""

    def test_agent_registration(self):
        """Test that decorated agents are registered."""
        @agent(model="claude-sonnet-4-20250514")
        async def test_agent_reg(query: str) -> str:
            """Test agent for registration."""
            pass

        spec = get_agent("test_agent_reg")
        assert spec is not None
        assert spec.name == "test_agent_reg"
        assert spec.model.model_id == "claude-sonnet-4-20250514"

    def test_agent_with_tools(self):
        """Test agent with tools."""
        @tool
        def search_tool(query: str) -> list:
            """Search for something."""
            return []

        @agent(model="claude-sonnet-4-20250514", tools=[search_tool])
        async def search_agent(query: str) -> str:
            """Agent with search tool."""
            pass

        spec = get_agent("search_agent")
        assert spec is not None
        assert len(spec.tools) == 1
        assert spec.tools[0].name == "search_tool"

    def test_agent_custom_name(self):
        """Test agent with custom name."""
        @agent(model="claude-sonnet-4-20250514", name="my_custom_agent")
        async def internal_name(query: str) -> str:
            """Agent with custom name."""
            pass

        spec = get_agent("my_custom_agent")
        assert spec is not None
        assert spec.name == "my_custom_agent"

    def test_agent_description_from_docstring(self):
        """Test that description comes from docstring."""
        @agent(model="claude-sonnet-4-20250514")
        async def documented_agent(query: str) -> str:
            """This is the agent description from docstring."""
            pass

        spec = get_agent("documented_agent")
        assert "description from docstring" in spec.description.lower()

    def test_agent_custom_description(self):
        """Test agent with custom description."""
        @agent(
            model="claude-sonnet-4-20250514",
            description="Custom description override",
        )
        async def desc_agent(query: str) -> str:
            """Docstring description."""
            pass

        spec = get_agent("desc_agent")
        assert spec.description == "Custom description override"

    def test_agent_max_iterations(self):
        """Test max_iterations setting."""
        @agent(model="claude-sonnet-4-20250514", max_iterations=20)
        async def iter_agent(query: str) -> str:
            """Agent with custom iterations."""
            pass

        spec = get_agent("iter_agent")
        assert spec.max_iterations == 20

    def test_agent_timeout(self):
        """Test timeout setting."""
        @agent(model="claude-sonnet-4-20250514", timeout=60.0)
        async def timeout_agent(query: str) -> str:
            """Agent with custom timeout."""
            pass

        spec = get_agent("timeout_agent")
        assert spec.timeout_seconds == 60.0

    def test_agent_system_prompt(self):
        """Test system_prompt setting."""
        @agent(
            model="claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant.",
        )
        async def prompt_agent(query: str) -> str:
            """Agent with system prompt."""
            pass

        spec = get_agent("prompt_agent")
        assert spec.system_prompt == "You are a helpful assistant."

    def test_agent_reasoning_strategy_string(self):
        """Test reasoning strategy as string."""
        @agent(model="claude-sonnet-4-20250514", reasoning_strategy="react")
        async def react_agent(query: str) -> str:
            """ReAct agent."""
            pass

        spec = get_agent("react_agent")
        assert spec.reasoning_strategy == ReasoningStrategy.REACT

    def test_agent_reasoning_strategy_enum(self):
        """Test reasoning strategy as enum."""
        @agent(
            model="claude-sonnet-4-20250514",
            reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )
        async def cot_agent(query: str) -> str:
            """CoT agent."""
            pass

        spec = get_agent("cot_agent")
        assert spec.reasoning_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT

    def test_agent_source_code_capture(self):
        """Test that source code is captured."""
        @agent(model="claude-sonnet-4-20250514")
        async def source_agent(query: str) -> str:
            """Agent with source capture."""
            pass

        spec = get_agent("source_agent")
        assert spec.source_code is not None
        assert "source_agent" in spec.source_code

    def test_agent_has_spec_attribute(self):
        """Test that decorated function has _agent_spec attribute."""
        @agent(model="claude-sonnet-4-20250514")
        async def attr_agent(query: str) -> str:
            """Agent with spec attribute."""
            pass

        assert hasattr(attr_agent, "_agent_spec")
        assert attr_agent._agent_spec.name == "attr_agent"


class TestModelParsing:
    """Tests for model string parsing."""

    def test_claude_sonnet_string(self):
        """Test parsing claude-sonnet-4-20250514 string."""
        @agent(model="claude-sonnet-4-20250514")
        async def sonnet_agent(q: str) -> str:
            pass

        spec = get_agent("sonnet_agent")
        assert spec.model.model_id == "claude-sonnet-4-20250514"

    def test_gpt4_string(self):
        """Test parsing gpt-4 string."""
        @agent(model="gpt-4")
        async def gpt_agent(q: str) -> str:
            pass

        spec = get_agent("gpt_agent")
        assert "gpt" in spec.model.model_id.lower()

    def test_unknown_model_defaults_to_anthropic(self):
        """Test that unknown models default to Anthropic."""
        from openagentflow.core.types import LLMProvider

        @agent(model="some-unknown-model")
        async def unknown_agent(q: str) -> str:
            pass

        spec = get_agent("unknown_agent")
        assert spec.model.provider == LLMProvider.ANTHROPIC


class TestGetAllAgents:
    """Tests for get_all_agents function."""

    def test_get_all_returns_dict(self):
        """Test that get_all_agents returns a dict."""
        agents = get_all_agents()
        assert isinstance(agents, dict)

    def test_get_all_includes_registered(self):
        """Test that get_all_agents includes registered agents."""
        @agent(model="claude-sonnet-4-20250514")
        async def listed_agent(q: str) -> str:
            pass

        agents = get_all_agents()
        assert "listed_agent" in agents
