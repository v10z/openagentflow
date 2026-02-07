"""Pytest configuration and fixtures for Open Agent Flow tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    from openagentflow import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return add


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    from openagentflow import agent

    @agent(model="claude-sonnet-4-20250514")
    async def test_agent(query: str) -> str:
        """Test agent."""
        return f"Response to: {query}"

    return test_agent


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "content": "This is a test response",
        "tool_calls": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
