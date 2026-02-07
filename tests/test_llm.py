"""Tests for LLM providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openagentflow.core.types import Message, ModelConfig, LLMProvider, ToolSpec, ToolCall
from openagentflow.llm.base import LLMResponse, StreamChunk


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_total_tokens(self):
        """Test total_tokens property."""
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=20,
        )
        assert response.total_tokens == 30

    def test_has_tool_calls_empty(self):
        """Test has_tool_calls with no tool calls."""
        response = LLMResponse(content="Hello")
        assert response.has_tool_calls is False

    def test_has_tool_calls_with_calls(self):
        """Test has_tool_calls with tool calls."""
        response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="1", tool_name="search", arguments={"query": "test"}),
            ],
        )
        assert response.has_tool_calls is True


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_name(self):
        """Test provider name."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        assert provider.provider_name == "anthropic"

    def test_supported_models(self):
        """Test supported models list."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        models = provider.supported_models
        assert "claude-sonnet-4-20250514" in models
        assert "claude-3-opus-20240229" in models

    def test_supports_model_valid(self):
        """Test supports_model with valid model."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        assert provider.supports_model("claude-sonnet-4-20250514") is True

    def test_supports_model_invalid(self):
        """Test supports_model with invalid model."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        assert provider.supports_model("gpt-4") is False

    def test_estimate_cost_sonnet(self):
        """Test cost estimation for Sonnet model."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        # 1000 input tokens, 500 output tokens
        cost = provider.estimate_cost(1000, 500, "claude-sonnet-4-20250514")
        # Sonnet: $3/1M input, $15/1M output
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_opus(self):
        """Test cost estimation for Opus model."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        cost = provider.estimate_cost(1000, 500, "claude-3-opus-20240229")
        # Opus: $15/1M input, $75/1M output
        expected = (1000 / 1_000_000) * 15.0 + (500 / 1_000_000) * 75.0
        assert abs(cost - expected) < 0.0001

    def test_count_tokens_approximation(self):
        """Test token counting approximation."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        # Without anthropic package, uses approximation
        text = "Hello, how are you today?"
        count = provider.count_tokens(text, "claude-sonnet-4-20250514")
        # Rough approximation: ~4 chars per token
        assert count > 0
        assert count < len(text)

    def test_convert_messages(self):
        """Test message conversion to Anthropic format."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        messages = [
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there!"),
        ]
        converted = provider._convert_messages(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello!"
        assert converted[1]["role"] == "assistant"

    def test_convert_tools(self):
        """Test tool conversion to Anthropic format."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider
        from openagentflow import tool

        provider = AnthropicProvider(api_key="test-key")

        @tool
        def search_web(query: str) -> list:
            """Search the web."""
            return []

        tools = [search_web._tool_spec]
        converted = provider._convert_tools(tools)
        assert len(converted) == 1
        assert converted[0]["name"] == "search_web"
        assert "Search the web" in converted[0]["description"]
        assert "input_schema" in converted[0]

    @pytest.mark.asyncio
    async def test_generate_missing_api_key(self):
        """Test generate fails without API key or anthropic package."""
        from openagentflow.llm.providers.anthropic_ import AnthropicProvider
        from openagentflow.exceptions import AuthenticationError, LLMError

        # Clear env var if set
        import os
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            provider = AnthropicProvider(api_key=None)
            # Will raise either LLMError (if anthropic not installed)
            # or AuthenticationError (if anthropic installed but no key)
            with pytest.raises((AuthenticationError, LLMError)):
                await provider.generate(
                    messages=[Message(role="user", content="Hello")],
                    config=ModelConfig(provider=LLMProvider.ANTHROPIC, model_id="claude-sonnet-4-20250514"),
                )
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_text_chunk(self):
        """Test text chunk."""
        chunk = StreamChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.is_tool_call_start is False
        assert chunk.is_final is False

    def test_tool_call_start_chunk(self):
        """Test tool call start chunk."""
        chunk = StreamChunk(
            is_tool_call_start=True,
            tool_call_id="123",
            tool_name="search",
        )
        assert chunk.is_tool_call_start is True
        assert chunk.tool_call_id == "123"
        assert chunk.tool_name == "search"

    def test_final_chunk(self):
        """Test final chunk."""
        chunk = StreamChunk(is_final=True)
        assert chunk.is_final is True
