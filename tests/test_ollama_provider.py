"""Regression tests for the Ollama LLM provider.

All HTTP calls are mocked via unittest.mock.patch so that no real Ollama
server is required.  Tests verify message parsing, tool call handling,
error propagation, model prefix stripping, and cost estimation.
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from openagentflow.core.types import Message, ModelConfig, ToolSpec, LLMProvider
from openagentflow.exceptions import LLMError
from openagentflow.llm.base import LLMResponse
from openagentflow.llm.providers.ollama_ import (
    DEFAULT_BASE_URL,
    OllamaProvider,
    is_ollama_available,
)


# =====================================================================
# Helper fixtures
# =====================================================================


@pytest.fixture
def provider() -> OllamaProvider:
    """Create an OllamaProvider pointed at localhost."""
    return OllamaProvider(base_url="http://localhost:11434")


@pytest.fixture
def config() -> ModelConfig:
    """Create a basic ModelConfig for Ollama tests."""
    return ModelConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llama3",
        temperature=0.7,
        max_tokens=1024,
    )


def _mock_ollama_response(
    content: str = "Hello!",
    tool_calls: list | None = None,
    done_reason: str = "stop",
    model: str = "llama3",
    prompt_eval_count: int = 10,
    eval_count: int = 20,
) -> dict:
    """Build a mock Ollama /api/chat JSON response."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": model,
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


# =====================================================================
# Generate tests
# =====================================================================


class TestOllamaGenerate:
    """Tests for OllamaProvider.generate() with mocked HTTP."""

    async def test_ollama_generate(self, provider: OllamaProvider, config: ModelConfig):
        """Verify that generate() correctly parses a standard text response.

        Mocks the HTTP POST to /api/chat and confirms the LLMResponse
        fields (content, token counts, stop_reason) are populated correctly.
        """
        mock_data = _mock_ollama_response(
            content="The answer is 42.",
            prompt_eval_count=15,
            eval_count=25,
        )

        with patch.object(provider, "_post_json", return_value=mock_data):
            messages = [Message.user("What is the answer?")]
            response = await provider.generate(messages, config)

        assert isinstance(response, LLMResponse)
        assert response.content == "The answer is 42."
        assert response.input_tokens == 15
        assert response.output_tokens == 25
        assert response.stop_reason == "stop"
        assert response.model_id == "llama3"

    async def test_ollama_with_tools(self, provider: OllamaProvider, config: ModelConfig):
        """Verify that tool call responses are parsed correctly.

        When Ollama returns tool_calls in the message, the provider should
        create ToolCall objects with the correct name and arguments.
        """
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "NYC", "unit": "celsius"},
                },
            }
        ]
        mock_data = _mock_ollama_response(
            content="",
            tool_calls=tool_calls,
        )

        with patch.object(provider, "_post_json", return_value=mock_data):
            messages = [Message.user("What's the weather in NYC?")]

            # Provide a matching tool spec.
            weather_tool = ToolSpec(
                name="get_weather",
                description="Get weather",
                func=lambda **kwargs: None,
                input_schema={"type": "object", "properties": {}},
            )
            response = await provider.generate(messages, config, tools=[weather_tool])

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "NYC", "unit": "celsius"}
        assert response.stop_reason == "tool_calls"

    async def test_ollama_with_tools_string_arguments(
        self, provider: OllamaProvider, config: ModelConfig
    ):
        """Verify that tool call arguments as JSON strings are parsed correctly.

        Some Ollama models return arguments as a JSON string rather than a dict.
        """
        tool_calls = [
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "python decorators"}',
                },
            }
        ]
        mock_data = _mock_ollama_response(content="", tool_calls=tool_calls)

        with patch.object(provider, "_post_json", return_value=mock_data):
            messages = [Message.user("Search for python decorators")]
            response = await provider.generate(messages, config)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {"query": "python decorators"}

    async def test_ollama_max_tokens_stop_reason(
        self, provider: OllamaProvider, config: ModelConfig
    ):
        """Verify that done_reason='length' maps to stop_reason='max_tokens'."""
        mock_data = _mock_ollama_response(
            content="Truncated...",
            done_reason="length",
        )

        with patch.object(provider, "_post_json", return_value=mock_data):
            messages = [Message.user("Tell me a very long story")]
            response = await provider.generate(messages, config)

        assert response.stop_reason == "max_tokens"

    async def test_ollama_connection_error(
        self, provider: OllamaProvider, config: ModelConfig
    ):
        """Verify that a connection error is raised as LLMError.

        When the Ollama server is not running, _post_json raises LLMError
        which should propagate to the caller.
        """
        with patch.object(
            provider,
            "_post_json",
            side_effect=LLMError("Failed to connect to Ollama"),
        ):
            messages = [Message.user("Hello")]
            with pytest.raises(LLMError) as exc_info:
                await provider.generate(messages, config)
            assert "connect" in str(exc_info.value).lower()

    async def test_ollama_system_prompt(
        self, provider: OllamaProvider, config: ModelConfig
    ):
        """Verify that system_prompt is included in the converted messages."""
        mock_data = _mock_ollama_response(content="I am helpful.")

        captured_payload = {}

        def capture_post(path, payload):
            captured_payload.update(payload)
            return mock_data

        with patch.object(provider, "_post_json", side_effect=capture_post):
            messages = [Message.user("Hi")]
            await provider.generate(
                messages, config, system_prompt="You are helpful."
            )

        # The converted messages should start with the system prompt.
        ollama_messages = captured_payload.get("messages", [])
        assert len(ollama_messages) >= 2
        assert ollama_messages[0]["role"] == "system"
        assert ollama_messages[0]["content"] == "You are helpful."


# =====================================================================
# Model prefix and utility tests
# =====================================================================


class TestOllamaModelPrefix:
    """Tests for model prefix stripping and static utility methods."""

    def test_ollama_model_prefix(self):
        """Verify that 'ollama/llama3' is stripped to 'llama3'.

        Users may specify model_id as 'ollama/model' to select the Ollama
        provider; the actual API call should use just the model name.
        """
        assert OllamaProvider._strip_model_prefix("ollama/llama3") == "llama3"
        assert OllamaProvider._strip_model_prefix("ollama/mistral:7b") == "mistral:7b"

    def test_ollama_model_no_prefix(self):
        """Verify that model names without the prefix are passed through unchanged."""
        assert OllamaProvider._strip_model_prefix("llama3") == "llama3"
        assert OllamaProvider._strip_model_prefix("mistral") == "mistral"

    def test_ollama_estimate_cost(self, provider: OllamaProvider):
        """Verify that estimate_cost always returns 0.0 for Ollama.

        Ollama models run locally so there is no monetary cost.
        """
        cost = provider.estimate_cost(
            input_tokens=1000, output_tokens=500, model_id="llama3"
        )
        assert cost == 0.0

    def test_ollama_estimate_cost_any_model(self, provider: OllamaProvider):
        """Verify that estimate_cost returns 0.0 regardless of model."""
        assert provider.estimate_cost(0, 0, "mistral") == 0.0
        assert provider.estimate_cost(99999, 99999, "codellama:34b") == 0.0

    def test_ollama_supports_model(self, provider: OllamaProvider):
        """Verify that supports_model returns True for any model ID.

        Ollama can run any pulled model, so supports_model always returns True.
        """
        assert provider.supports_model("llama3") is True
        assert provider.supports_model("some-custom-model:latest") is True
        assert provider.supports_model("completely-made-up") is True

    def test_ollama_provider_name(self, provider: OllamaProvider):
        """Verify the provider_name property."""
        assert provider.provider_name == "ollama"

    def test_ollama_supported_models_list(self, provider: OllamaProvider):
        """Verify that supported_models returns a non-empty list of known models."""
        models = provider.supported_models
        assert isinstance(models, list)
        assert len(models) > 0
        assert "llama3" in models

    @pytest.fixture
    def provider(self) -> OllamaProvider:
        return OllamaProvider()


# =====================================================================
# is_ollama_available tests
# =====================================================================


class TestOllamaAvailability:
    """Tests for the is_ollama_available() module-level function."""

    def test_is_ollama_available_running(self):
        """Verify that is_ollama_available returns True when server responds 200.

        Mocks urllib.request.urlopen to simulate a running Ollama server.
        """
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = is_ollama_available()

        assert result is True

    def test_is_ollama_available_not_running(self):
        """Verify that is_ollama_available returns False on connection error.

        Mocks urllib.request.urlopen to raise an exception simulating a
        connection refused error when Ollama is not running.
        """
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = is_ollama_available()

        assert result is False

    def test_is_ollama_available_timeout(self):
        """Verify that is_ollama_available returns False on timeout."""
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("Connection timed out"),
        ):
            result = is_ollama_available()

        assert result is False

    def test_is_ollama_available_custom_url(self):
        """Verify that is_ollama_available uses the provided base_url."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
            result = is_ollama_available(base_url="http://remote-host:11434")

        assert result is True
        # Verify the URL used includes the custom host.
        call_args = mock_open.call_args
        request_obj = call_args[0][0]
        assert "remote-host" in request_obj.full_url


# =====================================================================
# Message conversion tests
# =====================================================================


class TestOllamaMessageConversion:
    """Tests for internal message conversion logic."""

    def test_convert_messages_basic(self):
        """Verify that openagentflow Messages are converted to Ollama format."""
        provider = OllamaProvider()
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello!"),
            Message.assistant("Hi there!"),
        ]

        converted = provider._convert_messages(messages)
        assert len(converted) == 3
        assert converted[0] == {"role": "system", "content": "You are helpful."}
        assert converted[1] == {"role": "user", "content": "Hello!"}
        assert converted[2] == {"role": "assistant", "content": "Hi there!"}

    def test_convert_messages_with_system_prompt(self):
        """Verify that an extra system_prompt is prepended."""
        provider = OllamaProvider()
        messages = [Message.user("Hi")]

        converted = provider._convert_messages(messages, system_prompt="Be concise.")
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "Be concise."

    def test_convert_messages_tool_result(self):
        """Verify that tool result messages are converted correctly."""
        provider = OllamaProvider()
        messages = [Message.tool_result("call_1", "Result: 42", name="calculator")]

        converted = provider._convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["content"] == "Result: 42"


# =====================================================================
# Token counting tests
# =====================================================================


class TestOllamaTokenCounting:
    """Tests for the count_tokens method."""

    def test_count_tokens_fallback(self):
        """Verify that count_tokens uses the heuristic fallback.

        When the /api/tokenize endpoint is unavailable, count_tokens
        falls back to a ~4 chars per token approximation.
        """
        provider = OllamaProvider()
        text = "Hello, this is a test string with some words."

        with patch.object(
            provider,
            "_post_json",
            side_effect=Exception("Not available"),
        ):
            count = provider.count_tokens(text, "llama3")

        # Heuristic: len(text) // 4
        assert count == len(text) // 4

    def test_count_tokens_api(self):
        """Verify that count_tokens uses the API when available."""
        provider = OllamaProvider()
        mock_response = {"tokens": [1, 2, 3, 4, 5]}

        with patch.object(provider, "_post_json", return_value=mock_response):
            count = provider.count_tokens("hello world", "llama3")

        assert count == 5

    def test_count_tokens_strips_prefix(self):
        """Verify that count_tokens strips the 'ollama/' model prefix."""
        provider = OllamaProvider()
        mock_response = {"tokens": [1, 2, 3]}

        captured_args = {}

        def capture_post(path, payload):
            captured_args.update(payload)
            return mock_response

        with patch.object(provider, "_post_json", side_effect=capture_post):
            provider.count_tokens("test", "ollama/llama3")

        assert captured_args.get("model") == "llama3"
