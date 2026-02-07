"""Ollama provider for Open Agent Flow.

Local-first AI using Ollama's HTTP API. No external dependencies required --
uses only Python's stdlib (urllib.request, json) for HTTP communication.

Supports llama3, mistral, codellama, deepseek-coder, qwen, and any other
model available through Ollama.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, AsyncIterator

from openagentflow.exceptions import LLMError
from openagentflow.llm.base import BaseLLMProvider, LLMResponse, StreamChunk

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolSpec

# Well-known Ollama models (non-exhaustive -- Ollama supports any pulled model)
SUPPORTED_MODELS = [
    "llama3",
    "llama3:8b",
    "llama3:70b",
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.3",
    "llama3.3:70b",
    "mistral",
    "mistral:7b",
    "mixtral",
    "mixtral:8x7b",
    "codellama",
    "codellama:7b",
    "codellama:13b",
    "codellama:34b",
    "deepseek-coder",
    "deepseek-coder:6.7b",
    "deepseek-coder:33b",
    "deepseek-coder-v2",
    "qwen",
    "qwen:7b",
    "qwen:14b",
    "qwen2",
    "qwen2:7b",
    "qwen2.5-coder",
    "phi3",
    "phi3:mini",
    "gemma",
    "gemma:7b",
    "gemma2",
    "command-r",
    "command-r-plus",
]

DEFAULT_BASE_URL = "http://localhost:11434"


def is_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check whether an Ollama server is reachable.

    Sends a GET request to ``/api/tags`` and returns ``True`` if the server
    responds with HTTP 200.

    Args:
        base_url: Ollama server URL (default ``http://localhost:11434``).

    Returns:
        ``True`` if Ollama is running and reachable, ``False`` otherwise.
    """
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider.

    Communicates with a running Ollama server over its HTTP API.  Uses only
    Python stdlib -- no third-party packages required.

    Example:
        from openagentflow.llm.providers import OllamaProvider
        from openagentflow.core.types import Message, ModelConfig

        provider = OllamaProvider()  # Connects to localhost:11434
        response = await provider.generate(
            messages=[Message(role="user", content="Hello!")],
            config=ModelConfig(model_id="llama3"),
        )
        print(response.content)

    Args:
        base_url: Ollama server URL (default ``http://localhost:11434``).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> Any:
        """Send a POST request with a JSON body and return the parsed response.

        When *stream* is ``False`` the full response body is read, parsed as
        JSON and returned.  When *stream* is ``True`` the raw
        ``http.client.HTTPResponse`` object is returned so the caller can
        iterate over lines.
        """
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama HTTP {exc.code}: {body}")
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Failed to connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Error: {exc}"
            )

        if stream:
            return resp

        body = resp.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise LLMError(f"Ollama returned invalid JSON: {body[:500]}")

    def _convert_messages(
        self, messages: list[Message], system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert openagentflow Messages to Ollama chat format.

        Ollama uses the same message format as OpenAI:
        ``{"role": "...", "content": "..."}``.
        """
        result: list[dict[str, Any]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "tool":
                # Tool result message
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": msg.content,
                }
                if msg.tool_call_id:
                    tool_msg["tool_call_id"] = msg.tool_call_id
                result.append(tool_msg)
            elif msg.tool_calls:
                # Assistant message that contains tool calls
                ollama_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": ollama_tool_calls,
                }
                result.append(assistant_msg)
            else:
                result.append({"role": msg.role, "content": msg.content or ""})

        return result

    def _convert_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert openagentflow ToolSpecs to Ollama/OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _strip_model_prefix(model_id: str) -> str:
        """Remove the ``ollama/`` prefix if present.

        Users may specify ``model="ollama/llama3"`` to explicitly select the
        Ollama provider.  The actual Ollama API expects just ``"llama3"``.
        """
        if model_id.startswith("ollama/"):
            return model_id[len("ollama/"):]
        return model_id

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from a local Ollama model.

        Sends a non-streaming POST to ``/api/chat``.
        """
        import asyncio

        model_id = self._strip_model_prefix(config.model_id)

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "stream": False,
        }

        # Options
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.max_tokens:
            options["num_predict"] = config.max_tokens
        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = self._convert_tools(tools)

        # Run the synchronous HTTP call in a thread so we don't block the
        # event loop.
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None, lambda: self._post_json("/api/chat", payload)
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama request failed: {exc}")

        # Parse response
        from openagentflow.core.types import ToolCall

        if not isinstance(data, dict):
            raise LLMError(f"Ollama returned unexpected response type: {type(data).__name__}")

        message = data.get("message", {})
        content_text = message.get("content", "")
        tool_calls: list[ToolCall] = []

        # Ollama returns tool_calls in OpenAI-compatible format
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func_data = tc.get("function", {})
            arguments = func_data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    tool_name=func_data.get("name", ""),
                    arguments=arguments,
                )
            )

        # Determine stop reason
        done_reason = data.get("done_reason", "stop")
        if tool_calls:
            stop_reason = "tool_calls"
        elif done_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "stop"

        # Token usage -- Ollama reports these in the top-level response
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=data.get("model", model_id),
            raw_response=data,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from a local Ollama model.

        Sends a streaming POST to ``/api/chat`` (``stream: true``).  Each line
        of the response is a JSON object with an incremental delta.
        """
        import asyncio

        model_id = self._strip_model_prefix(config.model_id)

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._convert_messages(messages, system_prompt),
            "stream": True,
        }

        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.max_tokens:
            options["num_predict"] = config.max_tokens
        if options:
            payload["options"] = options

        if tools:
            payload["tools"] = self._convert_tools(tools)

        # Open the streaming connection in a thread executor
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None, lambda: self._post_json("/api/chat", payload, stream=True)
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama streaming request failed: {exc}")

        try:
            # Track tool call state across chunks
            seen_tool_call_ids: set[str] = set()

            while True:
                line = await loop.run_in_executor(None, resp.readline)
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    chunk_data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                message = chunk_data.get("message", {})

                # Text content delta
                text_delta = message.get("content", "")
                if text_delta:
                    yield StreamChunk(text=text_delta)

                # Tool calls in streaming mode
                raw_tool_calls = message.get("tool_calls", [])
                for tc in raw_tool_calls:
                    tc_id = tc.get("id", "")
                    func_data = tc.get("function", {})
                    func_name = func_data.get("name", "")
                    func_args = func_data.get("arguments", {})

                    if isinstance(func_args, dict):
                        func_args_str = json.dumps(func_args)
                    elif isinstance(func_args, str):
                        func_args_str = func_args
                    else:
                        func_args_str = str(func_args)

                    if tc_id not in seen_tool_call_ids:
                        seen_tool_call_ids.add(tc_id)
                        yield StreamChunk(
                            is_tool_call_start=True,
                            tool_call_id=tc_id,
                            tool_name=func_name,
                        )

                    yield StreamChunk(
                        tool_call_id=tc_id,
                        tool_name=func_name,
                        tool_input_delta=func_args_str,
                    )

                # Check if done
                if chunk_data.get("done", False):
                    yield StreamChunk(is_final=True)
                    break
        finally:
            resp.close()

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens for the given text.

        Attempts to use Ollama's ``/api/tokenize`` endpoint first.  Falls back
        to a rough heuristic (~4 characters per token for English text) if the
        endpoint is unavailable or the request fails.

        Args:
            text: The text to tokenize.
            model_id: The model to use for tokenization.

        Returns:
            Number of tokens.
        """
        model_id = self._strip_model_prefix(model_id)
        try:
            data = self._post_json(
                "/api/tokenize",
                {"model": model_id, "text": text},
            )
            tokens = data.get("tokens", [])
            if tokens:
                return len(tokens)
        except Exception:
            pass

        # Fallback: rough approximation (~4 chars per token for English)
        return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Estimate cost in USD.

        Ollama models run locally, so the cost is always 0.0.
        """
        return 0.0

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def supported_models(self) -> list[str]:
        return SUPPORTED_MODELS

    def supports_model(self, model_id: str) -> bool:
        """Check if this provider supports a model.

        Overridden because Ollama can run *any* pulled model -- the
        ``SUPPORTED_MODELS`` list is only a well-known subset.  We accept
        any model ID and let Ollama itself report an error if the model
        hasn't been pulled.
        """
        return True
