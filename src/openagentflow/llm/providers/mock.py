"""Mock LLM provider for testing and development.

This provider simulates LLM responses without requiring API keys.
It can be used to:
- Test agent workflows
- Develop and debug tool integrations
- Run demos without API costs
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, AsyncIterator

from openagentflow.core.types import ToolCall
from openagentflow.llm.base import BaseLLMProvider, LLMResponse, StreamChunk

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolSpec


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing and development.

    This provider simulates intelligent responses by:
    - Detecting tool calls from user queries
    - Generating appropriate tool invocations
    - Synthesizing responses from tool results

    Example:
        from openagentflow.llm.providers import MockProvider

        provider = MockProvider()
        response = await provider.generate(
            messages=[Message(role="user", content="What's the weather in Tokyo?")],
            config=ModelConfig(provider=LLMProvider.ANTHROPIC, model_id="mock"),
            tools=[weather_tool_spec],
        )
        # Response will include a tool call for the weather tool
    """

    def __init__(self, verbose: bool = False):
        """Initialize the mock provider.

        Args:
            verbose: If True, print debug information about mock responses.
        """
        self.verbose = verbose
        self._call_count = 0

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a mock response."""
        self._call_count += 1

        # Get the last user message
        last_user_msg = None
        last_tool_result = None

        for msg in reversed(messages):
            if msg.role == "user":
                if msg.tool_call_id:
                    # This is a tool result
                    last_tool_result = msg.content
                else:
                    last_user_msg = msg.content
                    break

        if self.verbose:
            print(f"[MockProvider] Call #{self._call_count}")
            print(f"[MockProvider] User message: {last_user_msg}")
            print(f"[MockProvider] Tool result: {last_tool_result}")
            print(f"[MockProvider] Available tools: {[t.name for t in (tools or [])]}")

        # If we have a tool result, generate a final response
        if last_tool_result:
            response_text = self._synthesize_response(last_user_msg or "", last_tool_result)
            return LLMResponse(
                content=response_text,
                tool_calls=[],
                stop_reason="end_turn",
                input_tokens=len(str(messages)) // 4,
                output_tokens=len(response_text) // 4,
                model_id="mock-provider",
            )

        # Check if we should call a tool
        if tools and last_user_msg:
            tool_call = self._detect_tool_call(last_user_msg, tools)
            if tool_call:
                thinking = f"I'll use the {tool_call.tool_name} tool to help answer this."
                return LLMResponse(
                    content=thinking,
                    tool_calls=[tool_call],
                    stop_reason="tool_use",
                    input_tokens=len(str(messages)) // 4,
                    output_tokens=len(thinking) // 4 + 50,
                    model_id="mock-provider",
                )

        # Default response without tools
        response_text = self._generate_default_response(last_user_msg or "")
        return LLMResponse(
            content=response_text,
            tool_calls=[],
            stop_reason="end_turn",
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(response_text) // 4,
            model_id="mock-provider",
        )

    def _detect_tool_call(
        self, user_msg: str, tools: list[ToolSpec]
    ) -> ToolCall | None:
        """Detect which tool to call based on user message."""
        user_lower = user_msg.lower()

        for tool in tools:
            # Check if tool name or description keywords match
            tool_keywords = tool.name.lower().replace("_", " ").split()
            desc_keywords = tool.description.lower().split()[:10]

            for keyword in tool_keywords + desc_keywords:
                if len(keyword) > 3 and keyword in user_lower:
                    # Extract arguments from the message
                    args = self._extract_tool_args(user_msg, tool)
                    return ToolCall(
                        id=f"call_{self._call_count}",
                        tool_name=tool.name,
                        arguments=args,
                    )

        return None

    def _extract_tool_args(self, user_msg: str, tool: ToolSpec) -> dict[str, Any]:
        """Extract tool arguments from user message."""
        args = {}
        schema = tool.input_schema

        if "properties" not in schema:
            return args

        for prop_name, prop_def in schema["properties"].items():
            prop_type = prop_def.get("type", "string")

            if prop_type == "string":
                # Check if this looks like a city/location parameter
                if prop_name in ("city", "location", "place"):
                    # Look for city names - common cities and capitalized words
                    cities = ["Tokyo", "London", "Paris", "New York", "Berlin", "Sydney"]
                    for city in cities:
                        if city.lower() in user_msg.lower():
                            args[prop_name] = city
                            break
                    if prop_name not in args:
                        # Look for capitalized words
                        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', user_msg)
                        if caps:
                            args[prop_name] = caps[0]

                # Check if this is an expression/query parameter
                elif prop_name in ("expression", "formula", "query", "search"):
                    # For expressions, extract math-like content
                    if prop_name == "expression":
                        # Extract everything after "calculate" or numbers with operators
                        math_match = re.search(r'[\d\s+\-*/().]+', user_msg)
                        if math_match:
                            expr = math_match.group().strip()
                            if any(c.isdigit() for c in expr):
                                args[prop_name] = expr
                    if prop_name not in args:
                        # Use the message content without common words
                        clean = re.sub(r'\b(what|is|the|calculate|search|find|for|in)\b', '', user_msg, flags=re.I)
                        args[prop_name] = clean.strip() or user_msg

                else:
                    # Default: use quoted strings or the whole message
                    quoted = re.findall(r'"([^"]+)"', user_msg)
                    if quoted:
                        args[prop_name] = quoted[0]
                    else:
                        args[prop_name] = user_msg

            elif prop_type in ("number", "integer"):
                # Extract numbers
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_msg)
                if numbers:
                    if prop_type == "integer":
                        args[prop_name] = int(float(numbers[0]))
                    else:
                        args[prop_name] = float(numbers[0])

        return args

    def _synthesize_response(self, user_msg: str, tool_result: str) -> str:
        """Synthesize a response from tool result."""
        # Try to parse as JSON
        try:
            data = json.loads(tool_result)
            if isinstance(data, dict):
                # Format dict nicely
                parts = []
                for key, value in data.items():
                    parts.append(f"{key}: {value}")
                formatted = ", ".join(parts)
                return f"Based on the information I found: {formatted}"
            elif isinstance(data, list):
                return f"I found {len(data)} results: {tool_result}"
        except json.JSONDecodeError:
            pass

        return f"Based on my research: {tool_result}"

    def _generate_default_response(self, user_msg: str) -> str:
        """Generate a default response without tools."""
        if not user_msg:
            return "Hello! How can I help you today?"

        if "?" in user_msg:
            return f"That's an interesting question about '{user_msg[:50]}...'. Let me help you with that."

        return f"I understand you're asking about '{user_msg[:50]}...'. I'm here to assist."

    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a mock response."""
        # Generate full response first
        response = await self.generate(messages, config, tools, system_prompt)

        # Stream it word by word
        words = response.content.split()
        for i, word in enumerate(words):
            yield StreamChunk(text=word + " ")

        # If there are tool calls, yield them
        for tc in response.tool_calls:
            yield StreamChunk(
                is_tool_call_start=True,
                tool_call_id=tc.id,
                tool_name=tc.tool_name,
            )
            yield StreamChunk(
                tool_call_id=tc.id,
                tool_input_delta=json.dumps(tc.arguments),
            )

        yield StreamChunk(is_final=True)

    def count_tokens(self, text: str, model_id: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Mock provider is free."""
        return 0.0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def supported_models(self) -> list[str]:
        return ["mock", "mock-provider", "test", "local"]
