"""Claude Code CLI provider for Open Agent Flow.

Uses the installed Claude Code CLI to make LLM calls.
No API key needed - uses Claude Code's built-in authentication.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, AsyncIterator

from openagentflow.exceptions import LLMError
from openagentflow.llm.base import BaseLLMProvider, LLMResponse, StreamChunk

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolSpec


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is installed and available."""
    return shutil.which("claude") is not None


def get_claude_code_path() -> str | None:
    """Get the path to Claude Code CLI."""
    return shutil.which("claude")


class ClaudeCodeProvider(BaseLLMProvider):
    """Claude Code CLI provider.

    Uses the installed Claude Code CLI to make LLM calls.
    No API key required - uses Claude Code's built-in authentication.

    Example:
        from openagentflow.llm.providers import ClaudeCodeProvider

        if ClaudeCodeProvider.is_available():
            provider = ClaudeCodeProvider()
            response = await provider.generate(
                messages=[Message(role="user", content="Hello!")],
                config=ModelConfig(model_id="claude-sonnet-4-20250514"),
            )
            print(response.content)
    """

    def __init__(self, timeout: float = 120.0):
        """Initialize the Claude Code provider.

        Args:
            timeout: Timeout for CLI calls in seconds.
        """
        self._timeout = timeout
        self._claude_path = get_claude_code_path()

    @classmethod
    def is_available(cls) -> bool:
        """Check if Claude Code CLI is available."""
        return is_claude_code_available()

    def _build_prompt(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> str:
        """Build a prompt string from messages."""
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}\n")
            elif msg.role == "tool":
                parts.append(f"Tool Result ({msg.name}): {msg.content}\n")

        # Add tool descriptions if provided
        if tools:
            tool_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tool_desc += f"- {tool.name}: {tool.description}\n"
                tool_desc += f"  Parameters: {json.dumps(tool.input_schema)}\n"
            tool_desc += "\nTo use a tool, respond with: TOOL_CALL: tool_name(args_json)\n"
            parts.insert(0, tool_desc)

        return "".join(parts)

    def _parse_tool_calls(self, content: str) -> tuple[str, list]:
        """Parse tool calls from response content."""
        from openagentflow.core.types import ToolCall

        tool_calls = []
        clean_content = content

        # Look for TOOL_CALL: pattern
        import re

        pattern = r"TOOL_CALL:\s*(\w+)\(({.*?})\)"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, (tool_name, args_json) in enumerate(matches):
            try:
                args = json.loads(args_json)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        tool_name=tool_name,
                        arguments=args,
                    )
                )
                # Remove tool call from content
                clean_content = re.sub(
                    rf"TOOL_CALL:\s*{tool_name}\({re.escape(args_json)}\)",
                    "",
                    clean_content,
                )
            except json.JSONDecodeError:
                pass

        return clean_content.strip(), tool_calls

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using Claude Code CLI."""
        if not self._claude_path:
            raise LLMError(
                "Claude Code CLI not found. Install it from: "
                "https://docs.anthropic.com/en/docs/claude-code"
            )

        # Build the prompt
        prompt = self._build_prompt(messages, system_prompt, tools)

        # Run Claude Code CLI
        try:
            # Use --print flag for simple output, -p for prompt
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
                "-p", prompt,
                "--output-format", "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise LLMError(f"Claude Code CLI error: {error_msg}")

            content = stdout.decode().strip()

        except asyncio.TimeoutError:
            raise LLMError(f"Claude Code CLI timed out after {self._timeout}s")
        except FileNotFoundError:
            raise LLMError("Claude Code CLI not found")

        # Parse response for tool calls
        clean_content, tool_calls = self._parse_tool_calls(content)

        # Estimate tokens (rough approximation)
        input_tokens = len(prompt) // 4
        output_tokens = len(content) // 4

        return LLMResponse(
            content=clean_content,
            tool_calls=tool_calls,
            stop_reason="end_turn",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id="claude-code",
            raw_response={"stdout": content},
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response using Claude Code CLI."""
        if not self._claude_path:
            raise LLMError("Claude Code CLI not found")

        prompt = self._build_prompt(messages, system_prompt, tools)

        try:
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
                "-p", prompt,
                "--output-format", "stream-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def read_stream():
                buffer = ""
                while True:
                    chunk = await process.stdout.read(100)
                    if not chunk:
                        break
                    text = chunk.decode()
                    buffer += text

                    # Try to parse JSON lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "content" in data:
                                    yield StreamChunk(text=data["content"])
                            except json.JSONDecodeError:
                                # Plain text output
                                yield StreamChunk(text=line)

                # Yield remaining buffer
                if buffer.strip():
                    yield StreamChunk(text=buffer)

                yield StreamChunk(is_final=True)

            async for chunk in read_stream():
                yield chunk

            await process.wait()

        except Exception as e:
            raise LLMError(f"Claude Code CLI streaming error: {e}")

    def count_tokens(self, text: str, model_id: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Claude Code CLI usage is included in subscription - no per-token cost."""
        return 0.0

    @property
    def provider_name(self) -> str:
        return "claude_code"

    @property
    def supported_models(self) -> list[str]:
        return ["claude-code", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]
