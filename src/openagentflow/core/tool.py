"""
@tool decorator for defining agent-callable tools.

Follows the TwinGraph 1.0 pattern of:
- Pure Python decorators (no DSL)
- Automatic schema generation from type hints
- Source code capture for reproducibility
- Works with or without decoration
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, get_type_hints, overload

from openagentflow.core.types import ToolCall, ToolResult, ToolSpec
from openagentflow.exceptions import ToolExecutionError, ToolTimeoutError, ToolValidationError

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# Tool registry (global)
_tool_registry: dict[str, ToolSpec] = {}


def get_tool(name: str) -> ToolSpec | None:
    """Get a registered tool by name."""
    return _tool_registry.get(name)


def get_all_tools() -> dict[str, ToolSpec]:
    """Get all registered tools."""
    return _tool_registry.copy()


def _generate_json_schema_from_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from function type hints."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, Any)
        param_schema = _type_to_json_schema(param_type)

        properties[param_name] = param_schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_json_schema(type_hint: Any) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    origin = getattr(type_hint, "__origin__", None)

    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is type(None):
        return {"type": "null"}
    elif origin is list:
        args = getattr(type_hint, "__args__", (Any,))
        return {"type": "array", "items": _type_to_json_schema(args[0])}
    elif origin is dict:
        args = getattr(type_hint, "__args__", (str, Any))
        return {
            "type": "object",
            "additionalProperties": _type_to_json_schema(args[1]) if len(args) > 1 else {},
        }
    elif origin is type(None) or type_hint is type(None):
        return {"type": "null"}
    else:
        # Default to any
        return {}


@overload
def tool(func: F) -> F: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30.0,
    retry: int = 3,
    requires_confirmation: bool = False,
) -> Callable[[F], F]: ...


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30.0,
    retry: int = 3,
    requires_confirmation: bool = False,
) -> F | Callable[[F], F]:
    """
    Decorator to define an agent-callable tool.

    Can be used with or without arguments:

        @tool
        def search(query: str) -> list[dict]:
            '''Search the web.'''
            ...

        @tool(timeout=60, retry=5)
        def slow_search(query: str) -> list[dict]:
            '''Slow search with custom timeout.'''
            ...

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        timeout: Execution timeout in seconds
        retry: Maximum retry attempts
        requires_confirmation: Whether human confirmation is needed
    """

    def decorator(fn: F) -> F:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or f"Tool: {tool_name}"

        # Capture source code (TwinGraph pattern)
        try:
            source_code = inspect.getsource(fn)
            source_file = inspect.getfile(fn)
        except (OSError, TypeError):
            source_code = None
            source_file = None

        # Generate schema from type hints
        input_schema = _generate_json_schema_from_hints(fn)

        # Get return type schema
        hints = get_type_hints(fn)
        return_type = hints.get("return", Any)
        output_schema = _type_to_json_schema(return_type)

        # Create tool spec
        spec = ToolSpec(
            name=tool_name,
            description=tool_description.strip(),
            func=fn,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout_seconds=timeout,
            max_retries=retry,
            requires_confirmation=requires_confirmation,
            source_code=source_code,
            source_file=source_file,
        )

        # Register tool
        _tool_registry[tool_name] = spec

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ToolResult:
            """Async wrapper that returns ToolResult."""
            call_id = f"{tool_name}_{datetime.now(timezone.utc).timestamp()}"
            start_time = datetime.now(timezone.utc)

            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(fn):
                    result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: fn(*args, **kwargs)),
                        timeout=timeout,
                    )

                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                return ToolResult.from_call_id(
                    call_id=call_id,
                    success=True,
                    output=result,
                    duration_ms=duration,
                )

            except asyncio.TimeoutError:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return ToolResult.from_call_id(
                    call_id=call_id,
                    success=False,
                    error=f"Tool '{tool_name}' timed out after {timeout}s",
                    duration_ms=duration,
                )

            except Exception as e:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return ToolResult.from_call_id(
                    call_id=call_id,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                )

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper that returns raw result (for direct use)."""
            return fn(*args, **kwargs)

        # Attach spec to wrapper
        sync_wrapper._tool_spec = spec  # type: ignore
        sync_wrapper._async_call = async_wrapper  # type: ignore

        return sync_wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


async def execute_tool(
    tool_call: ToolCall,
    available_tools: list[ToolSpec] | None = None,
) -> ToolResult:
    """
    Execute a tool call.

    Args:
        tool_call: The tool call to execute
        available_tools: List of available tools (uses registry if None)
    """
    start_time = datetime.now(timezone.utc)

    # Find the tool
    if available_tools:
        tool_map = {t.name: t for t in available_tools}
        spec = tool_map.get(tool_call.tool_name)
    else:
        spec = get_tool(tool_call.tool_name)

    if spec is None:
        return ToolResult(
            tool_call=tool_call,
            success=False,
            error=f"Tool not found: {tool_call.tool_name}",
            duration_ms=0,
        )

    # Execute with retries
    last_error: Exception | None = None
    for attempt in range(spec.max_retries):
        try:
            if asyncio.iscoroutinefunction(spec.func):
                result = await asyncio.wait_for(
                    spec.func(**tool_call.arguments),
                    timeout=spec.timeout_seconds,
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: spec.func(**tool_call.arguments)),
                    timeout=spec.timeout_seconds,
                )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return ToolResult(
                tool_call=tool_call,
                success=True,
                output=result,
                duration_ms=duration,
                metadata={"attempts": attempt + 1},
            )

        except asyncio.TimeoutError:
            last_error = ToolTimeoutError(
                f"Tool '{tool_call.tool_name}' timed out",
                tool_name=tool_call.tool_name,
            )
        except Exception as e:
            last_error = e

    # All retries failed
    duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    return ToolResult(
        tool_call=tool_call,
        success=False,
        error=str(last_error) if last_error else "Unknown error",
        duration_ms=duration,
        metadata={"attempts": spec.max_retries},
    )


def tools_to_anthropic_format(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert tools to Anthropic API format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def tools_to_openai_format(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert tools to OpenAI API format."""
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
