"""
ToolFactory - Creates new tools at runtime from source code.

The ToolFactory provides the ability to dynamically create, register, test,
and remove tools at runtime. It works in two modes:

1. Direct creation: Given Python source code, validate it via sandbox,
   wrap it as a ToolSpec, and register it in the global tool registry.

2. LLM-assisted creation: Given a natural language description, ask an
   LLM provider to generate the implementation, then validate and register.

All dynamically created tools are executed within the Sandbox to ensure
safety. The factory maintains its own registry of dynamic tools alongside
the global tool registry.
"""

from __future__ import annotations

from typing import Any

from openagentflow.core.tool import _tool_registry
from openagentflow.core.types import ToolSpec
from openagentflow.meta.sandbox import (
    Sandbox,
    SandboxExecutionError,
    SandboxValidationError,
)


class ToolFactoryError(Exception):
    """Raised when tool creation or management fails."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name


class ToolFactory:
    """
    Creates new tools at runtime from source code or natural language descriptions.

    The ToolFactory validates all code through the Sandbox before execution
    and registers created tools in both a local dynamic registry and the
    global OpenAgentFlow tool registry.

    Usage:
        factory = ToolFactory()

        # Create a tool from source code
        spec = factory.create_tool(
            name="add_numbers",
            description="Add two numbers together",
            source_code="def add_numbers(a: int, b: int) -> int:\\n    return a + b",
        )

        # Test the tool
        result = factory.test_tool("add_numbers", {"a": 3, "b": 4})
        assert result == 7

        # List all dynamic tools
        tools = factory.list_dynamic_tools()

        # Remove a tool
        factory.remove_tool("add_numbers")
    """

    # Type hint string to JSON Schema mapping for auto-schema generation
    _TYPE_HINT_TO_SCHEMA: dict[str, dict[str, Any]] = {
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "str": {"type": "string"},
        "bool": {"type": "boolean"},
        "None": {"type": "null"},
        "list": {"type": "array"},
        "dict": {"type": "object"},
        "Any": {},
    }

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        """
        Initialize the ToolFactory.

        Args:
            sandbox: A Sandbox instance for code validation and execution.
                If not provided, a new Sandbox with default settings is created.
        """
        self.sandbox = sandbox or Sandbox()
        self._dynamic_tools: dict[str, ToolSpec] = {}
        self._source_registry: dict[str, str] = {}

    def _type_hint_to_json_schema(self, type_str: str) -> dict[str, Any]:
        """
        Convert a type hint string to a JSON Schema fragment.

        Handles basic types, list[T], dict[K, V], and Optional patterns.

        Args:
            type_str: A Python type hint as a string (e.g., "int", "list[str]").

        Returns:
            JSON Schema dictionary.
        """
        type_str = type_str.strip()

        # Direct lookup
        if type_str in self._TYPE_HINT_TO_SCHEMA:
            return self._TYPE_HINT_TO_SCHEMA[type_str].copy()

        # Handle list[T]
        if type_str.startswith("list[") and type_str.endswith("]"):
            inner = type_str[5:-1]
            return {
                "type": "array",
                "items": self._type_hint_to_json_schema(inner),
            }

        # Handle dict[K, V]
        if type_str.startswith("dict[") and type_str.endswith("]"):
            inner = type_str[5:-1]
            # Simple split on first comma (doesn't handle nested generics)
            parts = inner.split(",", 1)
            if len(parts) == 2:
                return {
                    "type": "object",
                    "additionalProperties": self._type_hint_to_json_schema(
                        parts[1].strip()
                    ),
                }
            return {"type": "object"}

        # Handle Optional[T] or T | None
        if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner = type_str[9:-1]
            inner_schema = self._type_hint_to_json_schema(inner)
            return {**inner_schema, "nullable": True}

        if " | None" in type_str:
            inner = type_str.replace(" | None", "").strip()
            inner_schema = self._type_hint_to_json_schema(inner)
            return {**inner_schema, "nullable": True}

        if type_str.startswith("None | "):
            inner = type_str[7:].strip()
            inner_schema = self._type_hint_to_json_schema(inner)
            return {**inner_schema, "nullable": True}

        # Handle Union[T1, T2, ...]
        if type_str.startswith("Union[") and type_str.endswith("]"):
            inner = type_str[6:-1]
            parts = [p.strip() for p in inner.split(",")]
            non_none = [p for p in parts if p != "None"]
            has_none = len(non_none) < len(parts)
            if len(non_none) == 1:
                schema = self._type_hint_to_json_schema(non_none[0])
                if has_none:
                    return {**schema, "nullable": True}
                return schema
            # Multiple non-None types: use anyOf
            schemas = [self._type_hint_to_json_schema(p) for p in non_none]
            result: dict[str, Any] = {"anyOf": schemas}
            if has_none:
                result["nullable"] = True
            return result

        # Handle T1 | T2 (non-None union with pipe syntax)
        if " | " in type_str:
            parts = [p.strip() for p in type_str.split(" | ")]
            non_none = [p for p in parts if p != "None"]
            has_none = len(non_none) < len(parts)
            if len(non_none) == 1:
                schema = self._type_hint_to_json_schema(non_none[0])
                if has_none:
                    return {**schema, "nullable": True}
                return schema
            schemas = [self._type_hint_to_json_schema(p) for p in non_none]
            result_schema: dict[str, Any] = {"anyOf": schemas}
            if has_none:
                result_schema["nullable"] = True
            return result_schema

        # Handle tuple
        if type_str.startswith("tuple[") and type_str.endswith("]"):
            return {"type": "array"}

        # Handle set
        if type_str.startswith("set[") and type_str.endswith("]"):
            inner = type_str[4:-1]
            return {
                "type": "array",
                "items": self._type_hint_to_json_schema(inner),
                "uniqueItems": True,
            }

        # Default: unknown type
        return {}

    def _generate_input_schema(
        self, param_types: dict[str, str]
    ) -> dict[str, Any]:
        """
        Generate a JSON Schema from a dictionary of parameter type hints.

        Args:
            param_types: Dict mapping parameter names to type hint strings.

        Returns:
            JSON Schema with "type": "object", "properties", and "required".
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, type_str in param_types.items():
            properties[param_name] = self._type_hint_to_json_schema(type_str)
            # Assume all parameters are required unless type is Optional or nullable
            if "Optional" not in type_str and "| None" not in type_str and not type_str.startswith("None |"):
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _generate_output_schema(
        self, return_type: str | None
    ) -> dict[str, Any] | None:
        """
        Generate a JSON Schema for the return type.

        Args:
            return_type: Return type hint string, or None.

        Returns:
            JSON Schema dictionary, or None if no return type specified.
        """
        if return_type is None or return_type == "None":
            return None
        return self._type_hint_to_json_schema(return_type)

    def create_tool(
        self,
        name: str,
        description: str,
        source_code: str,
        input_schema: dict[str, Any] | None = None,
        output_type: str = "Any",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> ToolSpec:
        """
        Create and register a tool from Python source code.

        Steps:
        1. Validate the code via the sandbox
        2. Parse the function from source using AST to extract metadata
        3. Generate input_schema from type hints if not provided
        4. Create a wrapper function that executes via the sandbox
        5. Register in both the local dynamic registry and global tool registry
        6. Return the ToolSpec

        Args:
            name: Unique name for the tool.
            description: Human-readable description of what the tool does.
            source_code: Python source code defining a function. The function
                name in the source should match the ``name`` argument.
            input_schema: Optional JSON Schema for the input. If not provided,
                it will be auto-generated from the function's type hints.
            output_type: String representation of the return type (default: "Any").
            timeout: Execution timeout in seconds (default: 30.0).
            max_retries: Maximum retry attempts on failure (default: 3).

        Returns:
            The created ToolSpec.

        Raises:
            ToolFactoryError: If validation fails, the function is not found,
                or a tool with the same name already exists as a dynamic tool.
        """
        # Check for duplicate names in dynamic registry
        if name in self._dynamic_tools:
            raise ToolFactoryError(
                f"A dynamic tool named '{name}' already exists. "
                f"Remove it first with remove_tool('{name}') before recreating.",
                tool_name=name,
            )

        # Step 1: Validate the code
        is_safe, reason = self.sandbox.validate_source(source_code)
        if not is_safe:
            raise ToolFactoryError(
                f"Source code for tool '{name}' failed validation: {reason}",
                tool_name=name,
            )

        # Step 2: Extract function metadata from AST
        try:
            func_name, param_types, return_type_str = (
                self.sandbox.validate_and_extract_function(source_code)
            )
        except SandboxValidationError as e:
            raise ToolFactoryError(
                f"Failed to parse function from source for tool '{name}': {e}",
                tool_name=name,
            )

        # Step 3: Generate input schema if not provided
        if input_schema is None:
            input_schema = self._generate_input_schema(param_types)

        # Generate output schema
        effective_return_type = return_type_str or output_type
        output_schema = self._generate_output_schema(effective_return_type)

        # Step 4: Create sandbox-wrapped function
        sandbox = self.sandbox
        source = source_code
        target_func_name = func_name

        def sandboxed_func(**kwargs: Any) -> Any:
            """Dynamically created tool executed in sandbox."""
            return sandbox.execute(
                source=source,
                func_name=target_func_name,
                args=kwargs,
                timeout=timeout,
            )

        # Attach metadata for introspection
        sandboxed_func.__name__ = name
        sandboxed_func.__doc__ = description
        sandboxed_func.__qualname__ = f"ToolFactory.{name}"

        # Step 5: Create ToolSpec
        spec = ToolSpec(
            name=name,
            description=description,
            func=sandboxed_func,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout_seconds=timeout,
            max_retries=max_retries,
            requires_confirmation=False,
            source_code=source_code,
            source_file="<dynamic:ToolFactory>",
        )

        # Register in both registries
        self._dynamic_tools[name] = spec
        self._source_registry[name] = source_code
        _tool_registry[name] = spec

        return spec

    async def create_tool_from_prompt(
        self,
        prompt: str,
        llm_provider: Any,
        tool_name: str | None = None,
    ) -> ToolSpec:
        """
        Ask an LLM to generate a tool implementation from natural language.

        Sends a structured prompt to the LLM asking for a Python function
        implementation, parses the generated code, validates it via the
        sandbox, and registers the tool.

        Args:
            prompt: Natural language description of the desired tool.
            llm_provider: An LLM provider instance with a ``generate()`` method
                that accepts a list of Message-like dicts and returns a response
                with a ``.content`` attribute containing the text.
            tool_name: Optional name for the tool. If not provided, the LLM
                will be asked to choose an appropriate name.

        Returns:
            The created ToolSpec.

        Raises:
            ToolFactoryError: If the LLM fails to generate valid code, or
                the generated code fails validation.
        """
        # Build the structured prompt for the LLM
        system_message = {
            "role": "system",
            "content": (
                "You are a Python tool generator. When asked to create a tool, "
                "respond with ONLY a Python function definition. Rules:\n"
                "1. Define exactly ONE function\n"
                "2. Include full type hints on all parameters and return value\n"
                "3. Include a docstring describing what the function does\n"
                "4. Only use Python stdlib modules from this list: "
                "math, re, json, collections, itertools, functools, string, "
                "textwrap, unicodedata, datetime, decimal, fractions, random, "
                "statistics, operator, copy, dataclasses, enum, typing, abc, "
                "bisect, heapq\n"
                "5. Keep the function pure - no side effects, no I/O, no network\n"
                "6. Handle edge cases gracefully\n"
                "7. Do NOT include any explanation, markdown, or code fences\n"
                "8. Do NOT use exec, eval, open, __import__, or any other "
                "dangerous functions\n"
                "9. Respond with ONLY the Python code, nothing else\n"
            ),
        }

        name_instruction = ""
        if tool_name:
            name_instruction = f" Name the function '{tool_name}'."

        user_message = {
            "role": "user",
            "content": (
                f"Create a Python function that does the following: {prompt}"
                f"{name_instruction}"
            ),
        }

        # Call the LLM
        try:
            response = await llm_provider.generate(
                messages=[system_message, user_message]
            )
        except Exception as e:
            raise ToolFactoryError(
                f"LLM failed to generate tool code: {e}",
                tool_name=tool_name,
            )

        # Extract the generated code
        generated_code = response.content if hasattr(response, "content") else str(response)

        # Clean up: remove markdown code fences if present
        if "```python" in generated_code:
            generated_code = generated_code.split("```python", 1)[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```", 1)[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```", 1)[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```", 1)[0]

        generated_code = generated_code.strip()

        if not generated_code:
            raise ToolFactoryError(
                "LLM returned empty code",
                tool_name=tool_name,
            )

        # Extract function name and description from the generated code
        try:
            func_name, param_types, return_type = (
                self.sandbox.validate_and_extract_function(generated_code)
            )
        except SandboxValidationError as e:
            raise ToolFactoryError(
                f"LLM-generated code failed validation: {e}",
                tool_name=tool_name,
            )

        # Use the function name from the code if no name was provided
        effective_name = tool_name or func_name

        # Extract docstring for description
        import ast

        try:
            tree = ast.parse(generated_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node)
                    break
            else:
                docstring = None
        except SyntaxError:
            docstring = None

        description = docstring or f"Dynamically created tool: {prompt}"

        # Create the tool via the standard path
        return self.create_tool(
            name=effective_name,
            description=description,
            source_code=generated_code,
        )

    def list_dynamic_tools(self) -> list[ToolSpec]:
        """
        List all dynamically created tools.

        Returns:
            A list of ToolSpec objects for all tools created by this factory.
        """
        return list(self._dynamic_tools.values())

    def get_dynamic_tool(self, name: str) -> ToolSpec | None:
        """
        Get a specific dynamic tool by name.

        Args:
            name: The tool name.

        Returns:
            The ToolSpec if found, otherwise None.
        """
        return self._dynamic_tools.get(name)

    def get_tool_source(self, name: str) -> str | None:
        """
        Get the source code for a dynamically created tool.

        Args:
            name: The tool name.

        Returns:
            The source code string if found, otherwise None.
        """
        return self._source_registry.get(name)

    def remove_tool(self, name: str) -> bool:
        """
        Remove a dynamically created tool.

        Removes the tool from both the local dynamic registry and the
        global tool registry.

        Args:
            name: Name of the tool to remove.

        Returns:
            True if the tool was found and removed, False if not found.
        """
        if name not in self._dynamic_tools:
            return False

        del self._dynamic_tools[name]
        self._source_registry.pop(name, None)

        # Remove from global registry
        _tool_registry.pop(name, None)

        return True

    def test_tool(self, name: str, test_input: dict[str, Any]) -> Any:
        """
        Test a dynamically created tool with sample input.

        Executes the tool in the sandbox with the provided input and
        returns the result. This is useful for verifying tool behavior
        before using it in an agent workflow.

        Args:
            name: Name of the tool to test.
            test_input: Dictionary of keyword arguments to pass to the tool.

        Returns:
            The return value from the tool execution.

        Raises:
            ToolFactoryError: If the tool is not found.
            SandboxExecutionError: If the tool execution fails.
            SandboxTimeoutError: If the tool exceeds the timeout.
        """
        source = self._source_registry.get(name)
        if source is None:
            raise ToolFactoryError(
                f"Dynamic tool '{name}' not found. "
                f"Available dynamic tools: {list(self._dynamic_tools.keys())}",
                tool_name=name,
            )

        # Extract the function name from source
        try:
            func_name, _, _ = self.sandbox.validate_and_extract_function(source)
        except SandboxValidationError as e:
            raise ToolFactoryError(
                f"Failed to parse tool '{name}' source: {e}",
                tool_name=name,
            )

        return self.sandbox.execute(
            source=source,
            func_name=func_name,
            args=test_input,
        )

    def clear_all(self) -> int:
        """
        Remove all dynamically created tools.

        Returns:
            The number of tools removed.
        """
        count = len(self._dynamic_tools)

        for name in list(self._dynamic_tools.keys()):
            self.remove_tool(name)

        return count
