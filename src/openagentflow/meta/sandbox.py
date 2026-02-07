"""
Restricted execution environment for dynamically created tools.

Provides a sandboxed Python execution environment that:
- Validates source code via AST analysis before execution
- Restricts available modules to a safe subset of stdlib
- Blocks dangerous builtins (exec, eval, open, __import__, etc.)
- Prevents access to dunder attributes (except safe ones)
- Enforces execution timeout via threading
- Runs code in an isolated namespace

This is a security-critical component. All dynamically generated code
must pass through the Sandbox before execution.
"""

from __future__ import annotations

import ast
import math
import re
import json
import collections
import itertools
import functools
import string
import textwrap
import unicodedata
import datetime
import decimal
import fractions
import random
import statistics
import operator
import copy
import dataclasses
import enum
import typing
import abc
import bisect
import heapq
import threading
from typing import Any


class SandboxValidationError(Exception):
    """Raised when source code fails sandbox validation."""

    def __init__(self, message: str, *, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.violations = violations or []

    def __str__(self) -> str:
        if self.violations:
            details = "; ".join(self.violations)
            return f"{self.message}: {details}"
        return self.message


class SandboxExecutionError(Exception):
    """Raised when sandboxed code fails during execution."""

    def __init__(self, message: str, *, source_error: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.source_error = source_error


class SandboxTimeoutError(Exception):
    """Raised when sandboxed code exceeds the execution timeout."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Execution timed out after {timeout}s")
        self.timeout = timeout


class Sandbox:
    """
    Restricted execution environment for dynamically created tools.

    Validates Python source code using AST analysis, then executes it in
    an isolated namespace with only safe builtins and allowed stdlib modules.

    Usage:
        sandbox = Sandbox()

        # Validate code before execution
        is_safe, reason = sandbox.validate_source("def add(a: int, b: int) -> int:\\n    return a + b")
        assert is_safe

        # Execute safely
        result = sandbox.execute(
            source="def add(a: int, b: int) -> int:\\n    return a + b",
            func_name="add",
            args={"a": 1, "b": 2},
        )
        assert result == 3
    """

    # Allowed stdlib modules (safe, side-effect-free modules only)
    ALLOWED_MODULES: set[str] = {
        "math",
        "re",
        "json",
        "collections",
        "itertools",
        "functools",
        "string",
        "textwrap",
        "unicodedata",
        "datetime",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "operator",
        "copy",
        "dataclasses",
        "enum",
        "typing",
        "abc",
        "bisect",
        "heapq",
    }

    # Pre-imported module objects (avoids __import__ in sandbox)
    _MODULE_MAP: dict[str, Any] = {
        "math": math,
        "re": re,
        "json": json,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "string": string,
        "textwrap": textwrap,
        "unicodedata": unicodedata,
        "datetime": datetime,
        "decimal": decimal,
        "fractions": fractions,
        "random": random,
        "statistics": statistics,
        "operator": operator,
        "copy": copy,
        "dataclasses": dataclasses,
        "enum": enum,
        "typing": typing,
        "abc": abc,
        "bisect": bisect,
        "heapq": heapq,
    }

    # Builtins that are blocked in the sandbox
    BLOCKED_BUILTINS: set[str] = {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "delattr",
        "setattr",
        "getattr",
        "memoryview",
        "classmethod",
        "staticmethod",
        "property",
        "super",
        "type",
    }

    # Dunder attributes that are allowed (safe ones)
    ALLOWED_DUNDERS: set[str] = {
        "__init__",
        "__str__",
        "__repr__",
        "__len__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__bool__",
        "__contains__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__neg__",
        "__pos__",
        "__abs__",
        "__int__",
        "__float__",
        "__complex__",
        "__round__",
        "__enter__",
        "__exit__",
        "__call__",
        "__name__",
        "__doc__",
    }

    # Dangerous module names that must never appear anywhere in the code
    DANGEROUS_MODULES: set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "ctypes",
        "multiprocessing",
        "signal",
        "importlib",
        "pkgutil",
        "code",
        "codeop",
        "compileall",
        "py_compile",
        "zipimport",
        "pickle",
        "shelve",
        "marshal",
        "tempfile",
        "glob",
        "fnmatch",
        "io",
        "builtins",
        "_thread",
        "threading",
        "asyncio",
        "concurrent",
        "webbrowser",
        "ssl",
        "sqlite3",
    }

    def __init__(self, max_source_length: int = 10_000) -> None:
        """
        Initialize the Sandbox.

        Args:
            max_source_length: Maximum allowed source code length in characters.
                Prevents denial-of-service via extremely large code submissions.
        """
        self.max_source_length = max_source_length

    def validate_source(self, source: str) -> tuple[bool, str]:
        """
        Validate source code using AST analysis.

        Checks for:
        - Syntactically valid Python
        - No imports of non-allowed modules
        - No use of blocked builtins
        - No dangerous dunder attribute access
        - No os/sys/subprocess or other dangerous module usage
        - No file I/O operations
        - No network access patterns
        - Source code length within limits

        Args:
            source: Python source code to validate.

        Returns:
            A tuple of (is_safe, reason). If is_safe is True, reason is "OK".
            If is_safe is False, reason describes the violation(s).
        """
        violations: list[str] = []

        # Check source length
        if len(source) > self.max_source_length:
            return (
                False,
                f"Source code exceeds maximum length of {self.max_source_length} characters "
                f"(got {len(source)})",
            )

        # Check for empty source
        if not source.strip():
            return False, "Source code is empty"

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return False, f"Syntax error in source code: {e}"

        # Walk the AST and check each node
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in self.ALLOWED_MODULES:
                        violations.append(
                            f"Import of disallowed module '{alias.name}'"
                        )
                    if module_name in self.DANGEROUS_MODULES:
                        violations.append(
                            f"Import of dangerous module '{alias.name}'"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name not in self.ALLOWED_MODULES:
                        violations.append(
                            f"Import from disallowed module '{node.module}'"
                        )
                    if module_name in self.DANGEROUS_MODULES:
                        violations.append(
                            f"Import from dangerous module '{node.module}'"
                        )

            # Check function calls for blocked builtins
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        violations.append(
                            f"Call to blocked builtin '{node.func.id}'"
                        )
                    # Check for dangerous module references as function calls
                    if node.func.id in self.DANGEROUS_MODULES:
                        violations.append(
                            f"Reference to dangerous module '{node.func.id}'"
                        )

            # Check attribute access for dunders
            elif isinstance(node, ast.Attribute):
                attr = node.attr
                if attr.startswith("__") and attr.endswith("__"):
                    if attr not in self.ALLOWED_DUNDERS:
                        violations.append(
                            f"Access to disallowed dunder attribute '{attr}'"
                        )
                # Check for dangerous module attribute access (e.g., os.system)
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.DANGEROUS_MODULES:
                        violations.append(
                            f"Attribute access on dangerous module '{node.value.id}.{attr}'"
                        )

            # Check for Name nodes referencing dangerous modules
            elif isinstance(node, ast.Name):
                if node.id in self.DANGEROUS_MODULES:
                    # Only flag if used as a standalone reference (not in import)
                    # Imports are already checked above
                    parent_types = (ast.Import, ast.ImportFrom)
                    # We check this conservatively
                    pass

            # Check string literals for suspicious patterns
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                # Check for shell injection patterns
                suspicious_patterns = [
                    r"os\.system",
                    r"subprocess\.",
                    r"__import__",
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"open\s*\(",
                ]
                for pattern in suspicious_patterns:
                    if re.search(pattern, val):
                        violations.append(
                            f"Suspicious string literal containing '{pattern}'"
                        )

        # Additional full-text checks for patterns that might escape AST analysis
        # (e.g., constructed via string concatenation)
        dangerous_text_patterns = [
            (r"\bos\s*\.\s*system\b", "os.system call"),
            (r"\bos\s*\.\s*popen\b", "os.popen call"),
            (r"\bos\s*\.\s*exec", "os.exec call"),
            (r"\bsubprocess\b", "subprocess reference"),
            (r"\b__import__\b", "__import__ call"),
            (r"\bsocket\b", "socket reference"),
            (r"\burllib\b", "urllib reference"),
            (r"\bhttp\.client\b", "http.client reference"),
            (r"\bctypes\b", "ctypes reference"),
            (r"\bpickle\b", "pickle reference"),
            (r"\bmarshal\b", "marshal reference"),
        ]

        for pattern, description in dangerous_text_patterns:
            if re.search(pattern, source):
                # Only add if not already caught by AST analysis
                msg = f"Dangerous pattern detected: {description}"
                if msg not in violations:
                    violations.append(msg)

        if violations:
            return False, "; ".join(violations)

        return True, "OK"

    def _build_safe_builtins(self) -> dict[str, Any]:
        """
        Build a restricted builtins dictionary.

        Includes only safe builtins, excluding anything that could
        perform I/O, code execution, or namespace manipulation.

        Returns:
            Dictionary of allowed builtin names to their implementations.
        """
        import builtins

        safe_builtins: dict[str, Any] = {}

        # Whitelist approach: only include known-safe builtins
        allowed_builtin_names = {
            # Types and constructors
            "int",
            "float",
            "str",
            "bool",
            "bytes",
            "bytearray",
            "complex",
            "list",
            "tuple",
            "dict",
            "set",
            "frozenset",
            "slice",
            "range",
            "object",
            # Iteration and generators
            "iter",
            "next",
            "reversed",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            # Math and numeric
            "abs",
            "round",
            "min",
            "max",
            "sum",
            "pow",
            "divmod",
            "bin",
            "oct",
            "hex",
            # String and representation
            "repr",
            "str",
            "ascii",
            "chr",
            "ord",
            "format",
            "hash",
            # Type checking
            "isinstance",
            "issubclass",
            "callable",
            "id",
            "len",
            # Boolean
            "all",
            "any",
            # Exceptions (needed for try/except)
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "RuntimeError",
            "StopIteration",
            "ZeroDivisionError",
            "OverflowError",
            "ArithmeticError",
            "LookupError",
            "NotImplementedError",
            # Constants
            "True",
            "False",
            "None",
            # Other safe builtins
            "print",
            "dir",
            "hasattr",
        }

        for name in allowed_builtin_names:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Ensure True, False, None are available
        safe_builtins["True"] = True
        safe_builtins["False"] = False
        safe_builtins["None"] = None

        return safe_builtins

    def _build_restricted_globals(self, source: str) -> dict[str, Any]:
        """
        Build a restricted globals dictionary for code execution.

        Includes:
        - Safe builtins only
        - Pre-imported allowed modules (only those referenced in source)

        Args:
            source: The source code (used to determine which modules to include).

        Returns:
            Restricted globals dictionary.
        """
        restricted_globals: dict[str, Any] = {
            "__builtins__": self._build_safe_builtins(),
        }

        # Only include modules that are actually imported in the source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return restricted_globals

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in self.ALLOWED_MODULES and module_name in self._MODULE_MAP:
                        # Use the alias name if provided, otherwise the module name
                        key = alias.asname if alias.asname else alias.name
                        restricted_globals[key] = self._MODULE_MAP[module_name]

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in self.ALLOWED_MODULES and module_name in self._MODULE_MAP:
                        mod = self._MODULE_MAP[module_name]
                        # Handle submodule paths (e.g., "from collections import OrderedDict")
                        if node.names:
                            for alias in node.names:
                                attr_name = alias.name
                                key = alias.asname if alias.asname else attr_name
                                if hasattr(mod, attr_name):
                                    restricted_globals[key] = getattr(mod, attr_name)
                                else:
                                    # Module itself if attribute not found
                                    restricted_globals[key] = mod

        return restricted_globals

    def execute(
        self,
        source: str,
        func_name: str,
        args: dict[str, Any],
        timeout: float = 5.0,
    ) -> Any:
        """
        Execute code in a restricted environment with timeout.

        Steps:
        1. Validate the source code via AST analysis
        2. Create restricted globals (only allowed builtins + allowed modules)
        3. Execute the source in the restricted namespace
        4. Call the specified function with the provided arguments
        5. Return the result (with timeout enforcement via threading)

        Args:
            source: Python source code defining one or more functions.
            func_name: Name of the function to call after execution.
            args: Dictionary of keyword arguments to pass to the function.
            timeout: Maximum execution time in seconds (default: 5.0).

        Returns:
            The return value of the called function.

        Raises:
            SandboxValidationError: If the source code fails validation.
            SandboxExecutionError: If the code fails during execution.
            SandboxTimeoutError: If execution exceeds the timeout.
        """
        # Step 1: Validate source
        is_safe, reason = self.validate_source(source)
        if not is_safe:
            raise SandboxValidationError(
                f"Source code failed sandbox validation",
                violations=[reason],
            )

        # Step 2: Build restricted globals
        restricted_globals = self._build_restricted_globals(source)

        # Step 3 & 4: Execute with timeout using threading
        result_container: dict[str, Any] = {}
        error_container: dict[str, Exception] = {}

        def _run() -> None:
            try:
                # Execute the source in the restricted namespace
                exec(source, restricted_globals)  # noqa: S102

                # Find the function
                if func_name not in restricted_globals:
                    error_container["error"] = SandboxExecutionError(
                        f"Function '{func_name}' not found in executed source code. "
                        f"Available names: {[k for k in restricted_globals if not k.startswith('_')]}"
                    )
                    return

                func = restricted_globals[func_name]
                if not callable(func):
                    error_container["error"] = SandboxExecutionError(
                        f"'{func_name}' is not callable (type: {type(func).__name__})"
                    )
                    return

                # Call the function with provided arguments
                result_container["result"] = func(**args)

            except Exception as e:
                error_container["error"] = SandboxExecutionError(
                    f"Execution failed: {type(e).__name__}: {e}",
                    source_error=e,
                )

        # Step 5: Run with timeout enforcement
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running -- it exceeded the timeout.
            # Since we set daemon=True, it will be killed when the main thread exits.
            # We cannot forcefully kill a thread in Python, but daemon threads
            # won't block process exit.
            raise SandboxTimeoutError(timeout)

        if "error" in error_container:
            raise error_container["error"]

        if "result" not in result_container:
            raise SandboxExecutionError(
                f"Function '{func_name}' did not produce a result"
            )

        return result_container["result"]

    def validate_and_extract_function(
        self, source: str
    ) -> tuple[str, dict[str, str], str | None]:
        """
        Validate source code and extract function metadata.

        Parses the AST to find the primary function definition and extracts:
        - The function name
        - Parameter names with their type hint strings
        - The return type hint string (if present)

        This is useful for auto-generating input schemas from source code.

        Args:
            source: Python source code containing a function definition.

        Returns:
            A tuple of (func_name, param_types, return_type) where:
            - func_name is the name of the first function defined
            - param_types is a dict mapping parameter names to type hint strings
            - return_type is the return type hint string, or None

        Raises:
            SandboxValidationError: If validation fails or no function is found.
        """
        is_safe, reason = self.validate_source(source)
        if not is_safe:
            raise SandboxValidationError(
                "Source code failed sandbox validation",
                violations=[reason],
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise SandboxValidationError(
                f"Syntax error in source code: {e}"
            )

        # Find the first function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_name = node.name
                param_types: dict[str, str] = {}

                for arg in node.args.args:
                    arg_name = arg.arg
                    if arg.annotation:
                        param_types[arg_name] = ast.unparse(arg.annotation)
                    else:
                        param_types[arg_name] = "Any"

                return_type: str | None = None
                if node.returns:
                    return_type = ast.unparse(node.returns)

                return func_name, param_types, return_type

        raise SandboxValidationError(
            "No function definition found in source code"
        )
