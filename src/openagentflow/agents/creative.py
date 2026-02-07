"""Creative agents for code analysis, transformation, and improvement."""

import ast
import re
from typing import List, Dict
from openagentflow import agent


# ============================================================================
# Helper Tools
# ============================================================================


def explain_function(code: str) -> str:
    """Explain what a function does by analyzing its structure and logic."""
    try:
        tree = ast.parse(code)
        explanations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                explanation = f"Function '{node.name}' "

                # Analyze parameters
                if node.args.args:
                    params = [arg.arg for arg in node.args.args]
                    explanation += f"takes parameters: {', '.join(params)}. "
                else:
                    explanation += "takes no parameters. "

                # Check for return statements
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                if has_return:
                    explanation += "Returns a value. "

                # Count operations
                statements = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                explanation += f"Contains {statements} statements. "

                explanations.append(explanation)

        return "\n".join(explanations) if explanations else "No functions found in code."
    except SyntaxError:
        return "Code contains syntax errors. Unable to parse and explain."


def trace_flow(code: str) -> str:
    """Trace execution flow through the code."""
    try:
        tree = ast.parse(code)
        flow_trace = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                flow_trace.append("→ Conditional branch (if statement)")
            elif isinstance(node, ast.For):
                flow_trace.append("→ Loop iteration (for loop)")
            elif isinstance(node, ast.While):
                flow_trace.append("→ Loop iteration (while loop)")
            elif isinstance(node, ast.Try):
                flow_trace.append("→ Exception handling (try/except)")
            elif isinstance(node, ast.FunctionDef):
                flow_trace.append(f"→ Function definition: {node.name}")
            elif isinstance(node, ast.Return):
                flow_trace.append("→ Return statement")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                flow_trace.append(f"→ Function call: {node.func.id}")

        return "\n".join(flow_trace) if flow_trace else "No control flow detected."
    except SyntaxError:
        return "Code contains syntax errors. Unable to trace flow."


def visualize_logic(code: str) -> str:
    """Create ASCII visualization of code logic structure."""
    try:
        tree = ast.parse(code)
        visualization = ["Code Structure Visualization:", "=" * 40]

        def visualize_node(node, indent=0):
            prefix = "  " * indent
            if isinstance(node, ast.FunctionDef):
                visualization.append(f"{prefix}┌─ Function: {node.name}")
                for child in node.body:
                    visualize_node(child, indent + 1)
                visualization.append(f"{prefix}└─ End {node.name}")
            elif isinstance(node, ast.If):
                visualization.append(f"{prefix}├─ IF condition")
                for child in node.body:
                    visualize_node(child, indent + 1)
                if node.orelse:
                    visualization.append(f"{prefix}├─ ELSE")
                    for child in node.orelse:
                        visualize_node(child, indent + 1)
            elif isinstance(node, ast.For):
                visualization.append(f"{prefix}├─ FOR loop")
                for child in node.body:
                    visualize_node(child, indent + 1)
            elif isinstance(node, ast.While):
                visualization.append(f"{prefix}├─ WHILE loop")
                for child in node.body:
                    visualize_node(child, indent + 1)
            elif isinstance(node, ast.Return):
                visualization.append(f"{prefix}└─ RETURN")

        for node in tree.body:
            visualize_node(node)

        return "\n".join(visualization)
    except SyntaxError:
        return "Code contains syntax errors. Unable to visualize."


def brainstorm_features(code: str) -> List[str]:
    """Suggest new features based on code analysis."""
    features = []

    try:
        tree = ast.parse(code)

        # Check for functions that could use caching
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(isinstance(n, ast.For) for n in ast.walk(node)):
                    features.append(f"Add memoization/caching to {node.name} for better performance")

                if not any(isinstance(n, ast.Try) for n in ast.walk(node)):
                    features.append(f"Add error handling to {node.name}")

                if not node.args.args or len(node.args.args) < 2:
                    features.append(f"Add configuration parameters to {node.name}")

        # General suggestions
        features.extend([
            "Add logging for debugging and monitoring",
            "Add type hints for better code clarity",
            "Add docstrings for documentation",
            "Add unit tests for reliability",
            "Add configuration file support",
            "Add async support for I/O operations"
        ])
    except SyntaxError:
        features.append("Fix syntax errors first")

    return features[:10]  # Return top 10


def find_opportunities(code: str) -> List[str]:
    """Find improvement opportunities in the code."""
    opportunities = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function length
                statements = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                if statements > 20:
                    opportunities.append(f"Refactor {node.name}: too long ({statements} statements)")

                # Check nested loops
                loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))]
                if len(loops) > 2:
                    opportunities.append(f"Optimize nested loops in {node.name}")

                # Check for duplicate code patterns
                if isinstance(node, ast.If) and node.orelse:
                    opportunities.append("Consider using polymorphism instead of if/else chains")

        # Check for hardcoded values
        literals = [n for n in ast.walk(tree) if isinstance(n, (ast.Str, ast.Num))]
        if len(literals) > 5:
            opportunities.append("Extract magic numbers/strings into constants")

        opportunities.extend([
            "Add input validation",
            "Improve variable naming",
            "Reduce code complexity",
            "Add comprehensive error messages",
            "Consider design patterns"
        ])
    except SyntaxError:
        opportunities.append("Fix syntax errors")

    return opportunities[:8]


def suggest_integrations(code: str) -> List[str]:
    """Suggest integrations based on code patterns."""
    integrations = []

    try:
        tree = ast.parse(code)
        imports = [node.names[0].name for node in ast.walk(tree)
                  if isinstance(node, ast.Import)]

        # Suggest based on patterns
        has_async = any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(tree))
        if has_async:
            integrations.append("AsyncIO event loop integration")
            integrations.append("AIOHTTP for async HTTP requests")

        if "json" in str(code).lower():
            integrations.append("Pydantic for data validation")
            integrations.append("JSON Schema validation")

        if any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)):
            integrations.append("FastAPI for REST API")
            integrations.append("Click for CLI interface")
            integrations.append("Celery for background tasks")

        integrations.extend([
            "Redis for caching",
            "PostgreSQL for data persistence",
            "Prometheus for metrics",
            "Sentry for error tracking",
            "Docker for containerization"
        ])
    except SyntaxError:
        integrations.append("Fix syntax errors first")

    return integrations[:8]


def to_async(code: str) -> str:
    """Convert synchronous code to async/await style."""
    try:
        tree = ast.parse(code)
        converted_lines = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Convert function definition
                args = ", ".join([arg.arg for arg in node.args.args])
                converted_lines.append(f"async def {node.name}({args}):")
                converted_lines.append("    # Converted to async")

                # Check for I/O operations
                if any(isinstance(n, ast.Call) for n in ast.walk(node)):
                    converted_lines.append("    # Add await for async operations")
                    converted_lines.append("    result = await async_operation()")
                    converted_lines.append("    return result")
                else:
                    converted_lines.append("    # Original logic here (async-compatible)")
                    converted_lines.append("    pass")

        if not converted_lines:
            return "# No functions found to convert\n" + code

        return "\n".join(converted_lines)
    except SyntaxError:
        return "# Syntax error in original code\n" + code


def to_functional(code: str) -> str:
    """Convert to functional programming style."""
    try:
        tree = ast.parse(code)
        functional_code = ["# Functional programming style conversion", ""]

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = ", ".join([arg.arg for arg in node.args.args])
                functional_code.append(f"def {node.name}({args}):")
                functional_code.append("    # Using functional approach:")
                functional_code.append("    # - Pure functions (no side effects)")
                functional_code.append("    # - Immutable data")
                functional_code.append("    # - Higher-order functions (map, filter, reduce)")
                functional_code.append("    return result  # Pure computation")
                functional_code.append("")

        if len(functional_code) <= 2:
            return code + "\n# Consider using map(), filter(), reduce()"

        return "\n".join(functional_code)
    except SyntaxError:
        return "# Syntax error\n" + code


def to_oop(code: str) -> str:
    """Convert to object-oriented programming style."""
    try:
        tree = ast.parse(code)
        class_code = ["# Object-oriented style conversion", ""]

        # Extract functions
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if functions:
            class_code.append("class ConvertedClass:")
            class_code.append('    """Converted to OOP style."""')
            class_code.append("")
            class_code.append("    def __init__(self):")
            class_code.append("        # Initialize state")
            class_code.append("        pass")
            class_code.append("")

            for func in functions:
                args = ", ".join([arg.arg for arg in func.args.args])
                if args:
                    class_code.append(f"    def {func.name}(self, {args}):")
                else:
                    class_code.append(f"    def {func.name}(self):")
                class_code.append("        # Method implementation")
                class_code.append("        pass")
                class_code.append("")
        else:
            return code + "\n# No functions to convert to class"

        return "\n".join(class_code)
    except SyntaxError:
        return "# Syntax error\n" + code


def suggest_names(code: str) -> Dict[str, str]:
    """Suggest better names for variables and functions."""
    suggestions = {}

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name
                # Check for single letter or unclear names
                if len(name) <= 2:
                    suggestions[name] = f"{name}_descriptive_name"
                elif name.startswith("func") or name.startswith("do"):
                    suggestions[name] = f"descriptive_{name}"

            elif isinstance(node, ast.Name):
                name = node.id
                # Suggest improvements for short variables
                if len(name) == 1 and name not in ['i', 'j', 'k', 'x', 'y', 'z']:
                    suggestions[name] = f"{name}_variable"
                elif name in ['tmp', 'temp', 'data', 'val', 'var']:
                    suggestions[name] = f"specific_{name}_name"
    except SyntaxError:
        suggestions["error"] = "Fix syntax errors first"

    # Add general suggestions
    if not suggestions:
        suggestions.update({
            "example": "Use descriptive, intention-revealing names",
            "tip": "Avoid abbreviations unless widely known",
            "guideline": "Use verb phrases for functions, nouns for variables"
        })

    return suggestions


def expand_abbreviations(name: str) -> str:
    """Expand abbreviated names to full form."""
    abbreviations = {
        "cfg": "config",
        "ctx": "context",
        "db": "database",
        "doc": "document",
        "err": "error",
        "fn": "function",
        "idx": "index",
        "init": "initialize",
        "len": "length",
        "msg": "message",
        "num": "number",
        "obj": "object",
        "params": "parameters",
        "prev": "previous",
        "proc": "process",
        "req": "request",
        "res": "response",
        "str": "string",
        "temp": "temporary",
        "tmp": "temporary",
        "val": "value",
        "var": "variable",
    }

    # Check if name is an abbreviation
    lower_name = name.lower()
    if lower_name in abbreviations:
        return abbreviations[lower_name]

    # Check for common patterns
    for abbr, full in abbreviations.items():
        if lower_name.startswith(abbr):
            expanded = name.replace(abbr, full, 1)
            return expanded
        if lower_name.endswith(abbr):
            expanded = name[:len(name)-len(abbr)] + full
            return expanded

    return name  # Return original if no expansion found


def semantic_rename(code: str, old_name: str, new_name: str) -> str:
    """Rename with semantic awareness (preserves structure)."""
    try:
        tree = ast.parse(code)

        class NameReplacer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == old_name:
                    node.id = new_name
                return node

            def visit_FunctionDef(self, node):
                if node.name == old_name:
                    node.name = new_name
                self.generic_visit(node)
                return node

        replacer = NameReplacer()
        new_tree = replacer.visit(tree)
        return ast.unparse(new_tree)
    except SyntaxError:
        # Fallback to simple string replacement
        return code.replace(old_name, new_name)


# ============================================================================
# Creative Agents
# ============================================================================


@agent(
    model="claude-sonnet-4-20250514",
    tools=[explain_function, trace_flow, visualize_logic]
)
async def code_explainer(code: str) -> str:
    """
    Explain code in plain English with detailed analysis.

    This agent analyzes code and provides clear explanations of:
    - What the code does
    - How it works
    - The execution flow
    - Visual structure representation

    Args:
        code: The code to explain

    Returns:
        A comprehensive explanation in plain English
    """
    return f"""Please analyze this code and provide a detailed explanation:

{code}

Use the available tools to:
1. Explain what each function does
2. Trace the execution flow
3. Visualize the logic structure
4. Provide a plain English summary

Make the explanation clear and accessible for developers."""


@agent(
    model="claude-sonnet-4-20250514",
    tools=[brainstorm_features, find_opportunities, suggest_integrations]
)
async def idea_generator(code: str) -> str:
    """
    Generate feature ideas and improvement suggestions.

    This agent analyzes code and suggests:
    - New features to add
    - Improvement opportunities
    - Potential integrations
    - Enhancement ideas

    Args:
        code: The code to analyze for ideas

    Returns:
        Creative suggestions for features and improvements
    """
    return f"""Analyze this code and generate creative ideas:

{code}

Use the available tools to:
1. Brainstorm new features that could be added
2. Find opportunities for improvement
3. Suggest integrations with other tools/libraries
4. Provide innovative enhancement ideas

Be creative and think about how this code could be extended or improved."""


@agent(
    model="claude-sonnet-4-20250514",
    tools=[to_async, to_functional, to_oop]
)
async def code_translator(code: str) -> str:
    """
    Convert code between different programming styles.

    This agent can transform code to:
    - Async/await style
    - Functional programming style
    - Object-oriented programming style

    Args:
        code: The code to transform

    Returns:
        Converted code in different styles with explanations
    """
    return f"""Transform this code into different programming styles:

{code}

Use the available tools to show:
1. Async/await version (for concurrent operations)
2. Functional programming version (pure functions, immutability)
3. Object-oriented version (classes, encapsulation)

Explain the benefits and tradeoffs of each style."""


@agent(
    model="claude-sonnet-4-20250514",
    tools=[suggest_names, expand_abbreviations, semantic_rename]
)
async def name_suggester(code: str) -> str:
    """
    Suggest better variable and function names.

    This agent helps improve code readability by:
    - Suggesting better names for variables and functions
    - Expanding abbreviations
    - Performing semantic-aware renaming
    - Following naming best practices

    Args:
        code: The code to analyze for naming improvements

    Returns:
        Suggestions for better names and explanations
    """
    return f"""Analyze this code and suggest better names:

{code}

Use the available tools to:
1. Suggest better names for all variables and functions
2. Expand any abbreviations to full forms
3. Provide examples of semantic renaming
4. Explain naming best practices

Focus on making names more descriptive, clear, and intention-revealing."""
