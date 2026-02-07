"""Refactoring agents for code modernization and architecture improvements."""

import ast
import re
from typing import Any

from openagentflow import agent


# Helper Tools

def upgrade_type_hints(code: str) -> str:
    """
    Upgrade to modern type hints (Python 3.10+).

    Converts old-style type hints to modern syntax:
    - List[str] -> list[str]
    - Dict[str, int] -> dict[str, int]
    - Optional[str] -> str | None
    - Union[str, int] -> str | int

    Args:
        code: Python code with old-style type hints

    Returns:
        Code with modernized type hints
    """
    # Convert Optional[X] to X | None
    code = re.sub(r'Optional\[([^\]]+)\]', r'\1 | None', code)

    # Convert Union[X, Y, ...] to X | Y | ...
    def replace_union(match):
        types = match.group(1)
        return ' | '.join(t.strip() for t in types.split(','))
    code = re.sub(r'Union\[([^\]]+)\]', replace_union, code)

    # Convert typing generic collections to built-in versions
    replacements = {
        r'\bList\[': 'list[',
        r'\bDict\[': 'dict[',
        r'\bSet\[': 'set[',
        r'\bTuple\[': 'tuple[',
        r'\bFrozenSet\[': 'frozenset[',
    }

    for pattern, replacement in replacements.items():
        code = re.sub(pattern, replacement, code)

    return code


def use_match_statements(code: str) -> str:
    """
    Convert if-elif chains to match statements where beneficial.

    Identifies simple value-based if-elif-else chains and converts them
    to Python 3.10+ match statements for better readability.

    Args:
        code: Python code with if-elif chains

    Returns:
        Code with match statements where applicable
    """
    # This is a simplified implementation that provides suggestions
    # A full implementation would require AST manipulation

    # Pattern to detect if-elif chains checking the same variable
    pattern = r'if\s+(\w+)\s*==\s*([^\:]+):\s*\n\s+(.+)\n\s*elif\s+\1\s*==\s*([^\:]+):'

    if re.search(pattern, code):
        suggestion = (
            "# Consider converting if-elif chains to match statements:\n"
            "# match variable:\n"
            "#     case value1:\n"
            "#         action1\n"
            "#     case value2:\n"
            "#         action2\n"
            "#     case _:\n"
            "#         default_action\n"
        )
        return code + "\n\n" + suggestion

    return code


def apply_walrus(code: str) -> str:
    """
    Apply walrus operator (:=) where beneficial.

    Identifies patterns where the walrus operator can reduce redundancy:
    - Assignment followed by condition check
    - Repeated function calls in conditions

    Args:
        code: Python code without walrus operators

    Returns:
        Code with walrus operators applied where beneficial
    """
    # Pattern: value = func(); if value:
    # Convert to: if (value := func()):
    pattern = r'(\s*)(\w+)\s*=\s*([^\n]+)\n\s*if\s+\2\s*:'
    replacement = r'\1if (\2 := \3):'
    code = re.sub(pattern, replacement, code)

    # Pattern: value = func(); while value:
    pattern = r'(\s*)(\w+)\s*=\s*([^\n]+)\n\s*while\s+\2\s*:'
    replacement = r'\1while (\2 := \3):'
    code = re.sub(pattern, replacement, code)

    return code


def suggest_patterns(code: str) -> list[dict]:
    """
    Suggest design patterns applicable to the code.

    Analyzes code structure and suggests appropriate design patterns
    such as Factory, Strategy, Observer, Singleton, etc.

    Args:
        code: Python code to analyze

    Returns:
        List of pattern suggestions with rationale
    """
    suggestions = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"pattern": "Parse Error", "reason": "Unable to parse code"}]

    # Check for multiple conditional object creation -> Factory Pattern
    if re.search(r'if.*:\s*\n\s*.*=\s*\w+\(.*\)\s*\n\s*elif.*:\s*\n\s*.*=\s*\w+\(', code):
        suggestions.append({
            "pattern": "Factory Pattern",
            "reason": "Multiple conditional object instantiations detected",
            "benefit": "Encapsulate object creation logic"
        })

    # Check for many if-else for different behaviors -> Strategy Pattern
    if code.count('elif') > 3:
        suggestions.append({
            "pattern": "Strategy Pattern",
            "reason": "Multiple conditional branches for behavior selection",
            "benefit": "Replace conditionals with polymorphic strategy classes"
        })

    # Check for state management -> State Pattern
    if re.search(r'self\._?state|self\._?status', code, re.IGNORECASE):
        suggestions.append({
            "pattern": "State Pattern",
            "reason": "State-based behavior changes detected",
            "benefit": "Encapsulate state-specific behavior in separate classes"
        })

    # Check for callback/event patterns -> Observer Pattern
    if re.search(r'callback|listener|notify|subscribe', code, re.IGNORECASE):
        suggestions.append({
            "pattern": "Observer Pattern",
            "reason": "Event/callback mechanism detected",
            "benefit": "Implement formal observer pattern for better decoupling"
        })

    # Check for class with only class methods -> Singleton candidate
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            if methods and all(
                any(d.id == 'classmethod' for d in m.decorator_list if isinstance(d, ast.Name))
                for m in methods if m.name != '__init__'
            ):
                suggestions.append({
                    "pattern": "Singleton Pattern",
                    "reason": "Class with primarily class methods detected",
                    "benefit": "Consider singleton if single instance is needed"
                })

    # Check for data transformation pipeline -> Chain of Responsibility
    if re.search(r'process.*\(.*process.*\(', code):
        suggestions.append({
            "pattern": "Chain of Responsibility / Pipeline",
            "reason": "Nested processing calls detected",
            "benefit": "Create a processing pipeline with composable handlers"
        })

    return suggestions if suggestions else [
        {"pattern": "None", "reason": "No obvious pattern opportunities detected"}
    ]


def detect_violations(code: str) -> list[dict]:
    """
    Detect SOLID principle violations in the code.

    Analyzes code for violations of:
    - Single Responsibility Principle
    - Open/Closed Principle
    - Liskov Substitution Principle
    - Interface Segregation Principle
    - Dependency Inversion Principle

    Args:
        code: Python code to analyze

    Returns:
        List of detected violations with explanations
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"principle": "Parse Error", "violation": "Unable to parse code"}]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]

            # Single Responsibility Principle
            # Check for classes with many methods (possible multiple responsibilities)
            if len(methods) > 10:
                violations.append({
                    "principle": "Single Responsibility Principle (SRP)",
                    "violation": f"Class '{class_name}' has {len(methods)} methods",
                    "suggestion": "Consider splitting into multiple classes with focused responsibilities",
                    "severity": "medium"
                })

            # Check for mixed concerns (data access + business logic + presentation)
            method_names = [m.name for m in methods]
            has_db = any('save' in m or 'load' in m or 'query' in m for m in method_names)
            has_business = any('calculate' in m or 'process' in m or 'validate' in m for m in method_names)
            has_ui = any('render' in m or 'display' in m or 'format' in m for m in method_names)

            mixed_concerns = sum([has_db, has_business, has_ui])
            if mixed_concerns >= 2:
                violations.append({
                    "principle": "Single Responsibility Principle (SRP)",
                    "violation": f"Class '{class_name}' mixes multiple concerns (data/business/presentation)",
                    "suggestion": "Separate data access, business logic, and presentation concerns",
                    "severity": "high"
                })

            # Dependency Inversion Principle
            # Check for direct instantiation of concrete classes
            for method in methods:
                for subnode in ast.walk(method):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Name):
                            if subnode.func.id[0].isupper():  # Likely a class instantiation
                                violations.append({
                                    "principle": "Dependency Inversion Principle (DIP)",
                                    "violation": f"Direct instantiation in '{class_name}.{method.name}'",
                                    "suggestion": "Inject dependencies rather than creating them directly",
                                    "severity": "medium"
                                })
                                break

    # Open/Closed Principle
    # Check for type checking with isinstance in conditionals
    if re.search(r'isinstance\(.+\).+if|if.+isinstance\(', code):
        violations.append({
            "principle": "Open/Closed Principle (OCP)",
            "violation": "Type checking with isinstance() detected",
            "suggestion": "Use polymorphism instead of type checking",
            "severity": "medium"
        })

    # Check for type() comparisons
    if 'type(' in code and '==' in code:
        violations.append({
            "principle": "Open/Closed Principle (OCP)",
            "violation": "Type comparison with type() detected",
            "suggestion": "Use polymorphism and duck typing instead",
            "severity": "high"
        })

    # Interface Segregation Principle
    # Check for classes with NotImplementedError (possible fat interfaces)
    if 'NotImplementedError' in code or 'raise NotImplemented' in code:
        violations.append({
            "principle": "Interface Segregation Principle (ISP)",
            "violation": "Methods raising NotImplementedError detected",
            "suggestion": "Split interface into smaller, more specific interfaces",
            "severity": "medium"
        })

    return violations if violations else [
        {"principle": "None", "violation": "No obvious SOLID violations detected"}
    ]


def propose_restructuring(code: str) -> str:
    """
    Propose code restructuring improvements.

    Analyzes code structure and proposes refactoring to improve:
    - Function/method length and complexity
    - Code duplication
    - Naming conventions
    - Module organization

    Args:
        code: Python code to analyze

    Returns:
        Restructuring proposal as formatted text
    """
    proposals = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Unable to parse code for restructuring analysis."

    # Analyze function complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name

            # Count lines (rough complexity metric)
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                lines = node.end_lineno - node.lineno
                if lines > 50:
                    proposals.append(
                        f"Function '{func_name}' is {lines} lines long. "
                        f"Consider breaking it into smaller functions (target: <20 lines per function)."
                    )

            # Count nested depth
            max_depth = 0
            def count_depth(n, depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                for child in ast.iter_child_nodes(n):
                    if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                        count_depth(child, depth + 1)
                    else:
                        count_depth(child, depth)

            count_depth(node)
            if max_depth > 4:
                proposals.append(
                    f"Function '{func_name}' has nesting depth of {max_depth}. "
                    f"Consider extracting nested logic into separate functions."
                )

    # Check for code duplication (simple pattern matching)
    lines = code.split('\n')
    stripped_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

    # Find repeated blocks (3+ identical lines)
    for i in range(len(stripped_lines) - 3):
        block = tuple(stripped_lines[i:i+3])
        if block[0]:  # Not empty
            count = 0
            for j in range(len(stripped_lines) - 3):
                if tuple(stripped_lines[j:j+3]) == block:
                    count += 1
            if count > 1:
                proposals.append(
                    f"Code duplication detected: {count} occurrences of similar 3-line blocks. "
                    f"Consider extracting into a reusable function."
                )
                break

    # Check naming conventions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not node.name[0].isupper():
                proposals.append(
                    f"Class '{node.name}' should follow PascalCase naming convention."
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('_') and node.name.endswith('_'):
                continue  # Dunder methods are fine
            if not node.name.islower() and '_' not in node.name:
                proposals.append(
                    f"Function '{node.name}' should follow snake_case naming convention."
                )

    # Check for magic numbers
    for node in ast.walk(tree):
        if isinstance(node, ast.Num):
            if isinstance(node.n, (int, float)) and node.n not in (0, 1, -1):
                proposals.append(
                    f"Magic number '{node.n}' detected. Consider defining as a named constant."
                )
                break

    if not proposals:
        return "Code structure looks good. No major restructuring needed."

    return "RESTRUCTURING PROPOSALS:\n\n" + "\n\n".join(f"{i+1}. {p}" for i, p in enumerate(proposals))


# Agent 1: Code Modernizer

@agent(
    name="code_modernizer",
    model="claude-sonnet-4-20250514",
    tools=[upgrade_type_hints, use_match_statements, apply_walrus],
    system_prompt="""You are a Python code modernization expert specializing in Python 3.10+ features.

Your role is to update Python code to use modern syntax and features:

1. TYPE HINTS: Convert old typing module generics to built-in types
   - List[X] -> list[X]
   - Dict[K, V] -> dict[K, V]
   - Optional[X] -> X | None
   - Union[X, Y] -> X | Y

2. MATCH STATEMENTS: Replace verbose if-elif chains with match statements
   - Look for patterns checking the same variable against multiple values
   - Use structural pattern matching for complex conditions
   - Leverage guard clauses (case X if condition:)

3. WALRUS OPERATOR: Use := to reduce redundancy
   - Combine assignment with condition checks
   - Eliminate repeated function calls
   - Improve readability in comprehensions

4. OTHER MODERN FEATURES:
   - Use f-strings over .format() or %
   - Prefer pathlib over os.path
   - Use dataclasses or attrs for data containers
   - Apply type hints everywhere (PEP 484)

Always explain why each modernization improves the code (readability, performance, or maintainability).
Provide the modernized code along with a summary of changes made."""
)
async def code_modernizer(code: str) -> str:
    """
    Modernize Python code to use Python 3.10+ features.

    Args:
        code: Python code to modernize

    Returns:
        Modernized code with explanations
    """
    return code


# Agent 2: Architecture Advisor

@agent(
    name="architecture_advisor",
    model="claude-sonnet-4-20250514",
    tools=[suggest_patterns, detect_violations, propose_restructuring],
    system_prompt="""You are a software architecture expert specializing in clean code principles and design patterns.

Your role is to analyze code and provide architectural guidance:

1. SOLID PRINCIPLES:
   - Single Responsibility: Each class should have one reason to change
   - Open/Closed: Open for extension, closed for modification
   - Liskov Substitution: Subtypes must be substitutable for base types
   - Interface Segregation: Many specific interfaces over one general interface
   - Dependency Inversion: Depend on abstractions, not concretions

2. DESIGN PATTERNS:
   - Creational: Factory, Builder, Singleton, Prototype
   - Structural: Adapter, Decorator, Facade, Proxy
   - Behavioral: Strategy, Observer, Command, Template Method, Chain of Responsibility

3. CODE QUALITY:
   - Identify code smells (long methods, large classes, duplicated code)
   - Suggest refactoring opportunities
   - Recommend better abstractions and separation of concerns

4. BEST PRACTICES:
   - Dependency injection over tight coupling
   - Composition over inheritance
   - Clear naming and single-level abstractions
   - Testability and maintainability

Provide specific, actionable recommendations with clear reasoning.
Prioritize violations by severity (high/medium/low).
Suggest concrete refactoring steps with example code structure."""
)
async def architecture_advisor(code: str) -> str:
    """
    Analyze code architecture and suggest improvements.

    Args:
        code: Python code to analyze

    Returns:
        Architectural analysis and recommendations
    """
    return code
