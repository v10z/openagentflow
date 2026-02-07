"""
Code Quality Agents - 5 Specialized Agents for Python Code Analysis

This module provides 5 specialized agents for analyzing Python code quality:
1. style_enforcer - Enforces PEP 8 style, naming conventions, and formatting
2. complexity_analyzer - Analyzes cyclomatic complexity and nesting depth
3. dead_code_hunter - Finds unused imports, variables, and functions
4. pattern_detector - Detects anti-patterns and code smells
5. consistency_checker - Checks style consistency across codebase

Each agent uses tools from openagentflow.tools.code and returns a detailed
analysis report as a string.

Usage:
    from openagentflow.agents.code_quality import style_enforcer

    code = '''
    def myBadFunction():
        x = 42
        return x
    '''

    result = await style_enforcer(code)
    print(result.output)
"""

from openagentflow.core.agent import agent
from openagentflow.tools.code import (
    check_naming_convention,
    find_long_functions,
    find_magic_numbers,
    calculate_complexity,
    find_nested_loops,
    extract_imports,
    extract_functions,
    detect_global_variables,
    extract_classes,
    extract_docstrings,
    find_type_hints,
)


@agent(
    model="claude-sonnet-4-20250514",
    tools=[check_naming_convention, find_long_functions, find_magic_numbers],
    system_prompt="""You are a Python style enforcer focused on PEP 8 compliance and code formatting.

Your responsibilities:
1. Check naming conventions (snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants)
2. Identify functions that are too long (>50 lines)
3. Find magic numbers that should be named constants
4. Provide specific, actionable feedback on style violations

When analyzing code:
- Use check_naming_convention to find naming violations
- Use find_long_functions to identify overly long functions
- Use find_magic_numbers to locate hardcoded values
- Summarize all violations clearly with line numbers
- Suggest specific fixes for each issue

Your report should be well-structured and include:
- Overview of style issues found
- Detailed list of violations with line numbers
- Recommendations for fixes
- Priority level for each issue (critical, high, medium, low)

Be thorough but concise. Focus on issues that impact readability and maintainability."""
)
async def style_enforcer(code: str) -> str:
    """
    Enforce PEP 8 style guidelines, naming conventions, and formatting standards.

    Analyzes Python code for:
    - Naming convention violations (PEP 8)
    - Functions that exceed recommended length
    - Magic numbers that should be constants

    Args:
        code: Python source code to analyze

    Returns:
        Detailed style analysis report with violations and recommendations
    """
    return f"Analyzing code for PEP 8 style violations...\n\nCode:\n{code}"


@agent(
    model="claude-sonnet-4-20250514",
    tools=[calculate_complexity, find_nested_loops, find_long_functions],
    system_prompt="""You are a code complexity analyzer focused on identifying overly complex code.

Your responsibilities:
1. Calculate cyclomatic complexity for the code
2. Find deeply nested loops (depth > 3)
3. Identify long functions that may be hard to maintain
4. Provide recommendations to reduce complexity

When analyzing code:
- Use calculate_complexity to get the overall complexity score
- Use find_nested_loops to identify deep nesting (max_depth=3)
- Use find_long_functions to find functions exceeding 50 lines
- Assess the cognitive load of the code

Complexity guidelines:
- Cyclomatic complexity: 1-10 (simple), 11-20 (moderate), 21+ (complex)
- Loop nesting: max 3 levels deep
- Function length: max 50 lines

Your report should include:
- Overall complexity assessment
- Specific complexity hotspots with line numbers
- Refactoring suggestions to reduce complexity
- Impact assessment (how complexity affects maintainability)

Be constructive and provide actionable refactoring advice."""
)
async def complexity_analyzer(code: str) -> str:
    """
    Analyze code complexity including cyclomatic complexity and nesting depth.

    Evaluates:
    - Cyclomatic complexity (McCabe complexity)
    - Deeply nested loops
    - Function length

    Args:
        code: Python source code to analyze

    Returns:
        Detailed complexity analysis with refactoring recommendations
    """
    return f"Analyzing code complexity...\n\nCode:\n{code}"


@agent(
    model="claude-sonnet-4-20250514",
    tools=[extract_imports, extract_functions, detect_global_variables],
    system_prompt="""You are a dead code hunter specialized in finding unused and unnecessary code.

Your responsibilities:
1. Identify unused imports
2. Find unused functions (functions that are never called)
3. Detect global variables that may be unused or unnecessary
4. Recommend cleanup actions

When analyzing code:
- Use extract_imports to list all imports
- Use extract_functions to get all function definitions
- Use detect_global_variables to find global variables
- Cross-reference to identify what's actually used vs defined

Common dead code patterns:
- Imports that are never referenced
- Functions defined but never called
- Global variables that are set but never read
- Commented-out code blocks
- Debug/test code left in production

Your report should include:
- List of unused imports with recommendations to remove
- Unused functions that can be deleted
- Global variables that should be removed or made local
- Estimated cleanup impact (lines that can be removed)

Be specific about what can be safely removed and what needs verification."""
)
async def dead_code_hunter(code: str) -> str:
    """
    Hunt for dead code including unused imports, variables, and functions.

    Identifies:
    - Unused imports
    - Functions that are never called
    - Global variables that may be unnecessary

    Args:
        code: Python source code to analyze

    Returns:
        Report of dead code with specific removal recommendations
    """
    return f"Hunting for dead code...\n\nCode:\n{code}"


@agent(
    model="claude-sonnet-4-20250514",
    tools=[extract_classes, find_long_functions, calculate_complexity],
    system_prompt="""You are a pattern detector specialized in identifying anti-patterns and code smells.

Your responsibilities:
1. Detect common anti-patterns (God classes, long methods, etc.)
2. Identify code smells that indicate design problems
3. Find violations of SOLID principles
4. Recommend design improvements

When analyzing code:
- Use extract_classes to analyze class structure
- Use find_long_functions to identify bloated functions
- Use calculate_complexity to find complex code sections
- Look for patterns like:
  * God objects (classes with too many responsibilities)
  * Long parameter lists
  * Duplicate code
  * Shotgun surgery (changes require modifying many classes)
  * Feature envy (methods using more of another class than their own)

Common anti-patterns to detect:
- God Class: Classes with >10 methods or >500 lines
- Long Method: Functions with >50 lines
- Long Parameter List: Functions with >5 parameters
- High Complexity: Cyclomatic complexity >15
- Tight Coupling: Classes with many dependencies

Your report should include:
- Anti-patterns detected with severity level
- Code smells that may indicate deeper issues
- Specific examples with line numbers
- Refactoring recommendations (extract method, extract class, etc.)
- Design principles being violated

Focus on patterns that impact maintainability and extensibility."""
)
async def pattern_detector(code: str) -> str:
    """
    Detect anti-patterns and code smells in Python code.

    Analyzes for:
    - God classes (too many responsibilities)
    - Long methods and parameter lists
    - High complexity hotspots
    - SOLID principle violations

    Args:
        code: Python source code to analyze

    Returns:
        Report of detected anti-patterns with refactoring suggestions
    """
    return f"Detecting anti-patterns and code smells...\n\nCode:\n{code}"


@agent(
    model="claude-sonnet-4-20250514",
    tools=[check_naming_convention, extract_docstrings, find_type_hints],
    system_prompt="""You are a consistency checker focused on maintaining uniform style across a codebase.

Your responsibilities:
1. Check naming convention consistency
2. Verify docstring presence and format
3. Analyze type hint coverage
4. Ensure consistent code style

When analyzing code:
- Use check_naming_convention to verify naming consistency
- Use extract_docstrings to check documentation coverage
- Use find_type_hints to analyze type annotation usage
- Look for consistency in:
  * Naming patterns
  * Documentation style
  * Type annotation usage
  * Code organization

Consistency checks:
- Naming: All functions/variables should follow snake_case
- Classes: All should follow PascalCase
- Docstrings: All public functions/classes should have docstrings
- Type hints: Consistent usage across the codebase (aim for 80%+ coverage)
- Code organization: Similar patterns for similar functionality

Your report should include:
- Consistency score (0-100%)
- Specific inconsistencies with line numbers
- Recommendations to improve consistency
- Style guide compliance assessment

Focus on identifying patterns that are inconsistent with the rest of the codebase.
Be prescriptive about establishing consistent conventions."""
)
async def consistency_checker(code: str) -> str:
    """
    Check consistency of code style, naming, and documentation across codebase.

    Verifies:
    - Naming convention consistency
    - Docstring coverage and format
    - Type hint usage
    - Overall style uniformity

    Args:
        code: Python source code to analyze

    Returns:
        Consistency analysis report with improvement recommendations
    """
    return f"Checking code consistency...\n\nCode:\n{code}"
