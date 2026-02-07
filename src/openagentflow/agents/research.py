"""Research agents for code analysis and optimization."""

from openagentflow import agent, tool


# Helper Tools for Dependency Research

@tool
def find_alternatives(package: str) -> list[dict]:
    """Find alternative packages to the given package.

    Args:
        package: Name of the package to find alternatives for

    Returns:
        List of alternative packages with metadata
    """
    # Simulated alternatives database
    alternatives_db = {
        "requests": [
            {"name": "httpx", "stars": 12000, "async_support": True, "description": "Modern async HTTP client"},
            {"name": "aiohttp", "stars": 14000, "async_support": True, "description": "Async HTTP client/server"},
            {"name": "urllib3", "stars": 3500, "async_support": False, "description": "HTTP library with connection pooling"},
        ],
        "pandas": [
            {"name": "polars", "stars": 25000, "async_support": False, "description": "Fast DataFrame library in Rust"},
            {"name": "dask", "stars": 12000, "async_support": True, "description": "Parallel computing library"},
            {"name": "modin", "stars": 9000, "async_support": False, "description": "Parallelized pandas"},
        ],
        "flask": [
            {"name": "fastapi", "stars": 70000, "async_support": True, "description": "Modern async web framework"},
            {"name": "starlette", "stars": 9000, "async_support": True, "description": "Lightweight ASGI framework"},
            {"name": "django", "stars": 75000, "async_support": True, "description": "Full-featured web framework"},
        ],
    }

    return alternatives_db.get(package.lower(), [
        {"name": "No alternatives found", "stars": 0, "async_support": False, "description": "Consider searching PyPI"}
    ])


@tool
def compare_libraries(lib1: str, lib2: str) -> dict:
    """Compare two libraries across various metrics.

    Args:
        lib1: First library name
        lib2: Second library name

    Returns:
        Comparison dictionary with metrics
    """
    # Simulated comparison data
    library_data = {
        "requests": {"performance": 7, "ease_of_use": 9, "community": 10, "maintenance": 8},
        "httpx": {"performance": 8, "ease_of_use": 8, "community": 7, "maintenance": 9},
        "pandas": {"performance": 6, "ease_of_use": 8, "community": 10, "maintenance": 9},
        "polars": {"performance": 10, "ease_of_use": 7, "community": 7, "maintenance": 9},
        "flask": {"performance": 7, "ease_of_use": 9, "community": 10, "maintenance": 8},
        "fastapi": {"performance": 9, "ease_of_use": 9, "community": 9, "maintenance": 10},
    }

    lib1_data = library_data.get(lib1.lower(), {"performance": 5, "ease_of_use": 5, "community": 5, "maintenance": 5})
    lib2_data = library_data.get(lib2.lower(), {"performance": 5, "ease_of_use": 5, "community": 5, "maintenance": 5})

    return {
        "library_1": {"name": lib1, "metrics": lib1_data},
        "library_2": {"name": lib2, "metrics": lib2_data},
        "recommendation": lib1 if sum(lib1_data.values()) > sum(lib2_data.values()) else lib2,
        "summary": f"Compared {lib1} and {lib2} across performance, ease of use, community support, and maintenance"
    }


@tool
def check_maintenance(package: str) -> dict:
    """Check if a package is actively maintained.

    Args:
        package: Package name to check

    Returns:
        Maintenance status information
    """
    # Simulated maintenance data
    maintenance_db = {
        "requests": {"last_update": "2 months ago", "is_maintained": True, "contributors": 500, "open_issues": 150},
        "pandas": {"last_update": "1 week ago", "is_maintained": True, "contributors": 3000, "open_issues": 3500},
        "flask": {"last_update": "3 months ago", "is_maintained": True, "contributors": 600, "open_issues": 50},
        "numpy": {"last_update": "2 weeks ago", "is_maintained": True, "contributors": 1500, "open_issues": 800},
    }

    return maintenance_db.get(package.lower(), {
        "last_update": "Unknown",
        "is_maintained": False,
        "contributors": 0,
        "open_issues": 0,
        "note": "Package not found in database"
    })


# Helper Tools for Performance Profiling

@tool
def find_bottlenecks(code: str) -> list[dict]:
    """Find performance bottlenecks in code.

    Args:
        code: Source code to analyze

    Returns:
        List of identified bottlenecks
    """
    bottlenecks = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Check for nested loops
        if 'for' in line and any('for' in lines[j] for j in range(max(0, i-5), min(len(lines), i+5)) if j != i-1):
            bottlenecks.append({
                "line": i,
                "type": "nested_loops",
                "severity": "high",
                "description": "Nested loops can cause O(n^2) or worse complexity"
            })

        # Check for inefficient operations
        if '.append(' in line and 'for' in ''.join(lines[max(0, i-3):i]):
            bottlenecks.append({
                "line": i,
                "type": "repeated_append",
                "severity": "medium",
                "description": "Repeated list.append() in loop - consider list comprehension"
            })

        # Check for string concatenation in loops
        if '+=' in line and 'str' in line.lower():
            bottlenecks.append({
                "line": i,
                "type": "string_concatenation",
                "severity": "medium",
                "description": "String concatenation in loop - use join() instead"
            })

        # Check for global lookups
        if 'global' in line:
            bottlenecks.append({
                "line": i,
                "type": "global_access",
                "severity": "low",
                "description": "Global variable access is slower than local"
            })

    return bottlenecks if bottlenecks else [{"type": "none", "description": "No obvious bottlenecks detected"}]


@tool
def suggest_optimizations(code: str) -> list[str]:
    """Suggest performance optimizations for code.

    Args:
        code: Source code to analyze

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    if 'for' in code and '.append(' in code:
        suggestions.append("Use list comprehensions instead of for loops with append()")

    if '+=' in code and ('str' in code.lower() or '"' in code or "'" in code):
        suggestions.append("Use ''.join() for string concatenation instead of +=")

    if 'range(len(' in code:
        suggestions.append("Use enumerate() instead of range(len()) for cleaner iteration")

    if code.count('for') >= 2:
        suggestions.append("Consider vectorization with NumPy or using itertools for nested loops")

    if 'def ' in code and 'return' in code:
        suggestions.append("Consider using @lru_cache for expensive function calls with repeated arguments")

    if 'import' in code and 'pandas' in code:
        suggestions.append("Use pandas vectorized operations instead of iterating over DataFrame rows")

    if not suggestions:
        suggestions.append("Code looks reasonably optimized - consider profiling with cProfile for detailed analysis")

    return suggestions


@tool
def estimate_complexity(code: str) -> dict:
    """Estimate Big-O complexity of code.

    Args:
        code: Source code to analyze

    Returns:
        Complexity estimation
    """
    lines = code.split('\n')
    loop_count = sum(1 for line in lines if 'for' in line or 'while' in line)
    nested_loops = 0

    for i, line in enumerate(lines):
        if 'for' in line or 'while' in line:
            # Check if there's another loop nearby (simple heuristic)
            context = ''.join(lines[max(0, i-5):min(len(lines), i+5)])
            nested_loops += context.count('for') + context.count('while') - 1

    if nested_loops >= 3:
        complexity = "O(n^3) or worse"
        rating = "poor"
    elif nested_loops >= 2:
        complexity = "O(n^2)"
        rating = "moderate"
    elif loop_count >= 1:
        complexity = "O(n)"
        rating = "good"
    else:
        complexity = "O(1)"
        rating = "excellent"

    return {
        "time_complexity": complexity,
        "space_complexity": "O(n)" if 'append' in code or '[]' in code else "O(1)",
        "rating": rating,
        "loop_count": loop_count,
        "nested_depth": nested_loops
    }


# Helper Tools for Best Practices Analysis

@tool
def check_practices(code: str) -> list[dict]:
    """Check code against best practices.

    Args:
        code: Source code to check

    Returns:
        List of practice violations and recommendations
    """
    issues = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Check for missing docstrings
        if line.strip().startswith('def ') and i < len(lines):
            next_line = lines[i].strip() if i < len(lines) else ""
            if not next_line.startswith('"""') and not next_line.startswith("'''"):
                issues.append({
                    "line": i,
                    "type": "missing_docstring",
                    "severity": "medium",
                    "practice": "PEP 257 - Docstring Conventions",
                    "recommendation": "Add docstring to document function purpose and parameters"
                })

        # Check for line length
        if len(line) > 100:
            issues.append({
                "line": i,
                "type": "line_too_long",
                "severity": "low",
                "practice": "PEP 8 - Line Length",
                "recommendation": "Keep lines under 88-100 characters"
            })

        # Check for bare except
        if 'except:' in line and 'except ' not in line.replace('except:', ''):
            issues.append({
                "line": i,
                "type": "bare_except",
                "severity": "high",
                "practice": "Exception Handling Best Practices",
                "recommendation": "Catch specific exceptions instead of bare except"
            })

        # Check for mutable default arguments
        if 'def ' in line and ('=[]' in line.replace(' ', '') or '={}' in line.replace(' ', '')):
            issues.append({
                "line": i,
                "type": "mutable_default_argument",
                "severity": "high",
                "practice": "Python Best Practices",
                "recommendation": "Use None as default and initialize inside function"
            })

        # Check for proper naming
        if 'def ' in line:
            func_name = line.split('def ')[1].split('(')[0]
            if func_name and func_name[0].isupper():
                issues.append({
                    "line": i,
                    "type": "naming_convention",
                    "severity": "low",
                    "practice": "PEP 8 - Naming Conventions",
                    "recommendation": "Function names should be lowercase with underscores"
                })

    return issues if issues else [{"type": "none", "description": "No best practice violations detected"}]


@tool
def suggest_improvements(code: str) -> list[str]:
    """Suggest code quality improvements.

    Args:
        code: Source code to improve

    Returns:
        List of improvement suggestions
    """
    improvements = []

    if 'import' in code:
        improvements.append("Organize imports: stdlib, third-party, local (PEP 8)")

    if code.count('\n\n\n') > 0:
        improvements.append("Use exactly two blank lines between top-level definitions")

    if 'def ' in code and 'return' in code:
        improvements.append("Consider adding type hints for better code documentation and IDE support")

    if 'if' in code and 'else' in code:
        improvements.append("Consider using guard clauses to reduce nesting depth")

    if any(word in code for word in ['TODO', 'FIXME', 'XXX', 'HACK']):
        improvements.append("Address TODO/FIXME comments before production deployment")

    if 'print(' in code:
        improvements.append("Replace print() with proper logging using the logging module")

    if 'class' in code:
        improvements.append("Follow single responsibility principle - each class should have one clear purpose")

    if not improvements:
        improvements.append("Code follows good practices - consider adding comprehensive unit tests")

    return improvements


@tool
def rate_code_quality(code: str) -> dict:
    """Rate overall code quality on a scale of 1-10.

    Args:
        code: Source code to rate

    Returns:
        Quality rating and breakdown
    """
    score = 10
    feedback = []

    # Check documentation
    has_docstrings = '"""' in code or "'''" in code
    if not has_docstrings:
        score -= 2
        feedback.append("Missing docstrings (-2)")

    # Check complexity
    complexity_data = estimate_complexity(code)
    if complexity_data['rating'] == 'poor':
        score -= 3
        feedback.append("High complexity (-3)")
    elif complexity_data['rating'] == 'moderate':
        score -= 1
        feedback.append("Moderate complexity (-1)")

    # Check best practices
    issues = check_practices(code)
    high_severity = sum(1 for issue in issues if issue.get('severity') == 'high')
    if high_severity > 0:
        score -= min(2, high_severity)
        feedback.append(f"Best practice violations (-{min(2, high_severity)})")

    # Check code length (maintainability)
    line_count = len(code.split('\n'))
    if line_count > 200:
        score -= 1
        feedback.append("Consider breaking into smaller modules (-1)")

    # Bonus for good practices
    if 'typing' in code or ': str' in code or '-> ' in code:
        score += 1
        feedback.append("Uses type hints (+1)")

    score = max(1, min(10, score))

    if score >= 8:
        rating = "excellent"
    elif score >= 6:
        rating = "good"
    elif score >= 4:
        rating = "fair"
    else:
        rating = "needs improvement"

    return {
        "score": score,
        "rating": rating,
        "feedback": feedback,
        "summary": f"Code quality rated {score}/10 - {rating}"
    }


# Research Agents

@agent(
    model="claude-sonnet-4-20250514",
    tools=[find_alternatives, compare_libraries, check_maintenance],
    system_prompt="""You are a dependency research specialist who analyzes project dependencies.

Your responsibilities:
1. Identify dependencies used in the code
2. Find alternative packages with better features or maintenance
3. Compare libraries across performance, ease of use, and community metrics
4. Check if dependencies are actively maintained

When analyzing dependencies:
- Use find_alternatives to discover replacement options
- Use compare_libraries to evaluate options side-by-side
- Use check_maintenance to verify ongoing support
- Recommend migration paths for deprecated packages

Provide specific, actionable recommendations with clear reasoning.""",
)
async def dependency_researcher(code: str) -> str:
    """Analyze code dependencies and suggest alternatives or improvements.

    This agent examines the dependencies in your code, checks their maintenance
    status, finds alternatives, and provides recommendations for better options.

    Args:
        code: Source code to analyze for dependencies

    Returns:
        Analysis report with dependency recommendations
    """
    pass  # ReAct loop handles execution via LLM


@agent(
    model="claude-sonnet-4-20250514",
    tools=[find_bottlenecks, suggest_optimizations, estimate_complexity],
    system_prompt="""You are a performance profiling expert who identifies bottlenecks and suggests optimizations.

Your responsibilities:
1. Find performance bottlenecks in code
2. Estimate computational complexity (Big-O)
3. Suggest specific optimizations for better performance
4. Identify inefficient patterns and algorithms

When analyzing performance:
- Use find_bottlenecks to locate slow code paths
- Use estimate_complexity to assess algorithmic efficiency
- Use suggest_optimizations to provide improvement strategies
- Focus on the highest-impact optimizations first

Provide measurable improvement estimates where possible.""",
)
async def performance_profiler(code: str) -> str:
    """Analyze code performance and suggest optimizations.

    This agent identifies performance bottlenecks, estimates computational complexity,
    and provides specific optimization recommendations.

    Args:
        code: Source code to profile for performance

    Returns:
        Performance analysis report with optimization suggestions
    """
    pass  # ReAct loop handles execution via LLM


@agent(
    model="claude-sonnet-4-20250514",
    tools=[check_practices, suggest_improvements, rate_code_quality],
    system_prompt="""You are a best practices advisor who reviews code quality and recommends improvements.

Your responsibilities:
1. Check code against Python best practices and PEP guidelines
2. Suggest specific improvements for code quality
3. Rate overall code quality with detailed feedback
4. Prioritize recommendations by impact

When reviewing code:
- Use check_practices to audit against coding standards
- Use suggest_improvements to generate actionable recommendations
- Use rate_code_quality to provide an overall assessment
- Focus on maintainability, readability, and correctness

Be constructive and specific. Every recommendation should include a clear reason.""",
)
async def best_practices_advisor(code: str) -> str:
    """Review code against industry best practices and provide improvement suggestions.

    This agent checks code quality, adherence to style guides and best practices,
    and provides actionable recommendations for improvement.

    Args:
        code: Source code to review for best practices

    Returns:
        Code quality report with improvement recommendations
    """
    pass  # ReAct loop handles execution via LLM
