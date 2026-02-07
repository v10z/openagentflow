"""
Code Analysis Tools - 15 Pure Python Tools Using AST

A comprehensive collection of code analysis tools for Python source code.
All implementations use the `ast` module for parsing and analysis.

Tools:
1. count_lines_of_code - Count total, blank, comment, and code lines
2. extract_functions - Extract function definitions with signatures
3. extract_classes - Extract class definitions with methods
4. extract_imports - Extract all import statements
5. find_todos - Find TODO, FIXME, XXX, HACK comments
6. calculate_complexity - Calculate cyclomatic complexity
7. find_duplicate_code - Detect duplicate code blocks
8. extract_string_literals - Extract all string literals
9. find_magic_numbers - Find hardcoded numbers
10. check_naming_convention - Check PEP8 naming conventions
11. find_long_functions - Find functions exceeding line limit
12. detect_global_variables - Detect global variables
13. find_nested_loops - Find deeply nested loops
14. extract_docstrings - Extract all docstrings
15. find_type_hints - Analyze type hint coverage
"""

import ast
import re
from typing import Any
from collections import defaultdict

from openagentflow.core.tool import tool


@tool
def count_lines_of_code(code: str) -> dict:
    """
    Count total, blank, comment, and code lines in Python source code.

    Args:
        code: Python source code as string

    Returns:
        Dictionary with counts for total, blank, comment, and code lines
    """
    lines = code.split('\n')
    total = len(lines)
    blank = 0
    comment = 0
    code_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
        elif stripped.startswith('#'):
            comment += 1
        else:
            # Check if line has inline comment
            if '#' in stripped:
                # Don't count strings with # as comments
                in_string = False
                quote_char = None
                for i, char in enumerate(stripped):
                    if char in ('"', "'") and (i == 0 or stripped[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                    elif char == '#' and not in_string:
                        break
            code_lines += 1

    return {
        'total': total,
        'blank': blank,
        'comment': comment,
        'code': code_lines
    }


@tool
def extract_functions(code: str) -> list[dict]:
    """
    Extract function definitions with their signatures from Python code.

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing function information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    functions = []

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Extract parameters
            params = []
            for arg in node.args.args:
                annotation = ast.unparse(arg.annotation) if arg.annotation else None
                params.append({
                    'name': arg.arg,
                    'annotation': annotation
                })

            # Extract return type
            return_type = ast.unparse(node.returns) if node.returns else None

            # Extract decorators
            decorators = [ast.unparse(dec) for dec in node.decorator_list]

            # Extract docstring
            docstring = ast.get_docstring(node)

            functions.append({
                'name': node.name,
                'line': node.lineno,
                'params': params,
                'return_type': return_type,
                'decorators': decorators,
                'docstring': docstring,
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            })

            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

    visitor = FunctionVisitor()
    visitor.visit(tree)

    return functions


@tool
def extract_classes(code: str) -> list[dict]:
    """
    Extract class definitions with their methods from Python code.

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing class information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    classes = []

    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # Extract base classes
            bases = [ast.unparse(base) for base in node.bases]

            # Extract decorators
            decorators = [ast.unparse(dec) for dec in node.decorator_list]

            # Extract docstring
            docstring = ast.get_docstring(node)

            # Extract methods
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        'name': item.name,
                        'line': item.lineno,
                        'is_async': isinstance(item, ast.AsyncFunctionDef),
                        'is_property': any(
                            isinstance(dec, ast.Name) and dec.id == 'property'
                            for dec in item.decorator_list
                        )
                    })

            classes.append({
                'name': node.name,
                'line': node.lineno,
                'bases': bases,
                'decorators': decorators,
                'docstring': docstring,
                'methods': methods
            })

            self.generic_visit(node)

    visitor = ClassVisitor()
    visitor.visit(tree)

    return classes


@tool
def extract_imports(code: str) -> list[str]:
    """
    Extract all import statements from Python code.

    Args:
        code: Python source code as string

    Returns:
        List of import statements as strings
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f'Error: Syntax error: {str(e)}']

    imports = []

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.asname:
                    imports.append(f'import {alias.name} as {alias.asname}')
                else:
                    imports.append(f'import {alias.name}')

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ''
            level = '.' * node.level

            for alias in node.names:
                if alias.asname:
                    imports.append(f'from {level}{module} import {alias.name} as {alias.asname}')
                else:
                    imports.append(f'from {level}{module} import {alias.name}')

    visitor = ImportVisitor()
    visitor.visit(tree)

    return imports


@tool
def find_todos(code: str) -> list[dict]:
    """
    Find TODO, FIXME, XXX, HACK comments in Python code.

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing todo information
    """
    todos = []
    keywords = ['TODO', 'FIXME', 'XXX', 'HACK']

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        # Check for comments
        if '#' in line:
            comment_start = line.find('#')
            comment = line[comment_start+1:].strip()

            for keyword in keywords:
                if keyword in comment.upper():
                    todos.append({
                        'line': line_num,
                        'type': keyword,
                        'text': comment,
                        'full_line': line.strip()
                    })
                    break

    return todos


@tool
def calculate_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity of Python code.

    McCabe's cyclomatic complexity measures the number of linearly
    independent paths through code. Higher values indicate more complex code.

    Args:
        code: Python source code as string

    Returns:
        Cyclomatic complexity score
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1

    complexity = 1  # Base complexity

    class ComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.complexity = 1

        def visit_If(self, node: ast.If) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            self.complexity += 1
            self.generic_visit(node)

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            # Each additional condition adds complexity
            if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):
                self.complexity += len(node.values) - 1
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            self.complexity += 1
            self.generic_visit(node)

    visitor = ComplexityVisitor()
    visitor.visit(tree)

    return visitor.complexity


@tool
def find_duplicate_code(code: str, min_lines: int = 5) -> list[dict]:
    """
    Detect duplicate code blocks in Python source.

    Args:
        code: Python source code as string
        min_lines: Minimum number of lines for a duplicate block

    Returns:
        List of dictionaries containing duplicate code information
    """
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    duplicates = []

    # Simple approach: look for identical consecutive line sequences
    seen_blocks = {}

    for i in range(len(lines) - min_lines + 1):
        block = '\n'.join(lines[i:i+min_lines])

        if block in seen_blocks:
            duplicates.append({
                'first_occurrence': seen_blocks[block],
                'second_occurrence': i + 1,
                'lines': min_lines,
                'code': block
            })
        else:
            seen_blocks[block] = i + 1

    return duplicates


@tool
def extract_string_literals(code: str) -> list[str]:
    """
    Extract all string literals from Python code.

    Args:
        code: Python source code as string

    Returns:
        List of string literals found in the code
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f'Error: Syntax error: {str(e)}']

    strings = []

    class StringVisitor(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str):
                strings.append(node.value)
            self.generic_visit(node)


    visitor = StringVisitor()
    visitor.visit(tree)

    return strings


@tool
def find_magic_numbers(code: str) -> list[dict]:
    """
    Find hardcoded numbers (magic numbers) in Python code.

    Excludes common acceptable values like 0, 1, -1, 2, 10, 100.

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing magic number information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    # Common acceptable values
    acceptable = {0, 1, -1, 2, 10, 100}
    magic_numbers = []

    class NumberVisitor(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, (int, float)) and node.value not in acceptable:
                magic_numbers.append({
                    'value': node.value,
                    'line': node.lineno,
                    'col': node.col_offset
                })
            self.generic_visit(node)


    visitor = NumberVisitor()
    visitor.visit(tree)

    return magic_numbers


@tool
def check_naming_convention(code: str) -> list[dict]:
    """
    Check PEP8 naming conventions in Python code.

    Checks for:
    - Functions and variables should be snake_case
    - Classes should be PascalCase
    - Constants should be UPPER_CASE

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing naming violations
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    violations = []

    def is_snake_case(name: str) -> bool:
        return name.islower() or '_' in name and name.replace('_', '').islower()

    def is_pascal_case(name: str) -> bool:
        return name[0].isupper() and '_' not in name

    def is_upper_case(name: str) -> bool:
        return name.isupper() or name.replace('_', '').isupper()

    class NamingVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if not node.name.startswith('_') and not is_snake_case(node.name):
                violations.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'issue': 'Should be snake_case'
                })
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if not node.name.startswith('_') and not is_snake_case(node.name):
                violations.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'issue': 'Should be snake_case'
                })
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if not is_pascal_case(node.name):
                violations.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'issue': 'Should be PascalCase'
                })
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            # Check for module-level constants
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Simple heuristic: all uppercase suggests it should be a constant
                    if target.id.isupper() and len(target.id) > 1:
                        # This is intentionally a constant, which is good
                        pass
                    elif '_' not in target.id and target.id[0].isupper():
                        violations.append({
                            'type': 'variable',
                            'name': target.id,
                            'line': node.lineno,
                            'issue': 'Should be snake_case (or UPPER_CASE for constants)'
                        })
            self.generic_visit(node)

    visitor = NamingVisitor()
    visitor.visit(tree)

    return violations


@tool
def find_long_functions(code: str, max_lines: int = 50) -> list[dict]:
    """
    Find functions that exceed the specified line limit.

    Args:
        code: Python source code as string
        max_lines: Maximum allowed lines for a function

    Returns:
        List of dictionaries containing long function information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    long_functions = []

    class FunctionLengthVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Calculate function length
            if hasattr(node, 'end_lineno') and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
            else:
                # Fallback for older Python versions
                length = 0
                if node.body:
                    last_node = node.body[-1]
                    if hasattr(last_node, 'lineno'):
                        length = last_node.lineno - node.lineno + 1

            if length > max_lines:
                long_functions.append({
                    'name': node.name,
                    'start_line': node.lineno,
                    'length': length,
                    'max_allowed': max_lines
                })

            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if hasattr(node, 'end_lineno') and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
            else:
                length = 0
                if node.body:
                    last_node = node.body[-1]
                    if hasattr(last_node, 'lineno'):
                        length = last_node.lineno - node.lineno + 1

            if length > max_lines:
                long_functions.append({
                    'name': node.name,
                    'start_line': node.lineno,
                    'length': length,
                    'max_allowed': max_lines
                })

            self.generic_visit(node)

    visitor = FunctionLengthVisitor()
    visitor.visit(tree)

    return long_functions


@tool
def detect_global_variables(code: str) -> list[str]:
    """
    Detect global variables in Python code.

    Args:
        code: Python source code as string

    Returns:
        List of global variable names
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f'Error: Syntax error: {str(e)}']

    global_vars = []

    # Get module-level assignments
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Exclude constants (all uppercase)
                    if not target.id.isupper():
                        global_vars.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                if not node.target.id.isupper():
                    global_vars.append(node.target.id)

    return global_vars


@tool
def find_nested_loops(code: str, max_depth: int = 3) -> list[dict]:
    """
    Find deeply nested loops in Python code.

    Args:
        code: Python source code as string
        max_depth: Maximum allowed nesting depth

    Returns:
        List of dictionaries containing nested loop information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    nested_loops = []

    class LoopNestingVisitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_stack = []

        def check_loop(self, node: ast.AST, loop_type: str) -> None:
            self.loop_stack.append((node.lineno, loop_type))
            depth = len(self.loop_stack)

            if depth > max_depth:
                nested_loops.append({
                    'line': node.lineno,
                    'depth': depth,
                    'max_allowed': max_depth,
                    'type': loop_type,
                    'nesting_path': [lt for _, lt in self.loop_stack]
                })

            self.generic_visit(node)
            self.loop_stack.pop()

        def visit_For(self, node: ast.For) -> None:
            self.check_loop(node, 'for')

        def visit_While(self, node: ast.While) -> None:
            self.check_loop(node, 'while')

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.check_loop(node, 'async for')

    visitor = LoopNestingVisitor()
    visitor.visit(tree)

    return nested_loops


@tool
def extract_docstrings(code: str) -> list[dict]:
    """
    Extract all docstrings from Python code.

    Args:
        code: Python source code as string

    Returns:
        List of dictionaries containing docstring information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [{'error': f'Syntax error: {str(e)}'}]

    docstrings = []

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docstrings.append({
            'type': 'module',
            'name': '<module>',
            'line': 1,
            'docstring': module_doc
        })

    class DocstringVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'docstring': doc
                })
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append({
                    'type': 'async_function',
                    'name': node.name,
                    'line': node.lineno,
                    'docstring': doc
                })
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'docstring': doc
                })
            self.generic_visit(node)

    visitor = DocstringVisitor()
    visitor.visit(tree)

    return docstrings


@tool
def find_type_hints(code: str) -> dict:
    """
    Analyze type hint coverage in Python code.

    Args:
        code: Python source code as string

    Returns:
        Dictionary with type hint statistics
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'error': f'Syntax error: {str(e)}'}

    stats = {
        'total_functions': 0,
        'functions_with_return_type': 0,
        'total_parameters': 0,
        'parameters_with_type': 0,
        'functions': []
    }

    class TypeHintVisitor(ast.NodeVisitor):
        def analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            stats['total_functions'] += 1

            # Check return type
            has_return = node.returns is not None
            if has_return:
                stats['functions_with_return_type'] += 1

            # Check parameter types
            params_with_type = 0
            total_params = len(node.args.args)

            for arg in node.args.args:
                stats['total_parameters'] += 1
                if arg.annotation:
                    stats['parameters_with_type'] += 1
                    params_with_type += 1

            stats['functions'].append({
                'name': node.name,
                'line': node.lineno,
                'has_return_type': has_return,
                'total_params': total_params,
                'params_with_type': params_with_type,
                'type_hint_coverage': (params_with_type / total_params * 100) if total_params > 0 else 100
            })

            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.analyze_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.analyze_function(node)

    visitor = TypeHintVisitor()
    visitor.visit(tree)

    # Calculate overall coverage
    if stats['total_functions'] > 0:
        stats['return_type_coverage'] = (
            stats['functions_with_return_type'] / stats['total_functions'] * 100
        )
    else:
        stats['return_type_coverage'] = 0

    if stats['total_parameters'] > 0:
        stats['parameter_type_coverage'] = (
            stats['parameters_with_type'] / stats['total_parameters'] * 100
        )
    else:
        stats['parameter_type_coverage'] = 100

    return stats
