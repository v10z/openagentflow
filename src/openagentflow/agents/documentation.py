"""Documentation agents for generating docstrings, READMEs, and API documentation."""

import ast
import re

from openagentflow import agent, tool


# Helper Tools

@tool
def generate_docstring(code: str) -> str:
    """
    Generate docstring for function/class.

    Args:
        code: Source code of function or class

    Returns:
        Generated docstring in appropriate format
    """
    try:
        tree = ast.parse(code)

        # Find the first function or class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function signature
                func_name = node.name
                args = [arg.arg for arg in node.args.args]

                # Check if return annotation exists
                returns = "Any"
                if node.returns:
                    returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else "Any"

                # Build docstring
                docstring = f'''"""
    {func_name.replace('_', ' ').title()}.

    Args:
'''
                for arg in args:
                    if arg != 'self':
                        docstring += f"        {arg}: Description of {arg}\n"

                docstring += f'''
    Returns:
        {returns}: Description of return value
    """'''
                return docstring

            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                docstring = f'''"""
    {class_name} class.

    This class provides functionality for {class_name.lower()}.

    Attributes:
        Add relevant attributes here

    Example:
        >>> obj = {class_name}()
        >>> obj.method()
    """'''
                return docstring

        return '"""Add documentation here."""'

    except Exception as e:
        return f'"""Error generating docstring: {str(e)}"""'


@tool
def infer_types(code: str) -> dict:
    """
    Infer types from code analysis.

    Args:
        code: Source code to analyze

    Returns:
        Dictionary mapping variable/function names to inferred types
    """
    type_info = {}

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Function definitions with annotations
            if isinstance(node, ast.FunctionDef):
                func_types = {
                    'name': node.name,
                    'args': {},
                    'return': None
                }

                # Argument types
                for arg in node.args.args:
                    if arg.annotation:
                        try:
                            func_types['args'][arg.arg] = ast.unparse(arg.annotation)
                        except:
                            func_types['args'][arg.arg] = 'Any'
                    else:
                        func_types['args'][arg.arg] = 'Any'

                # Return type
                if node.returns:
                    try:
                        func_types['return'] = ast.unparse(node.returns)
                    except:
                        func_types['return'] = 'Any'
                else:
                    func_types['return'] = 'Any'

                type_info[node.name] = func_types

            # Class definitions
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [],
                    'methods': []
                }

                for base in node.bases:
                    try:
                        class_info['bases'].append(ast.unparse(base))
                    except:
                        class_info['bases'].append('Unknown')

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info['methods'].append(item.name)

                type_info[node.name] = class_info

    except Exception as e:
        type_info['error'] = str(e)

    return type_info


@tool
def extract_examples(code: str) -> list[str]:
    """
    Extract usage examples from code.

    Args:
        code: Source code to extract examples from

    Returns:
        List of usage example strings
    """
    examples = []

    try:
        tree = ast.parse(code)

        # Look for docstrings with examples
        for node in ast.walk(tree):
            docstring = ast.get_docstring(node)
            if docstring:
                # Find Example: or Examples: sections
                example_pattern = r'Example[s]?:\s*\n((?:.*\n)*?)(?:\n\n|\Z)'
                matches = re.finditer(example_pattern, docstring, re.MULTILINE)

                for match in matches:
                    example_text = match.group(1).strip()
                    if example_text:
                        examples.append(example_text)

        # Look for if __name__ == '__main__' blocks
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                try:
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__'):
                        # Extract code from this block
                        example_code = ast.unparse(node) if hasattr(ast, 'unparse') else "# Main example block found"
                        examples.append(f"Main execution example:\n{example_code}")
                except:
                    pass

    except Exception as e:
        examples.append(f"Error extracting examples: {str(e)}")

    return examples if examples else ["No examples found in code"]


@tool
def generate_readme(code: str) -> str:
    """
    Generate README content from code.

    Args:
        code: Source code to document

    Returns:
        Generated README content in Markdown format
    """
    readme = "# Project Documentation\n\n"

    try:
        tree = ast.parse(code)
        module_docstring = ast.get_docstring(tree)

        if module_docstring:
            readme += f"## Overview\n\n{module_docstring}\n\n"

        # Extract classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if classes:
            readme += "## Classes\n\n"
            for cls in classes:
                class_doc = ast.get_docstring(cls)
                readme += f"### {cls.name}\n\n"
                if class_doc:
                    readme += f"{class_doc}\n\n"
                else:
                    readme += f"Class for {cls.name.lower()} functionality.\n\n"

                # List methods
                methods = [m.name for m in cls.body if isinstance(m, ast.FunctionDef)]
                if methods:
                    readme += "**Methods:**\n"
                    for method in methods:
                        readme += f"- `{method}()`\n"
                    readme += "\n"

        # Extract functions
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if functions:
            readme += "## Functions\n\n"
            for func in functions:
                func_doc = ast.get_docstring(func)
                args = ', '.join([arg.arg for arg in func.args.args])
                readme += f"### `{func.name}({args})`\n\n"
                if func_doc:
                    readme += f"{func_doc}\n\n"
                else:
                    readme += f"Function: {func.name}\n\n"

        # Installation section
        readme += "## Installation\n\n"
        readme += "```bash\n"
        readme += "pip install <package-name>\n"
        readme += "```\n\n"

        # Usage section
        readme += "## Usage\n\n"
        readme += "```python\n"
        readme += "# Import the module\n"
        readme += "from module import ClassName\n\n"
        readme += "# Create instance and use\n"
        readme += "instance = ClassName()\n"
        readme += "result = instance.method()\n"
        readme += "```\n\n"

        # License section
        readme += "## License\n\n"
        readme += "Specify your license here.\n"

    except Exception as e:
        readme += f"\nError generating README: {str(e)}\n"

    return readme


@tool
def create_changelog(changes: list[str]) -> str:
    """
    Create changelog entry from list of changes.

    Args:
        changes: List of change descriptions

    Returns:
        Formatted changelog entry
    """
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    changelog = f"## [Unreleased] - {today}\n\n"

    # Categorize changes
    added = []
    changed = []
    fixed = []
    removed = []
    deprecated = []
    security = []

    for change in changes:
        change_lower = change.lower()
        if any(word in change_lower for word in ['add', 'new', 'introduce']):
            added.append(change)
        elif any(word in change_lower for word in ['fix', 'bug', 'resolve']):
            fixed.append(change)
        elif any(word in change_lower for word in ['remove', 'delete']):
            removed.append(change)
        elif any(word in change_lower for word in ['deprecate']):
            deprecated.append(change)
        elif any(word in change_lower for word in ['security', 'vulnerability']):
            security.append(change)
        else:
            changed.append(change)

    if added:
        changelog += "### Added\n"
        for item in added:
            changelog += f"- {item}\n"
        changelog += "\n"

    if changed:
        changelog += "### Changed\n"
        for item in changed:
            changelog += f"- {item}\n"
        changelog += "\n"

    if fixed:
        changelog += "### Fixed\n"
        for item in fixed:
            changelog += f"- {item}\n"
        changelog += "\n"

    if deprecated:
        changelog += "### Deprecated\n"
        for item in deprecated:
            changelog += f"- {item}\n"
        changelog += "\n"

    if removed:
        changelog += "### Removed\n"
        for item in removed:
            changelog += f"- {item}\n"
        changelog += "\n"

    if security:
        changelog += "### Security\n"
        for item in security:
            changelog += f"- {item}\n"
        changelog += "\n"

    return changelog


@tool
def document_api(code: str) -> str:
    """
    Document API endpoints from code.

    Args:
        code: Source code containing API definitions

    Returns:
        API documentation in Markdown format
    """
    api_docs = "# API Documentation\n\n"

    try:
        tree = ast.parse(code)

        # Look for route decorators (Flask/FastAPI style)
        routes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for route decorators
                http_methods = []
                route_path = None

                for decorator in node.decorator_list:
                    decorator_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)

                    # Common patterns: @app.route, @router.get, @app.get, etc.
                    if 'route' in decorator_str.lower() or any(m in decorator_str.lower() for m in ['get', 'post', 'put', 'delete', 'patch']):
                        # Try to extract path
                        if isinstance(decorator, ast.Call):
                            if decorator.args:
                                try:
                                    route_path = ast.literal_eval(decorator.args[0])
                                except:
                                    route_path = ast.unparse(decorator.args[0]) if hasattr(ast, 'unparse') else "/path"

                        # Determine HTTP method
                        for method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']:
                            if method.lower() in decorator_str.lower():
                                http_methods.append(method)

                if route_path or http_methods:
                    routes.append({
                        'name': node.name,
                        'path': route_path or '/unknown',
                        'methods': http_methods or ['GET'],
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args if arg.arg not in ['self', 'cls']]
                    })

        if routes:
            api_docs += "## Endpoints\n\n"

            for route in routes:
                methods_str = ', '.join(route['methods'])
                api_docs += f"### {methods_str} `{route['path']}`\n\n"
                api_docs += f"**Function:** `{route['name']}()`\n\n"

                if route['docstring']:
                    api_docs += f"**Description:**\n{route['docstring']}\n\n"

                if route['args']:
                    api_docs += "**Parameters:**\n"
                    for arg in route['args']:
                        api_docs += f"- `{arg}`: Description needed\n"
                    api_docs += "\n"

                api_docs += "**Response:**\n```json\n{\n  \"status\": \"success\",\n  \"data\": {}\n}\n```\n\n"
                api_docs += "---\n\n"
        else:
            api_docs += "No API routes detected. This module may not contain API endpoint definitions.\n\n"
            api_docs += "## General API Information\n\n"
            api_docs += "Document your API endpoints, request/response formats, authentication, and error codes here.\n"

    except Exception as e:
        api_docs += f"\nError generating API documentation: {str(e)}\n"

    return api_docs


# Agent Definitions

@agent(
    name="docstring_generator",
    model="claude-sonnet-4-20250514",
    system_prompt="""You are an expert documentation generator specializing in creating high-quality docstrings.

Your role is to:
1. Analyze code structure and functionality
2. Generate clear, comprehensive docstrings following best practices
3. Use appropriate documentation style (Google, NumPy, or Sphinx)
4. Include parameter types, return types, and descriptions
5. Add usage examples where helpful
6. Ensure consistency across the codebase

Best Practices:
- Be concise but complete
- Document all parameters and return values
- Include type hints when possible
- Add examples for complex functions
- Use proper formatting and indentation
- Follow PEP 257 docstring conventions

Always analyze the code carefully and generate documentation that helps developers understand both what the code does and how to use it.""",
    tools=[generate_docstring, infer_types, extract_examples]
)
async def docstring_generator(code: str) -> str:
    """
    Generate or improve docstrings for given code.

    This agent analyzes Python code and generates comprehensive docstrings
    following best practices and documentation standards.

    Args:
        code: Source code to document

    Returns:
        Improved code with generated docstrings
    """
    pass  # ReAct loop handles execution via LLM


@agent(
    name="readme_writer",
    model="claude-sonnet-4-20250514",
    system_prompt="""You are an expert technical writer specializing in project documentation.

Your role is to:
1. Create comprehensive README files
2. Generate clear API documentation
3. Write well-organized changelogs
4. Document installation and usage instructions
5. Provide helpful examples and use cases
6. Maintain consistent documentation style

Documentation Standards:
- Use clear, accessible language
- Structure content logically (Overview, Installation, Usage, API, Examples)
- Include code examples and snippets
- Add badges and visual elements where appropriate
- Document dependencies and requirements
- Provide troubleshooting guidance
- Keep changelogs organized by version and category

Focus on creating documentation that helps users quickly understand and effectively use the project. Make it comprehensive but easy to navigate.""",
    tools=[generate_readme, create_changelog, document_api, extract_examples]
)
async def readme_writer(code: str) -> str:
    """
    Generate README, changelog, and API documentation.

    This agent creates comprehensive project documentation including
    README files, changelogs, and API documentation from source code.

    Args:
        code: Source code or change list to document

    Returns:
        Generated documentation content
    """
    pass  # ReAct loop handles execution via LLM
