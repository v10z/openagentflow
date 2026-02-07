"""System and file utility tools for OpenAgentFlow.

This module provides various system and file manipulation utilities including
file extension handling, filename sanitization, size formatting, pattern matching,
and configuration file parsing.
"""

import os
import re
from pathlib import Path
from typing import Dict, List
import fnmatch

from openagentflow import tool


@tool
def get_file_extension(filename: str) -> str:
    """Extract file extension from a filename.

    Args:
        filename: The filename or path to extract extension from

    Returns:
        The file extension including the dot (e.g., '.txt'), or empty string if no extension

    Examples:
        >>> get_file_extension("document.pdf")
        '.pdf'
        >>> get_file_extension("archive.tar.gz")
        '.gz'
        >>> get_file_extension("README")
        ''
    """
    return Path(filename).suffix


@tool
def change_file_extension(filename: str, new_ext: str) -> str:
    """Change the file extension of a filename.

    Args:
        filename: The original filename or path
        new_ext: The new extension (with or without leading dot)

    Returns:
        The filename with the new extension

    Examples:
        >>> change_file_extension("document.txt", ".pdf")
        'document.pdf'
        >>> change_file_extension("photo.jpg", "png")
        'photo.png'
    """
    path = Path(filename)
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return str(path.with_suffix(new_ext))


@tool
def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename to make it safe for filesystem.

    Removes or replaces characters that are invalid in Windows, macOS, and Linux filesystems.

    Args:
        filename: The filename to sanitize

    Returns:
        A sanitized filename safe for use on most filesystems

    Examples:
        >>> sanitize_filename("my/file:name*.txt")
        'my_file_name_.txt'
        >>> sanitize_filename("report<2024>.pdf")
        'report_2024_.pdf'
    """
    # Characters that are invalid in Windows filenames
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'

    # Replace invalid characters with underscore
    sanitized = re.sub(invalid_chars, '_', filename)

    # Remove leading/trailing spaces and dots (problematic on Windows)
    sanitized = sanitized.strip('. ')

    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed'

    # Truncate if too long (Windows has 255 char limit)
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        name = name[:max_length - len(ext)]
        sanitized = name + ext

    return sanitized


@tool
def human_readable_size(bytes: int) -> str:
    """Convert bytes to human readable size format (KB, MB, GB, etc.).

    Args:
        bytes: The number of bytes to convert

    Returns:
        A human-readable string representation of the size

    Examples:
        >>> human_readable_size(1024)
        '1.0 KB'
        >>> human_readable_size(1048576)
        '1.0 MB'
        >>> human_readable_size(1234567890)
        '1.15 GB'
    """
    if bytes < 0:
        raise ValueError("Bytes must be non-negative")

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
    size = float(bytes)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


@tool
def parse_size(size_str: str) -> int:
    """Parse a size string (e.g., '10MB', '1.5GB') to bytes.

    Args:
        size_str: A size string with optional unit (B, KB, MB, GB, TB)

    Returns:
        The size in bytes

    Examples:
        >>> parse_size("10MB")
        10485760
        >>> parse_size("1.5GB")
        1610612736
        >>> parse_size("512")
        512
    """
    size_str = size_str.strip().upper()

    # Define unit multipliers
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        'PB': 1024 ** 5,
        'EB': 1024 ** 6,
    }

    # Try to match number and unit
    match = re.match(r'^([\d.]+)\s*([A-Z]*)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number_str, unit = match.groups()

    try:
        number = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid number in size string: {number_str}")

    # Default to bytes if no unit specified
    if not unit:
        unit = 'B'

    if unit not in units:
        raise ValueError(f"Unknown size unit: {unit}")

    return int(number * units[unit])


@tool
def generate_tree_structure(paths: List[str]) -> str:
    """Generate an ASCII tree structure from a list of file paths.

    Args:
        paths: A list of file paths (can be relative or absolute)

    Returns:
        A string containing an ASCII tree representation

    Examples:
        >>> paths = ["src/main.py", "src/utils/helper.py", "README.md"]
        >>> print(generate_tree_structure(paths))
        .
        ├── src
        │   ├── main.py
        │   └── utils
        │       └── helper.py
        └── README.md
    """
    if not paths:
        return "."

    # Build a tree structure
    tree = {}

    for path in sorted(paths):
        parts = Path(path).parts
        current = tree

        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {}
            current = current[part]

    def build_tree_string(node: dict, prefix: str = "", is_last: bool = True) -> List[str]:
        """Recursively build tree string representation."""
        lines = []
        items = list(node.items())

        for i, (name, children) in enumerate(items):
            is_last_item = (i == len(items) - 1)

            # Determine the connectors
            if prefix == "":
                connector = ""
                lines.append(f"{name}")
            else:
                connector = "└── " if is_last_item else "├── "
                lines.append(f"{prefix}{connector}{name}")

            # Recursively add children
            if children:
                if prefix == "":
                    new_prefix = ""
                else:
                    extension = "    " if is_last_item else "│   "
                    new_prefix = prefix + extension

                lines.extend(build_tree_string(children, new_prefix, is_last_item))

        return lines

    result = ".\n" + "\n".join(build_tree_string(tree))
    return result


@tool
def glob_to_regex(glob_pattern: str) -> str:
    """Convert a glob pattern to a regular expression pattern.

    Args:
        glob_pattern: A glob pattern (e.g., '*.txt', 'test_*.py')

    Returns:
        A regular expression pattern string

    Examples:
        >>> glob_to_regex("*.txt")
        '.*\\.txt'
        >>> glob_to_regex("test_?.py")
        'test_.\\.py'
    """
    # Escape special regex characters except glob wildcards
    pattern = glob_pattern

    # Escape special regex characters
    for char in '.^$+{}[]|()':
        pattern = pattern.replace(char, '\\' + char)

    # Convert glob wildcards to regex
    pattern = pattern.replace('?', '.')  # ? matches single character
    pattern = pattern.replace('*', '.*')  # * matches any characters

    # Anchor the pattern
    pattern = '^' + pattern + '$'

    return pattern


@tool
def match_gitignore(pattern: str, path: str) -> bool:
    """Check if a path matches a gitignore pattern.

    Implements basic gitignore pattern matching rules.

    Args:
        pattern: A gitignore pattern (e.g., '*.pyc', 'node_modules/', '!important.txt')
        path: The file path to check

    Returns:
        True if the path matches the pattern, False otherwise

    Examples:
        >>> match_gitignore("*.pyc", "test.pyc")
        True
        >>> match_gitignore("node_modules/", "node_modules/package.json")
        True
        >>> match_gitignore("*.txt", "README.md")
        False
    """
    # Normalize path separators
    path = path.replace('\\', '/')
    pattern = pattern.replace('\\', '/')

    # Handle negation patterns (return False for negated patterns that match)
    if pattern.startswith('!'):
        return not match_gitignore(pattern[1:], path)

    # Remove trailing spaces
    pattern = pattern.rstrip()

    # Skip comments and empty lines
    if not pattern or pattern.startswith('#'):
        return False

    # Directory-only patterns (ending with /)
    if pattern.endswith('/'):
        pattern = pattern.rstrip('/')
        # Check if path starts with the directory
        return path.startswith(pattern + '/') or path == pattern

    # Patterns with slash must match from root
    if '/' in pattern:
        # Handle leading slash
        if pattern.startswith('/'):
            pattern = pattern[1:]
        return fnmatch.fnmatch(path, pattern)

    # Patterns without slash match against basename and full path
    basename = os.path.basename(path)

    # Match against basename or any part of the path
    if fnmatch.fnmatch(basename, pattern):
        return True

    # Also check if any component matches
    parts = path.split('/')
    for i in range(len(parts)):
        subpath = '/'.join(parts[i:])
        if fnmatch.fnmatch(subpath, pattern):
            return True

    return False


@tool
def parse_ini(ini_content: str) -> Dict[str, Dict[str, str]]:
    """Parse INI file content to a dictionary.

    Args:
        ini_content: The content of an INI file as a string

    Returns:
        A dictionary with sections as keys and section contents as nested dictionaries

    Examples:
        >>> content = "[section1]\\nkey1=value1\\n[section2]\\nkey2=value2"
        >>> parse_ini(content)
        {'section1': {'key1': 'value1'}, 'section2': {'key2': 'value2'}}
    """
    result = {}
    current_section = None

    for line in ini_content.split('\n'):
        # Remove leading/trailing whitespace
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(';') or line.startswith('#'):
            continue

        # Check for section header
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip()
            result[current_section] = {}
            continue

        # Parse key-value pairs
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            if current_section is not None:
                result[current_section][key] = value
            else:
                # Handle key-value pairs before any section
                if '' not in result:
                    result[''] = {}
                result[''][key] = value

    return result


@tool
def parse_env_file(env_content: str) -> Dict[str, str]:
    """Parse .env file content to a dictionary.

    Supports basic .env file format with comments and quoted values.

    Args:
        env_content: The content of a .env file as a string

    Returns:
        A dictionary of environment variables

    Examples:
        >>> content = "DB_HOST=localhost\\nDB_PORT=5432\\n# Comment\\nAPI_KEY='secret'"
        >>> parse_env_file(content)
        {'DB_HOST': 'localhost', 'DB_PORT': '5432', 'API_KEY': 'secret'}
    """
    result = {}

    for line in env_content.split('\n'):
        # Remove leading/trailing whitespace
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Parse key-value pairs
        if '=' in line:
            # Split only on first = to handle values with =
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if len(value) >= 2:
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

            # Handle inline comments (but not in quoted strings)
            # Only remove comments if value wasn't originally quoted
            if not (line.split('=', 1)[1].strip().startswith('"') or
                    line.split('=', 1)[1].strip().startswith("'")):
                if '#' in value:
                    value = value.split('#')[0].strip()

            # Process escape sequences in double-quoted strings
            value = value.replace('\\n', '\n').replace('\\t', '\t')

            result[key] = value

    return result
