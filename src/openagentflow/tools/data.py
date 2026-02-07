"""Data transformation tools for OpenAgentFlow.

This module provides 15 data transformation tools for converting between
different data formats and encodings using only Python standard library.
"""

import json
import csv
import base64
import urllib.parse
import xml.etree.ElementTree as ET
from io import StringIO
from typing import Any, Dict, List
from openagentflow import tool


@tool
def json_to_csv(json_data: str) -> str:
    """Convert JSON array to CSV format.

    Args:
        json_data: JSON string containing an array of objects

    Returns:
        CSV formatted string

    Example:
        >>> json_to_csv('[{"name":"Alice","age":30},{"name":"Bob","age":25}]')
        'name,age\\nAlice,30\\nBob,25\\n'
    """
    data = json.loads(json_data)

    if not isinstance(data, list) or not data:
        raise ValueError("JSON data must be a non-empty array of objects")

    output = StringIO()

    # Get headers from first object
    fieldnames = list(data[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

    return output.getvalue()


@tool
def csv_to_json(csv_data: str) -> str:
    """Convert CSV to JSON array format.

    Args:
        csv_data: CSV formatted string

    Returns:
        JSON string containing an array of objects

    Example:
        >>> csv_to_json('name,age\\nAlice,30\\nBob,25')
        '[{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]'
    """
    input_data = StringIO(csv_data)
    reader = csv.DictReader(input_data)

    rows = list(reader)
    return json.dumps(rows, indent=2)


@tool
def flatten_json(json_data: str) -> dict:
    """Flatten nested JSON structure into a single-level dictionary.

    Args:
        json_data: JSON string with nested structure

    Returns:
        Flattened dictionary with dot-separated keys

    Example:
        >>> flatten_json('{"a":{"b":{"c":1}}}')
        {'a.b.c': 1}
    """
    data = json.loads(json_data)

    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        """Recursively flatten nested dictionary."""
        items = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                new_key = f"{parent_key}[{idx}]"
                items.extend(_flatten(value, new_key).items())
        else:
            items.append((parent_key, obj))

        return dict(items)

    return _flatten(data)


@tool
def unflatten_json(flat_dict: dict) -> dict:
    """Unflatten a dictionary with dot-separated keys into nested structure.

    Args:
        flat_dict: Dictionary with dot-separated keys

    Returns:
        Nested dictionary structure

    Example:
        >>> unflatten_json({'a.b.c': 1})
        {'a': {'b': {'c': 1}}}
    """
    result = {}

    for key, value in flat_dict.items():
        parts = key.split('.')
        current = result

        for i, part in enumerate(parts[:-1]):
            # Handle array indices
            if '[' in part:
                base_key = part[:part.index('[')]
                if base_key not in current:
                    current[base_key] = []
                current = current[base_key]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Set the final value
        final_key = parts[-1]
        if '[' in final_key:
            base_key = final_key[:final_key.index('[')]
            if base_key not in current:
                current[base_key] = []
        else:
            current[final_key] = value

    return result


@tool
def xml_to_dict(xml_string: str) -> dict:
    """Parse XML string to dictionary structure.

    Args:
        xml_string: XML formatted string

    Returns:
        Dictionary representation of XML

    Example:
        >>> xml_to_dict('<root><name>Alice</name><age>30</age></root>')
        {'root': {'name': 'Alice', 'age': '30'}}
    """
    def _parse_element(element: ET.Element) -> Any:
        """Recursively parse XML element to dict."""
        result = {}

        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib

        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # No children
                return element.text.strip()
            result['#text'] = element.text.strip()

        # Add children
        for child in element:
            child_data = _parse_element(child)

            if child.tag in result:
                # Convert to list if multiple elements with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result if result else (element.text or '')

    root = ET.fromstring(xml_string)
    return {root.tag: _parse_element(root)}


@tool
def dict_to_xml(data: dict, root: str = "root") -> str:
    """Convert dictionary to XML string.

    Args:
        data: Dictionary to convert
        root: Root element name (default: "root")

    Returns:
        XML formatted string

    Example:
        >>> dict_to_xml({'name': 'Alice', 'age': 30})
        '<root><name>Alice</name><age>30</age></root>'
    """
    def _build_element(parent: ET.Element, data: Any, tag: str = None) -> None:
        """Recursively build XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == '@attributes':
                    parent.attrib.update(value)
                elif key == '#text':
                    parent.text = str(value)
                else:
                    child = ET.SubElement(parent, key)
                    _build_element(child, value)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, tag or 'item')
                _build_element(child, item)
        else:
            parent.text = str(data)

    # Handle case where data has single root key
    if len(data) == 1 and not root == "root":
        root_key = list(data.keys())[0]
        root_element = ET.Element(root)
        _build_element(root_element, data[root_key])
    elif len(data) == 1:
        root_key = list(data.keys())[0]
        root_element = ET.Element(root_key)
        _build_element(root_element, data[root_key])
    else:
        root_element = ET.Element(root)
        _build_element(root_element, data)

    return ET.tostring(root_element, encoding='unicode')


@tool
def yaml_to_json(yaml_string: str) -> str:
    """Convert simple YAML to JSON format.

    Basic YAML parser supporting key-value pairs, lists, and nested structures.
    Does not support all YAML features.

    Args:
        yaml_string: YAML formatted string

    Returns:
        JSON formatted string

    Example:
        >>> yaml_to_json('name: Alice\\nage: 30')
        '{"name": "Alice", "age": 30}'
    """
    def _parse_value(value: str) -> Any:
        """Parse YAML value to Python type."""
        value = value.strip()

        # Boolean
        if value.lower() in ('true', 'yes', 'on'):
            return True
        if value.lower() in ('false', 'no', 'off'):
            return False

        # Null
        if value.lower() in ('null', 'none', '~', ''):
            return None

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String (remove quotes if present)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]

        return value

    def _parse_yaml(lines: List[str], start_indent: int = 0) -> Any:
        """Recursively parse YAML lines."""
        result = {}
        current_key = None
        i = 0

        while i < len(lines):
            line = lines[i]

            if not line.strip() or line.strip().startswith('#'):
                i += 1
                continue

            indent = len(line) - len(line.lstrip())

            if indent < start_indent:
                break

            if indent > start_indent:
                i += 1
                continue

            # List item
            if line.strip().startswith('- '):
                if not isinstance(result, list):
                    result = []

                item_value = line.strip()[2:]
                if ':' in item_value:
                    # Object in list
                    nested_lines = [item_value]
                    i += 1
                    while i < len(lines):
                        next_indent = len(lines[i]) - len(lines[i].lstrip())
                        if next_indent > indent:
                            nested_lines.append(lines[i])
                            i += 1
                        else:
                            break
                    result.append(_parse_yaml(nested_lines, indent + 2))
                    continue
                else:
                    result.append(_parse_value(item_value))

            # Key-value pair
            elif ':' in line:
                key, _, value = line.strip().partition(':')
                key = key.strip()
                value = value.strip()

                if not value:
                    # Nested object
                    nested_lines = []
                    i += 1
                    while i < len(lines):
                        next_indent = len(lines[i]) - len(lines[i].lstrip())
                        if lines[i].strip() and next_indent > indent:
                            nested_lines.append(lines[i])
                            i += 1
                        elif lines[i].strip():
                            break
                        else:
                            i += 1

                    if nested_lines:
                        result[key] = _parse_yaml(nested_lines, indent + 2)
                    else:
                        result[key] = None
                    continue
                else:
                    result[key] = _parse_value(value)

            i += 1

        return result

    lines = yaml_string.split('\n')
    parsed = _parse_yaml(lines)
    return json.dumps(parsed, indent=2)


@tool
def json_to_yaml(json_string: str) -> str:
    """Convert JSON to YAML format.

    Args:
        json_string: JSON formatted string

    Returns:
        YAML formatted string

    Example:
        >>> json_to_yaml('{"name": "Alice", "age": 30}')
        'name: Alice\\nage: 30\\n'
    """
    data = json.loads(json_string)

    def _to_yaml(obj: Any, indent: int = 0) -> str:
        """Recursively convert object to YAML."""
        lines = []
        prefix = '  ' * indent

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(_to_yaml(value, indent + 1))
                else:
                    # Format value
                    if value is None:
                        formatted_value = 'null'
                    elif isinstance(value, bool):
                        formatted_value = 'true' if value else 'false'
                    elif isinstance(value, str):
                        # Quote strings with special characters
                        if any(c in value for c in [':', '#', '-', '[', ']', '{', '}']):
                            formatted_value = f'"{value}"'
                        else:
                            formatted_value = value
                    else:
                        formatted_value = str(value)

                    lines.append(f"{prefix}{key}: {formatted_value}")

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(_to_yaml(item, indent + 1))
                else:
                    # Format value
                    if item is None:
                        formatted_value = 'null'
                    elif isinstance(item, bool):
                        formatted_value = 'true' if item else 'false'
                    elif isinstance(item, str):
                        if any(c in item for c in [':', '#', '-', '[', ']', '{', '}']):
                            formatted_value = f'"{item}"'
                        else:
                            formatted_value = item
                    else:
                        formatted_value = str(item)

                    lines.append(f"{prefix}- {formatted_value}")

        return '\n'.join(lines)

    return _to_yaml(data) + '\n'


@tool
def base64_encode(data: str) -> str:
    """Encode string to base64.

    Args:
        data: String to encode

    Returns:
        Base64 encoded string

    Example:
        >>> base64_encode('Hello World')
        'SGVsbG8gV29ybGQ='
    """
    encoded_bytes = base64.b64encode(data.encode('utf-8'))
    return encoded_bytes.decode('utf-8')


@tool
def base64_decode(data: str) -> str:
    """Decode base64 string.

    Args:
        data: Base64 encoded string

    Returns:
        Decoded string

    Example:
        >>> base64_decode('SGVsbG8gV29ybGQ=')
        'Hello World'
    """
    decoded_bytes = base64.b64decode(data.encode('utf-8'))
    return decoded_bytes.decode('utf-8')


@tool
def url_encode(text: str) -> str:
    """URL encode a string.

    Args:
        text: String to encode

    Returns:
        URL encoded string

    Example:
        >>> url_encode('hello world?')
        'hello%20world%3F'
    """
    return urllib.parse.quote(text)


@tool
def url_decode(text: str) -> str:
    """Decode URL encoded string.

    Args:
        text: URL encoded string

    Returns:
        Decoded string

    Example:
        >>> url_decode('hello%20world%3F')
        'hello world?'
    """
    return urllib.parse.unquote(text)


@tool
def hex_encode(text: str) -> str:
    """Convert text to hexadecimal representation.

    Args:
        text: String to convert

    Returns:
        Hexadecimal string

    Example:
        >>> hex_encode('Hello')
        '48656c6c6f'
    """
    return text.encode('utf-8').hex()


@tool
def hex_decode(hex_string: str) -> str:
    """Convert hexadecimal string to text.

    Args:
        hex_string: Hexadecimal string

    Returns:
        Decoded text

    Example:
        >>> hex_decode('48656c6c6f')
        'Hello'
    """
    return bytes.fromhex(hex_string).decode('utf-8')


@tool
def binary_to_text(binary: str) -> str:
    """Convert binary string to text.

    Args:
        binary: Binary string (e.g., '01001000 01100101')

    Returns:
        Decoded text

    Example:
        >>> binary_to_text('01001000 01100101 01101100 01101100 01101111')
        'Hello'
    """
    # Remove spaces and split into 8-bit chunks
    binary = binary.replace(' ', '')

    if len(binary) % 8 != 0:
        raise ValueError("Binary string length must be a multiple of 8")

    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        chars.append(chr(int(byte, 2)))

    return ''.join(chars)
