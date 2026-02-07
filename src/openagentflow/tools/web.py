"""Web and HTTP utility tools for URL parsing, validation, and HTML processing."""

import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, quote, unquote
from typing import Optional

from openagentflow import tool


@tool
def parse_url(url: str) -> dict:
    """
    Parse URL into components (scheme, host, path, query, fragment).

    Args:
        url: The URL string to parse

    Returns:
        Dictionary with URL components: scheme, netloc, host, port, path, params, query, fragment

    Example:
        >>> parse_url("https://example.com:8080/path?key=value#section")
        {
            'scheme': 'https',
            'netloc': 'example.com:8080',
            'host': 'example.com',
            'port': 8080,
            'path': '/path',
            'params': '',
            'query': 'key=value',
            'fragment': 'section'
        }
    """
    parsed = urlparse(url)

    # Extract host and port from netloc
    host = parsed.hostname or parsed.netloc
    port = parsed.port

    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'host': host,
        'port': port,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }


@tool
def build_url(scheme: str, host: str, path: str = "", params: Optional[dict] = None) -> str:
    """
    Build URL from components.

    Args:
        scheme: URL scheme (e.g., 'http', 'https')
        host: Hostname or netloc (can include port like 'example.com:8080')
        path: URL path (default: "")
        params: Query parameters as dictionary (default: None)

    Returns:
        Complete URL string

    Example:
        >>> build_url("https", "example.com", "/api/users", {"page": "1", "limit": "10"})
        'https://example.com/api/users?page=1&limit=10'
    """
    # Ensure path starts with / if not empty
    if path and not path.startswith('/'):
        path = '/' + path

    # Build query string from params
    query = ""
    if params:
        query = urlencode(params)

    # Use urlunparse to build the complete URL
    # urlunparse expects: (scheme, netloc, path, params, query, fragment)
    url = urlunparse((scheme, host, path, '', query, ''))

    return url


@tool
def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: The URL string

    Returns:
        Domain name (hostname) without port

    Example:
        >>> extract_domain("https://www.example.com:8080/path?query=1")
        'www.example.com'
    """
    parsed = urlparse(url)
    domain = parsed.hostname if parsed.hostname else parsed.netloc.split(':')[0]
    return domain if domain else ""


@tool
def is_valid_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: The URL string to validate

    Returns:
        True if URL is valid, False otherwise

    Example:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("not a url")
        False
    """
    try:
        parsed = urlparse(url)
        # A valid URL should have at least a scheme and netloc (or path for relative URLs)
        # For absolute URLs, we expect scheme and netloc
        if parsed.scheme and parsed.netloc:
            return True
        # For relative URLs or file paths
        if parsed.path:
            # Check if it looks like a relative URL
            return True
        return False
    except Exception:
        return False


@tool
def parse_query_string(query: str) -> dict:
    """
    Parse query string into dictionary.

    Args:
        query: Query string (with or without leading '?')

    Returns:
        Dictionary of query parameters (values are lists to handle multiple values)

    Example:
        >>> parse_query_string("key1=value1&key2=value2&key1=value3")
        {'key1': ['value1', 'value3'], 'key2': ['value2']}
    """
    # Remove leading '?' if present
    if query.startswith('?'):
        query = query[1:]

    # parse_qs returns a dictionary with lists as values
    parsed = parse_qs(query)

    return parsed


@tool
def build_query_string(params: dict) -> str:
    """
    Build query string from dictionary.

    Args:
        params: Dictionary of query parameters

    Returns:
        Query string (without leading '?')

    Example:
        >>> build_query_string({"key1": "value1", "key2": "value2"})
        'key1=value1&key2=value2'
    """
    # urlencode handles the conversion and URL encoding
    # doseq=True allows passing lists as values for repeated parameters
    query_string = urlencode(params, doseq=True)
    return query_string


@tool
def extract_links(html: str) -> list[str]:
    """
    Extract all href links from HTML using regex.

    Args:
        html: HTML content as string

    Returns:
        List of URLs found in href attributes

    Example:
        >>> extract_links('<a href="http://example.com">Link</a>')
        ['http://example.com']
    """
    # Pattern to match href attributes in anchor tags
    # Matches: href="url" or href='url'
    pattern = r'href\s*=\s*["\']([^"\']+)["\']'

    links = re.findall(pattern, html, re.IGNORECASE)

    return links


@tool
def extract_images(html: str) -> list[str]:
    """
    Extract all image src URLs from HTML.

    Args:
        html: HTML content as string

    Returns:
        List of image URLs found in src attributes

    Example:
        >>> extract_images('<img src="image.jpg" /><img src="photo.png">')
        ['image.jpg', 'photo.png']
    """
    # Pattern to match src attributes in img tags
    # Matches: src="url" or src='url'
    pattern = r'<img[^>]+src\s*=\s*["\']([^"\']+)["\']'

    images = re.findall(pattern, html, re.IGNORECASE)

    return images


@tool
def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown (basic conversion).

    Supports: headings (h1-h6), paragraphs, bold, italic, links, images, lists, code, blockquotes.

    Args:
        html: HTML content as string

    Returns:
        Markdown formatted string

    Example:
        >>> html_to_markdown('<h1>Title</h1><p>Text with <strong>bold</strong></p>')
        '# Title\\n\\nText with **bold**'
    """
    text = html

    # Headings (h1-h6)
    text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h5[^>]*>(.*?)</h5>', r'##### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<h6[^>]*>(.*?)</h6>', r'###### \1\n\n', text, flags=re.IGNORECASE | re.DOTALL)

    # Bold
    text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.IGNORECASE | re.DOTALL)

    # Italic
    text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.IGNORECASE | re.DOTALL)

    # Links
    text = re.sub(r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.IGNORECASE | re.DOTALL)

    # Images
    text = re.sub(r'<img[^>]+src\s*=\s*["\']([^"\']+)["\'][^>]*alt\s*=\s*["\']([^"\']*)["\'][^>]*/?>', r'![\2](\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'<img[^>]+alt\s*=\s*["\']([^"\']*)["\'][^>]*src\s*=\s*["\']([^"\']+)["\'][^>]*/?>', r'![\1](\2)', text, flags=re.IGNORECASE)
    text = re.sub(r'<img[^>]+src\s*=\s*["\']([^"\']+)["\'][^>]*/?>', r'![](\1)', text, flags=re.IGNORECASE)

    # Code blocks
    text = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```\n', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```\n', text, flags=re.IGNORECASE | re.DOTALL)

    # Inline code
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.IGNORECASE | re.DOTALL)

    # Blockquotes
    text = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', lambda m: '\n'.join('> ' + line for line in m.group(1).strip().split('\n')) + '\n\n', text, flags=re.IGNORECASE | re.DOTALL)

    # Unordered lists
    text = re.sub(r'<ul[^>]*>(.*?)</ul>', lambda m: '\n' + re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', m.group(1), flags=re.IGNORECASE | re.DOTALL) + '\n', text, flags=re.IGNORECASE | re.DOTALL)

    # Ordered lists
    def replace_ol(match):
        items = re.findall(r'<li[^>]*>(.*?)</li>', match.group(1), flags=re.IGNORECASE | re.DOTALL)
        result = '\n'
        for i, item in enumerate(items, 1):
            result += f'{i}. {item.strip()}\n'
        return result + '\n'

    text = re.sub(r'<ol[^>]*>(.*?)</ol>', replace_ol, text, flags=re.IGNORECASE | re.DOTALL)

    # Paragraphs
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.IGNORECASE | re.DOTALL)

    # Line breaks
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Horizontal rules
    text = re.sub(r'<hr\s*/?>', '\n---\n\n', text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()

    return text


@tool
def validate_email(email: str) -> bool:
    """
    Validate email format with comprehensive rules.

    Validates according to RFC 5322 simplified rules:
    - Local part (before @) can contain alphanumeric, dots, hyphens, underscores
    - Domain part must have at least one dot
    - TLD must be at least 2 characters

    Args:
        email: Email address string to validate

    Returns:
        True if email is valid, False otherwise

    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
    """
    # Comprehensive email regex pattern
    # Pattern breakdown:
    # - Local part: alphanumeric, dots, hyphens, underscores, plus signs
    # - Must have @ symbol
    # - Domain: alphanumeric and hyphens, with at least one dot
    # - TLD: at least 2 characters
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, email):
        return False

    # Additional validation rules
    local_part, domain = email.rsplit('@', 1)

    # Local part cannot start or end with a dot
    if local_part.startswith('.') or local_part.endswith('.'):
        return False

    # Local part cannot have consecutive dots
    if '..' in local_part:
        return False

    # Domain cannot start or end with a hyphen
    if domain.startswith('-') or domain.endswith('-'):
        return False

    # Domain parts (between dots) cannot start or end with hyphen
    for part in domain.split('.'):
        if part.startswith('-') or part.endswith('-'):
            return False
        if not part:  # Empty part (consecutive dots)
            return False

    return True
