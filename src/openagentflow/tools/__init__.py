"""
OpenAgentFlow Tools - 100+ Creative Python Tools

A comprehensive collection of pure Python tools for AI agents.

Categories:
- text: Text processing (extract emails, slugify, statistics, etc.)
- code: Code analysis (complexity, functions, imports, etc.)
- data: Data transformation (JSON, CSV, XML, YAML conversion)
- web: Web/HTTP utilities (URL parsing, HTML extraction)
- math_: Mathematical operations (primes, statistics, units)
- media: Media processing (colors, images, QR codes)
- datetime_: Date/time utilities (parsing, formatting, arithmetic)
- ai: AI/ML helpers (tokens, chunks, prompts, costs)
- system: System/file utilities (paths, sizes, patterns)

Usage:
    from openagentflow.tools import text, code, data

    # Extract emails from text
    emails = text.extract_emails("Contact: hello@example.com")

    # Analyze code complexity
    complexity = code.calculate_complexity(source_code)

    # Convert CSV to JSON
    json_data = data.csv_to_json(csv_content)
"""

from openagentflow.tools import text
from openagentflow.tools import code
from openagentflow.tools import data
from openagentflow.tools import web
from openagentflow.tools import math_ as math
from openagentflow.tools import media
from openagentflow.tools import datetime_ as datetime
from openagentflow.tools import ai
from openagentflow.tools import system

__all__ = [
    "text",
    "code",
    "data",
    "web",
    "math",
    "media",
    "datetime",
    "ai",
    "system",
]
