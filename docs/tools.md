# Tools Reference

OpenAgentFlow includes **99 pure Python tools** across **9 categories**, designed for use within agentic workflows. All tools use the `@tool` decorator, which auto-generates JSON Schema from type hints, making them immediately compatible with any LLM function-calling interface.

**Key design principles:**

- Zero external dependencies (standard library only)
- Deterministic outputs (no network calls, no side effects)
- Full type annotations with auto-generated JSON Schema
- Consistent error handling via return values

---

## Table of Contents

1. [Text Processing (15 tools)](#text-processing)
2. [Code Analysis (15 tools)](#code-analysis)
3. [Data Transformation (15 tools)](#data-transformation)
4. [Web/HTTP (10 tools)](#webhttp)
5. [Math/Science (10 tools)](#mathscience)
6. [Media (8 tools)](#media)
7. [Date/Time (8 tools)](#datetime)
8. [AI/ML Helpers (8 tools)](#aiml-helpers)
9. [System/File (10 tools)](#systemfile)
10. [Creating Custom Tools](#creating-custom-tools)

---

## Text Processing

**Module:** `openagentflow.tools.text`

15 tools for string manipulation, encoding, pattern extraction, and text analysis.

### Quick Start

```python
from openagentflow.tools import text

text.extract_emails("Contact: hello@example.com")  # ['hello@example.com']
text.text_to_slug("Hello World!")                   # 'hello-world'
text.text_to_morse("SOS")                           # '... --- ...'
text.detect_language("Bonjour le monde")            # 'french'
text.find_palindromes("A man a plan a canal")       # ['a', 'a', 'a']
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `extract_emails(text)` | Extract all email addresses from a string using regex pattern matching. |
| `extract_urls(text)` | Extract all URLs (http/https) from a string. |
| `text_to_slug(text)` | Convert text to a URL-safe slug (lowercase, hyphens, no special characters). |
| `text_to_morse(text)` | Encode a string into International Morse Code. |
| `morse_to_text(morse)` | Decode a Morse Code string back into readable text. |
| `detect_language(text)` | Detect the likely language of a text sample using character frequency and common word analysis. |
| `count_words(text)` | Count the total number of words in a string, with optional unique word count. |
| `find_palindromes(text)` | Find all palindromic words within a body of text. |
| `caesar_cipher(text, shift)` | Apply a Caesar cipher shift to alphabetic characters. |
| `rot13(text)` | Apply ROT13 encoding (Caesar cipher with shift of 13). |
| `reverse_text(text)` | Reverse the characters in a string. |
| `text_to_binary(text)` | Convert a text string to its binary (8-bit) representation. |
| `binary_to_text(binary)` | Convert a binary string (space-separated bytes) back to text. |
| `levenshtein_distance(s1, s2)` | Calculate the Levenshtein (edit) distance between two strings. |
| `hamming_distance(s1, s2)` | Calculate the Hamming distance between two equal-length strings. |

### Examples

```python
# Encoding and decoding
text.caesar_cipher("HELLO", 3)          # 'KHOOR'
text.rot13("Hello")                     # 'Uryyb'
text.text_to_binary("Hi")              # '01001000 01101001'
text.binary_to_text("01001000 01101001")  # 'Hi'

# Distance metrics
text.levenshtein_distance("kitten", "sitting")  # 3
text.hamming_distance("karolin", "kathrin")      # 3

# Extraction
text.extract_urls("Visit https://example.com")  # ['https://example.com']
text.count_words("one two three")                # 3
text.reverse_text("hello")                       # 'olleh'
```

---

## Code Analysis

**Module:** `openagentflow.tools.code`

15 tools for static analysis, code quality metrics, and source code introspection.

### Quick Start

```python
from openagentflow.tools import code

code.calculate_complexity(source)      # Cyclomatic complexity score
code.extract_functions(source)         # List of functions with signatures
code.find_todos(source)                # TODO/FIXME/XXX comments
code.find_magic_numbers(source)        # Hardcoded numeric literals
code.check_naming_convention(source)   # PEP 8 naming violations
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `calculate_complexity(source)` | Calculate the cyclomatic complexity of a Python source string. |
| `extract_functions(source)` | Extract all function definitions with their names, arguments, and line numbers. |
| `extract_classes(source)` | Extract all class definitions with their names, base classes, and methods. |
| `extract_imports(source)` | Extract all import statements, returning module names and aliases. |
| `find_todos(source)` | Find all TODO, FIXME, HACK, and XXX comments with line numbers. |
| `find_magic_numbers(source)` | Detect hardcoded numeric literals (excluding 0 and 1) that should be named constants. |
| `check_naming_convention(source)` | Check identifiers against PEP 8 naming conventions (snake_case functions, PascalCase classes). |
| `count_lines_of_code(source)` | Count total lines, code lines, comment lines, and blank lines. |
| `find_long_functions(source, max_lines)` | Identify functions that exceed a specified line count threshold. |
| `find_deeply_nested(source, max_depth)` | Find code blocks with excessive nesting depth. |
| `find_duplicate_code(source)` | Detect duplicate or near-duplicate code blocks within a source file. |
| `extract_docstrings(source)` | Extract all docstrings from modules, classes, and functions. |
| `find_global_variables(source)` | Identify global variable assignments at module scope. |
| `detect_code_language(source)` | Detect the programming language of a source snippet based on syntax patterns. |
| `calculate_maintainability_index(source)` | Compute a maintainability index score based on Halstead volume, complexity, and LOC. |

### Examples

```python
source = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    # TODO: add memoization
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

code.calculate_complexity(source)
# {'function': 'fibonacci', 'complexity': 3}

code.extract_functions(source)
# [{'name': 'fibonacci', 'args': ['n'], 'line': 2}]

code.find_todos(source)
# [{'line': 4, 'type': 'TODO', 'text': 'add memoization'}]

code.extract_docstrings(source)
# [{'type': 'function', 'name': 'fibonacci', 'docstring': 'Calculate fibonacci number.'}]

code.count_lines_of_code(source)
# {'total': 7, 'code': 4, 'comments': 1, 'blank': 2}
```

---

## Data Transformation

**Module:** `openagentflow.tools.data`

15 tools for converting between data formats, encoding/decoding, and structural transformations.

### Quick Start

```python
from openagentflow.tools import data

data.json_to_csv(json_str)     # Convert JSON array to CSV
data.csv_to_json(csv_str)      # Convert CSV to JSON array
data.flatten_json(nested)      # Flatten nested JSON to dot-notation keys
data.yaml_to_json(yaml_str)    # YAML to JSON string
data.xml_to_dict(xml_str)      # XML to Python dictionary
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `json_to_csv(json_str)` | Convert a JSON array of objects to CSV format. |
| `csv_to_json(csv_str)` | Convert a CSV string to a JSON array of objects. |
| `flatten_json(nested)` | Flatten a nested JSON structure using dot-notation keys. |
| `unflatten_json(flat)` | Restore a flat dot-notation dictionary to nested JSON structure. |
| `yaml_to_json(yaml_str)` | Convert a YAML string to JSON format. |
| `json_to_yaml(json_str)` | Convert a JSON string to YAML format. |
| `xml_to_dict(xml_str)` | Parse an XML string into a Python dictionary. |
| `dict_to_xml(data, root)` | Convert a Python dictionary to an XML string. |
| `base64_encode(text)` | Encode a string to Base64. |
| `base64_decode(encoded)` | Decode a Base64 string back to plain text. |
| `url_encode(text)` | Percent-encode a string for safe use in URLs. |
| `url_decode(encoded)` | Decode a percent-encoded URL string. |
| `json_schema_validate(data, schema)` | Validate a data structure against a JSON Schema definition. |
| `diff_texts(text1, text2)` | Generate a unified diff between two text strings. |
| `merge_dicts(dict1, dict2)` | Deep merge two dictionaries, with `dict2` values taking precedence. |

### Examples

```python
# Format conversion
data.json_to_csv('[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]')
# 'name,age\nAlice,30\nBob,25'

data.csv_to_json("name,age\nAlice,30\nBob,25")
# [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]

# Flattening and unflattening
data.flatten_json({"a": {"b": {"c": 1}}, "d": 2})
# {'a.b.c': 1, 'd': 2}

data.unflatten_json({"a.b.c": 1, "d": 2})
# {'a': {'b': {'c': 1}}, 'd': 2}

# Encoding
data.base64_encode("Hello, World!")   # 'SGVsbG8sIFdvcmxkIQ=='
data.base64_decode("SGVsbG8sIFdvcmxkIQ==")  # 'Hello, World!'
data.url_encode("hello world&foo=bar")  # 'hello%20world%26foo%3Dbar'

# Validation
schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
data.json_schema_validate({"name": "Alice"}, schema)  # {'valid': True}

# Deep merge
data.merge_dicts({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}, "e": 4})
# {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
```

---

## Web/HTTP

**Module:** `openagentflow.tools.web`

10 tools for URL manipulation, HTML processing, and HTTP-related utilities.

### Quick Start

```python
from openagentflow.tools import web

web.parse_url("https://example.com/path?q=1")  # URL components dict
web.extract_links(html)                         # All href links
web.html_to_markdown(html)                      # Convert HTML to Markdown
web.validate_email("user@example.com")          # Email format validation
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `parse_url(url)` | Decompose a URL into its components: scheme, host, port, path, query params, and fragment. |
| `build_url(scheme, host, path, params)` | Construct a URL from individual components. |
| `extract_links(html)` | Extract all `<a href="...">` links from an HTML string. |
| `extract_meta_tags(html)` | Extract all `<meta>` tag attributes from an HTML document head. |
| `html_to_text(html)` | Strip HTML tags and return plain text content. |
| `html_to_markdown(html)` | Convert an HTML string to Markdown format. |
| `validate_email(email)` | Validate an email address format using RFC 5322 pattern matching. |
| `parse_user_agent(ua_string)` | Parse a User-Agent string into browser, OS, and device components. |
| `generate_curl(method, url, headers, body)` | Generate a cURL command string from request parameters. |
| `parse_cookies(cookie_string)` | Parse a `Cookie` or `Set-Cookie` header string into a dictionary. |

### Examples

```python
# URL handling
web.parse_url("https://example.com:8080/api/v1?key=abc&limit=10")
# {'scheme': 'https', 'host': 'example.com', 'port': 8080,
#  'path': '/api/v1', 'params': {'key': 'abc', 'limit': '10'}}

web.build_url("https", "api.example.com", "/v2/users", {"page": "1"})
# 'https://api.example.com/v2/users?page=1'

# HTML processing
html = '<html><head><meta name="description" content="A site"></head>'
html += '<body><a href="/about">About</a><a href="/contact">Contact</a></body></html>'

web.extract_links(html)
# ['/about', '/contact']

web.extract_meta_tags(html)
# [{'name': 'description', 'content': 'A site'}]

web.html_to_text("<p>Hello <b>world</b></p>")
# 'Hello world'

# Email validation
web.validate_email("user@example.com")   # True
web.validate_email("not-an-email")       # False

# cURL generation
web.generate_curl("POST", "https://api.example.com/data",
                  headers={"Content-Type": "application/json"},
                  body='{"key": "value"}')
# "curl -X POST 'https://api.example.com/data' -H 'Content-Type: application/json' -d '{\"key\": \"value\"}'"

# Cookie parsing
web.parse_cookies("session=abc123; theme=dark; lang=en")
# {'session': 'abc123', 'theme': 'dark', 'lang': 'en'}
```

---

## Math/Science

**Module:** `openagentflow.tools.math`

10 tools for numeric computation, number theory, statistics, and unit conversion.

### Quick Start

```python
from openagentflow.tools import math

math.prime_factors(84)           # [2, 2, 3, 7]
math.is_prime(17)                # True
math.fibonacci(10)               # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
math.statistics_summary([1,2,3]) # mean, median, mode, std_dev
math.convert_units(100, "km", "miles")  # 62.137...
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `is_prime(n)` | Test whether a positive integer is prime. |
| `prime_factors(n)` | Compute the complete prime factorization of an integer. |
| `gcd(a, b)` | Calculate the greatest common divisor of two integers. |
| `lcm(a, b)` | Calculate the least common multiple of two integers. |
| `fibonacci(n)` | Generate the first `n` numbers in the Fibonacci sequence. |
| `factorial(n)` | Compute the factorial of a non-negative integer. |
| `statistics_summary(numbers)` | Compute mean, median, mode, standard deviation, variance, min, and max. |
| `convert_units(value, from_unit, to_unit)` | Convert between common units (length, weight, temperature, volume, etc.). |
| `evaluate_expression(expr)` | Safely evaluate a mathematical expression string (no code execution). |
| `matrix_multiply(a, b)` | Multiply two matrices represented as lists of lists. |

### Examples

```python
# Number theory
math.is_prime(97)          # True
math.prime_factors(360)    # [2, 2, 2, 3, 3, 5]
math.gcd(48, 18)           # 6
math.lcm(12, 8)            # 24
math.factorial(10)         # 3628800

# Sequences
math.fibonacci(8)          # [0, 1, 1, 2, 3, 5, 8, 13]

# Statistics
math.statistics_summary([4, 8, 15, 16, 23, 42])
# {'mean': 18.0, 'median': 15.5, 'mode': 4, 'std_dev': 12.84,
#  'variance': 164.89, 'min': 4, 'max': 42}

# Unit conversion
math.convert_units(100, "celsius", "fahrenheit")  # 212.0
math.convert_units(1, "kg", "lbs")                # 2.20462

# Safe expression evaluation
math.evaluate_expression("2 ** 10 + sqrt(144)")   # 1036.0

# Matrix operations
math.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# [[19, 22], [43, 50]]
```

---

## Media

**Module:** `openagentflow.tools.media`

8 tools for color conversion, image dimension calculations, and media data utilities.

### Quick Start

```python
from openagentflow.tools import media

media.color_hex_to_rgb("#FF5733")    # (255, 87, 51)
media.color_rgb_to_hex(255, 87, 51)  # '#FF5733'
media.aspect_ratio(1920, 1080)       # '16:9'
media.generate_qr_data("https://...")  # QR code matrix data
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `color_hex_to_rgb(hex_color)` | Convert a hex color string to an RGB tuple. |
| `color_rgb_to_hex(r, g, b)` | Convert RGB integer values to a hex color string. |
| `color_rgb_to_hsl(r, g, b)` | Convert RGB values to HSL (Hue, Saturation, Lightness). |
| `color_hsl_to_rgb(h, s, l)` | Convert HSL values back to RGB. |
| `aspect_ratio(width, height)` | Calculate the simplified aspect ratio of given dimensions. |
| `resize_dimensions(width, height, max_width, max_height)` | Calculate new dimensions that fit within bounds while preserving aspect ratio. |
| `generate_qr_data(text)` | Generate QR code data matrix for a given text string. |
| `image_to_ascii_dimensions(width, height, max_cols)` | Calculate the grid dimensions needed for ASCII art representation of an image. |

### Examples

```python
# Color conversions
media.color_hex_to_rgb("#3498DB")      # (52, 152, 219)
media.color_rgb_to_hex(52, 152, 219)   # '#3498DB'
media.color_rgb_to_hsl(52, 152, 219)   # (204, 70, 53)
media.color_hsl_to_rgb(204, 70, 53)    # (52, 152, 219)

# Dimension calculations
media.aspect_ratio(2560, 1440)         # '16:9'
media.aspect_ratio(1080, 1080)         # '1:1'

media.resize_dimensions(3000, 2000, max_width=1200, max_height=800)
# (1200, 800)

media.resize_dimensions(800, 2400, max_width=1000, max_height=1000)
# (333, 1000)

# ASCII art grid sizing
media.image_to_ascii_dimensions(1920, 1080, max_cols=80)
# {'cols': 80, 'rows': 22}
```

---

## Date/Time

**Module:** `openagentflow.tools.datetime`

8 tools for date parsing, formatting, arithmetic, and business day calculations.

### Quick Start

```python
from openagentflow.tools import datetime

datetime.parse_date("Jan 15, 2025")                       # '2025-01-15'
datetime.date_difference("2025-01-01", "2025-12-31")      # 364
datetime.get_weekday("2025-01-15")                         # 'Wednesday'
datetime.timestamp_to_date(1704067200)                     # '2024-01-01T00:00:00'
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `parse_date(date_string)` | Parse a date string in various formats and return an ISO 8601 date. |
| `format_date(iso_date, fmt)` | Format an ISO 8601 date string into a specified output format. |
| `date_difference(date1, date2)` | Calculate the number of days between two dates. |
| `add_duration(date, days, months, years)` | Add a duration to a date and return the resulting date. |
| `get_weekday(date)` | Return the day of the week for a given date. |
| `is_business_day(date)` | Check whether a date falls on a weekday (Monday through Friday). |
| `timestamp_to_date(timestamp)` | Convert a Unix timestamp to an ISO 8601 date string. |
| `date_to_timestamp(date)` | Convert an ISO 8601 date string to a Unix timestamp. |

### Examples

```python
# Parsing various formats
datetime.parse_date("January 15, 2025")    # '2025-01-15'
datetime.parse_date("15/01/2025")          # '2025-01-15'
datetime.parse_date("2025.01.15")          # '2025-01-15'

# Formatting
datetime.format_date("2025-01-15", "%B %d, %Y")  # 'January 15, 2025'
datetime.format_date("2025-01-15", "%m/%d/%Y")    # '01/15/2025'

# Arithmetic
datetime.date_difference("2025-01-01", "2025-03-01")  # 59
datetime.add_duration("2025-01-15", days=30)           # '2025-02-14'
datetime.add_duration("2025-01-15", months=3)          # '2025-04-15'

# Business day checks
datetime.is_business_day("2025-01-15")  # True  (Wednesday)
datetime.is_business_day("2025-01-18")  # False (Saturday)

# Timestamp conversions
datetime.date_to_timestamp("2025-01-01")  # 1735689600
datetime.timestamp_to_date(1735689600)    # '2025-01-01T00:00:00'
```

---

## AI/ML Helpers

**Module:** `openagentflow.tools.ai`

8 tools for token counting, text chunking, prompt formatting, and cost estimation for LLM workflows.

### Quick Start

```python
from openagentflow.tools import ai

ai.count_tokens("Hello world", model="gpt-4")     # Token count estimate
ai.split_into_chunks(long_text, chunk_size=1000)   # Text chunking
ai.extract_keywords(text, top_n=10)                # Key term extraction
ai.estimate_cost(1000, 500, "gpt-4")               # API cost estimate
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `count_tokens(text, model)` | Estimate the token count of a text string for a given model. |
| `split_into_chunks(text, chunk_size, overlap)` | Split text into overlapping chunks of a specified token size, suitable for RAG pipelines. |
| `extract_keywords(text, top_n)` | Extract the most significant keywords from a text using TF-based scoring. |
| `cosine_similarity(vec1, vec2)` | Calculate the cosine similarity between two numeric vectors. |
| `estimate_cost(input_tokens, output_tokens, model)` | Estimate the API cost for a request based on token counts and model pricing. |
| `format_prompt(template, variables)` | Render a prompt template by substituting variable placeholders. |
| `truncate_to_tokens(text, max_tokens, model)` | Truncate text to fit within a maximum token budget for a given model. |
| `detect_pii(text)` | Scan text for personally identifiable information (emails, phone numbers, SSNs, credit card numbers). |

### Examples

```python
# Token management
ai.count_tokens("The quick brown fox jumps over the lazy dog.", model="gpt-4")
# 10

ai.truncate_to_tokens("A very long document...", max_tokens=100, model="gpt-4")
# 'A very long document...'  (truncated to fit)

# Chunking for RAG
chunks = ai.split_into_chunks(long_document, chunk_size=500, overlap=50)
# [{'text': '...', 'start': 0, 'end': 500},
#  {'text': '...', 'start': 450, 'end': 950}, ...]

# Keyword extraction
ai.extract_keywords("Machine learning is a subset of artificial intelligence...", top_n=5)
# ['machine', 'learning', 'artificial', 'intelligence', 'subset']

# Vector similarity
ai.cosine_similarity([1, 0, 1], [1, 1, 0])
# 0.5

# Cost estimation
ai.estimate_cost(input_tokens=2000, output_tokens=500, model="gpt-4")
# {'input_cost': 0.06, 'output_cost': 0.03, 'total_cost': 0.09}

# Prompt templates
ai.format_prompt(
    "Summarize the following {doc_type}: {content}",
    {"doc_type": "article", "content": "..."}
)
# 'Summarize the following article: ...'

# PII detection
ai.detect_pii("Contact John at john@example.com or 555-123-4567")
# [{'type': 'email', 'value': 'john@example.com', 'start': 16, 'end': 32},
#  {'type': 'phone', 'value': '555-123-4567', 'start': 36, 'end': 48}]
```

---

## System/File

**Module:** `openagentflow.tools.system`

10 tools for filesystem utilities, hashing, ID generation, and environment handling.

### Quick Start

```python
from openagentflow.tools import system

system.human_readable_size(1048576)     # '1.00 MB'
system.sanitize_filename("file?.txt")   # 'file.txt'
system.glob_to_regex("*.py")            # Regex pattern string
system.parse_env_file(env_content)      # Dict of env vars
```

### Complete Tool List

| Tool | Description |
|------|-------------|
| `human_readable_size(size_bytes)` | Convert a byte count to a human-readable string (KB, MB, GB, etc.). |
| `sanitize_filename(filename)` | Remove or replace characters that are invalid in filenames. |
| `generate_uuid()` | Generate a new UUID v4 string. |
| `hash_string(text, algorithm)` | Compute a hash digest of a string using the specified algorithm (md5, sha1, sha256, sha512). |
| `glob_to_regex(pattern)` | Convert a glob pattern (e.g., `*.py`) to an equivalent regular expression. |
| `parse_env_file(content)` | Parse a `.env` file content string into a dictionary of key-value pairs. |
| `tree_directory(path, max_depth)` | Generate an ASCII tree representation of a directory structure. |
| `get_file_info(path)` | Return metadata about a file: size, modification time, extension, and MIME type. |
| `compare_versions(v1, v2)` | Compare two semantic version strings, returning -1, 0, or 1. |
| `generate_password(length, uppercase, digits, special)` | Generate a cryptographically random password with configurable character sets. |

### Examples

```python
# File size formatting
system.human_readable_size(0)              # '0 B'
system.human_readable_size(1024)           # '1.00 KB'
system.human_readable_size(1048576)        # '1.00 MB'
system.human_readable_size(5368709120)     # '5.00 GB'

# Filename safety
system.sanitize_filename('my file?.txt')        # 'my file.txt'
system.sanitize_filename('report <2025>.pdf')   # 'report 2025.pdf'

# Hashing
system.hash_string("hello", "sha256")
# 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

# UUID generation
system.generate_uuid()
# 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'

# Glob to regex
system.glob_to_regex("src/**/*.py")
# 'src/.*/[^/]*\\.py$'

# Env file parsing
system.parse_env_file("DB_HOST=localhost\nDB_PORT=5432\n# comment\nAPI_KEY=secret")
# {'DB_HOST': 'localhost', 'DB_PORT': '5432', 'API_KEY': 'secret'}

# Version comparison
system.compare_versions("1.2.3", "1.2.4")   # -1
system.compare_versions("2.0.0", "2.0.0")   # 0
system.compare_versions("3.1.0", "2.9.9")   # 1

# Password generation
system.generate_password(length=16, uppercase=True, digits=True, special=True)
# 'aX9#kL2$mP7!nQ4@'

# Directory tree
system.tree_directory("/project", max_depth=2)
# /project
# +-- src/
# |   +-- main.py
# |   +-- utils.py
# +-- tests/
# |   +-- test_main.py
# +-- README.md
```

---

## Creating Custom Tools

All built-in tools use the `@tool` decorator, and you can create your own tools the same way. The decorator auto-generates a JSON Schema from your function's type hints and docstring.

### Basic Custom Tool

```python
from openagentflow import tool

@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate the final price after applying a percentage discount.

    Args:
        price: The original price.
        discount_percent: The discount percentage (0-100).

    Returns:
        The discounted price.
    """
    return price * (1 - discount_percent / 100)
```

This automatically generates the following JSON Schema:

```json
{
  "name": "calculate_discount",
  "description": "Calculate the final price after applying a percentage discount.",
  "parameters": {
    "type": "object",
    "properties": {
      "price": {
        "type": "number",
        "description": "The original price."
      },
      "discount_percent": {
        "type": "number",
        "description": "The discount percentage (0-100)."
      }
    },
    "required": ["price", "discount_percent"]
  }
}
```

### Custom Tool with Optional Parameters

```python
from openagentflow import tool
from typing import Optional

@tool
def search_records(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None
) -> list:
    """Search records by query string.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        category: Optional category filter.

    Returns:
        A list of matching records.
    """
    results = perform_search(query, category=category)
    return results[:max_results]
```

### Custom Tool with Complex Types

```python
from openagentflow import tool
from typing import List, Dict

@tool
def analyze_dataset(
    data: List[Dict[str, float]],
    columns: List[str],
    normalize: bool = False
) -> Dict[str, Dict[str, float]]:
    """Analyze numeric columns in a dataset.

    Args:
        data: List of data rows as dictionaries.
        columns: Column names to analyze.
        normalize: Whether to normalize values to 0-1 range.

    Returns:
        Per-column statistics including mean, min, and max.
    """
    stats = {}
    for col in columns:
        values = [row[col] for row in data if col in row]
        stats[col] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return stats
```

### Registering Custom Tools with an Agent

```python
from openagentflow import Agent, tool

@tool
def lookup_inventory(product_id: str) -> dict:
    """Look up current inventory for a product.

    Args:
        product_id: The product identifier.

    Returns:
        Inventory details including quantity and warehouse location.
    """
    return {"product_id": product_id, "quantity": 42, "warehouse": "A3"}

agent = Agent(
    model="gpt-4",
    tools=[lookup_inventory],
)

response = agent.run("How many units of SKU-1234 are in stock?")
```

### Guidelines for Writing Tools

- **Use clear type hints.** The `@tool` decorator relies on them to generate accurate schemas. Prefer `str`, `int`, `float`, `bool`, `list`, `dict`, and their `typing` equivalents.
- **Write a docstring.** The first line becomes the tool description. `Args:` sections become parameter descriptions.
- **Keep tools focused.** Each tool should do one thing well. Compose multiple tools in an agent workflow rather than building monolithic functions.
- **Return serializable data.** Tool return values must be JSON-serializable (strings, numbers, lists, dicts, booleans, and None).
- **Handle errors gracefully.** Return descriptive error messages instead of raising exceptions, so the LLM can reason about failures.
- **Avoid side effects when possible.** Pure functions are easier to test, cache, and reason about.
