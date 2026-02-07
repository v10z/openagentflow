"""
OpenAgentFlow Example 01: Tools

Demonstrates:
- Creating custom tools with @tool
- Auto-generated JSON Schema from type hints
- Using built-in tools (text, code, math, data)
- Tool specs and metadata
"""
import json
from openagentflow import tool
from openagentflow.tools import text, code, math, data


# --- Custom Tools ---

@tool
def analyze_text(content: str) -> dict:
    """Analyze text and return statistics."""
    words = content.split()
    sentences = content.split(".")
    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "unique_words": len(set(w.lower() for w in words)),
    }


@tool
def transform_data(items: list[dict], sort_by: str = "name") -> list[dict]:
    """Sort a list of dictionaries by a given key."""
    return sorted(items, key=lambda x: x.get(sort_by, ""))


def main():
    print("=" * 60)
    print("OpenAgentFlow Example 01: Tools")
    print("=" * 60)

    # Custom tool usage
    print("\n--- Custom Tool: analyze_text ---")
    result = analyze_text("The quick brown fox jumps over the lazy dog. It was a fine day.")
    print(f"Result: {json.dumps(result, indent=2)}")

    # Tool spec (auto-generated from type hints)
    print(f"\nTool name: {analyze_text._tool_spec.name}")
    print(f"Description: {analyze_text._tool_spec.description}")
    print(f"Input schema: {json.dumps(analyze_text._tool_spec.input_schema, indent=2)}")

    # Custom tool with default params
    print("\n--- Custom Tool: transform_data ---")
    items = [{"name": "Charlie", "age": 30}, {"name": "Alice", "age": 25}, {"name": "Bob", "age": 35}]
    sorted_items = transform_data(items, sort_by="name")
    print(f"Sorted by name: {json.dumps(sorted_items, indent=2)}")

    # Built-in tools
    print("\n--- Built-in Tools: Text ---")
    print(f"Extract emails: {text.extract_emails('Contact bob@example.com or alice@test.org')}")
    print(f"Text to slug: {text.text_to_slug('Hello World! This is OpenAgentFlow')}")
    print(f"Morse code: {text.text_to_morse('SOS')}")
    print(f"Detect language: {text.detect_language('Bonjour le monde')}")

    print("\n--- Built-in Tools: Code ---")
    sample_code = '''
def fibonacci(n):
    # TODO: Add memoization
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def unused_function():
    x = 42
    pass
'''
    print(f"Complexity: {code.calculate_complexity(sample_code)}")
    print(f"TODOs: {code.find_todos(sample_code)}")
    print(f"Functions: {[f['name'] for f in code.extract_functions(sample_code)]}")

    print("\n--- Built-in Tools: Math ---")
    print(f"Is 17 prime? {math.is_prime(17)}")
    print(f"Prime factors of 84: {math.prime_factors(84)}")
    print(f"Fibonacci(10): {math.fibonacci(10)}")
    stats = math.statistics_summary([10, 20, 30, 40, 50])
    print(f"Statistics: mean={stats['mean']}, median={stats['median']}")

    print("\n--- Built-in Tools: Data ---")
    json_str = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
    csv_result = data.json_to_csv(json_str)
    print(f"JSON to CSV:\n{csv_result}")

    print("\n" + "=" * 60)
    print("All tools executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
