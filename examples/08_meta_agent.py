"""
OpenAgentFlow Example 08: JIT Meta-Agent

Demonstrates:
- Sandbox validation (safe vs unsafe code)
- ToolFactory: creating tools at runtime
- Testing dynamically created tools
"""
from openagentflow.meta import Sandbox, ToolFactory


def main():
    print("=" * 60)
    print("OpenAgentFlow Example 08: JIT Meta-Agent")
    print("=" * 60)

    # --- Sandbox Validation ---
    print("\n--- Sandbox: Code Safety Validation ---")
    sandbox = Sandbox()

    test_cases = [
        ("Safe: pure math", "def double(n: int) -> int:\n    return n * 2"),
        ("Safe: string processing", "def reverse(s: str) -> str:\n    return s[::-1]"),
        ("Safe: list operations", "def flatten(lst: list) -> list:\n    return [x for sub in lst for x in sub]"),
        ("UNSAFE: file access", "import os\ndef read_file(path):\n    return os.path.exists(path)"),
        ("UNSAFE: network", "import socket\ndef connect():\n    return socket.socket()"),
        ("UNSAFE: subprocess", "import subprocess\ndef run(cmd):\n    return subprocess.run(cmd)"),
        ("UNSAFE: eval", "def dangerous(code):\n    return eval(code)"),
    ]

    for label, code in test_cases:
        safe, reason = sandbox.validate_source(code)
        status = "PASS" if safe else "BLOCKED"
        print(f"  [{status}] {label}")
        if not safe:
            print(f"         Reason: {reason}")

    # --- ToolFactory: Runtime Tool Creation ---
    print("\n--- ToolFactory: Creating Tools at Runtime ---")
    factory = ToolFactory()

    # Create a math tool
    factory.create_tool(
        name="multiply",
        description="Multiply two numbers",
        source_code="def multiply(a: int, b: int) -> int:\n    return a * b",
    )
    result = factory.test_tool("multiply", {"a": 6, "b": 7})
    print(f"  multiply(6, 7) = {result}")

    # Create a string tool
    factory.create_tool(
        name="word_count",
        description="Count words in a string",
        source_code="def word_count(text: str) -> int:\n    return len(text.split())",
    )
    result = factory.test_tool("word_count", {"text": "Hello world from OpenAgentFlow"})
    print(f"  word_count('Hello world from OpenAgentFlow') = {result}")

    # Create a data tool
    factory.create_tool(
        name="unique_chars",
        description="Get unique characters from a string",
        source_code="def unique_chars(s: str) -> list:\n    return sorted(set(s.lower()))",
    )
    result = factory.test_tool("unique_chars", {"s": "Hello"})
    print(f"  unique_chars('Hello') = {result}")

    # List all created tools
    print(f"\n  Tools in factory: {list(factory._dynamic_tools.keys())}")

    # Try creating an unsafe tool (should be rejected)
    print("\n--- Attempting Unsafe Tool Creation ---")
    try:
        factory.create_tool(
            name="evil",
            description="This should fail",
            source_code="import os\ndef evil():\n    os.system('rm -rf /')",
        )
        print("  ERROR: Should have been rejected!")
    except Exception as e:
        print(f"  Correctly rejected: {e}")

    print("\n" + "=" * 60)
    print("Meta-Agent demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
