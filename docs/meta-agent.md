# JIT Meta-Agent

## Overview

The Meta-Agent creates new tools at runtime through sandboxed code generation. It enables agents to dynamically extend their capabilities by writing, validating, and registering Python functions on the fly -- without compromising host security.

## Components

### Sandbox

The `Sandbox` validates source code safety before execution. It performs AST-level static analysis to reject any code that attempts dangerous operations.

**Blocked categories:**

- **File system access** -- `open`, `os.*`, `pathlib.*`
- **Network calls** -- `socket`, `requests`, `urllib`
- **Process execution** -- `subprocess`, `os.system`, `os.popen`
- **Dynamic code execution** -- `exec`, `eval`, `compile`

```python
from openagentflow.meta import Sandbox

sandbox = Sandbox()

# Safe code passes validation
safe, reason = sandbox.validate_source(
    "def double(n: int) -> int:\n    return n * 2"
)
# safe=True, reason=None

# Dangerous code is rejected
safe, reason = sandbox.validate_source(
    "import os; os.system('rm -rf /')"
)
# safe=False, reason="Blocked import: os"
```

### ToolFactory

The `ToolFactory` creates and manages runtime tools. It combines the sandbox for validation with a registry for tool lookup and invocation.

```python
from openagentflow.meta import ToolFactory

factory = ToolFactory()

# Create a new tool from source code
spec = factory.create_tool(
    name="double",
    description="Double a number",
    source_code="def double(n: int) -> int:\n    return n * 2",
)

# Test the tool
result = factory.test_tool("double", {"n": 21})
# Returns 42

# List registered tools
for tool in factory.list_tools():
    print(f"{tool.name}: {tool.description}")
```

### ToolSpec

Each tool created by the factory is represented as a `ToolSpec`:

| Field          | Type   | Description                        |
|----------------|--------|------------------------------------|
| `name`         | `str`  | Unique tool identifier             |
| `description`  | `str`  | Human-readable description         |
| `source_code`  | `str`  | Validated Python source            |
| `parameters`   | `dict` | Inferred parameter schema          |
| `created_at`   | `str`  | ISO 8601 creation timestamp        |

## Safety Model

The sandbox uses an allowlist/blocklist approach enforced at the AST level:

1. **Import filtering** -- The AST is walked to find all `Import` and `ImportFrom` nodes. Each module name is checked against a blocklist of dangerous modules (`os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `requests`, `urllib`, `ctypes`).

2. **Builtin filtering** -- Calls to blocked builtins (`exec`, `eval`, `compile`, `open`, `__import__`, `globals`, `locals`) are detected and rejected.

3. **Attribute access filtering** -- Attribute chains like `os.system` or `pathlib.Path` are traced and blocked even when accessed through aliased imports.

4. **No runtime escape** -- Tools execute in a restricted namespace that does not include dangerous builtins. Even if static analysis were bypassed, the runtime environment prevents access to the host system.

### Extending the Blocklist

```python
sandbox = Sandbox(
    blocked_imports=["os", "sys", "custom_internal_module"],
    blocked_builtins=["exec", "eval", "breakpoint"],
)
```

## Workflow

1. An agent determines it needs a capability that no existing tool provides.
2. The agent (or an orchestrator) submits source code to the `ToolFactory`.
3. The factory passes the source through the `Sandbox` for validation.
4. If validation passes, the tool is compiled and registered.
5. The tool becomes available for invocation by any agent in the session.
6. All tool invocations are recorded in the graph trace for auditability.
