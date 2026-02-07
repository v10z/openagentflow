"""Security agents for vulnerability scanning and secrets detection."""

import re
from openagentflow import agent, tool


# Security Tools

@tool
def detect_sql_injection(code: str) -> list[dict]:
    """Find SQL injection vulnerabilities in code.

    Args:
        code: Source code to analyze

    Returns:
        List of detected SQL injection vulnerabilities
    """
    vulnerabilities = []

    # Pattern for string concatenation in SQL queries
    sql_concat_patterns = [
        r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s+.*?\+\s*\w+',
        r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s+.*?%s',
        r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s+.*?\.format\(',
        r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s+.*?f["\']',
        r'execute\s*\(\s*["\'].*?\+',
        r'cursor\.execute\s*\(\s*f["\']',
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern in sql_concat_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'SQL Injection',
                    'severity': 'HIGH',
                    'line': line_num,
                    'code': line.strip(),
                    'description': 'Potential SQL injection via string concatenation'
                })
                break

    return vulnerabilities


@tool
def find_xss_vectors(code: str) -> list[dict]:
    """Find Cross-Site Scripting (XSS) vulnerabilities.

    Args:
        code: Source code to analyze

    Returns:
        List of detected XSS vulnerabilities
    """
    vulnerabilities = []

    # Pattern for unsafe HTML rendering
    xss_patterns = [
        r'innerHTML\s*=',
        r'outerHTML\s*=',
        r'document\.write\s*\(',
        r'\.html\s*\(\s*[^)]*\+',
        r'dangerouslySetInnerHTML',
        r'<script>.*?</script>',
        r'eval\s*\(',
        r'render_template_string\s*\(',
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern in xss_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'Cross-Site Scripting (XSS)',
                    'severity': 'HIGH',
                    'line': line_num,
                    'code': line.strip(),
                    'description': 'Potential XSS vulnerability - unsafe HTML rendering'
                })
                break

    return vulnerabilities


@tool
def check_command_injection(code: str) -> list[dict]:
    """Find command injection vulnerabilities.

    Args:
        code: Source code to analyze

    Returns:
        List of detected command injection vulnerabilities
    """
    vulnerabilities = []

    # Pattern for unsafe command execution
    command_patterns = [
        r'os\.system\s*\(',
        r'subprocess\.call\s*\(\s*["\'].*?\+',
        r'subprocess\.run\s*\(\s*["\'].*?\+',
        r'subprocess\.Popen\s*\(\s*["\'].*?\+',
        r'eval\s*\(',
        r'exec\s*\(',
        r'shell\s*=\s*True',
        r'Runtime\.getRuntime\(\)\.exec',
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern in command_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'Command Injection',
                    'severity': 'CRITICAL',
                    'line': line_num,
                    'code': line.strip(),
                    'description': 'Potential command injection via unsafe command execution'
                })
                break

    return vulnerabilities


@tool
def find_hardcoded_secrets(code: str) -> list[dict]:
    """Find hardcoded secrets and passwords in code.

    Args:
        code: Source code to analyze

    Returns:
        List of detected hardcoded secrets
    """
    secrets = []

    # Pattern for hardcoded credentials
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded Password'),
        (r'passwd\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded Password'),
        (r'pwd\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded Password'),
        (r'secret\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded Secret'),
        (r'token\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded Token'),
        (r'auth\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded Auth'),
        (r'private_key\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded Private Key'),
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern, secret_type in secret_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                secrets.append({
                    'type': secret_type,
                    'severity': 'CRITICAL',
                    'line': line_num,
                    'code': line.strip(),
                    'description': f'{secret_type} found in source code'
                })
                break

    return secrets


@tool
def detect_api_keys(code: str) -> list[dict]:
    """Detect API keys and tokens in code.

    Args:
        code: Source code to analyze

    Returns:
        List of detected API keys
    """
    api_keys = []

    # Pattern for common API key formats
    api_key_patterns = [
        (r'api_key\s*=\s*["\'][^"\']{16,}["\']', 'API Key'),
        (r'apikey\s*=\s*["\'][^"\']{16,}["\']', 'API Key'),
        (r'access_token\s*=\s*["\'][^"\']{16,}["\']', 'Access Token'),
        (r'client_secret\s*=\s*["\'][^"\']{16,}["\']', 'Client Secret'),
        (r'aws_access_key_id\s*=\s*["\']AKIA[0-9A-Z]{16}["\']', 'AWS Access Key'),
        (r'sk-[a-zA-Z0-9]{32,}', 'OpenAI API Key'),
        (r'ghp_[a-zA-Z0-9]{36,}', 'GitHub Personal Access Token'),
        (r'ya29\.[a-zA-Z0-9_-]{68,}', 'Google OAuth Token'),
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern, key_type in api_key_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                api_keys.append({
                    'type': key_type,
                    'severity': 'CRITICAL',
                    'line': line_num,
                    'code': line.strip(),
                    'description': f'{key_type} exposed in source code'
                })
                break

    return api_keys


@tool
def scan_env_leaks(code: str) -> list[dict]:
    """Find environment variable leaks and misconfigurations.

    Args:
        code: Source code to analyze

    Returns:
        List of detected environment leaks
    """
    leaks = []

    # Pattern for environment variable misuse
    env_patterns = [
        (r'print\s*\(\s*os\.environ', 'Environment Variable Leak via Print'),
        (r'console\.log\s*\(\s*process\.env', 'Environment Variable Leak via Console'),
        (r'logger\.\w+\s*\(\s*os\.environ', 'Environment Variable Leak via Logger'),
        (r'DEBUG\s*=\s*True', 'Debug Mode Enabled'),
        (r'debug\s*:\s*true', 'Debug Mode Enabled'),
    ]

    lines = code.split('\n')
    for line_num, line in enumerate(lines, 1):
        for pattern, leak_type in env_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                leaks.append({
                    'type': leak_type,
                    'severity': 'MEDIUM',
                    'line': line_num,
                    'code': line.strip(),
                    'description': f'{leak_type} detected - may expose sensitive data'
                })
                break

    return leaks


# Security Agents

@agent(
    name="vulnerability_scanner",
    model="claude-sonnet-4-20250514",
    system_prompt="""You are a security vulnerability scanner specialized in detecting SQL injection,
Cross-Site Scripting (XSS), and command injection vulnerabilities in source code.

Your responsibilities:
1. Analyze code for SQL injection vulnerabilities (string concatenation in queries, unparameterized queries)
2. Detect XSS vulnerabilities (unsafe HTML rendering, innerHTML usage, unescaped user input)
3. Identify command injection risks (shell=True, os.system with user input, unsafe subprocess calls)

For each vulnerability found:
- Clearly identify the type and severity
- Explain the security risk
- Provide remediation recommendations
- Suggest secure coding alternatives

Be thorough and precise. Focus on actionable findings.""",
    tools=[detect_sql_injection, find_xss_vectors, check_command_injection]
)
async def vulnerability_scanner(code: str) -> str:
    """Scan code for security vulnerabilities including SQL injection, XSS, and command injection.

    Args:
        code: Source code to analyze for vulnerabilities

    Returns:
        Detailed vulnerability analysis report
    """
    # The agent framework will handle the actual execution
    # This docstring describes what the agent does
    pass


@agent(
    name="secrets_detector",
    model="claude-sonnet-4-20250514",
    system_prompt="""You are a secrets detection specialist focused on identifying exposed credentials,
API keys, and sensitive data in source code.

Your responsibilities:
1. Find hardcoded passwords, secrets, and authentication tokens
2. Detect API keys from various providers (AWS, OpenAI, GitHub, Google, etc.)
3. Identify environment variable leaks and misconfigurations
4. Spot private keys and certificates in code

For each secret found:
- Classify the type and severity
- Explain the security implications
- Recommend proper secret management practices
- Suggest using environment variables, secret managers, or key vaults

Be vigilant and comprehensive. Even seemingly minor exposures can lead to major breaches.""",
    tools=[find_hardcoded_secrets, detect_api_keys, scan_env_leaks]
)
async def secrets_detector(code: str) -> str:
    """Detect hardcoded secrets, API keys, and exposed credentials in code.

    Args:
        code: Source code to analyze for secrets

    Returns:
        Detailed secrets detection report
    """
    # The agent framework will handle the actual execution
    # This docstring describes what the agent does
    pass
