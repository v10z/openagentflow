"""
Math and Science Tools

A collection of mathematical and scientific computation tools.
All implementations are pure Python with no external dependencies.
"""

from __future__ import annotations

from openagentflow.core.tool import tool


@tool
def calculate_expression(expr: str) -> float:
    """
    Safely evaluate a mathematical expression without using eval().

    Supports: +, -, *, /, //, %, **, (, )
    Examples: "2 + 3 * 4", "(5 + 3) ** 2", "10 / 2 - 1"

    Args:
        expr: Mathematical expression as a string

    Returns:
        Result of the calculation as a float
    """
    # Remove whitespace
    expr = expr.replace(" ", "")

    def parse_number(s: str, pos: int) -> tuple[float, int]:
        """Parse a number starting at position pos."""
        start = pos
        if pos < len(s) and s[pos] in "+-":
            pos += 1
        while pos < len(s) and (s[pos].isdigit() or s[pos] == "."):
            pos += 1
        if start == pos:
            raise ValueError(f"Expected number at position {pos}")
        return float(s[start:pos]), pos

    def parse_factor(s: str, pos: int) -> tuple[float, int]:
        """Parse a factor (number or parenthesized expression)."""
        if pos < len(s) and s[pos] == "(":
            pos += 1
            result, pos = parse_expr(s, pos)
            if pos >= len(s) or s[pos] != ")":
                raise ValueError("Mismatched parentheses")
            return result, pos + 1
        return parse_number(s, pos)

    def parse_power(s: str, pos: int) -> tuple[float, int]:
        """Parse power operations (right-associative)."""
        left, pos = parse_factor(s, pos)
        if pos < len(s) and s[pos:pos+2] == "**":
            right, pos = parse_power(s, pos + 2)
            return left ** right, pos
        return left, pos

    def parse_term(s: str, pos: int) -> tuple[float, int]:
        """Parse multiplication, division, modulo."""
        left, pos = parse_power(s, pos)
        while pos < len(s):
            if s[pos:pos+2] == "//":
                right, pos = parse_power(s, pos + 2)
                left = left // right
            elif s[pos] == "*":
                right, pos = parse_power(s, pos + 1)
                left = left * right
            elif s[pos] == "/":
                right, pos = parse_power(s, pos + 1)
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            elif s[pos] == "%":
                right, pos = parse_power(s, pos + 1)
                left = left % right
            else:
                break
        return left, pos

    def parse_expr(s: str, pos: int) -> tuple[float, int]:
        """Parse addition and subtraction."""
        left, pos = parse_term(s, pos)
        while pos < len(s):
            if s[pos] == "+":
                right, pos = parse_term(s, pos + 1)
                left = left + right
            elif s[pos] == "-":
                right, pos = parse_term(s, pos + 1)
                left = left - right
            else:
                break
        return left, pos

    result, pos = parse_expr(expr, 0)
    if pos != len(expr):
        raise ValueError(f"Unexpected character at position {pos}")
    return result


@tool
def prime_factors(n: int) -> list[int]:
    """
    Find all prime factors of a number.

    Args:
        n: Integer to factorize

    Returns:
        List of prime factors in ascending order
    """
    if n < 2:
        return []

    factors = []

    # Check for 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Check odd numbers from 3 onwards
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2

    # If n is still greater than 1, it's prime
    if n > 1:
        factors.append(n)

    return factors


@tool
def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n: Integer to check

    Returns:
        True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2

    return True


@tool
def gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor using Euclidean algorithm.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Greatest common divisor
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


@tool
def lcm(a: int, b: int) -> int:
    """
    Calculate the least common multiple.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Least common multiple
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


@tool
def fibonacci(n: int) -> list[int]:
    """
    Generate Fibonacci sequence up to n terms.

    Args:
        n: Number of terms to generate

    Returns:
        List of Fibonacci numbers
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])

    return fib


@tool
def factorial(n: int) -> int:
    """
    Calculate factorial of a number.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")

    result = 1
    for i in range(2, n + 1):
        result *= i

    return result


@tool
def statistics_summary(numbers: list[float]) -> dict:
    """
    Calculate statistical summary of a list of numbers.

    Args:
        numbers: List of numbers to analyze

    Returns:
        Dictionary with mean, median, mode, std_dev, variance
    """
    if not numbers:
        return {
            "mean": 0.0,
            "median": 0.0,
            "mode": None,
            "std_dev": 0.0,
            "variance": 0.0,
            "count": 0
        }

    n = len(numbers)

    # Mean
    mean = sum(numbers) / n

    # Median
    sorted_nums = sorted(numbers)
    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]

    # Mode (most frequent value)
    from collections import Counter
    freq = Counter(numbers)
    max_freq = max(freq.values())
    modes = [num for num, count in freq.items() if count == max_freq]
    mode = modes[0] if len(modes) == 1 else modes

    # Variance and Standard Deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "std_dev": std_dev,
        "variance": variance,
        "count": n
    }


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between different units of measurement.

    Supports:
    - Length: m, km, mi, ft, in
    - Weight: kg, lb, oz, g
    - Temperature: C, F, K

    Args:
        value: Numeric value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value
    """
    # Length conversions (to meters)
    length_to_m = {
        "m": 1.0,
        "km": 1000.0,
        "mi": 1609.34,
        "ft": 0.3048,
        "in": 0.0254,
    }

    # Weight conversions (to kilograms)
    weight_to_kg = {
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
        "oz": 0.0283495,
    }

    # Temperature conversions
    if from_unit in ["C", "F", "K"] and to_unit in ["C", "F", "K"]:
        # Convert to Celsius first
        if from_unit == "F":
            celsius = (value - 32) * 5/9
        elif from_unit == "K":
            celsius = value - 273.15
        else:
            celsius = value

        # Convert from Celsius to target
        if to_unit == "F":
            return celsius * 9/5 + 32
        elif to_unit == "K":
            return celsius + 273.15
        else:
            return celsius

    # Length conversions
    if from_unit in length_to_m and to_unit in length_to_m:
        meters = value * length_to_m[from_unit]
        return meters / length_to_m[to_unit]

    # Weight conversions
    if from_unit in weight_to_kg and to_unit in weight_to_kg:
        kilograms = value * weight_to_kg[from_unit]
        return kilograms / weight_to_kg[to_unit]

    raise ValueError(f"Cannot convert from '{from_unit}' to '{to_unit}'")


@tool
def roman_to_int(roman: str) -> int:
    """
    Convert Roman numeral to integer.

    Supports: I, V, X, L, C, D, M

    Args:
        roman: Roman numeral string (e.g., "XIV", "MCMXCIV")

    Returns:
        Integer value
    """
    roman_values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    roman = roman.upper()
    result = 0
    prev_value = 0

    # Process from right to left
    for char in reversed(roman):
        if char not in roman_values:
            raise ValueError(f"Invalid Roman numeral character: {char}")

        value = roman_values[char]

        # If current value is less than previous, subtract it
        if value < prev_value:
            result -= value
        else:
            result += value

        prev_value = value

    return result
