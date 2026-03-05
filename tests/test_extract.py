from persona_bench.runner.extract import ensure_indented, extract_body, extract_code

# --- extract_code tests ---


def test_extract_raw_code():
    raw = "    return x + 1"
    assert extract_code(raw) == "    return x + 1"


def test_extract_from_markdown_fences():
    raw = "```python\n    return x + 1\n```"
    assert extract_code(raw) == "    return x + 1"


def test_extract_from_fences_no_language():
    raw = "```\n    return x + 1\n```"
    assert extract_code(raw) == "    return x + 1"


def test_extract_with_surrounding_text():
    raw = "Here is the code:\n```python\n    return x + 1\n```\nDone."
    assert extract_code(raw) == "    return x + 1"


def test_extract_multiline():
    raw = "```python\n    if x > 0:\n        return x\n    return -x\n```"
    assert extract_code(raw) == "    if x > 0:\n        return x\n    return -x"


def test_extract_single_backtick_wrapping():
    raw = "`return x + 1\nreturn y`"
    assert extract_code(raw) == "return x + 1\nreturn y"


# --- extract_body tests ---


def test_extract_body_complete_function():
    """Extract body from a complete function with docstring."""
    code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b'
    imports, body = extract_body(code)
    assert imports == ""
    assert body == "    return a + b"


def test_extract_body_multiline_docstring():
    """Extract body from function with multi-line docstring."""
    code = (
        "def parse(s: str) -> list:\n"
        '    """Parse a string.\n'
        "\n"
        "    Returns a list.\n"
        '    """\n'
        "    return s.split()"
    )
    imports, body = extract_body(code)
    assert imports == ""
    assert body == "    return s.split()"


def test_extract_body_no_docstring():
    """Extract body from function without docstring."""
    code = "def add(a, b):\n    return a + b"
    imports, body = extract_body(code)
    assert imports == ""
    assert body == "    return a + b"


def test_extract_body_preserves_imports():
    """Imports before the def line are returned separately."""
    code = (
        "from typing import List, Tuple\n"
        "import math\n"
        "\n"
        "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        '    """Return sum and product."""\n'
        "    return (sum(numbers), math.prod(numbers))"
    )
    imports, body = extract_body(code)
    assert imports == "from typing import List, Tuple\nimport math\n"
    assert body == "    return (sum(numbers), math.prod(numbers))"


def test_extract_body_imports_assemble_correctly():
    """Imports go above the prompt, body goes inside it."""
    code = (
        "import math\n"
        "\n"
        "def f(n: int) -> int:\n"
        '    """Compute factorial."""\n'
        "    return math.factorial(n)"
    )
    prompt = 'def f(n: int) -> int:\n    """Compute factorial."""\n'

    imports, body = extract_body(code)
    full_code = imports + prompt + ensure_indented(body)

    assert full_code == (
        "import math\n"
        "def f(n: int) -> int:\n"
        '    """Compute factorial."""\n'
        "    return math.factorial(n)"
    )
    # Verify it compiles
    compile(full_code, "<test>", "exec")


def test_extract_body_already_body_only():
    """When given just a body (no def line), return unchanged."""
    code = "    return a + b"
    imports, body = extract_body(code)
    assert imports == ""
    assert body == "    return a + b"


def test_extract_body_multiline_body():
    """Extract multi-line body from complete function."""
    code = (
        "def has_close_elements(numbers, threshold):\n"
        '    """Check if any two numbers are closer than threshold."""\n'
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False"
    )
    expected = (
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False"
    )
    imports, body = extract_body(code)
    assert imports == ""
    assert body == expected


def test_extract_body_different_function_name():
    """Still works for any function name."""
    code = 'def multiply(a, b):\n    """Multiply."""\n    return a * b'
    imports, body = extract_body(code)
    assert imports == ""
    assert body == "    return a * b"


# --- ensure_indented tests ---


def test_ensure_indented_already_indented():
    code = "    return x + 1"
    assert ensure_indented(code) == "    return x + 1"


def test_ensure_indented_not_indented():
    code = "return x + 1"
    assert ensure_indented(code) == "    return x + 1"


def test_ensure_indented_multiline():
    code = "result = []\nfor x in items:\n    result.append(x)\nreturn result"
    expected = "    result = []\n    for x in items:\n        result.append(x)\n    return result"
    assert ensure_indented(code) == expected


def test_ensure_indented_empty_lines():
    code = "x = 1\n\nreturn x"
    expected = "    x = 1\n\n    return x"
    assert ensure_indented(code) == expected
