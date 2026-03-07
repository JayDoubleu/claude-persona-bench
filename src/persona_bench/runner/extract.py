from __future__ import annotations

import re


def extract_code(raw: str) -> str:
    """Extract Python code from a response, stripping markdown fences if present."""
    # Strip <think>...</think> blocks (e.g., Qwen3 inline reasoning)
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)

    # Try triple-backtick markdown fences
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # Try single-backtick wrapping (must contain newlines to avoid inline code)
    match = re.search(r"`(.*?)`", raw, re.DOTALL)
    if match and "\n" in match.group(1):
        return match.group(1).strip("\n")

    # Raw code: strip only leading/trailing newlines, preserve internal whitespace
    return raw.strip("\n")


def extract_body(code: str) -> tuple[str, str]:
    """Extract the function body and imports from a complete function definition.

    Returns (imports, body) where imports is any import/from lines before the
    def, and body is everything after the docstring. Imports go above the
    prompt; body goes inside it.

    If the code doesn't start with `def`, returns ("", code).
    """
    lines = code.split("\n")

    # Find the def line
    def_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_idx = i
            break

    if def_idx is None:
        return "", code

    # Collect import lines before the def
    imports = []
    for line in lines[:def_idx]:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)

    # Find the end of the docstring (if any)
    body_start = def_idx + 1
    in_docstring = False

    for i in range(def_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue

        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = '"""' if stripped.startswith('"""') else "'''"
                rest = stripped[3:]
                if quote in rest:
                    # Single-line docstring: """text"""
                    body_start = i + 1
                    break
                else:
                    in_docstring = True
            else:
                # No docstring, body starts here
                body_start = i
                break
        else:
            if '"""' in stripped or "'''" in stripped:
                body_start = i + 1
                break

    body = "\n".join(lines[body_start:])
    if not body.strip():
        return "", code

    imports_str = "\n".join(imports) + "\n" if imports else ""
    return imports_str, body


def ensure_indented(code: str, indent: str = "    ") -> str:
    """Ensure code is indented as a function body.

    Adds one indent level to all non-empty lines if the first non-empty line
    starts at column 0. If code is already indented, returns it unchanged.
    """
    lines = code.split("\n")

    # Check if already indented
    for line in lines:
        if line.strip():
            if line.startswith((" ", "\t")):
                return code
            break
    else:
        return code

    # Add indent to all non-empty lines
    return "\n".join(indent + line if line.strip() else line for line in lines)
