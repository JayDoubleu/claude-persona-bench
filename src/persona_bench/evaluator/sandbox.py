from __future__ import annotations

import multiprocessing
from typing import Any

TIMEOUT_SECONDS = 10

# Builtins/modules to disable inside the sandbox
_DISABLED_BUILTINS = ["exit", "quit"]
_DISABLED_MODULES = [
    "os",
    "shutil",
    "subprocess",
    "signal",
    "socket",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "webbrowser",
    "ctypes",
]

# Python's exec() is used intentionally here to run HumanEval completions
# inside an isolated subprocess with restricted builtins. This is the standard
# approach for HumanEval evaluation and runs in a separate spawned process.
_EXEC = exec


def _run_in_sandbox(
    code: str,
    test: str,
    entry_point: str,
    result_queue: multiprocessing.Queue,  # type: ignore[type-arg]
) -> None:
    """Execute code + tests inside a restricted environment."""
    try:
        # Build restricted global namespace
        raw_builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        safe_globals: dict[str, Any] = {"__builtins__": dict(raw_builtins)}

        # Remove dangerous builtins
        for name in _DISABLED_BUILTINS:
            safe_globals["__builtins__"].pop(name, None)

        # Override __import__ to block dangerous modules
        original_import = safe_globals["__builtins__"]["__import__"]

        def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] in _DISABLED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return original_import(name, *args, **kwargs)

        safe_globals["__builtins__"]["__import__"] = restricted_import

        # Run the completion code, then the test harness
        _EXEC(code, safe_globals)
        _EXEC(test, safe_globals)

        # Call the check function (HumanEval convention)
        if "check" in safe_globals:
            safe_globals["check"](safe_globals[entry_point])

        result_queue.put(("pass", None))
    except Exception as e:
        result_queue.put(("fail", f"{type(e).__name__}: {e}"))


def run_sandboxed(code: str, test: str, entry_point: str) -> tuple[bool, str | None]:
    """Run code against tests in a sandboxed subprocess with timeout.

    Returns (passed, error_message).
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[tuple[str, str | None]] = ctx.Queue()
    process = ctx.Process(
        target=_run_in_sandbox,
        args=(code, test, entry_point, result_queue),
    )
    process.start()
    process.join(timeout=TIMEOUT_SECONDS)

    if process.is_alive():
        process.kill()
        process.join()
        return False, "Timed out"

    if process.exitcode != 0 and result_queue.empty():
        return False, f"Process crashed with exit code {process.exitcode}"

    if result_queue.empty():
        return False, "No result returned from sandbox"

    status, error = result_queue.get_nowait()
    return status == "pass", error
