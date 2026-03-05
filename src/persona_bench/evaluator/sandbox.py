from __future__ import annotations

import multiprocessing
import os
import tempfile
from multiprocessing.connection import Connection
from typing import Any

TIMEOUT_SECONDS = 10

# Builtins/modules to disable inside the sandbox
_DISABLED_BUILTINS = ["exit", "quit", "input"]
_DISABLED_MODULES = [
    "builtins",
    "importlib",
    "io",
    "os",
    "pathlib",
    "shutil",
    "subprocess",
    "signal",
    "socket",
    "sys",
    "tempfile",
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


def _blocked_open(*args: Any, **kwargs: Any) -> Any:
    raise PermissionError("File access is not allowed in sandbox")


def _run_in_sandbox(
    code: str,
    test: str,
    entry_point: str,
    result_conn: Connection,
) -> None:
    """Execute code + tests inside a restricted environment."""
    try:
        # Build restricted global namespace
        raw_builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        safe_globals: dict[str, Any] = {"__builtins__": dict(raw_builtins)}

        # Remove dangerous builtins
        for name in _DISABLED_BUILTINS:
            safe_globals["__builtins__"].pop(name, None)
        safe_globals["__builtins__"]["open"] = _blocked_open

        # Override __import__ to block dangerous modules
        original_import = safe_globals["__builtins__"]["__import__"]

        def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] in _DISABLED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return original_import(name, *args, **kwargs)

        safe_globals["__builtins__"]["__import__"] = restricted_import

        # Run user code in an empty temp working directory to avoid incidental file access.
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)

            # Run the completion code, then the test harness
            _EXEC(code, safe_globals)
            _EXEC(test, safe_globals)

            # Call the check function (HumanEval convention)
            if "check" in safe_globals:
                safe_globals["check"](safe_globals[entry_point])

        result_conn.send(("pass", None))
    except Exception as e:
        result_conn.send(("fail", f"{type(e).__name__}: {e}"))
    finally:
        result_conn.close()


def run_sandboxed(code: str, test: str, entry_point: str) -> tuple[bool, str | None]:
    """Run code against tests in a sandboxed subprocess with timeout.

    Returns (passed, error_message).
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_in_sandbox,
        args=(code, test, entry_point, child_conn),
    )
    process.start()
    child_conn.close()
    process.join(timeout=TIMEOUT_SECONDS)

    if process.is_alive():
        process.kill()
        process.join()
        parent_conn.close()
        return False, "Timed out"

    if process.exitcode != 0 and not parent_conn.poll():
        parent_conn.close()
        return False, f"Process crashed with exit code {process.exitcode}"

    if not parent_conn.poll():
        parent_conn.close()
        return False, "No result returned from sandbox"

    status, error = parent_conn.recv()
    parent_conn.close()
    return status == "pass", error
