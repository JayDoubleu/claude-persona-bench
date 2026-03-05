from persona_bench.evaluator.sandbox import run_sandboxed


def test_passing_code():
    code = "def add(a, b):\n    return a + b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n    assert f(0, 0) == 0\n"
    passed, error = run_sandboxed(code, test, "add")
    assert passed is True
    assert error is None


def test_failing_code():
    code = "def add(a, b):\n    return a - b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n"
    passed, error = run_sandboxed(code, test, "add")
    assert passed is False
    assert error is not None


def test_timeout():
    code = "def loop():\n    while True: pass\n"
    test = "def check(f):\n    f()\n"
    passed, error = run_sandboxed(code, test, "loop")
    assert passed is False
    assert "Timed out" in (error or "")


def test_blocked_import():
    code = "import os\ndef dangerous():\n    return os.listdir('.')\n"
    test = "def check(f):\n    f()\n"
    passed, _error = run_sandboxed(code, test, "dangerous")
    assert passed is False
