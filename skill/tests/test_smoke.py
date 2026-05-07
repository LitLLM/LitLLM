"""Smoke tests — package imports cleanly and CLI parser works."""

import subprocess
import sys


def test_import():
    import litllm

    assert litllm.__version__


def test_cli_help():
    top = subprocess.run(
        [sys.executable, "-m", "litllm.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "related-work" in top.stdout
    assert "keywords" in top.stdout

    rw = subprocess.run(
        [sys.executable, "-m", "litllm.cli", "related-work", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--deep-research" in rw.stdout
