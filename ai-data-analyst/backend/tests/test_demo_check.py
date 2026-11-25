"""The interview demo must keep working: run its checker (without a language model) in a scratch
workspace. Steps that need the LLM planner must still answer, with a clear message."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


@pytest.mark.integration
def test_the_demo_passes_without_a_language_model(tmp_path):
    env = {**os.environ, "DEMO_WORKSPACE": str(tmp_path / "demo"), "PYTHONIOENCODING": "utf-8"}
    env.pop("OPENAI_API_KEY", None)
    run = subprocess.run([sys.executable, "-m", "demo", "check", "--no-llm"], cwd=BACKEND, env=env,
                         capture_output=True, text=True, timeout=600)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out[-4000:]
    assert "demo steps passed" in out
    assert "PASS" in out and "Score this file with my churn model" in out
