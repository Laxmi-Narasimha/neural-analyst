#!/usr/bin/env python3
"""
AI Data Adequacy Agent — Single Python launcher

Starts the FastAPI backend and Streamlit frontend from one script.

Usage (Windows / PowerShell):
  python run_dev.py

Optional flags:
  --backend-port 8000     Backend port (default 8000)
  --frontend-port 8501    Frontend port (default 8501)
  --install               Install/verify dependencies before launch

Notes:
- Expects backend/.env to exist. FastAPI loads it via dotenv.
- Graceful shutdown: Ctrl+C will terminate both child processes.
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
BACKEND_ENV = BACKEND / ".env"
REQUIREMENTS = BACKEND / "requirements.txt"


def print_header():
    print("=" * 60)
    print("AI Data Adequacy Agent — Dev Runner")
    print("=" * 60)


def ensure_env_file():
    if not BACKEND_ENV.exists():
        print("ERROR: backend/.env not found.\n"
              "Create it from backend/.env.template and fill keys.")
        sys.exit(1)


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.Popen:
    # Start a child process inheriting this console
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


def stream_output(prefix: str, proc: subprocess.Popen):
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")


def check_and_install(args):
    """Optionally install required packages."""
    if not args.install:
        return

    print("[setup] Installing/validating Python dependencies…")
    # Install backend requirements
    if REQUIREMENTS.exists():
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if rc != 0:
            print("[setup] Failed to install backend requirements.")
            sys.exit(rc)
    else:
        print(f"[setup] WARNING: {REQUIREMENTS} not found.")

    # Ensure streamlit present
    try:
        import streamlit  # noqa: F401
    except Exception:
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "streamlit"])
        if rc != 0:
            print("[setup] Failed to install Streamlit.")
            sys.exit(rc)



def main():
    parser = argparse.ArgumentParser(description="Run backend and frontend together")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--install", action="store_true", help="Install/verify dependencies before launch")
    args = parser.parse_args()

    print_header()

    # Basic checks
    if not BACKEND.exists() or not FRONTEND.exists():
        print("ERROR: Expected backend/ and frontend/ folders at repo root.")
        sys.exit(1)

    ensure_env_file()
    check_and_install(args)

    # Commands
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--reload",
        "--host", args.host,
        "--port", str(args.backend_port),
    ]
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(args.frontend_port),
        "--server.address", "localhost",
    ]

    print(f"Backend:  http://localhost:{args.backend_port}")
    print(f"Docs:     http://localhost:{args.backend_port}/docs")
    print(f"Frontend: http://localhost:{args.frontend_port}")

    # Launch
    print("[1/2] Starting backend…")
    backend_proc = run_cmd(backend_cmd, BACKEND)

    # Give backend a moment
    time.sleep(2)

    print("[2/2] Starting frontend…")
    frontend_proc = run_cmd(frontend_cmd, FRONTEND)

    # Open browser
    try:
        webbrowser.open_new_tab(f"http://localhost:{args.frontend_port}")
    except Exception:
        pass

    try:
        # Stream combined output until either process exits
        while True:
            code_b = backend_proc.poll()
            code_f = frontend_proc.poll()
            if code_b is not None or code_f is not None:
                break
            # Non-blocking small sleep
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        for proc, name in [(backend_proc, "backend"), (frontend_proc, "frontend")]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        # Give them time to exit; then kill if needed
        time.sleep(1.0)
        for proc in (backend_proc, frontend_proc):
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

        print("Done.")


if __name__ == "__main__":
    main()
