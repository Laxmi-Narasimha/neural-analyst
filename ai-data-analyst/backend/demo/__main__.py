"""Neural Analyst demo: one command to set up, verify and run the interview demo.

Run from ai-data-analyst/backend with the backend's virtual environment:

    python -m demo serve      # seed the demo workspace, then start backend + frontend (Ctrl+C stops both)
    python -m demo check      # ask every demo question and verify every number independently
    python -m demo check --url http://127.0.0.1:8000   # same, against the running server the UI uses
    python -m demo seed       # load the demo datasets and train the churn model (idempotent)
    python -m demo api        # seed, then run only the backend (for frontend development)
    python -m demo reset      # delete the demo workspace (your own data is never touched)

The demo uses its own workspace (demo_workspace/): database, uploads, artifacts and models.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.workspace import BACKEND, DATA, WORKSPACE, apply_demo_env, demo_env  # noqa: E402

FRONTEND = BACKEND.parent / "frontend"
OK, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _say(text: str = "") -> None:
    print(text, flush=True)


def _quiet_logs() -> None:
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass


def _llm_status() -> tuple[bool, str]:
    from app.planning.llm_planner import planner_enabled

    if not planner_enabled():
        return False, "the language-model planner is switched off (LLM_PLANNER=off)"
    try:
        from app.core.config import settings

        model = os.getenv("LLM_PLANNER_MODEL") or settings.llm_planner_model or settings.llm_model
        key = settings.openai.api_key
        if key is None or not key.get_secret_value().strip():
            if not any(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OLLAMA_BASE_URL")):
                return False, "no LLM API key is configured (set OPENAI_API_KEY in backend/.env)"
        return True, f"planner model {model}"
    except Exception as e:  # pragma: no cover - configuration problems are reported, not raised
        return False, f"LLM configuration could not be read ({e})"


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

async def seed(url: str | None = None) -> dict[str, str]:
    from demo import client as c
    from demo.script import DATASETS

    async with c.open_client(url) as client:
        have = {d.get("name"): str(d.get("id")) for d in await c.list_datasets(client)}
        ids: dict[str, str] = {}
        for name, file in DATASETS.items():
            if name in have:
                ids[name] = have[name]
                _say(f"  dataset {name:22s} already loaded")
                continue
            ids[name] = await c.upload(client, DATA / file, name, in_process=url is None)
            _say(f"  dataset {name:22s} loaded")
        models = {m.get("name") for m in await c.list_models(client)}
        if not any(str(n).startswith("churned (") for n in models):
            t0 = time.perf_counter()
            await c.train(client, ids["customer_churn"], "churned", None)
            _say(f"  model   churned (customer_churn) trained in {time.perf_counter() - t0:.0f}s")
        else:
            _say("  model   churned (customer_churn) already trained")
        return ids


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

async def check(url: str | None = None, *, llm: bool = True) -> bool:
    from demo import client as c
    from demo.script import DEMO

    ids = await seed(url)
    _say("")
    results: list[tuple[str, str, float, str]] = []
    async with c.open_client(url) as client:
        for i, step in enumerate(DEMO, 1):
            label = f"{i:2d}. [{step.dataset}] {step.question}"
            if step.needs_llm and not llm:
                t0 = time.perf_counter()
                answer = await c.chat(client, ids[step.dataset], step.question)
                graceful = bool(str(answer.get("content") or "").strip())
                results.append((label, SKIP if graceful else FAIL, time.perf_counter() - t0,
                                "needs the LLM planner; answered with a clear message" if graceful else "empty answer"))
                continue
            t0 = time.perf_counter()
            try:
                answer = await c.chat(client, ids[step.dataset], step.question)
            except Exception as e:
                results.append((label, FAIL, time.perf_counter() - t0, f"request failed: {e}"))
                continue
            took = time.perf_counter() - t0
            content = str(answer.get("content") or "")
            ops = [a.get("operator") for a in answer.get("agent_actions") or []]
            problems = []
            if step.operator not in ops:
                problems.append(f"ran {ops or 'nothing'} instead of {step.operator}")
            missing = [e for e in step.expect() if e not in content]
            if missing:
                problems.append("answer lacks " + "; ".join(repr(m) for m in missing[:3]))
            if step.operator == "ml_score" and not problems:
                problems += await _check_download(client, answer)
            detail = "; ".join(problems) if problems else (content.splitlines()[0][:90] if content else "")
            results.append((label, FAIL if problems else OK, took, detail))
            if problems:
                _say(f"\n--- answer to {step.question!r} ---\n{content[:1200]}\n---")

    _say("")
    for label, state, took, detail in results:
        _say(f"{state}  {took:5.1f}s  {label}")
        if detail:
            _say(f"              {detail}")
    failed = [r for r in results if r[1] == FAIL]
    _say("")
    _say(f"{len(results) - len(failed)}/{len(results)} demo steps passed." if not failed
         else f"{len(failed)} of {len(results)} demo steps FAILED. Fix these before demoing.")
    return not failed


async def _check_download(client, answer: dict) -> list[str]:
    """The predictions table the answer points to must open, like the UI's download does."""
    for action in answer.get("agent_actions") or []:
        for art in action.get("artifacts") or []:
            if str(art.get("name") or "").endswith(":predictions"):
                res = await client.get(f"/api/v1/artifacts/{art['artifact_id']}/rows")
                if res.status_code != 200 or "predicted_" not in res.text:
                    return [f"predictions table did not open ({res.status_code})"]
                return []
    return ["no predictions table was attached"]


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def _frontend_needs_build() -> bool:
    build_id = FRONTEND / ".next" / "BUILD_ID"
    if not build_id.exists():
        return True
    built = build_id.stat().st_mtime
    marker = FRONTEND / ".next" / "demo-api-url"
    if not marker.exists():
        return True
    newest = max((p.stat().st_mtime for p in (FRONTEND / "src").rglob("*") if p.is_file()), default=0)
    return newest > built


def _wait_for(url: str, seconds: int) -> bool:
    import httpx

    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def serve(*, backend_port: int, frontend_port: int, dev: bool) -> int:
    # On Windows, "npm" without an extension is a shell script that CreateProcess cannot run.
    names = ("npm.cmd", "npm") if sys.platform == "win32" else ("npm",)
    npm = next((found for name in names if (found := shutil.which(name))), None)
    if npm is None:
        _say("npm was not found. Install Node.js 20.9+ (see docs/SETUP.md).")
        return 1
    if not (FRONTEND / "node_modules").exists():
        _say("Installing frontend packages (first run only)...")
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)

    _say("Seeding the demo workspace...")
    asyncio.run(seed())

    api_url = f"http://localhost:{backend_port}/api/v1"
    env = {**os.environ, **demo_env(), "NEXT_PUBLIC_API_URL": api_url, "NEXT_PUBLIC_PRIVATE_BETA": "true"}
    if not dev and _frontend_needs_build():
        _say("Building the frontend for a fast, stable demo (about a minute, first run only)...")
        subprocess.run([npm, "run", "build"], cwd=FRONTEND, env=env, check=True)
        (FRONTEND / ".next" / "demo-api-url").write_text(api_url)

    procs = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
            cwd=BACKEND, env=env))
        if not _wait_for(f"http://127.0.0.1:{backend_port}/health", 90):
            _say("The backend did not start; see the log above.")
            return 1
        cmd = [npm, "run", "dev" if dev else "start", "--", "--port", str(frontend_port)]
        procs.append(subprocess.Popen(cmd, cwd=FRONTEND, env=env))
        app_url = f"http://localhost:{frontend_port}/app/datasets"
        if not _wait_for(app_url, 120):
            _say("The frontend did not start; see the log above.")
            return 1
        ok, why = _llm_status()
        _say("")
        _say(f"Neural Analyst demo is running: {app_url}")
        _say(f"API docs: http://localhost:{backend_port}/docs")
        _say(f"Language model: {'on, ' + why if ok else 'OFF: ' + why}")
        _say("Demo script: docs/DEMO.md. Press Ctrl+C to stop.")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed", help="load demo datasets and train the churn model")
    s.add_argument("--url", help="seed a running backend instead of the demo workspace")
    c = sub.add_parser("check", help="ask every demo question and verify every number")
    c.add_argument("--url", help="check a running backend (what the web app uses)")
    c.add_argument("--no-llm", action="store_true", help="check without the language-model planner")
    v = sub.add_parser("serve", help="seed, then run backend and frontend for the demo")
    v.add_argument("--backend-port", type=int, default=8000)
    v.add_argument("--frontend-port", type=int, default=3002)
    v.add_argument("--dev", action="store_true", help="frontend in dev mode (slower first page loads)")
    a = sub.add_parser("api", help="seed, then run only the backend on the demo workspace")
    a.add_argument("--port", type=int, default=8000)
    c.add_argument("--repeat", type=int, default=1, help="run the whole check N times (catches flaky planning)")
    sub.add_parser("reset", help="delete the demo workspace")
    args = parser.parse_args(argv)

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if args.command == "reset":
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        _say(f"Deleted {WORKSPACE}")
        return 0

    if not getattr(args, "url", None):
        apply_demo_env()
    if getattr(args, "no_llm", False):
        os.environ["LLM_PLANNER"] = "off"
    _quiet_logs()

    if args.command == "seed":
        asyncio.run(seed(args.url))
        return 0
    if args.command == "check":
        ok, why = _llm_status()
        llm = ok and not args.no_llm
        _say(f"Language model: {'on, ' + why if llm else 'off: ' + why}")
        runs = [asyncio.run(check(args.url, llm=llm)) for _ in range(max(1, args.repeat))]
        if len(runs) > 1:
            _say(f"\n{sum(runs)}/{len(runs)} full runs passed.")
        return 0 if all(runs) else 1
    if args.command == "api":
        import uvicorn

        asyncio.run(seed())
        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)
        return 0
    return serve(backend_port=args.backend_port, frontend_port=args.frontend_port, dev=args.dev)


if __name__ == "__main__":
    raise SystemExit(main())
