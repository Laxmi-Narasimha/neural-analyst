"""Command line companion for an existing Neural Analyst dataset."""

import argparse
import json
import os
import sys
from uuid import UUID

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    questions = commands.add_parser("questions")
    questions.add_argument("dataset", type=UUID)
    ask = commands.add_parser("ask")
    ask.add_argument("dataset", type=UUID)
    ask.add_argument("message")
    ask.add_argument("--conversation", type=UUID)
    rows = commands.add_parser("rows")
    rows.add_argument("artifact", type=UUID)
    rows.add_argument("--offset", type=int, default=0)
    rows.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    url = os.environ.get("NEURAL_ANALYST_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
    token = os.environ.get("NEURAL_ANALYST_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(base_url=url, headers=headers, timeout=120) as client:
            if args.command == "questions":
                response = client.get(f"datasets/{args.dataset}")
            elif args.command == "ask":
                body = {"dataset_id": str(args.dataset), "message": args.message}
                if args.conversation:
                    body["conversation_id"] = str(args.conversation)
                response = client.post("chat", json=body)
            else:
                response = client.get(f"artifacts/{args.artifact}/rows", params={"offset": args.offset, "limit": args.limit})
            response.raise_for_status()
            data = response.json()["data"]
    except httpx.HTTPStatusError as exc:
        print(f"API returned HTTP {exc.response.status_code}: {exc.response.text[:600]}", file=sys.stderr)
        return 1
    except (httpx.RequestError, ValueError, KeyError) as exc:
        print(f"Could not read the Neural Analyst response: {exc}", file=sys.stderr)
        return 1

    if args.command == "questions":
        data = {key: data.get(key) for key in ("id", "name", "status", "error_message", "row_count", "starter_questions")}
    elif args.command == "ask":
        data = {
            **{key: data.get(key) for key in ("conversation_id", "content", "clarification", "metadata", "suggestions")},
            "artifacts": [
                {key: artifact.get(key) for key in ("artifact_id", "artifact_type", "name", "operator_name", "dataset_version", "operator_params")}
                for action in data.get("agent_actions", []) for artifact in action.get("artifacts", [])
            ],
        }
    print(json.dumps(data, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
