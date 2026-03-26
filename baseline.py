#!/usr/bin/env python3
"""
Baseline Inference Script — SRE Incident Response OpenEnv

Runs the GPT-4o-mini model against all 3 tasks and produces reproducible scores.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--env-url http://localhost:7860] [--model gpt-4o-mini]

Output:
    JSON to stdout with scores for all 3 tasks.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import datetime
import time
import requests
from typing import Any


DEFAULT_ENV_URL = "http://localhost:7860"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE).
You are operating an incident response environment via API. At each step you will receive
the current system state and must return a single JSON action. No markdown, no explanation.

JSON format (only this, nothing else):
{
  "action_type": "<diagnose|query_logs|query_metrics|run_command|apply_config|escalate|close_incident>",
  "target_service": "<service name>",
  "command": "<command string or null>",
  "rationale": "<brief reasoning, max 200 chars>"
}

Resolution strategy:
1. Diagnose/query_logs on the MOST SUSPICIOUS service first
2. Trace root cause to the correct service (not just symptoms)
3. Apply the correct config fix with the right command keywords
4. Query metrics post-fix to verify recovery  
5. Close incident only after confirmed resolution
"""


def call_env(url: str, method: str, path: str, body: dict | None = None) -> dict:
    full_url = f"{url}{path}"
    if method == "GET":
        resp = requests.get(full_url, timeout=30)
    else:
        resp = requests.post(full_url, json=body or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_episode(
    env_url: str,
    task_id: str,
    openai_client: Any,
    model: str,
) -> dict[str, Any]:
    """Run one full episode for a task. Returns score dict."""

    # Reset
    reset_resp = call_env(env_url, "POST", "/reset", {"task_id": task_id})
    task_desc = reset_resp["task_description"]
    obs = reset_resp["observation"]

    conversation: list[dict[str, str]] = []
    done = False
    steps_used = 0

    while not done:
        step_count = obs["step_count"]
        max_steps = obs["max_steps"]

        if step_count >= max_steps:
            break

        # Format observation
        alerts_text = "\n".join(
            f"  [{a['severity']}] {a['service']}: {a['title']}"
            for a in obs["active_alerts"]
        )
        topology_text = "\n".join(
            f"  {name}: status={svc['status']}, error_rate={svc['error_rate']:.2f}, latency={svc['latency_p99_ms']:.0f}ms"
            for name, svc in obs["system_topology"].items()
        )

        obs_text = f"INCIDENT: {obs['incident_context']}\n\nALERTS:\n{alerts_text}\n\nSERVICES:\n{topology_text}"

        if obs.get("last_action_feedback"):
            obs_text += f"\n\nFEEDBACK: {obs['last_action_feedback']['output']}"

        if step_count == 0:
            obs_text = f"TASK: {task_desc}\n\n" + obs_text

        conversation.append({"role": "user", "content": obs_text})

        # LLM call
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            max_tokens=300,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        conversation.append({"role": "assistant", "content": raw})

        # Parse action
        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            action_data = json.loads(clean)
        except Exception:
            print(f"  [WARN] JSON parse error at step {step_count}, using fallback diagnose", file=sys.stderr)
            action_data = {
                "action_type": "diagnose",
                "target_service": list(obs["system_topology"].keys())[0],
                "command": None,
                "rationale": "parse error fallback",
            }

        # Step
        step_resp = call_env(env_url, "POST", "/step", action_data)
        obs = step_resp["observation"]
        done = step_resp["done"]
        steps_used = obs["step_count"]

        print(
            f"  Step {steps_used}: {action_data.get('action_type')}:{action_data.get('target_service')} "
            f"→ reward={step_resp['reward']['total']:.3f}",
            file=sys.stderr,
        )

        time.sleep(0.1)  # rate limit courtesy

    # Grade
    grade_resp = call_env(env_url, "POST", "/grader", {})

    return {
        "task_id": task_id,
        "score": grade_resp["score"],
        "breakdown": grade_resp["breakdown"],
        "verdict": grade_resp["verdict"],
        "steps_used": steps_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SRE OpenEnv Baseline Inference Script")
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL, help="Base URL of the OpenEnv server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Verify server is up
    try:
        health = call_env(args.env_url, "GET", "/health")
        print(f"[INFO] Server healthy: {health}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {args.env_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get task list
    task_list = call_env(args.env_url, "GET", "/tasks")
    task_ids = [t["task_id"] for t in task_list]

    results = []
    for task_id in task_ids:
        print(f"\n[INFO] Running task: {task_id}", file=sys.stderr)
        result = run_episode(args.env_url, task_id, client, args.model)
        results.append(result)
        print(f"  Score: {result['score']:.4f} — {result['verdict']}", file=sys.stderr)

    mean_score = round(sum(r["score"] for r in results) / len(results), 4)

    output = {
        "model": args.model,
        "env_url": args.env_url,
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "mean_score": mean_score,
        "scores": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
