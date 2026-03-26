"""
SRE Incident Response — FastAPI Application

Endpoints (OpenEnv spec required):
  POST /reset      → ResetResponse
  POST /step       → StepResponse
  GET  /state      → Observation
  GET  /tasks      → list[TaskSchema]
  POST /grader     → GraderResponse
  POST /baseline   → BaselineResponse

Additional:
  GET  /           → health check
  GET  /health     → health check
"""
from __future__ import annotations
import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import (
    Action, ActionType, ResetResponse, StepResponse, Observation,
    GraderResponse, TaskSchema, BaselineResponse, BaselineScore,
)
from app.env import SREEnvironment
from app.tasks import TASK_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global environment instance (single-session for hackathon scope) ─────────
_env = SREEnvironment()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("SRE Incident Response environment starting up.")
    yield
    logger.info("SRE Incident Response environment shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SRE Incident Response OpenEnv",
    description=(
        "Real-world SRE incident triage environment. "
        "An AI agent diagnoses and remediates cloud infrastructure incidents "
        "through a standardized step()/reset()/state() API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "pod-oom-restart"


class GraderRequest(BaseModel):
    """Empty body — grades the current completed episode."""
    pass


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": "sre-incident-response", "version": "1.0.0"}


# ─── OpenEnv Core Endpoints ───────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(request: ResetRequest) -> ResetResponse:
    """
    Reset the environment to the initial state for a given task.
    Returns the first Observation and task metadata.
    """
    try:
        response = _env.reset(request.task_id)
        logger.info(f"Environment reset — task_id={request.task_id}")
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(action: Action) -> StepResponse:
    """
    Execute one action in the environment.
    Returns (observation, reward, done, info).
    """
    try:
        response = _env.step(action)
        logger.info(
            f"step={response.observation.step_count} "
            f"action={action.action_type.value}:{action.target_service} "
            f"reward={response.reward.total:.3f} "
            f"done={response.done}"
        )
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/state", response_model=Observation, tags=["openenv"])
def state() -> Observation:
    """
    Return the current environment observation without advancing the episode.
    """
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/tasks", response_model=list[TaskSchema], tags=["openenv"])
def tasks() -> list[TaskSchema]:
    """
    Return all available tasks with action schema and success criteria.
    """
    action_schema = {
        "type": "object",
        "required": ["action_type", "target_service", "rationale"],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": [a.value for a in ActionType],
                "description": "Type of action to perform",
            },
            "target_service": {
                "type": "string",
                "description": "Service or component to act on",
            },
            "command": {
                "type": "string",
                "nullable": True,
                "description": "Required for run_command and apply_config actions",
            },
            "rationale": {
                "type": "string",
                "description": "Agent's stated reasoning for this action (max 500 chars)",
            },
        },
    }

    return [
        TaskSchema(
            task_id=task.task_id,
            difficulty=task.difficulty,
            description=task.description,
            max_steps=task.max_steps,
            action_schema=action_schema,
            success_criteria=task.success_criteria,
        )
        for task in TASK_REGISTRY.values()
    ]


@app.post("/grader", response_model=GraderResponse, tags=["openenv"])
def grader(_: GraderRequest = Body(default=GraderRequest())) -> GraderResponse:
    """
    Grade the completed episode. Returns a deterministic score [0.0, 1.0]
    with a breakdown of partial credit components.
    Call after episode is done (done=True from /step).
    """
    try:
        result = _env.grade()
        logger.info(f"Grader called — task={result.task_id} score={result.score:.4f}")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/baseline", response_model=BaselineResponse, tags=["openenv"])
def baseline() -> BaselineResponse:
    """
    Run the baseline inference script against all 3 tasks using the OpenAI API.
    Reads OPENAI_API_KEY from environment variables.
    Returns reproducible baseline scores.
    """
    import datetime

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY environment variable not set. Cannot run baseline.",
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise HTTPException(status_code=503, detail="openai package not installed.")

    scores: list[BaselineScore] = []
    model_name = "gpt-4o-mini"

    for task_id in TASK_REGISTRY.keys():
        try:
            score = _run_baseline_episode(client, task_id, model_name)
            scores.append(score)
        except Exception as e:
            logger.error(f"Baseline episode failed for task={task_id}: {e}")
            scores.append(BaselineScore(task_id=task_id, score=0.0, steps_used=0, resolved=False))

    mean = round(sum(s.score for s in scores) / len(scores), 4)

    return BaselineResponse(
        model=model_name,
        scores=scores,
        mean_score=mean,
        run_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ─── Baseline Episode Runner ──────────────────────────────────────────────────

def _run_baseline_episode(client: Any, task_id: str, model: str) -> BaselineScore:
    """
    Runs a single baseline episode for a given task.
    Uses a ReAct-style prompt: Observation → Think → Act loop.
    """
    from app.models import Action, ActionType

    # Reset environment for this task
    reset_resp = _env.reset(task_id)
    task_desc = reset_resp.task_description
    obs = reset_resp.observation

    SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE).
You are operating an incident response environment. At each step you will receive:
- The current system state (alerts, service health, action history)
- Feedback from your last action

You must respond with a single JSON object representing your next action. No other text.

JSON format:
{
  "action_type": "<one of: diagnose, query_logs, query_metrics, run_command, apply_config, escalate, close_incident>",
  "target_service": "<service name>",
  "command": "<command string or null>",
  "rationale": "<brief reasoning>"
}

Strategy:
1. Diagnose or query logs on services showing alerts
2. Identify the ROOT CAUSE service (not just symptomatic services)  
3. Apply the correct config fix to the root cause service
4. Verify recovery with query_metrics
5. Close the incident only after the fix is confirmed
"""

    conversation: list[dict[str, str]] = []
    done = False
    steps_used = 0
    resolved = False

    while not done and obs.step_count < obs.max_steps:
        # Build observation message
        obs_text = (
            f"INCIDENT CONTEXT: {obs.incident_context}\n\n"
            f"ACTIVE ALERTS:\n"
            + "\n".join(
                f"  [{a.severity.value}] {a.service}: {a.title}" for a in obs.active_alerts
            )
            + f"\n\nSERVICE HEALTH:\n"
            + "\n".join(
                f"  {name}: status={svc.status}, error_rate={svc.error_rate:.2f}, latency={svc.latency_p99_ms:.0f}ms"
                for name, svc in obs.system_topology.items()
            )
        )
        if obs.last_action_feedback:
            obs_text += f"\n\nLAST ACTION FEEDBACK:\n{obs.last_action_feedback.output}"

        if obs.step_count == 0:
            obs_text = f"TASK: {task_desc}\n\n" + obs_text

        conversation.append({"role": "user", "content": obs_text})

        # Call LLM
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            max_tokens=256,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        conversation.append({"role": "assistant", "content": raw})

        # Parse action
        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            action_data = json.loads(clean)
            action = Action(
                action_type=ActionType(action_data.get("action_type", "diagnose")),
                target_service=action_data.get("target_service", "unknown"),
                command=action_data.get("command"),
                rationale=action_data.get("rationale", "No rationale provided.")[:500],
            )
        except Exception:
            action = Action(
                action_type=ActionType.DIAGNOSE,
                target_service=list(obs.system_topology.keys())[0],
                rationale="Parse error fallback — diagnosing first service.",
            )

        step_resp = _env.step(action)
        obs = step_resp.observation
        done = step_resp.done
        steps_used = obs.step_count

        if done and step_resp.info.get("incident_resolved"):
            resolved = True

    grader_resp = _env.grade()
    return BaselineScore(
        task_id=task_id,
        score=grader_resp.score,
        steps_used=steps_used,
        resolved=resolved,
    )
