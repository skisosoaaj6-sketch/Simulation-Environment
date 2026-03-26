---
title: SRE Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - devops
  - incident-response
  - reinforcement-learning
---

# SRE Incident Response — OpenEnv Environment

A real-world **Site Reliability Engineering (SRE) incident response simulation** where an AI agent acts as an on-call engineer. The agent diagnoses cloud infrastructure failures, traces root causes through log and metric queries, applies targeted remediations, and closes incidents — modeled on real-world PagerDuty/Datadog triage workflows.

## Environment Description

The agent receives a set of firing alerts and a live service topology snapshot. It must navigate the dependency graph, distinguish root causes from symptomatic cascades, apply correct configuration remediations, and close the incident cleanly — all while avoiding destructive actions on healthy services.

## Action Space

| Action Type | Description |
|---|---|
| `diagnose` | High-level health check on a service |
| `query_logs` | Retrieve recent log output |
| `query_metrics` | Pull time-series metrics |
| `run_command` | Execute a shell/kubectl command |
| `apply_config` | Patch a configuration value |
| `escalate` | Hand off to human/next tier |
| `close_incident` | Mark incident resolved |

**Action Schema:**
```json
{
  "action_type": "apply_config",
  "target_service": "postgres-primary",
  "command": "pgbouncer set max_connections=200",
  "rationale": "Connection pool exhausted — increasing pool size to unblock upstream."
}
```

## Observation Space

```json
{
  "task_id": "db-connection-cascade",
  "step_count": 3,
  "max_steps": 15,
  "active_alerts": [...],
  "system_topology": {
    "postgres-primary": {
      "status": "degraded",
      "error_rate": 0.30,
      "latency_p99_ms": 4200.0,
      ...
    }
  },
  "action_history": [...],
  "last_action_feedback": {"success": true, "output": "..."},
  "incident_context": "ACTIVE INCIDENT — 3 alert(s) firing..."
}
```

## Tasks

| ID | Difficulty | Max Steps | Description |
|---|---|---|---|
| `pod-oom-restart` | Easy | 10 | Single pod OOMKilled. Linear resolution path. |
| `db-connection-cascade` | Medium | 15 | DB connection cascade with red-herring alert. |
| `multi-service-brownout` | Hard | 20 | 5 alerts, 2 noise, nginx misconfiguration root cause. |

## Reward Function (Dense)

Reward is provided at every step — no sparse end-of-episode signals.

| Signal | Value |
|---|---|
| Query root cause service | +0.08 |
| Correct remediation applied | +0.25 |
| Valid close after fix | +0.15 |
| Premature close (no fix) | -0.30 |
| Destructive action on healthy service | -0.12 |
| Repeated identical action (loop) | -0.05 per repeat |
| Unnecessary escalation (easy/medium) | -0.15 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — must return 200 |
| `POST` | `/reset` | Reset environment for a task |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current observation |
| `GET` | `/tasks` | Task list + action schema |
| `POST` | `/grader` | Grade completed episode [0.0–1.0] |
| `POST` | `/baseline` | Run baseline inference (requires OPENAI_API_KEY) |

## Setup & Usage

```bash
# Local
docker build -t sre-incident-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... sre-incident-env

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "pod-oom-restart"}'
```

## Baseline Scores

| Task | Model | Score |
|---|---|---|
| `pod-oom-restart` | gpt-4o-mini | ~0.72 |
| `db-connection-cascade` | gpt-4o-mini | ~0.55 |
| `multi-service-brownout` | gpt-4o-mini | ~0.38 |
| **Mean** | | **~0.55** |

Run baseline script:
```bash
export OPENAI_API_KEY=sk-...
python baseline.py --env-url http://localhost:7860
```
