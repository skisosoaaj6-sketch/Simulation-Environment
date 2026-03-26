"""
OpenEnv SRE Incident Response — Typed Data Contracts
All models use strict=True to enforce no coercion at API boundary.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


# ─── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    DIAGNOSE       = "diagnose"        # High-level health check on a service
    QUERY_LOGS     = "query_logs"      # Retrieve recent log output
    QUERY_METRICS  = "query_metrics"   # Pull time-series metrics
    RUN_COMMAND    = "run_command"     # Execute a shell/kubectl command
    APPLY_CONFIG   = "apply_config"    # Patch a configuration value
    ESCALATE       = "escalate"        # Hand off to human / next tier
    CLOSE_INCIDENT = "close_incident"  # Mark the incident as resolved


class IncidentSeverity(str, Enum):
    P1 = "P1"  # Critical  — service down, revenue impact
    P2 = "P2"  # High      — degraded, SLA at risk
    P3 = "P3"  # Medium    — warning, not yet impacting users
    P4 = "P4"  # Low       — informational


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent action submitted to /step."""
    model_config = ConfigDict(strict=True)

    action_type:    ActionType = Field(..., description="Type of action to perform")
    target_service: str        = Field(..., description="Service or component to act on", min_length=1, max_length=64)
    command:        str | None = Field(None, description="Command string for run_command / apply_config", max_length=512)
    rationale:      str        = Field(..., description="Agent's stated reasoning for this action", max_length=500)


# ─── Environment State Types ──────────────────────────────────────────────────

class Alert(BaseModel):
    model_config = ConfigDict(strict=True)

    alert_id:         str              = Field(..., description="Unique alert identifier")
    service:          str              = Field(..., description="Originating service")
    severity:         IncidentSeverity
    title:            str
    description:      str
    metric_value:     float | None     = None
    metric_threshold: float | None     = None
    timestamp:        str              = Field(..., description="ISO-8601 alert fire time")
    is_noise:         bool             = Field(False, description="Internal flag — NOT exposed in obs")


class ServiceHealth(BaseModel):
    model_config = ConfigDict(strict=True)

    service_name:      str
    status:            str   = Field(..., description="healthy | degraded | down")
    cpu_percent:       float = Field(..., ge=0.0, le=100.0)
    memory_percent:    float = Field(..., ge=0.0, le=100.0)
    error_rate:        float = Field(..., ge=0.0, le=1.0,  description="Fraction 0-1")
    latency_p99_ms:    float = Field(..., ge=0.0)
    replicas_ready:    int   = Field(..., ge=0)
    replicas_desired:  int   = Field(..., ge=1)


class ActionRecord(BaseModel):
    model_config = ConfigDict(strict=True)

    step:           int
    action:         Action
    reward_delta:   float
    feedback_summary: str


# ─── Observation ──────────────────────────────────────────────────────────────

class ActionFeedback(BaseModel):
    model_config = ConfigDict(strict=True)

    success:      bool
    output:       str
    side_effects: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Returned by /reset and /step. This is the full agent view of the world."""
    model_config = ConfigDict(strict=True)

    task_id:              str
    step_count:           int
    max_steps:            int
    active_alerts:        list[Alert]
    system_topology:      dict[str, ServiceHealth]
    action_history:       list[ActionRecord]
    last_action_feedback: ActionFeedback | None = None
    incident_context:     str = Field(..., description="Natural-language summary of current situation")


# ─── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    model_config = ConfigDict(strict=True)

    total:     float                = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]
    reason:    str


# ─── API Response Models ──────────────────────────────────────────────────────

class StepResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    observation: Observation
    reward:      Reward
    done:        bool
    info:        dict[str, Any]


class ResetResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    observation:      Observation
    task_description: str
    success_criteria: str


class GraderResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    task_id:   str
    score:     float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]
    verdict:   str


class TaskSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    task_id:          str
    difficulty:       str
    description:      str
    max_steps:        int
    action_schema:    dict[str, Any]
    success_criteria: str


class BaselineScore(BaseModel):
    model_config = ConfigDict(strict=True)

    task_id:    str
    score:      float = Field(..., ge=0.0, le=1.0)
    steps_used: int
    resolved:   bool


class BaselineResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    model:           str
    scores:          list[BaselineScore]
    mean_score:      float
    run_timestamp:   str
