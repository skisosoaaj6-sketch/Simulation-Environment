"""
Task Registry — 3 SRE Incident Tasks (Easy / Medium / Hard)

Each task defines:
  - initial_state():  factory returning fresh ServiceHealth + Alert snapshots
  - grade(trajectory): deterministic grader → float[0.0, 1.0]
  - RESOLUTION_CRITERIA: ground truth for grader
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any

from app.models import (
    Alert, ServiceHealth, IncidentSeverity, ActionRecord, ActionType
)


# ─── Task Definition Dataclass ────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    task_id:           str
    difficulty:        str
    description:       str
    max_steps:         int
    success_criteria:  str
    # Ground truth signals — used by grader, never exposed in Observation
    root_cause_service:     str
    correct_remediation_keywords: list[str]   # substrings that must appear in command/target
    correct_remediation_action: ActionType
    destructive_services:   list[str]          # services that SHOULD NOT be touched
    noise_alert_services:   list[str]          # alerts that are irrelevant noise
    resolution_requires_verification: bool = False

    def initial_alerts(self) -> list[Alert]:
        raise NotImplementedError

    def initial_topology(self) -> dict[str, ServiceHealth]:
        raise NotImplementedError

    def grade(self, trajectory: list[ActionRecord], episode_info: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError


# ─── Task 1: Easy — Pod OOM Restart ──────────────────────────────────────────

class PodOOMTask(TaskDefinition):
    """
    api-gateway is OOMKilled. Agent must query logs → apply memory config → close.
    Single service, linear path. Optimal: 3 steps.
    """

    def __init__(self) -> None:
        super().__init__(
            task_id="pod-oom-restart",
            difficulty="easy",
            description=(
                "The api-gateway service has been OOMKilled and is crash-looping. "
                "Diagnose the root cause and apply the correct remediation to restore service."
            ),
            max_steps=10,
            success_criteria=(
                "Query logs on api-gateway to identify OOM, apply a memory limit increase "
                "via apply_config, and close the incident after recovery is confirmed."
            ),
            root_cause_service="api-gateway",
            correct_remediation_keywords=["memory", "mem_limit", "resources", "limit"],
            correct_remediation_action=ActionType.APPLY_CONFIG,
            destructive_services=["postgres-primary", "redis-cache"],
            noise_alert_services=[],
            resolution_requires_verification=False,
        )

    def initial_alerts(self) -> list[Alert]:
        return [
            Alert(
                alert_id="ALT-001",
                service="api-gateway",
                severity=IncidentSeverity.P1,
                title="api-gateway OOMKilled — replicas 0/2",
                description="Container api-gateway exceeded memory limit (512Mi) and was killed by the kernel OOM reaper. Pod is crash-looping.",
                metric_value=512.0,
                metric_threshold=512.0,
                timestamp="2026-03-26T10:00:00Z",
                is_noise=False,
            ),
        ]

    def initial_topology(self) -> dict[str, ServiceHealth]:
        return {
            "api-gateway": ServiceHealth(
                service_name="api-gateway",
                status="down",
                cpu_percent=0.0,
                memory_percent=100.0,
                error_rate=1.0,
                latency_p99_ms=0.0,
                replicas_ready=0,
                replicas_desired=2,
            ),
            "postgres-primary": ServiceHealth(
                service_name="postgres-primary",
                status="healthy",
                cpu_percent=22.0,
                memory_percent=45.0,
                error_rate=0.001,
                latency_p99_ms=8.0,
                replicas_ready=1,
                replicas_desired=1,
            ),
            "redis-cache": ServiceHealth(
                service_name="redis-cache",
                status="healthy",
                cpu_percent=8.0,
                memory_percent=30.0,
                error_rate=0.0,
                latency_p99_ms=2.0,
                replicas_ready=1,
                replicas_desired=1,
            ),
        }

    def grade(self, trajectory: list[ActionRecord], episode_info: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {
            "root_cause_queried":   0.0,
            "correct_remediation":  0.0,
            "no_destructive_action": 1.0,
            "clean_close":          0.0,
            "efficiency":           0.0,
        }

        queried_root_cause = False
        correct_fix_applied = False
        destructive_touched = False
        incident_closed_correctly = False

        for rec in trajectory:
            a = rec.action
            target = a.target_service.lower()
            cmd = (a.command or "").lower()

            # Root cause query
            if a.action_type in (ActionType.QUERY_LOGS, ActionType.QUERY_METRICS, ActionType.DIAGNOSE):
                if target == self.root_cause_service:
                    queried_root_cause = True

            # Correct remediation
            if a.action_type == ActionType.APPLY_CONFIG and target == self.root_cause_service:
                if any(kw in cmd for kw in self.correct_remediation_keywords):
                    correct_fix_applied = True

            # Destructive action
            if a.action_type in (ActionType.RUN_COMMAND, ActionType.APPLY_CONFIG):
                if target in self.destructive_services:
                    destructive_touched = True

            # Correct close
            if a.action_type == ActionType.CLOSE_INCIDENT and correct_fix_applied:
                incident_closed_correctly = True

        if queried_root_cause:
            scores["root_cause_queried"] = 0.25
        if correct_fix_applied:
            scores["correct_remediation"] = 0.45
        if destructive_touched:
            scores["no_destructive_action"] = 0.0
        if incident_closed_correctly:
            scores["clean_close"] = 0.20

        # Efficiency: optimal is 3 steps, max is 10
        steps = episode_info.get("steps_used", self.max_steps)
        optimal = 3
        if correct_fix_applied:
            efficiency = max(0.0, (self.max_steps - steps) / (self.max_steps - optimal))
            scores["efficiency"] = round(min(0.10, efficiency * 0.10), 4)

        return scores


# ─── Task 2: Medium — DB Connection Cascade ───────────────────────────────────

class DBConnectionCascadeTask(TaskDefinition):
    """
    postgres-primary connection pool exhausted → order-service 503s.
    Red herring: worker-service high CPU (unrelated). Optimal: 4 steps.
    """

    def __init__(self) -> None:
        super().__init__(
            task_id="db-connection-cascade",
            difficulty="medium",
            description=(
                "order-service is returning 503 errors. Multiple alerts are firing. "
                "Identify the true root cause in the dependency chain and remediate it. "
                "Not all alerts are causally related."
            ),
            max_steps=15,
            success_criteria=(
                "Identify that postgres-primary connection pool is exhausted (not order-service itself). "
                "Apply a connection pool config fix to postgres-primary. "
                "Do NOT restart or modify worker-service. Close the incident."
            ),
            root_cause_service="postgres-primary",
            correct_remediation_keywords=["max_connections", "pool_size", "connection", "pgbouncer"],
            correct_remediation_action=ActionType.APPLY_CONFIG,
            destructive_services=["worker-service", "order-service"],
            noise_alert_services=["worker-service"],
            resolution_requires_verification=False,
        )

    def initial_alerts(self) -> list[Alert]:
        return [
            Alert(
                alert_id="ALT-010",
                service="order-service",
                severity=IncidentSeverity.P1,
                title="order-service error rate 45% — HTTP 503",
                description="order-service is returning HTTP 503 to 45% of requests. Database timeout errors in logs.",
                metric_value=0.45,
                metric_threshold=0.05,
                timestamp="2026-03-26T11:00:00Z",
                is_noise=False,
            ),
            Alert(
                alert_id="ALT-011",
                service="postgres-primary",
                severity=IncidentSeverity.P2,
                title="postgres-primary connection pool 100% utilized",
                description="All 100 connections in pgbouncer pool are active. New connection requests are queuing.",
                metric_value=100.0,
                metric_threshold=90.0,
                timestamp="2026-03-26T10:58:00Z",
                is_noise=False,
            ),
            Alert(
                alert_id="ALT-012",
                service="worker-service",
                severity=IncidentSeverity.P3,
                title="worker-service CPU at 85%",
                description="worker-service CPU utilization elevated. Cause: scheduled batch job started at 10:55.",
                metric_value=85.0,
                metric_threshold=80.0,
                timestamp="2026-03-26T10:55:00Z",
                is_noise=True,
            ),
        ]

    def initial_topology(self) -> dict[str, ServiceHealth]:
        return {
            "order-service": ServiceHealth(
                service_name="order-service",
                status="degraded",
                cpu_percent=40.0,
                memory_percent=55.0,
                error_rate=0.45,
                latency_p99_ms=8500.0,
                replicas_ready=3,
                replicas_desired=3,
            ),
            "postgres-primary": ServiceHealth(
                service_name="postgres-primary",
                status="degraded",
                cpu_percent=70.0,
                memory_percent=60.0,
                error_rate=0.30,
                latency_p99_ms=4200.0,
                replicas_ready=1,
                replicas_desired=1,
            ),
            "worker-service": ServiceHealth(
                service_name="worker-service",
                status="healthy",
                cpu_percent=85.0,
                memory_percent=40.0,
                error_rate=0.001,
                latency_p99_ms=120.0,
                replicas_ready=2,
                replicas_desired=2,
            ),
            "redis-cache": ServiceHealth(
                service_name="redis-cache",
                status="healthy",
                cpu_percent=10.0,
                memory_percent=25.0,
                error_rate=0.0,
                latency_p99_ms=1.5,
                replicas_ready=1,
                replicas_desired=1,
            ),
        }

    def grade(self, trajectory: list[ActionRecord], episode_info: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {
            "identified_root_cause_service": 0.0,
            "correct_remediation_applied":   0.0,
            "avoided_red_herring":           0.20,   # starts full, deducted on violation
            "avoided_symptom_fix":           0.20,   # penalize fixing order-service directly
            "clean_close":                   0.0,
            "efficiency":                    0.0,
        }

        queried_db = False
        correct_fix = False
        touched_worker = False
        restarted_order_service = False
        incident_closed_correctly = False

        for rec in trajectory:
            a = rec.action
            target = a.target_service.lower()
            cmd = (a.command or "").lower()

            # Query root cause service
            if a.action_type in (ActionType.QUERY_METRICS, ActionType.QUERY_LOGS, ActionType.DIAGNOSE):
                if target == "postgres-primary":
                    queried_db = True

            # Correct fix
            if a.action_type == ActionType.APPLY_CONFIG and target == "postgres-primary":
                if any(kw in cmd for kw in self.correct_remediation_keywords):
                    correct_fix = True

            # Red herring — touching worker
            if target == "worker-service" and a.action_type in (
                ActionType.RUN_COMMAND, ActionType.APPLY_CONFIG
            ):
                touched_worker = True

            # Symptom fix — restarting order-service (wrong layer)
            if target == "order-service" and a.action_type in (
                ActionType.RUN_COMMAND, ActionType.APPLY_CONFIG
            ):
                restarted_order_service = True

            # Close
            if a.action_type == ActionType.CLOSE_INCIDENT and correct_fix:
                incident_closed_correctly = True

        if queried_db:
            scores["identified_root_cause_service"] = 0.20
        if correct_fix:
            scores["correct_remediation_applied"] = 0.35
        if touched_worker:
            scores["avoided_red_herring"] = 0.0
        if restarted_order_service:
            scores["avoided_symptom_fix"] = 0.0
        if incident_closed_correctly:
            scores["clean_close"] = 0.15

        steps = episode_info.get("steps_used", self.max_steps)
        optimal = 4
        if correct_fix:
            eff = max(0.0, (self.max_steps - steps) / (self.max_steps - optimal))
            scores["efficiency"] = round(min(0.10, eff * 0.10), 4)

        return scores


# ─── Task 3: Hard — Multi-Service Brownout ────────────────────────────────────

class MultiServiceBrownoutTask(TaskDefinition):
    """
    nginx rate limit misconfigured after deploy → 3 services show elevated errors.
    5 alerts: 2 symptomatic, 1 correlated (nginx), 1 noise (disk), 1 misleading (memory).
    Agent must: trace to nginx → verify deploy event → rollback/patch config → 
    verify 2 downstream services recover → close. Optimal: 7 steps.
    """

    def __init__(self) -> None:
        super().__init__(
            task_id="multi-service-brownout",
            difficulty="hard",
            description=(
                "Multiple services are showing elevated error rates and latency. "
                "5 alerts are firing simultaneously. Identify the single root cause "
                "in the shared infrastructure layer, apply the correct fix, "
                "verify downstream recovery, and close the incident."
            ),
            max_steps=20,
            success_criteria=(
                "Identify nginx-proxy as root cause (rate_limit misconfiguration from recent deploy). "
                "Apply config rollback/fix to nginx-proxy. "
                "Verify api-gateway AND checkout-service recover (query metrics on both). "
                "Do NOT act on storage-service (noise) or user-service (misleading). "
                "Close incident after verification."
            ),
            root_cause_service="nginx-proxy",
            correct_remediation_keywords=["rate_limit", "rate-limit", "rollback", "nginx.conf", "limit_req"],
            correct_remediation_action=ActionType.APPLY_CONFIG,
            destructive_services=["storage-service", "user-service"],
            noise_alert_services=["storage-service"],
            resolution_requires_verification=True,
        )

    def initial_alerts(self) -> list[Alert]:
        return [
            Alert(
                alert_id="ALT-020",
                service="api-gateway",
                severity=IncidentSeverity.P1,
                title="api-gateway latency P99 > 3000ms",
                description="api-gateway P99 latency spiked from 120ms to 3200ms at 14:02. Correlates with increased HTTP 429 responses from upstream.",
                metric_value=3200.0,
                metric_threshold=500.0,
                timestamp="2026-03-26T14:02:00Z",
                is_noise=False,
            ),
            Alert(
                alert_id="ALT-021",
                service="checkout-service",
                severity=IncidentSeverity.P1,
                title="checkout-service HTTP 429 rate 38%",
                description="38% of requests to checkout-service are being rate-limited (HTTP 429). Checkout completion rate dropped to 61%.",
                metric_value=0.38,
                metric_threshold=0.01,
                timestamp="2026-03-26T14:02:30Z",
                is_noise=False,
            ),
            Alert(
                alert_id="ALT-022",
                service="nginx-proxy",
                severity=IncidentSeverity.P2,
                title="nginx-proxy rate_limit config changed at 14:00",
                description="Deployment nginx-proxy:v2.1.4 changed rate_limit from 1000r/s to 100r/s at 14:00. Deploy by CI pipeline.",
                metric_value=None,
                metric_threshold=None,
                timestamp="2026-03-26T14:00:00Z",
                is_noise=False,
            ),
            Alert(
                alert_id="ALT-023",
                service="storage-service",
                severity=IncidentSeverity.P3,
                title="storage-service disk usage 78%",
                description="storage-service disk at 78%. Pre-existing trend; not related to current incident. Threshold crossed at 13:45.",
                metric_value=78.0,
                metric_threshold=75.0,
                timestamp="2026-03-26T13:45:00Z",
                is_noise=True,
            ),
            Alert(
                alert_id="ALT-024",
                service="user-service",
                severity=IncidentSeverity.P3,
                title="user-service memory 72% — elevated",
                description="user-service memory elevated after separate feature flag deploy. Service is healthy; no error rate impact.",
                metric_value=72.0,
                metric_threshold=70.0,
                timestamp="2026-03-26T14:01:00Z",
                is_noise=True,
            ),
        ]

    def initial_topology(self) -> dict[str, ServiceHealth]:
        return {
            "nginx-proxy": ServiceHealth(
                service_name="nginx-proxy",
                status="degraded",
                cpu_percent=55.0,
                memory_percent=40.0,
                error_rate=0.38,
                latency_p99_ms=250.0,
                replicas_ready=2,
                replicas_desired=2,
            ),
            "api-gateway": ServiceHealth(
                service_name="api-gateway",
                status="degraded",
                cpu_percent=35.0,
                memory_percent=50.0,
                error_rate=0.35,
                latency_p99_ms=3200.0,
                replicas_ready=3,
                replicas_desired=3,
            ),
            "checkout-service": ServiceHealth(
                service_name="checkout-service",
                status="degraded",
                cpu_percent=30.0,
                memory_percent=45.0,
                error_rate=0.38,
                latency_p99_ms=2800.0,
                replicas_ready=2,
                replicas_desired=2,
            ),
            "storage-service": ServiceHealth(
                service_name="storage-service",
                status="healthy",
                cpu_percent=20.0,
                memory_percent=78.0,
                error_rate=0.001,
                latency_p99_ms=15.0,
                replicas_ready=3,
                replicas_desired=3,
            ),
            "user-service": ServiceHealth(
                service_name="user-service",
                status="healthy",
                cpu_percent=25.0,
                memory_percent=72.0,
                error_rate=0.002,
                latency_p99_ms=80.0,
                replicas_ready=2,
                replicas_desired=2,
            ),
        }

    def grade(self, trajectory: list[ActionRecord], episode_info: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {
            "root_cause_identified":    0.0,  # Queried/diagnosed nginx-proxy
            "deploy_event_traced":      0.0,  # Queried logs on nginx for deploy event
            "correct_fix_applied":      0.0,  # apply_config to nginx with rate_limit keyword
            "downstream_verified":      0.0,  # Queried metrics on api-gateway AND checkout-service post-fix
            "noise_alerts_ignored":     0.20, # Did NOT act on storage-service / user-service
            "clean_close":              0.0,
            "efficiency":               0.0,
        }

        diagnosed_nginx = False
        queried_nginx_logs = False
        correct_fix = False
        verified_api_gateway = False
        verified_checkout = False
        acted_on_noise = False
        incident_closed = False
        fix_step_index = None

        for i, rec in enumerate(trajectory):
            a = rec.action
            target = a.target_service.lower()
            cmd = (a.command or "").lower()

            if a.action_type == ActionType.DIAGNOSE and target == "nginx-proxy":
                diagnosed_nginx = True

            if a.action_type == ActionType.QUERY_LOGS and target == "nginx-proxy":
                queried_nginx_logs = True

            if a.action_type == ActionType.APPLY_CONFIG and target == "nginx-proxy":
                if any(kw in cmd for kw in self.correct_remediation_keywords):
                    correct_fix = True
                    fix_step_index = i

            # Verification: must happen AFTER fix is applied
            if correct_fix and fix_step_index is not None and i > fix_step_index:
                if a.action_type == ActionType.QUERY_METRICS:
                    if target == "api-gateway":
                        verified_api_gateway = True
                    if target == "checkout-service":
                        verified_checkout = True

            if target in ("storage-service", "user-service") and a.action_type in (
                ActionType.RUN_COMMAND, ActionType.APPLY_CONFIG
            ):
                acted_on_noise = True

            if a.action_type == ActionType.CLOSE_INCIDENT:
                incident_closed = True

        if diagnosed_nginx:
            scores["root_cause_identified"] = 0.15
        if queried_nginx_logs:
            scores["deploy_event_traced"] = 0.10
        if correct_fix:
            scores["correct_fix_applied"] = 0.30
        if verified_api_gateway and verified_checkout:
            scores["downstream_verified"] = 0.15
        elif verified_api_gateway or verified_checkout:
            scores["downstream_verified"] = 0.07
        if acted_on_noise:
            scores["noise_alerts_ignored"] = 0.0

        if incident_closed and correct_fix and (verified_api_gateway or verified_checkout):
            scores["clean_close"] = 0.15

        steps = episode_info.get("steps_used", self.max_steps)
        optimal = 7
        if correct_fix:
            eff = max(0.0, (self.max_steps - steps) / (self.max_steps - optimal))
            scores["efficiency"] = round(min(0.15, eff * 0.15), 4)

        return scores


# ─── Task Registry ─────────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, TaskDefinition] = {
    "pod-oom-restart":         PodOOMTask(),
    "db-connection-cascade":   DBConnectionCascadeTask(),
    "multi-service-brownout":  MultiServiceBrownoutTask(),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]
