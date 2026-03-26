"""
SRE Incident Response Environment — State Machine

Architecture: Cognitive Decoupling
  - _compute_step_reward()  : reward logic only, no state mutation
  - _apply_action()         : state mutation only, no reward logic
  - step()                  : orchestrates both, returns StepResponse
"""
from __future__ import annotations
import copy
from typing import Any

from app.models import (
    Action, ActionType, Observation, ActionFeedback, ActionRecord,
    Reward, StepResponse, ResetResponse, GraderResponse,
    Alert, ServiceHealth
)
from app.tasks import TaskDefinition, get_task


# ─── Feedback Templates ────────────────────────────────────────────────────────

_DIAGNOSE_TEMPLATES: dict[str, str] = {
    "api-gateway":       "api-gateway: status=down, memory_percent=100%, OOMKilled events in last 5 restarts. Container exit code: 137 (OOM).",
    "postgres-primary":  "postgres-primary: active_connections=100/100, queue_depth=47, avg_query_time=4200ms. pgbouncer pool exhausted.",
    "nginx-proxy":       "nginx-proxy: deployment v2.1.4 applied at 14:00. rate_limit changed from 1000r/s to 100r/s. 38% of requests being rate-limited.",
    "order-service":     "order-service: healthy replicas=3/3. Downstream errors from postgres-primary propagating as 503s.",
    "worker-service":    "worker-service: healthy. CPU elevation due to scheduled batch job (cron: '0 11 * * *'). No error impact.",
    "storage-service":   "storage-service: healthy. Disk 78% — pre-existing growth trend. No error rate impact.",
    "user-service":      "user-service: healthy. Memory elevated after feature-flag deploy. No error rate impact.",
    "checkout-service":  "checkout-service: degraded. 38% HTTP 429 responses from upstream nginx rate limiter.",
    "redis-cache":       "redis-cache: healthy. No anomalies detected.",
}

_LOG_TEMPLATES: dict[str, str] = {
    "api-gateway":       "[ERROR] 2026-03-26T09:59:55Z java.lang.OutOfMemoryError: Java heap space\n[WARN]  Container killed by OOM reaper (exit 137)\n[INFO]  Last successful request: 09:59:52Z",
    "postgres-primary":  "[WARN]  2026-03-26T10:57:00Z pgbouncer: pool 'main' full (100/100)\n[ERROR] remaining clients queued: 47\n[ERROR] 2026-03-26T10:58:00Z connection timeout after 5000ms",
    "nginx-proxy":       "[INFO]  2026-03-26T14:00:02Z nginx reloaded with config v2.1.4\n[WARN]  rate_limit: limit_req_zone changed from '1000r/s' to '100r/s'\n[ERROR] 14:02:15Z limit_req: 38% requests exceeded rate (429)",
    "order-service":     "[ERROR] 2026-03-26T10:58:30Z pq: sorry, too many clients already\n[ERROR] database connection pool exhausted — retry queue backing up",
    "worker-service":    "[INFO]  2026-03-26T10:55:00Z batch-job: customer_report_daily STARTED\n[INFO]  Processing 142,000 records. CPU-bound phase.",
    "checkout-service":  "[ERROR] 2026-03-26T14:02:45Z upstream returned 429 Too Many Requests\n[WARN]  retry attempt 1/3 failed: still 429",
    "storage-service":   "[INFO]  disk_usage: 78.2% — growth rate 0.3%/day (normal)",
    "user-service":      "[INFO]  feature_flag 'new_session_manager' enabled at 14:01. Memory +120MB (expected).",
    "redis-cache":       "[INFO]  All connections healthy. Hit rate: 94.2%",
}

_METRICS_TEMPLATES: dict[str, str] = {
    "api-gateway":       "api-gateway metrics — error_rate: 0.35, latency_p99: 3200ms, rps: 180, replicas_ready: 3/3",
    "postgres-primary":  "postgres-primary metrics — connections: 100/100, query_latency_p99: 4200ms, deadlocks: 3, cache_hit: 0.72",
    "nginx-proxy":       "nginx-proxy metrics — rps: 1200, rate_limited_pct: 38%, upstream_errors: 0.38, active_connections: 420",
    "order-service":     "order-service metrics — error_rate: 0.45, latency_p99: 8500ms, db_timeout_rate: 0.44",
    "worker-service":    "worker-service metrics — cpu: 85%, memory: 40%, job_progress: 31%",
    "checkout-service":  "checkout-service metrics — error_rate: 0.38, http_429_rate: 0.38, completion_rate: 0.61",
    "storage-service":   "storage-service metrics — disk: 78%, iops: 420, error_rate: 0.001",
    "user-service":      "user-service metrics — memory: 72%, cpu: 25%, error_rate: 0.002",
    "redis-cache":       "redis-cache metrics — hit_rate: 0.942, latency_p99: 1.5ms, evictions: 0",
}


# ─── Environment ──────────────────────────────────────────────────────────────

class SREEnvironment:
    """
    State machine for SRE Incident Response OpenEnv environment.

    Public interface: reset(task_id) → ResetResponse
                      step(action)   → StepResponse
                      state()        → Observation
                      grade()        → GraderResponse
    """

    def __init__(self) -> None:
        self._task:          TaskDefinition | None = None
        self._topology:      dict[str, ServiceHealth] = {}
        self._alerts:        list[Alert] = []
        self._history:       list[ActionRecord] = []
        self._step_count:    int = 0
        self._episode_done:  bool = False
        self._cumulative_reward: float = 0.0
        self._loop_counts:   dict[str, int] = {}
        self._destructive_count: int = 0
        self._root_cause_diagnosed: bool = False
        self._correct_fix_applied: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> ResetResponse:
        task = get_task(task_id)
        self._task = task
        self._topology = task.initial_topology()
        self._alerts = task.initial_alerts()
        self._history = []
        self._step_count = 0
        self._episode_done = False
        self._cumulative_reward = 0.0
        self._loop_counts = {}
        self._destructive_count = 0
        self._root_cause_diagnosed = False
        self._correct_fix_applied = False

        obs = self._build_observation()
        return ResetResponse(
            observation=obs,
            task_description=task.description,
            success_criteria=task.success_criteria,
        )

    def step(self, action: Action) -> StepResponse:
        if self._task is None:
            raise RuntimeError("Environment not reset. Call /reset first.")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call /reset to start a new episode.")

        self._step_count += 1

        # Phase 1: Compute reward (read-only view of state before mutation)
        reward = self._compute_step_reward(action)

        # Phase 2: Apply action (mutate state)
        feedback = self._apply_action(action)

        # Phase 3: Record in trajectory
        record = ActionRecord(
            step=self._step_count,
            action=action,
            reward_delta=reward.total,
            feedback_summary=feedback.output[:120],
        )
        self._history.append(record)
        self._cumulative_reward = min(1.0, self._cumulative_reward + reward.total)

        # Phase 4: Check termination
        done = (
            action.action_type == ActionType.CLOSE_INCIDENT
            or self._step_count >= self._task.max_steps
        )
        self._episode_done = done

        obs = self._build_observation(last_feedback=feedback)
        info: dict[str, Any] = {
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_used": self._step_count,
            "max_steps": self._task.max_steps,
            "episode_done": done,
        }

        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Observation:
        if self._task is None:
            raise RuntimeError("Environment not reset.")
        return self._build_observation()

    def grade(self) -> GraderResponse:
        if self._task is None:
            raise RuntimeError("Environment not reset.")

        episode_info = {
            "steps_used": self._step_count,
            "cumulative_reward": self._cumulative_reward,
        }
        breakdown = self._task.grade(self._history, episode_info)
        score = round(min(1.0, max(0.0, sum(breakdown.values()))), 4)

        if score >= 0.80:
            verdict = "EXCELLENT — full resolution with minimal destructive actions"
        elif score >= 0.60:
            verdict = "GOOD — root cause identified and fixed, minor inefficiencies"
        elif score >= 0.40:
            verdict = "PARTIAL — some correct diagnosis but incomplete or erroneous remediation"
        elif score >= 0.20:
            verdict = "POOR — some relevant actions but root cause missed"
        else:
            verdict = "FAILED — no meaningful progress toward resolution"

        return GraderResponse(
            task_id=self._task.task_id,
            score=score,
            breakdown=breakdown,
            verdict=verdict,
        )

    # ── Private: Reward Computation (pure, no state mutation) ─────────────────

    def _compute_step_reward(self, action: Action) -> Reward:
        assert self._task is not None
        breakdown: dict[str, float] = {}
        target = action.target_service.lower()
        cmd = (action.command or "").lower()

        # --- Positive signals ---

        # Diagnose/query on the root cause service
        if action.action_type in (ActionType.DIAGNOSE, ActionType.QUERY_LOGS, ActionType.QUERY_METRICS):
            if target == self._task.root_cause_service:
                breakdown["relevant_diagnosis"] = 0.08
                self._root_cause_diagnosed = True

        # Correct remediation action on correct service
        if action.action_type == self._task.correct_remediation_action:
            if target == self._task.root_cause_service:
                if any(kw in cmd for kw in self._task.correct_remediation_keywords):
                    breakdown["correct_remediation"] = 0.25
                    self._correct_fix_applied = True
                else:
                    breakdown["correct_service_wrong_cmd"] = 0.02

        # Clean close after correct fix
        if action.action_type == ActionType.CLOSE_INCIDENT:
            if self._correct_fix_applied:
                breakdown["valid_close"] = 0.15
            else:
                breakdown["premature_close_penalty"] = -0.30

        # --- Penalties ---

        # Destructive action on healthy/irrelevant service
        if action.action_type in (ActionType.RUN_COMMAND, ActionType.APPLY_CONFIG):
            if target in self._task.destructive_services:
                breakdown["destructive_action"] = -0.12
                self._destructive_count += 1

        # Loop detection: same (action_type, target) repeated
        loop_key = f"{action.action_type.value}:{target}"
        count = self._loop_counts.get(loop_key, 0) + 1
        self._loop_counts[loop_key] = count
        if count > 1:
            breakdown[f"loop_penalty_{loop_key}"] = -0.05 * (count - 1)

        # Escalate on easy/medium (penalize avoidance of resolution)
        if action.action_type == ActionType.ESCALATE:
            if self._task.difficulty in ("easy", "medium"):
                breakdown["unnecessary_escalation"] = -0.15

        total = round(min(1.0, max(-1.0, sum(breakdown.values()))), 4)
        # Clamp per-step contribution to [-0.5, 0.5]
        total = round(max(-0.5, min(0.5, total)), 4)

        reason_parts = [f"{k}={v:+.2f}" for k, v in breakdown.items() if v != 0]
        reason = "; ".join(reason_parts) if reason_parts else "no signal this step"

        return Reward(total=max(0.0, total), breakdown=breakdown, reason=reason)

    # ── Private: State Mutation ────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> ActionFeedback:
        assert self._task is not None
        target = action.target_service.lower()
        cmd = (action.command or "").lower()

        if action.action_type == ActionType.DIAGNOSE:
            output = _DIAGNOSE_TEMPLATES.get(target, f"No diagnostic data available for '{target}'.")
            return ActionFeedback(success=True, output=output)

        if action.action_type == ActionType.QUERY_LOGS:
            output = _LOG_TEMPLATES.get(target, f"No logs found for '{target}'.")
            return ActionFeedback(success=True, output=output)

        if action.action_type == ActionType.QUERY_METRICS:
            output = _METRICS_TEMPLATES.get(target, f"No metrics available for '{target}'.")
            # If post-fix verification on downstream service, show recovery
            if self._correct_fix_applied and target in ("api-gateway", "checkout-service", "order-service"):
                output += "\n[POST-FIX] Metrics recovering — error_rate dropping toward baseline."
            return ActionFeedback(success=True, output=output)

        if action.action_type == ActionType.APPLY_CONFIG:
            if target == self._task.root_cause_service and any(
                kw in cmd for kw in self._task.correct_remediation_keywords
            ):
                # Correct fix: heal the root cause service
                if target in self._topology:
                    svc = self._topology[target]
                    self._topology[target] = ServiceHealth(
                        service_name=svc.service_name,
                        status="healthy",
                        cpu_percent=svc.cpu_percent * 0.6,
                        memory_percent=svc.memory_percent * 0.7,
                        error_rate=0.002,
                        latency_p99_ms=svc.latency_p99_ms * 0.05,
                        replicas_ready=svc.replicas_desired,
                        replicas_desired=svc.replicas_desired,
                    )
                return ActionFeedback(
                    success=True,
                    output=f"Config applied to {target}. Service recovering. Downstream metrics should normalize within 60s.",
                    side_effects={"service_healed": target},
                )
            elif target in self._task.destructive_services:
                return ActionFeedback(
                    success=False,
                    output=f"WARNING: {target} is not the root cause service. Config change applied but will not resolve the incident and may introduce instability.",
                    side_effects={"warning": "unnecessary_config_change"},
                )
            else:
                return ActionFeedback(
                    success=False,
                    output=f"Config change applied to {target} but no matching remediation keywords found. State unchanged.",
                )

        if action.action_type == ActionType.RUN_COMMAND:
            if target in self._task.destructive_services:
                return ActionFeedback(
                    success=False,
                    output=f"Command executed on {target} — this service is not the root cause. Potential instability introduced.",
                    side_effects={"warning": "destructive_command_on_healthy_service"},
                )
            return ActionFeedback(
                success=True,
                output=f"Command '{action.command}' executed on {target}. Output: OK",
            )

        if action.action_type == ActionType.ESCALATE:
            return ActionFeedback(
                success=True,
                output="Incident escalated to on-call team lead. Note: This incident may have been resolvable at this tier.",
            )

        if action.action_type == ActionType.CLOSE_INCIDENT:
            if self._correct_fix_applied:
                return ActionFeedback(
                    success=True,
                    output="Incident closed. Root cause resolved. Post-mortem auto-generated.",
                    side_effects={"incident_resolved": True},
                )
            return ActionFeedback(
                success=False,
                output="WARNING: Incident closed without confirmed resolution. Root cause may still be active.",
                side_effects={"incident_resolved": False, "premature_close": True},
            )

        return ActionFeedback(success=False, output=f"Unknown action type: {action.action_type}")

    # ── Private: Observation Builder ───────────────────────────────────────────

    def _build_observation(self, last_feedback: ActionFeedback | None = None) -> Observation:
        assert self._task is not None

        # Filter is_noise from the exposed alerts (noise is visible but not labeled as such)
        public_alerts = [
            Alert(
                alert_id=a.alert_id,
                service=a.service,
                severity=a.severity,
                title=a.title,
                description=a.description,
                metric_value=a.metric_value,
                metric_threshold=a.metric_threshold,
                timestamp=a.timestamp,
                is_noise=False,  # never expose internal flag to agent
            )
            for a in self._alerts
        ]

        # Build incident context narrative
        degraded = [s for s in self._topology.values() if s.status != "healthy"]
        if degraded:
            context = (
                f"ACTIVE INCIDENT — {len(self._alerts)} alert(s) firing. "
                f"{len(degraded)} service(s) degraded/down: "
                + ", ".join(f"{s.service_name}({s.status})" for s in degraded)
                + f". Step {self._step_count}/{self._task.max_steps}."
            )
        else:
            context = f"All services healthy. Incident may be resolved. Step {self._step_count}/{self._task.max_steps}."

        return Observation(
            task_id=self._task.task_id,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            active_alerts=public_alerts,
            system_topology=self._topology,
            action_history=self._history[-10:],  # last 10 for context window efficiency
            last_action_feedback=last_feedback,
            incident_context=context,
        )
