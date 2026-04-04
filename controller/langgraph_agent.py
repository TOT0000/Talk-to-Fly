from __future__ import annotations

import math
import time
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .benchmark_layout import BENCHMARK_CHECKPOINTS_BY_ID

FrameworkMode = Literal["typefly_baseline", "langgraph_agent"]
MissionStatus = Literal["running", "completed", "failed"]
RouteDecision = Literal["continue", "retry_plan", "reselect_subgoal", "end", "abort"]


class AgentState(TypedDict, total=False):
    # task layer
    user_task: str
    task_id: str
    framework_mode: FrameworkMode
    mission_status: MissionStatus

    # objective layer
    active_zone_ids: list[str]
    active_checkpoint_ids: list[str]
    completed_checkpoint_ids: list[str]
    remaining_checkpoint_ids: list[str]
    current_subgoal_type: str
    current_subgoal_id: str | None
    subgoal_queue: list[str]

    # observation layer
    latest_snapshot: dict[str, Any] | None
    uav_est_xy: tuple[float, float] | None
    worker_states: list[dict[str, Any]]
    current_collision_risk: float
    historical_max_collision_risk: float
    per_worker_collision_risks: dict[str, float]
    dominant_risky_worker: str | None
    mission_collision_count: int

    # execution layer
    last_plan_text: str
    last_action_text: str
    last_action_result: dict[str, Any]
    execution_history: list[dict[str, Any]]
    replan_count: int
    agent_step_count: int

    # protection layer
    auto_replan_armed: bool
    replan_protection_remaining_steps: int
    max_agent_steps: int
    max_replan_attempts: int
    last_error: str | None

    # internal routing
    route_decision: RouteDecision
    last_progress_signature: str
    last_subgoal_distance_m: float | None
    no_progress_steps: int
    repeated_action_count: int


class LangGraphOrchestrationRunner:
    def __init__(self, controller):
        self.controller = controller
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def run_task(self, task_description: str, task_id: str, active_objective_set: dict[str, Any]) -> AgentState:
        active_checkpoint_ids = [str(v).upper() for v in active_objective_set.get("active_checkpoint_ids", [])]
        active_zone_ids = [str(v) for v in active_objective_set.get("active_zone_ids", [])]
        decomposed = self.controller.planner.decompose_task_for_langgraph(
            task_description=task_description,
            active_checkpoint_ids=active_checkpoint_ids,
            active_zone_ids=active_zone_ids,
        )
        if not decomposed:
            decomposed = list(active_checkpoint_ids)
        initial_state: AgentState = {
            "user_task": task_description,
            "task_id": task_id,
            "framework_mode": "langgraph_agent",
            "mission_status": "running",
            "active_zone_ids": active_zone_ids,
            "active_checkpoint_ids": active_checkpoint_ids,
            "completed_checkpoint_ids": [],
            "remaining_checkpoint_ids": list(decomposed),
            "current_subgoal_type": "checkpoint",
            "current_subgoal_id": None,
            "subgoal_queue": list(decomposed),
            "latest_snapshot": None,
            "uav_est_xy": None,
            "worker_states": [],
            "current_collision_risk": 0.0,
            "historical_max_collision_risk": 0.0,
            "per_worker_collision_risks": {},
            "dominant_risky_worker": None,
            "mission_collision_count": 0,
            "last_plan_text": "",
            "last_action_text": "",
            "last_action_result": {},
            "execution_history": [],
            "replan_count": 0,
            "agent_step_count": 0,
            "auto_replan_armed": bool(getattr(self.controller, "auto_replan_armed", True)),
            "replan_protection_remaining_steps": int(getattr(self.controller, "auto_replan_protection_remaining", 0)),
            "max_agent_steps": 64,
            "max_replan_attempts": 8,
            "last_error": None,
            "route_decision": "continue",
            "last_progress_signature": "",
            "last_subgoal_distance_m": None,
            "no_progress_steps": 0,
            "repeated_action_count": 0,
        }
        self._emit_agent_message(
            f"[AGENT] subgoal queue: {' -> '.join(list(decomposed)) if decomposed else '(empty)'}"
        )
        config = {"configurable": {"thread_id": task_id}}
        return self.graph.invoke(initial_state, config=config)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_runtime_state", self._node_load_runtime_state)
        builder.add_node("refresh_progress", self._node_refresh_progress)
        builder.add_node("select_subgoal", self._node_select_subgoal)
        builder.add_node("plan_step", self._node_plan_step)
        builder.add_node("execute_step", self._node_execute_step)
        builder.add_node("evaluate_outcome", self._node_evaluate_outcome)

        builder.add_edge(START, "load_runtime_state")
        builder.add_edge("load_runtime_state", "refresh_progress")
        builder.add_edge("refresh_progress", "select_subgoal")
        builder.add_edge("select_subgoal", "plan_step")
        builder.add_edge("plan_step", "execute_step")
        builder.add_edge("execute_step", "evaluate_outcome")
        builder.add_conditional_edges(
            "evaluate_outcome",
            self._route_from_evaluation,
            {
                "continue": "load_runtime_state",
                "retry_plan": "plan_step",
                "reselect_subgoal": "select_subgoal",
                "end": END,
                "abort": END,
            },
        )
        return builder.compile(checkpointer=self.checkpointer)

    def _node_load_runtime_state(self, state: AgentState) -> AgentState:
        snapshot = self.controller.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        workers = list(snapshot.get("workers") or []) if isinstance(snapshot, dict) else []
        per_worker = {}
        if safety_context is not None:
            per_worker = {
                str(item.get("id")): float(item.get("collision_probability", 0.0))
                for item in (getattr(safety_context, "per_worker_collision_probabilities", []) or [])
            }
        uav_est = None
        if isinstance(snapshot, dict):
            uav_pos = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est")
            if uav_pos is not None:
                uav_est = (float(uav_pos[0]), float(uav_pos[1]))
        return {
            "latest_snapshot": snapshot,
            "uav_est_xy": uav_est,
            "worker_states": workers,
            "current_collision_risk": 0.0 if safety_context is None else float(getattr(safety_context, "current_collision_probability", 0.0)),
            "historical_max_collision_risk": 0.0 if safety_context is None else float(getattr(safety_context, "historical_max_collision_probability", 0.0)),
            "per_worker_collision_risks": per_worker,
            "dominant_risky_worker": None if safety_context is None else str(getattr(safety_context, "dominant_threat_id", "unknown")),
            "mission_collision_count": int(state.get("mission_collision_count", 0)),
        }

    def _node_refresh_progress(self, state: AgentState) -> AgentState:
        snapshot = state.get("latest_snapshot") or {}
        completed = set(str(v).upper() for v in state.get("completed_checkpoint_ids", []))
        progress = snapshot.get("benchmark_progress") if isinstance(snapshot, dict) else None
        if isinstance(progress, dict):
            completed.update(str(v).upper() for v in progress.get("completed", []))
        active_ids = [str(v).upper() for v in state.get("active_checkpoint_ids", [])]
        remaining = [cid for cid in active_ids if cid not in completed]
        return {
            "completed_checkpoint_ids": sorted(completed),
            "remaining_checkpoint_ids": remaining,
            "subgoal_queue": list(remaining),
        }

    def _node_select_subgoal(self, state: AgentState) -> AgentState:
        remaining = list(state.get("remaining_checkpoint_ids", []))
        current_subgoal = state.get("current_subgoal_id")
        completed = set(state.get("completed_checkpoint_ids", []))
        if current_subgoal is None or current_subgoal in completed:
            current_subgoal = remaining[0] if remaining else None
        if (
            state.get("route_decision") == "reselect_subgoal"
            and current_subgoal in remaining
            and len(remaining) > 1
            and int(state.get("no_progress_steps", 0)) >= 2
        ):
            idx = remaining.index(current_subgoal)
            current_subgoal = remaining[(idx + 1) % len(remaining)]
        return {
            "current_subgoal_type": "checkpoint",
            "current_subgoal_id": current_subgoal,
        }

    def _node_plan_step(self, state: AgentState) -> AgentState:
        subgoal = state.get("current_subgoal_id")
        if subgoal is None:
            return {"last_plan_text": "", "last_action_text": "", "route_decision": "end"}

        collision_risk = float(state.get("current_collision_risk", 0.0))
        plan = self.controller.planner.plan_langgraph_step_action(
            task_description=str(state.get("user_task", "")),
            current_subgoal=str(subgoal),
            remaining_checkpoints=list(state.get("remaining_checkpoint_ids", [])),
            current_collision_risk=collision_risk,
            last_action=str(state.get("last_action_text", "")),
            last_result=str((state.get("last_action_result") or {}).get("message", "")),
            stall_count=int(state.get("no_progress_steps", 0)),
            recent_history=list(state.get("execution_history", [])),
        )
        if not plan:
            plan = f'go_checkpoint("{subgoal}");'
        action_text = plan
        self._emit_agent_message(f"[STEP] current subgoal: {subgoal}")
        self._emit_agent_message(f"[ACTION] {plan}")
        return {
            "last_plan_text": plan,
            "last_action_text": action_text,
            "route_decision": "continue",
        }

    def _node_execute_step(self, state: AgentState) -> AgentState:
        plan = str(state.get("last_plan_text", "")).strip()
        if not plan:
            return {
                "last_action_result": {"ok": False, "recoverable": False, "message": "empty_plan"},
                "last_error": "empty_plan",
            }
        try:
            ret = self.controller.execute_minispec(plan, silent=True)
            ok = True
            recoverable = False
            if isinstance(ret, tuple) and len(ret) >= 2:
                ok = bool(ret[0] is not False)
            if hasattr(ret, "replan") and bool(ret.replan):
                ok = False
                recoverable = True
            return {
                "last_action_result": {
                    "ok": ok,
                    "recoverable": recoverable,
                    "message": str(getattr(ret, "value", ret)),
                },
                "last_error": None if ok else str(getattr(ret, "value", "step_failed")),
            }
        except Exception as exc:
            return {
                "last_action_result": {"ok": False, "recoverable": True, "message": str(exc)},
                "last_error": str(exc),
            }

    def _node_evaluate_outcome(self, state: AgentState) -> AgentState:
        step_count = int(state.get("agent_step_count", 0)) + 1
        replan_count = int(state.get("replan_count", 0))
        result = dict(state.get("last_action_result") or {})
        history = list(state.get("execution_history", []))
        history.append(
            {
                "step": step_count,
                "subgoal": state.get("current_subgoal_id"),
                "plan": state.get("last_plan_text"),
                "ok": bool(result.get("ok", False)),
                "recoverable": bool(result.get("recoverable", False)),
                "message": result.get("message", ""),
                "ts": time.time(),
            }
        )
        if not bool(result.get("ok", False)):
            replan_count += 1
        self._emit_agent_message(
            f"[RESULT] {'ok' if bool(result.get('ok', False)) else 'failed'}: {str(result.get('message', ''))[:120]}"
        )

        max_steps = int(state.get("max_agent_steps", 64))
        max_replans = int(state.get("max_replan_attempts", 8))
        if step_count >= max_steps:
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "failed",
                "route_decision": "abort",
                "last_error": f"max_agent_steps_exceeded({max_steps})",
            }
        if replan_count > max_replans:
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "failed",
                "route_decision": "abort",
                "last_error": f"max_replan_attempts_exceeded({max_replans})",
            }

        remaining = list(state.get("remaining_checkpoint_ids", []))
        subgoal = state.get("current_subgoal_id")
        progress_signature = f"{subgoal}:{','.join(remaining)}"
        prev_signature = str(state.get("last_progress_signature", ""))
        no_progress_steps = int(state.get("no_progress_steps", 0))
        repeated_action_count = int(state.get("repeated_action_count", 0))
        if len(history) >= 2 and str(history[-1].get("plan", "")) == str(history[-2].get("plan", "")):
            repeated_action_count += 1
        else:
            repeated_action_count = 0
        last_subgoal_distance = state.get("last_subgoal_distance_m")
        if subgoal in remaining:
            snapshot = state.get("latest_snapshot") or {}
            drone_xy = None
            if isinstance(snapshot, dict):
                drone_gt = snapshot.get("drone_gt")
                if drone_gt is not None:
                    drone_xy = (float(drone_gt[0]), float(drone_gt[1]))
            checkpoint = BENCHMARK_CHECKPOINTS_BY_ID.get(str(subgoal))
            if checkpoint is not None and drone_xy is not None:
                distance_m = math.hypot(drone_xy[0] - float(checkpoint.x), drone_xy[1] - float(checkpoint.y))
                previous_distance = state.get("last_subgoal_distance_m")
                if previous_distance is not None and (float(previous_distance) - float(distance_m)) < 0.05:
                    no_progress_steps += 1
                else:
                    no_progress_steps = 0
                if distance_m <= float(checkpoint.radius_m):
                    completed = set(state.get("completed_checkpoint_ids", []))
                    completed.add(str(subgoal))
                    remaining = [cid for cid in remaining if cid != subgoal]
                    state = {
                        **state,
                        "completed_checkpoint_ids": sorted(completed),
                        "remaining_checkpoint_ids": remaining,
                        "current_subgoal_id": None,
                    }
                    no_progress_steps = 0
                    repeated_action_count = 0
                    self._emit_agent_message(f"[RESULT] checkpoint completed: {subgoal}")
                progress_signature = f"{state.get('current_subgoal_id')}:{','.join(remaining)}"
                if progress_signature == prev_signature:
                    no_progress_steps += 1
                last_subgoal_distance = float(distance_m)
                if repeated_action_count >= 2 and no_progress_steps >= 2:
                    self._emit_agent_message("[AGENT] planning stalled, trigger subgoal reselection.")
                    return {
                        "route_decision": "reselect_subgoal",
                        "mission_status": "running",
                        "agent_step_count": step_count,
                        "replan_count": replan_count,
                        "execution_history": history,
                        "last_subgoal_distance_m": last_subgoal_distance,
                        "no_progress_steps": no_progress_steps,
                        "repeated_action_count": repeated_action_count,
                        "last_progress_signature": progress_signature,
                    }

        if not remaining:
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "completed",
                "route_decision": "end",
                "last_progress_signature": progress_signature,
                "last_subgoal_distance_m": last_subgoal_distance,
                "no_progress_steps": no_progress_steps,
                "repeated_action_count": repeated_action_count,
            }
        if not bool(result.get("ok", False)):
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "running",
                "route_decision": "retry_plan" if bool(result.get("recoverable", True)) else "reselect_subgoal",
                "last_progress_signature": progress_signature,
                "last_subgoal_distance_m": last_subgoal_distance,
                "no_progress_steps": no_progress_steps,
                "repeated_action_count": repeated_action_count,
            }
        return {
            "agent_step_count": step_count,
            "replan_count": replan_count,
            "execution_history": history,
            "mission_status": "running",
            "route_decision": "continue",
            "last_progress_signature": progress_signature,
            "last_subgoal_distance_m": last_subgoal_distance,
            "no_progress_steps": no_progress_steps,
            "repeated_action_count": repeated_action_count,
        }

    def _route_from_evaluation(self, state: AgentState) -> RouteDecision:
        return str(state.get("route_decision", "continue"))  # type: ignore[return-value]

    def _emit_agent_message(self, message: str):
        if hasattr(self.controller, "append_message"):
            self.controller.append_message(message)
