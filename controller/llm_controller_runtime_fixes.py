"""Runtime safety fixes for LLMController.

This module patches two small control-flow regressions without changing public APIs:
1. gc() must resume the same checkpoint after heartbeat `continue` responses.
2. threshold auto-replan must return a pure bool and must not depend on heartbeat state.
"""

from __future__ import annotations

import inspect
import textwrap
import time

from . import llm_controller as _lc
from .utils import print_debug, print_t


def _patched_pause_gc_for_llm_wait(self, checkpoint_id=None) -> bool:
    """Pause gc() while an asynchronous heartbeat request is in flight.

    Return True only when gc() must preempt for a pending replan.
    Return False when the LLM wait ended with continue/discarded/no-replan and
    gc() should resume its original checkpoint loop.
    """
    if not (self.awaiting_llm_response or self.llm_wait_hover_active or self._planning_inflight):
        return bool(self._pending_heartbeat_replan_plan)

    start_ts = time.time()
    self.gc_llm_wait_pause_count += 1
    print_debug(
        f"[GC-LLM-WAIT] pause reason=heartbeat_request_started checkpoint={checkpoint_id}",
        env_var="TYPEFLY_VERBOSE_DEBUG",
    )

    while self.awaiting_llm_response or self.llm_wait_hover_active or self._planning_inflight:
        # Consume first: the worker may have queued the response just before it
        # cleared the wait flags. A full_replan_plan should preempt immediately.
        if self._consume_planning_response_queue():
            self.gc_llm_wait_total_sec += max(0.0, time.time() - start_ts)
            self.gc_llm_wait_replan_preempt_count += 1
            print_debug("[GC-LLM-WAIT] response_received response=full_replan_plan", env_var="TYPEFLY_VERBOSE_DEBUG")
            print_debug("[GC-LLM-WAIT] preempt_for_replan", env_var="TYPEFLY_VERBOSE_DEBUG")
            return True

        try:
            if hasattr(self.drone, "hold_position"):
                held = self.drone.hold_position()
            elif hasattr(self.drone, "set_hold_position_from_current"):
                held = self.drone.set_hold_position_from_current()
            else:
                held = None
            print_debug(f"[GC-LLM-WAIT] holding position={held}", env_var="TYPEFLY_VERBOSE_DEBUG")
        except Exception:
            pass
        time.sleep(0.05)

    # The worker may have cleared awaiting/inflight before this loop consumed the
    # response. Consume once more so continue/stale responses are recorded and a
    # late full_replan_plan can still preempt correctly.
    if self._consume_planning_response_queue():
        self.gc_llm_wait_total_sec += max(0.0, time.time() - start_ts)
        self.gc_llm_wait_replan_preempt_count += 1
        print_debug("[GC-LLM-WAIT] response_received response=full_replan_plan", env_var="TYPEFLY_VERBOSE_DEBUG")
        print_debug("[GC-LLM-WAIT] preempt_for_replan", env_var="TYPEFLY_VERBOSE_DEBUG")
        return True

    self.gc_llm_wait_total_sec += max(0.0, time.time() - start_ts)
    if self._pending_heartbeat_replan_plan:
        self.gc_llm_wait_replan_preempt_count += 1
        print_debug("[GC-LLM-WAIT] preempt_for_replan", env_var="TYPEFLY_VERBOSE_DEBUG")
        return True

    self.gc_llm_wait_resume_count += 1
    print_debug("[GC-LLM-WAIT] response_received response=continue_or_discarded", env_var="TYPEFLY_VERBOSE_DEBUG")
    print_debug("[GC-LLM-WAIT] resume_original_plan", env_var="TYPEFLY_VERBOSE_DEBUG")
    return False


def _patched_should_trigger_auto_replan(self, predicted_p: float, source: str) -> bool:
    """Pure boolean threshold gate for Event-PredRisk mode."""
    if not self._is_threshold_replan_mode():
        return False

    if int(getattr(self, "_replan_attempts", 0)) >= int(self.replan_limit):
        print_debug(
            "[REPLAN_DEBUG] "
            f"auto_replan_suppressed p={float(predicted_p):.6f} reason=max_replan_attempts_reached "
            f"current={int(getattr(self, '_replan_attempts', 0))} limit={int(self.replan_limit)} source={source}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return False

    predicted_p = float(predicted_p)
    threshold = float(self.predicted_collision_replan_threshold)
    rearm_threshold = float(self.predicted_collision_rearm_threshold)

    if self.auto_replan_protection_remaining > 0:
        if str(source) == "go_checkpoint_loop" and predicted_p >= threshold:
            print_debug(
                "[REPLAN_DEBUG] "
                f"protection_window_bypassed_for_gc p={predicted_p:.6f} "
                f"remaining_statements={self.auto_replan_protection_remaining} source={source}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        else:
            print_debug(
                "[REPLAN_DEBUG] "
                f"auto_replan_suppressed p={predicted_p:.6f} reason=protection_window "
                f"remaining_statements={self.auto_replan_protection_remaining} source={source}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
            return False

    if not self.auto_replan_armed:
        if predicted_p <= rearm_threshold:
            self.auto_replan_armed = True
            print_debug(
                "[REPLAN_DEBUG] "
                f"auto_replan_rearmed p={predicted_p:.6f} threshold={rearm_threshold:.2f}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        else:
            print_debug(
                "[REPLAN_DEBUG] "
                f"auto_replan_suppressed p={predicted_p:.6f} reason=disarmed source={source}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        return False

    trigger_replan = (
        predicted_p > threshold
        if bool(self.predicted_collision_replan_strictly_greater)
        else predicted_p >= threshold
    )
    if trigger_replan:
        self.auto_replan_armed = False
        print_debug(
            "[REPLAN_DEBUG] "
            f"auto_replan_triggered p={predicted_p:.6f} armed=True source={source} "
            f"trigger_threshold={threshold:.2f}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        print_debug("[REPLAN_DEBUG] auto_replan_armed=False", env_var="TYPEFLY_VERBOSE_DEBUG")
        return True
    return False


def _patch_skill_go_checkpoint_return_type() -> None:
    """Fix the nested _should_preempt_for_replan() no-op return value.

    The regression returned the non-empty string "none", which is truthy and
    therefore made gc() break even when no preemption was requested.
    """
    cls = _lc.LLMController
    source = textwrap.dedent(inspect.getsource(cls.skill_go_checkpoint))
    old = '            return "none"\n\n        for idx in range(max_iterations):'
    new = '            return False\n\n        for idx in range(max_iterations):'
    if old not in source:
        # Already fixed or source changed; leave it alone.
        return
    patched_source = source.replace(old, new, 1)
    namespace = dict(_lc.__dict__)
    exec(patched_source, namespace)
    cls.skill_go_checkpoint = namespace["skill_go_checkpoint"]


def apply_runtime_fixes() -> None:
    cls = _lc.LLMController
    cls._pause_gc_for_llm_wait = _patched_pause_gc_for_llm_wait
    cls._should_trigger_auto_replan = _patched_should_trigger_auto_replan
    _patch_skill_go_checkpoint_return_type()


__all__ = ["apply_runtime_fixes"]
