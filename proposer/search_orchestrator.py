from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


FAILURE_POLICY = {
    "proposal_error": {
        "budget_delta": 1,
        "freeze_branch": False,
        "allow_local_retry": True,
        "terminate_family": False,
        "next_state": "validator_failed",
    },
    "runtime_wiring_error": {
        "budget_delta": 1,
        "freeze_branch": False,
        "allow_local_retry": True,
        "terminate_family": False,
        "next_state": "validator_failed",
    },
    "benchmark_semantics_error": {
        "budget_delta": 0,
        "freeze_branch": True,
        "allow_local_retry": False,
        "terminate_family": False,
        "next_state": "system_error",
    },
    "trace_incomplete_error": {
        "budget_delta": 0,
        "freeze_branch": False,
        "allow_local_retry": True,
        "terminate_family": False,
        "next_state": "validator_failed",
    },
    "infra_crash_error": {
        "budget_delta": 0,
        "freeze_branch": False,
        "allow_local_retry": True,
        "terminate_family": False,
        "next_state": "screening_failed",
    },
    "genuine_harness_regression": {
        "budget_delta": 1,
        "freeze_branch": False,
        "allow_local_retry": False,
        "terminate_family": False,
        "next_state": "screening_failed",
    },
    "unsafe_candidate_regression": {
        "budget_delta": 1,
        "freeze_branch": False,
        "allow_local_retry": False,
        "terminate_family": True,
        "next_state": "formal_failed",
    },
}


@dataclass
class CandidateLifecycle:
    candidate_id: str
    state: str
    revise_budget_remaining: int
    branch_frozen: bool = False
    system_error_count: int = 0
    family_terminated: bool = False


class SearchOrchestrator:
    """Manages candidate lifecycle and screening->formal promotion with explicit failure handling policy."""

    def __init__(self, revise_budget: int = 3, system_error_freeze_threshold: int = 2):
        self.revise_budget = int(revise_budget)
        self.system_error_freeze_threshold = int(system_error_freeze_threshold)
        self._states: Dict[str, CandidateLifecycle] = {}

    def register_candidate(self, candidate_id: str) -> CandidateLifecycle:
        life = CandidateLifecycle(candidate_id=candidate_id, state="proposed", revise_budget_remaining=self.revise_budget)
        self._states[candidate_id] = life
        return life

    def transition(self, candidate_id: str, new_state: str) -> CandidateLifecycle:
        life = self._states[candidate_id]
        if life.branch_frozen:
            life.state = "system_error"
            return life
        life.state = str(new_state)
        if new_state in {"validator_failed", "screening_failed", "formal_failed"}:
            life.revise_budget_remaining = max(0, life.revise_budget_remaining - 1)
        if new_state == "system_error":
            life.system_error_count += 1
            if life.system_error_count >= self.system_error_freeze_threshold:
                life.branch_frozen = True
        return life

    def apply_failure(self, candidate_id: str, failure_type: str) -> Dict:
        life = self._states[candidate_id]
        policy = dict(FAILURE_POLICY.get(str(failure_type), FAILURE_POLICY["genuine_harness_regression"]))

        if life.branch_frozen:
            return {"candidate_id": candidate_id, "applied": False, "reason": "branch_frozen", "state": life.state}

        budget_delta = int(policy["budget_delta"])
        if budget_delta > 0:
            life.revise_budget_remaining = max(0, life.revise_budget_remaining - budget_delta)

        next_state = str(policy["next_state"])
        life.state = next_state

        if next_state == "system_error":
            life.system_error_count += 1
        if bool(policy["freeze_branch"]) or life.system_error_count >= self.system_error_freeze_threshold:
            life.branch_frozen = True
            life.state = "system_error"
        if bool(policy["terminate_family"]):
            life.family_terminated = True

        out = {
            "candidate_id": candidate_id,
            "failure_type": failure_type,
            "state": life.state,
            "budget_delta": budget_delta,
            "revise_budget_remaining": life.revise_budget_remaining,
            "allow_local_retry": bool(policy["allow_local_retry"]),
            "freeze_branch": bool(life.branch_frozen),
            "terminate_family": bool(policy["terminate_family"]),
        }
        return out

    def promote_to_formal(self, candidate_id: str) -> CandidateLifecycle:
        return self.transition(candidate_id, "promoted_to_formal")

    def snapshot(self) -> List[Dict]:
        return [vars(v) for v in self._states.values()]
