from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CandidateLifecycle:
    candidate_id: str
    state: str
    revise_budget_remaining: int


class SearchOrchestrator:
    """Manages candidate lifecycle and screening->formal promotion."""

    def __init__(self, revise_budget: int = 3):
        self.revise_budget = int(revise_budget)
        self._states: Dict[str, CandidateLifecycle] = {}

    def register_candidate(self, candidate_id: str) -> CandidateLifecycle:
        life = CandidateLifecycle(candidate_id=candidate_id, state="proposed", revise_budget_remaining=self.revise_budget)
        self._states[candidate_id] = life
        return life

    def transition(self, candidate_id: str, new_state: str) -> CandidateLifecycle:
        life = self._states[candidate_id]
        life.state = str(new_state)
        if new_state in {"validator_failed", "screening_failed", "formal_failed"}:
            life.revise_budget_remaining = max(0, life.revise_budget_remaining - 1)
        return life

    def promote_to_formal(self, candidate_id: str) -> CandidateLifecycle:
        return self.transition(candidate_id, "promoted_to_formal")

    def snapshot(self) -> List[Dict]:
        return [vars(v) for v in self._states.values()]
