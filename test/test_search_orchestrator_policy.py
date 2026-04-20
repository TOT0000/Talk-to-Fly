from __future__ import annotations

from proposer.search_orchestrator import SearchOrchestrator


def test_revise_budget_consumption_by_failure_type():
    orch = SearchOrchestrator(revise_budget=3)
    orch.register_candidate("c1")

    out = orch.apply_failure("c1", "proposal_error")
    assert out["revise_budget_remaining"] == 2

    out = orch.apply_failure("c1", "infra_crash_error")
    assert out["revise_budget_remaining"] == 2  # no budget consumed for infra retry

    out = orch.apply_failure("c1", "trace_incomplete_error")
    assert out["revise_budget_remaining"] == 2  # no budget consumed for incomplete traces


def test_repeated_system_error_freezes_branch():
    orch = SearchOrchestrator(revise_budget=3, system_error_freeze_threshold=2)
    orch.register_candidate("c2")

    first = orch.transition("c2", "system_error")
    assert first.branch_frozen is False

    second = orch.transition("c2", "system_error")
    assert second.branch_frozen is True


def test_unsafe_regression_terminates_candidate_family():
    orch = SearchOrchestrator(revise_budget=2)
    orch.register_candidate("c3")

    out = orch.apply_failure("c3", "unsafe_candidate_regression")
    assert out["state"] == "formal_failed"
    assert out["terminate_family"] is True
    assert out["allow_local_retry"] is False
