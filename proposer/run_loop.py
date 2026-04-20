from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from proposer.evaluate_candidate import evaluate_candidate_live
from proposer.propose_candidate import propose_next_candidate, rebuild_index
from proposer.consistency import validate_candidate_contract_alignment
from proposer.registry import HarnessRegistry
from proposer.registry import validate_candidate_boundary


VALIDATOR_REVISE_MAX_ROUNDS = 3
EVALUATOR_REVISE_MAX_ROUNDS = 2
TOTAL_REPAIR_MAX_ROUNDS = 5


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _classify_system_error(text: str) -> bool:
    raw = str(text or "").lower()
    system_markers = [
        "mission_success",
        "evaluate pipeline",
        "planner logger",
        "heartbeat loop not executed",
        "archive",
        "run summary",
        "takeoff",
        "sim",
        "controller",
        "completion scope",
    ]
    return any(m in raw for m in system_markers)


def _validator_feedback_from_exception(exc: Exception, *, evidence_paths: List[str]) -> Dict:
    summary = str(exc)
    is_system_error = _classify_system_error(summary)
    return {
        "error_stage": "validator",
        "error_type": "validator_exception",
        "severity": "error",
        "is_system_error": bool(is_system_error),
        "is_proposer_fixable": bool(not is_system_error),
        "summary": summary,
        "expected": "Candidate passes boundary/contract/runtime-wiring validation checks.",
        "observed": summary,
        "evidence_paths": evidence_paths,
        "repair_hint": "Revise candidate spec/modules to align runtime wiring and contract fields.",
        "blocking": True,
    }


def _run_validator(candidate_dir: Path, parent_dir: Path) -> List[Dict]:
    feedback: List[Dict] = []
    evidence_paths = [
        (candidate_dir / "spec.json").as_posix(),
        (candidate_dir / "runtime_wiring_verification.json").as_posix(),
        (candidate_dir / "parent_diff.patch").as_posix(),
    ]
    try:
        validate_candidate_boundary(candidate_dir)
        spec = _load_json(candidate_dir / "spec.json")
        validate_candidate_contract_alignment(candidate_dir, parent_dir=parent_dir, proposal_contract=spec.get("proposal_contract"))
        wiring = _load_json(candidate_dir / "runtime_wiring_verification.json")
        if not bool(wiring.get("passed", False)):
            raise RuntimeError("runtime_wiring_mismatch: runtime_wiring_verification.passed=false")
    except Exception as exc:
        feedback.append(_validator_feedback_from_exception(exc, evidence_paths=evidence_paths))
    return feedback


def _extract_screening_feedback(eval_result) -> List[Dict]:
    rows = list(getattr(eval_result, "run_artifacts", []) or [])
    feedback: List[Dict] = []
    for row in rows:
        run_status = str(row.get("run_status") or "").lower()
        if bool(row.get("mission_success")) and run_status in {"ok", "success", "completed"}:
            continue
        metadata_path = Path(str(row.get("metadata_path") or ""))
        report = {}
        if metadata_path.exists():
            meta = _load_json(metadata_path)
            report = dict(meta.get("evaluate_error_report") or {})
        summary = str(report.get("failure_reason") or report.get("termination_reason") or row.get("run_status") or "screening failure")
        error_type = str(report.get("error_type") or "task_aborted")
        is_system_error = _classify_system_error(f"{error_type} {summary}")
        feedback.append(
            {
                "error_stage": "evaluator_screening",
                "error_type": error_type,
                "severity": "error",
                "is_system_error": bool(is_system_error),
                "is_proposer_fixable": bool(not is_system_error),
                "scene": str(row.get("scene_id") or ""),
                "run_id": str(row.get("run_id") or ""),
                "summary": summary,
                "observed": {
                    "run_status": row.get("run_status"),
                    "mission_success": row.get("mission_success"),
                    "collision_count": row.get("collision_count"),
                    "near_miss_count": row.get("near_miss_count"),
                },
                "evidence_paths": [str(row.get("runtime_trace_path") or ""), str(row.get("planning_trace_path") or ""), str(row.get("metadata_path") or "")],
                "repair_hint": "Revise runtime-effect modules/spec contract to address screening failure evidence.",
                "blocking": True,
            }
        )
        break
    return feedback


def _update_loop_metadata(
    candidate_dir: Path,
    *,
    proposal_iteration_id: str,
    proposal_revision_round: int,
    validator_round_index: int,
    evaluator_round_index: int,
    proposer_loop_status: str,
    latest_validator_feedback_path: str = "",
    latest_evaluator_feedback_path: str = "",
    revision_history_path: str = "",
) -> None:
    spec_path = candidate_dir / "spec.json"
    if not spec_path.exists():
        return
    spec = _load_json(spec_path)
    runtime_meta = dict(spec.get("runtime_metadata") or {})
    runtime_meta.update(
        {
            "proposal_iteration_id": proposal_iteration_id,
            "proposal_revision_round": int(proposal_revision_round),
            "validator_round_index": int(validator_round_index),
            "evaluator_round_index": int(evaluator_round_index),
            "proposer_loop_status": proposer_loop_status,
            "latest_validator_feedback_path": latest_validator_feedback_path,
            "latest_evaluator_feedback_path": latest_evaluator_feedback_path,
            "revision_history_path": revision_history_path,
        }
    )
    spec["runtime_metadata"] = runtime_meta
    _write_json(spec_path, spec)


def run_once(
    repo_root: Path,
    evaluate_baselines: bool = False,
    focus_text: str = "Improve safety-aware replan timing while avoiding unnecessary detours.",
    allow_fallback_heuristic: bool = False,
) -> str:
    repo_root = Path(repo_root)
    archive_v2 = repo_root / "proposer_archive_v2"

    reg = HarnessRegistry(repo_root)

    if evaluate_baselines:
        for baseline in reg.list_baselines():
            evaluate_candidate_live(
                repo_root=repo_root,
                harness_id=baseline.harness_id,
                archive_root=archive_v2,
            )

    proposal_iteration_id = datetime.now(timezone.utc).strftime("proposal_iter_%Y%m%dT%H%M%S")
    history_dir = archive_v2 / "proposer_iterations" / proposal_iteration_id
    history_path = history_dir / "revision_history.json"
    revision_history: List[Dict] = []

    candidate_dir = propose_next_candidate(
        repo_root,
        focus_text=focus_text,
        allow_fallback_heuristic=allow_fallback_heuristic,
        max_revision_rounds=0,
        proposal_iteration_id=proposal_iteration_id,
        proposer_loop_status="proposed",
    )
    current_candidate_dir = candidate_dir
    current_candidate_id = current_candidate_dir.name
    parent_dir = reg.get(_load_json(current_candidate_dir / "spec.json").get("parent")).dir_path
    validator_round = 0
    evaluator_round = 0
    total_repair_rounds = 0

    while True:
        print(f"[proposer-loop] candidate={current_candidate_id} phase=validator round={validator_round}")
        validator_feedback = _run_validator(current_candidate_dir, parent_dir)
        validator_feedback_path = ""
        if validator_feedback:
            validator_feedback_path = (current_candidate_dir / f"validator_feedback_round_{validator_round}.json").as_posix()
            _write_json(Path(validator_feedback_path), {"items": validator_feedback})
            status = "system_error" if any(x.get("is_system_error") for x in validator_feedback) else "validator_failed"
            _update_loop_metadata(
                current_candidate_dir,
                proposal_iteration_id=proposal_iteration_id,
                proposal_revision_round=total_repair_rounds,
                validator_round_index=validator_round,
                evaluator_round_index=evaluator_round,
                proposer_loop_status=status,
                latest_validator_feedback_path=validator_feedback_path,
                revision_history_path=history_path.as_posix(),
            )
            revision_history.append(
                {
                    "iteration_id": proposal_iteration_id,
                    "parent_proposal_version": current_candidate_id,
                    "validator_feedback_path": validator_feedback_path,
                    "evaluator_feedback_path": "",
                    "revised_proposal_version": "",
                    "final_status": status,
                }
            )
            _write_json(history_path, revision_history)
            if status == "system_error":
                break
            if validator_round >= VALIDATOR_REVISE_MAX_ROUNDS or total_repair_rounds >= TOTAL_REPAIR_MAX_ROUNDS:
                _update_loop_metadata(
                    current_candidate_dir,
                    proposal_iteration_id=proposal_iteration_id,
                    proposal_revision_round=total_repair_rounds,
                    validator_round_index=validator_round,
                    evaluator_round_index=evaluator_round,
                    proposer_loop_status="max_rounds_exhausted",
                    latest_validator_feedback_path=validator_feedback_path,
                    revision_history_path=history_path.as_posix(),
                )
                break
            total_repair_rounds += 1
            validator_round += 1
            revised = propose_next_candidate(
                repo_root,
                focus_text=focus_text,
                allow_fallback_heuristic=allow_fallback_heuristic,
                max_revision_rounds=0,
                parent_harness_override=current_candidate_id,
                external_feedback_context={"validator_feedback": validator_feedback},
                proposal_iteration_id=proposal_iteration_id,
                proposal_revision_round=total_repair_rounds,
                validator_round_index=validator_round,
                evaluator_round_index=evaluator_round,
                proposer_loop_status="validator_revised",
            )
            revision_history[-1]["revised_proposal_version"] = revised.name
            _write_json(history_path, revision_history)
            current_candidate_dir = revised
            current_candidate_id = revised.name
            parent_dir = reg.get(_load_json(current_candidate_dir / "spec.json").get("parent")).dir_path
            continue

        print(f"[proposer-loop] candidate={current_candidate_id} phase=screening round={evaluator_round}")
        out = evaluate_candidate_live(
            repo_root=repo_root,
            harness_id=current_candidate_id,
            archive_root=archive_v2,
            evaluation_mode="screening",
        )
        screening_feedback = _extract_screening_feedback(out)
        if screening_feedback:
            evaluator_feedback_path = (current_candidate_dir / f"evaluator_feedback_round_{evaluator_round}.json").as_posix()
            _write_json(Path(evaluator_feedback_path), {"items": screening_feedback})
            status = "system_error" if any(x.get("is_system_error") for x in screening_feedback) else "screening_failed"
            _update_loop_metadata(
                current_candidate_dir,
                proposal_iteration_id=proposal_iteration_id,
                proposal_revision_round=total_repair_rounds,
                validator_round_index=validator_round,
                evaluator_round_index=evaluator_round,
                proposer_loop_status=status,
                latest_evaluator_feedback_path=evaluator_feedback_path,
                revision_history_path=history_path.as_posix(),
            )
            revision_history.append(
                {
                    "iteration_id": proposal_iteration_id,
                    "parent_proposal_version": current_candidate_id,
                    "validator_feedback_path": "",
                    "evaluator_feedback_path": evaluator_feedback_path,
                    "revised_proposal_version": "",
                    "final_status": status,
                }
            )
            _write_json(history_path, revision_history)
            if status == "system_error":
                break
            if evaluator_round >= EVALUATOR_REVISE_MAX_ROUNDS or total_repair_rounds >= TOTAL_REPAIR_MAX_ROUNDS:
                _update_loop_metadata(
                    current_candidate_dir,
                    proposal_iteration_id=proposal_iteration_id,
                    proposal_revision_round=total_repair_rounds,
                    validator_round_index=validator_round,
                    evaluator_round_index=evaluator_round,
                    proposer_loop_status="max_rounds_exhausted",
                    latest_evaluator_feedback_path=evaluator_feedback_path,
                    revision_history_path=history_path.as_posix(),
                )
                break
            total_repair_rounds += 1
            evaluator_round += 1
            revised = propose_next_candidate(
                repo_root,
                focus_text=focus_text,
                allow_fallback_heuristic=allow_fallback_heuristic,
                max_revision_rounds=0,
                parent_harness_override=current_candidate_id,
                external_feedback_context={"evaluator_feedback": screening_feedback},
                proposal_iteration_id=proposal_iteration_id,
                proposal_revision_round=total_repair_rounds,
                validator_round_index=validator_round,
                evaluator_round_index=evaluator_round,
                proposer_loop_status="screening_revised",
            )
            revision_history[-1]["revised_proposal_version"] = revised.name
            _write_json(history_path, revision_history)
            current_candidate_dir = revised
            current_candidate_id = revised.name
            parent_dir = reg.get(_load_json(current_candidate_dir / "spec.json").get("parent")).dir_path
            continue

        _update_loop_metadata(
            current_candidate_dir,
            proposal_iteration_id=proposal_iteration_id,
            proposal_revision_round=total_repair_rounds,
            validator_round_index=validator_round,
            evaluator_round_index=evaluator_round,
            proposer_loop_status="screening_passed",
            revision_history_path=history_path.as_posix(),
        )
        revision_history.append(
            {
                "iteration_id": proposal_iteration_id,
                "parent_proposal_version": current_candidate_id,
                "validator_feedback_path": "",
                "evaluator_feedback_path": "",
                "revised_proposal_version": current_candidate_id,
                "final_status": "screening_passed",
            }
        )
        _write_json(history_path, revision_history)
        break

    rebuild_index(archive_v2)
    return current_candidate_id


if __name__ == "__main__":
    cid = run_once(Path(__file__).resolve().parents[1], evaluate_baselines=False)
    print(cid)
