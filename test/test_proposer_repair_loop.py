from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from proposer import run_loop


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write_json(repo / "harnesses" / "baseline1" / "spec.json", {"id": "baseline1", "kind": "baseline"})
    return repo


def _make_candidate(repo: Path, cid: str, parent: str) -> Path:
    cdir = repo / "harnesses" / "candidates" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    _write_json(
        cdir / "spec.json",
        {"id": cid, "kind": "candidate", "parent": parent, "runtime_metadata": {}},
    )
    (cdir / "runtime_wiring_verification.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (cdir / "parent_diff.patch").write_text("", encoding="utf-8")
    (cdir / "proposer_note.txt").write_text("note\n", encoding="utf-8")
    return cdir


def test_validator_failure_revise_then_screening_pass(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    created = []

    def _propose(*args, **kwargs):
        idx = len(created) + 1
        parent = kwargs.get("parent_harness_override") or "baseline1"
        cdir = _make_candidate(repo, f"candidate_{idx:04d}", parent)
        created.append(cdir.name)
        return cdir

    validator_calls = {"n": 0}

    def _validator(candidate_dir, parent_dir):
        validator_calls["n"] += 1
        if validator_calls["n"] == 1:
            return [{"error_stage": "validator", "is_system_error": False, "is_proposer_fixable": True, "blocking": True}]
        return []

    def _eval(**kwargs):
        return SimpleNamespace(run_artifacts=[{"run_id": "r1", "run_status": "ok", "mission_success": True}])

    monkeypatch.setattr(run_loop, "propose_next_candidate", _propose)
    monkeypatch.setattr(run_loop, "_run_validator", _validator)
    monkeypatch.setattr(run_loop, "evaluate_candidate_live", _eval)
    monkeypatch.setattr(run_loop, "rebuild_index", lambda _: {"entries": []})

    cid = run_loop.run_once(repo)
    assert cid == "candidate_0002"
    assert created == ["candidate_0001", "candidate_0002"]
    spec = json.loads((repo / "harnesses" / "candidates" / cid / "spec.json").read_text(encoding="utf-8"))
    assert spec["runtime_metadata"]["proposer_loop_status"] == "screening_passed"
    assert Path(spec["runtime_metadata"]["revision_history_path"]).exists()


def test_validator_system_error_stops_without_revise(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    first = _make_candidate(repo, "candidate_0001", "baseline1")
    monkeypatch.setattr(run_loop, "propose_next_candidate", lambda *a, **k: first)
    monkeypatch.setattr(
        run_loop,
        "_run_validator",
        lambda *a, **k: [{"error_stage": "validator", "is_system_error": True, "is_proposer_fixable": False, "blocking": True}],
    )
    monkeypatch.setattr(run_loop, "evaluate_candidate_live", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not evaluate")))
    monkeypatch.setattr(run_loop, "rebuild_index", lambda _: {"entries": []})

    cid = run_loop.run_once(repo)
    assert cid == "candidate_0001"
    spec = json.loads((first / "spec.json").read_text(encoding="utf-8"))
    assert spec["runtime_metadata"]["proposer_loop_status"] == "system_error"


def test_screening_failure_revise_and_system_error(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    created = []

    def _propose(*args, **kwargs):
        idx = len(created) + 1
        parent = kwargs.get("parent_harness_override") or "baseline1"
        cdir = _make_candidate(repo, f"candidate_{idx:04d}", parent)
        created.append(cdir.name)
        return cdir

    eval_calls = {"n": 0}

    def _eval(**kwargs):
        eval_calls["n"] += 1
        cid = kwargs["harness_id"]
        run_dir = repo / "proposer_archive_v2" / "candidates" / cid / "runs" / "run_0001"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = {"error_type": "mission_success_scope_bug", "failure_reason": "mission_success scope broken"}
        _write_json(run_dir / "metadata.json", {"evaluate_error_report": report})
        if eval_calls["n"] == 1:
            return SimpleNamespace(
                run_artifacts=[
                    {
                        "run_id": "run_0001",
                        "scene_id": "SCENE1",
                        "run_status": "failed",
                        "mission_success": False,
                        "collision_count": 0,
                        "near_miss_count": 0,
                        "runtime_trace_path": "",
                        "planning_trace_path": "",
                        "metadata_path": (run_dir / "metadata.json").as_posix(),
                    }
                ]
            )
        return SimpleNamespace(run_artifacts=[{"run_id": "ok", "run_status": "ok", "mission_success": True}])

    monkeypatch.setattr(run_loop, "propose_next_candidate", _propose)
    monkeypatch.setattr(run_loop, "_run_validator", lambda *a, **k: [])
    monkeypatch.setattr(run_loop, "evaluate_candidate_live", _eval)
    monkeypatch.setattr(run_loop, "rebuild_index", lambda _: {"entries": []})

    cid = run_loop.run_once(repo)
    assert cid == "candidate_0001"
    assert created == ["candidate_0001"]  # screening error is system_error, no revise
    spec = json.loads((repo / "harnesses" / "candidates" / cid / "spec.json").read_text(encoding="utf-8"))
    assert spec["runtime_metadata"]["proposer_loop_status"] == "system_error"


def test_max_rounds_exhausted(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    created = []

    def _propose(*args, **kwargs):
        idx = len(created) + 1
        parent = kwargs.get("parent_harness_override") or "baseline1"
        cdir = _make_candidate(repo, f"candidate_{idx:04d}", parent)
        created.append(cdir.name)
        return cdir

    monkeypatch.setattr(run_loop, "propose_next_candidate", _propose)
    monkeypatch.setattr(
        run_loop,
        "_run_validator",
        lambda *a, **k: [{"error_stage": "validator", "is_system_error": False, "is_proposer_fixable": True, "blocking": True}],
    )
    monkeypatch.setattr(run_loop, "evaluate_candidate_live", lambda **kwargs: (_ for _ in ()).throw(AssertionError("no eval")))
    monkeypatch.setattr(run_loop, "rebuild_index", lambda _: {"entries": []})

    cid = run_loop.run_once(repo)
    spec = json.loads((repo / "harnesses" / "candidates" / cid / "spec.json").read_text(encoding="utf-8"))
    assert spec["runtime_metadata"]["proposer_loop_status"] == "max_rounds_exhausted"
    assert len(created) == run_loop.VALIDATOR_REVISE_MAX_ROUNDS + 1
