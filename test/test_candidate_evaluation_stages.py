from __future__ import annotations

import json
from pathlib import Path

from controller.harness_protocol import get_evaluation_protocol, resolve_evaluation_mode
from proposer.evaluate_candidate import evaluate_candidate_live
from proposer.live_benchmark_runner import RunArtifact
from proposer.propose_candidate import rebuild_index


def _write_spec(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write_spec(repo / "harnesses" / "baseline1" / "spec.json", {"id": "baseline1", "kind": "baseline"})
    _write_spec(
        repo / "harnesses" / "candidates" / "candidate_0001" / "spec.json",
        {"id": "candidate_0001", "kind": "candidate", "parent": "baseline1"},
    )
    return repo


class FakeRunner:
    def __init__(self, repo_root, output_root, harness_id, evaluation_protocol):
        self.output_root = Path(output_root)
        self.harness_id = harness_id
        self.evaluation_protocol = dict(evaluation_protocol)

    def run(self):
        artifacts = []
        idx = 0
        for pair in self.evaluation_protocol["pairs"]:
            for _ in range(int(pair["runs"])):
                idx += 1
                run_id = f"run_{idx:04d}"
                run_dir = self.output_root / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "runtime_trace.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "planning_trace.jsonl").write_text("{}\n", encoding="utf-8")
                (run_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
                artifacts.append(
                    RunArtifact(
                        run_id=run_id,
                        scene_id=str(pair["scene_id"]),
                        task_zone=str(pair["task_zone"]),
                        harness_id=self.harness_id,
                        run_status="ok",
                        mission_success=True,
                        collision_count=0,
                        near_miss_count=0,
                        completion_time_mission_sec=10.0,
                        llm_call_count=1,
                        replan_count=1,
                        seed=0,
                        runtime_trace_path=(run_dir / "runtime_trace.jsonl").as_posix(),
                        planning_trace_path=(run_dir / "planning_trace.jsonl").as_posix(),
                        metadata_path=(run_dir / "metadata.json").as_posix(),
                    )
                )
        return artifacts


def test_protocol_modes_baseline_and_candidate_defaults():
    assert resolve_evaluation_mode(kind="baseline", requested_mode=None) == "formal"
    assert resolve_evaluation_mode(kind="candidate", requested_mode=None) == "screening"
    assert get_evaluation_protocol(kind="candidate", requested_mode="screening")["total_runs"] == 6
    assert get_evaluation_protocol(kind="candidate", requested_mode="formal")["total_runs"] == 24


def test_baseline_protocol_remains_formal_24_runs(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", FakeRunner)

    out = evaluate_candidate_live(repo_root=repo, harness_id="baseline1", archive_root=repo / "proposer_archive_v2")
    assert out.eval_summary["evaluation_stage"] == "formal"
    assert out.eval_summary["evaluation_protocol"]["runs_per_scene"] == 8
    assert out.eval_summary["total_runs_expected"] == 24


def test_candidate_default_evaluate_uses_screening(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", FakeRunner)

    out = evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=repo / "proposer_archive_v2")
    target = repo / "proposer_archive_v2" / "candidates" / "candidate_0001"

    assert out.eval_summary["evaluation_stage"] == "screening"
    assert out.eval_summary["evaluation_protocol"]["runs_per_scene"] == 2
    assert out.eval_summary["total_runs_expected"] == 6
    assert (target / "eval_summary_screening.json").exists()
    assert not (target / "eval_summary.json").exists()


def test_candidate_formal_promotion_and_stage_metadata(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", FakeRunner)
    archive = repo / "proposer_archive_v2"

    evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=archive)
    formal = evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=archive, evaluation_mode="formal")

    assert formal.eval_summary["evaluation_stage"] == "formal"
    assert formal.eval_summary["promoted_to_formal"] is True
    assert formal.eval_summary["total_runs_expected"] == 24

    index = rebuild_index(archive)
    by_id = {e["candidate_id"]: e for e in index["entries"]}
    cand = by_id["candidate_0001"]
    assert cand["evaluation_stage"] == "formal"
    assert cand["promoted_to_formal"] is True
    assert sorted(cand["stage_summaries"].keys()) == ["formal", "screening"]


def test_screening_metrics_not_treated_as_formal_in_index(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", FakeRunner)
    archive = repo / "proposer_archive_v2"

    evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=archive)
    evaluate_candidate_live(repo_root=repo, harness_id="baseline1", archive_root=archive)

    index = rebuild_index(archive)
    by_id = {e["candidate_id"]: e for e in index["entries"]}
    assert by_id["candidate_0001"]["evaluation_stage"] == "screening"
    assert by_id["candidate_0001"]["pareto_frontier"] is False
    assert by_id["baseline1"]["evaluation_stage"] == "formal"
