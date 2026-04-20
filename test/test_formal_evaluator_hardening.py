from __future__ import annotations

import json
from pathlib import Path

from proposer.evaluate_candidate import evaluate_candidate_live
from proposer.live_benchmark_runner import RunArtifact


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


class BaselineRunner:
    def __init__(self, repo_root, output_root, harness_id, evaluation_protocol):
        self.output_root = Path(output_root)
        self.harness_id = harness_id
        self.evaluation_protocol = dict(evaluation_protocol)

    def run(self):
        artifacts = []
        idx = 0
        for pair in self.evaluation_protocol["pairs"]:
            for seed in range(int(pair["runs"])):
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
                        completion_time_mission_sec=15.0,
                        llm_call_count=4,
                        replan_count=1,
                        seed=seed,
                        runtime_trace_path=(run_dir / "runtime_trace.jsonl").as_posix(),
                        planning_trace_path=(run_dir / "planning_trace.jsonl").as_posix(),
                        metadata_path=(run_dir / "metadata.json").as_posix(),
                    )
                )
        return artifacts


class UnsafeCandidateRunner(BaselineRunner):
    def run(self):
        rows = super().run()
        for r in rows:
            r.completion_time_mission_sec = 8.0
            r.llm_call_count = 2
            # Higher success but unsafe in one paired seed/scene.
            r.mission_success = True
            if r.scene_id == "SCENE1" and r.seed == 0:
                r.collision_count = 1
            else:
                r.collision_count = 0
            r.near_miss_count = 0
        return rows


def test_formal_evaluator_emits_scene_seed_pairwise_artifacts(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    archive = repo / "proposer_archive_v2"

    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", BaselineRunner)
    evaluate_candidate_live(repo_root=repo, harness_id="baseline1", archive_root=archive)

    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", UnsafeCandidateRunner)
    out = evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=archive, evaluation_mode="formal")

    target = archive / "candidates" / "candidate_0001"
    assert (target / "formal_summary.json").exists()
    assert (target / "formal_pairwise_deltas.json").exists()
    assert (target / "formal_safety_report.json").exists()
    assert (target / "formal_dossier.json").exists()

    summary = json.loads((target / "formal_summary.json").read_text(encoding="utf-8"))
    pairwise = json.loads((target / "formal_pairwise_deltas.json").read_text(encoding="utf-8"))
    assert "candidate" in summary["per_scene_metrics_summary"]
    assert "candidate" in summary["per_seed_metrics_summary"]
    assert "per_seed_delta_report" in pairwise
    assert "per_scene_delta_report" in pairwise
    assert out.eval_summary["formal_decision"] == "formal_fail"


def test_unsafe_candidate_not_selected_by_average_success(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    archive = repo / "proposer_archive_v2"

    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", BaselineRunner)
    evaluate_candidate_live(repo_root=repo, harness_id="baseline1", archive_root=archive)

    monkeypatch.setattr("proposer.evaluate_candidate.LiveBenchmarkRunner", UnsafeCandidateRunner)
    evaluate_candidate_live(repo_root=repo, harness_id="candidate_0001", archive_root=archive, evaluation_mode="formal")

    safety = json.loads((archive / "candidates" / "candidate_0001" / "formal_safety_report.json").read_text(encoding="utf-8"))
    summary = json.loads((archive / "candidates" / "candidate_0001" / "formal_summary.json").read_text(encoding="utf-8"))

    assert safety["unsafe_regression_pair_count"] > 0
    assert summary["decision"] == "formal_fail"
