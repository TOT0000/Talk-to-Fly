from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from controller.harness_protocol import (
    EVALUATION_PROTOCOL_SEQUENCE,
    EVALUATION_PROTOCOL_VERSION,
    TOTAL_EVAL_RUNS,
)
from proposer.live_benchmark_runner import LiveBenchmarkRunner, RunArtifact
from proposer.registry import HarnessRegistry


PARETO_MINIMIZE_KEYS = [
    "collision_count_avg",
    "near_miss_count_avg",
    "completion_time_mission_sec_avg",
    "llm_call_count_avg",
]


@dataclass(frozen=True)
class EvaluationResult:
    eval_summary: Dict
    per_scene_metrics: Dict
    run_artifacts: List[Dict]


def _metric_get(d: Dict, key: str) -> float:
    if key in d and d[key] is not None:
        return float(d[key])
    if key == "completion_time_mission_sec_avg" and ("completion_time_sec_avg" in d):
        return float(d["completion_time_sec_avg"])
    if key == "completion_time_sec_avg" and ("completion_time_mission_sec_avg" in d):
        return float(d["completion_time_mission_sec_avg"])
    raise KeyError(key)


def dominates(a: Dict, b: Dict, keys: Iterable[str]) -> bool:
    keys = list(keys)
    return all(_metric_get(a, k) <= _metric_get(b, k) for k in keys) and any(_metric_get(a, k) < _metric_get(b, k) for k in keys)


def mark_pareto(entries: List[Dict]) -> List[Dict]:
    out = []
    for i, cur in enumerate(entries):
        dominated = False
        for j, other in enumerate(entries):
            if i == j:
                continue
            if dominates(other["metrics"], cur["metrics"], PARETO_MINIMIZE_KEYS):
                dominated = True
                break
        enriched = dict(cur)
        enriched["pareto_frontier"] = not dominated
        out.append(enriched)
    return out


def _avg_completion_success_only(rows: List[RunArtifact]) -> float | None:
    successful = [r for r in rows if r.mission_success and (r.completion_time_mission_sec is not None)]
    if not successful:
        return None
    return mean(float(r.completion_time_mission_sec) for r in successful)


def _build_scene_metrics(rows: List[RunArtifact], scene_id: str, zone: str, expected_runs: int) -> Dict:
    success_count = sum(1 for r in rows if r.mission_success)
    return {
        "scene_id": scene_id,
        "task_zone": zone,
        "success_count": success_count,
        "total_runs": len(rows),
        "expected_runs": expected_runs,
        "success_rate": (float(success_count) / float(len(rows))) if rows else 0.0,
        "collision_count_avg": mean(r.collision_count for r in rows) if rows else 0.0,
        "near_miss_count_avg": mean(r.near_miss_count for r in rows) if rows else 0.0,
        "completion_time_mission_sec_avg_success_only": _avg_completion_success_only(rows),
        "llm_call_count_avg": mean(r.llm_call_count for r in rows) if rows else 0.0,
        "replan_count_avg": mean(r.replan_count for r in rows) if rows else 0.0,
    }


def evaluate_candidate_live(repo_root: Path, harness_id: str, archive_root: Path) -> EvaluationResult:
    repo_root = Path(repo_root)
    archive_root = Path(archive_root)
    harness_entry = HarnessRegistry(repo_root).get(harness_id)

    target = archive_root / ("baselines" if harness_id.startswith("baseline") else "candidates") / harness_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "code_or_spec").mkdir(parents=True, exist_ok=True)

    runner = LiveBenchmarkRunner(repo_root=repo_root, output_root=target, harness_id=harness_id)
    runs = runner.run()

    # copy harness source/spec snapshot
    for name in [
        "spec.json",
        "state_encoder.py",
        "trigger_policy.py",
        "prompt_builder.py",
        "state_features.py",
        "trigger_logic.py",
        "prompt_composer.py",
        "archive_selector.py",
        "validator_rules.py",
        "proposer_note.txt",
    ]:
        src = harness_entry.dir_path / name
        if src.exists():
            shutil.copy2(src, target / "code_or_spec" / name)

    by_scene: Dict[str, List[RunArtifact]] = {}
    for r in runs:
        by_scene.setdefault(r.scene_id, []).append(r)

    per_scene = {}
    for pair in EVALUATION_PROTOCOL_SEQUENCE:
        scene = str(pair["scene_id"])
        zone = str(pair["task_zone"])
        cnt = int(pair["runs"])
        per_scene[scene] = _build_scene_metrics(by_scene.get(scene, []), scene, zone, cnt)

    success_total = sum(1 for r in runs if r.mission_success)
    overall_completion_avg = _avg_completion_success_only(runs)
    eval_summary = {
        "harness_id": harness_id,
        "kind": harness_entry.kind,
        "status": "evaluated_live",
        "parent_id": harness_entry.spec.get("parent"),
        "parent_kind": ("baseline" if str(harness_entry.spec.get("parent", "")).startswith("baseline") else "candidate") if harness_entry.spec.get("parent") else None,
        "derived_from": harness_entry.spec.get("parent"),
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "pairs": EVALUATION_PROTOCOL_SEQUENCE,
            "total_runs": TOTAL_EVAL_RUNS,
        },
        "total_runs": len(runs),
        "metrics": {
            "success_rate": (float(success_total) / float(len(runs))) if runs else 0.0,
            "collision_count_avg": mean(r.collision_count for r in runs) if runs else 0.0,
            "near_miss_count_avg": mean(r.near_miss_count for r in runs) if runs else 0.0,
            "completion_time_mission_sec_avg": overall_completion_avg,
            "llm_call_count_avg": mean(r.llm_call_count for r in runs) if runs else 0.0,
            "replan_count_avg": mean(r.replan_count for r in runs) if runs else 0.0,
        },
    }

    per_run_payload = [r.__dict__ for r in runs]
    with (target / "per_run_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_run_payload, f, ensure_ascii=False, indent=2)
    with (target / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    with (target / "per_scene_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_scene, f, ensure_ascii=False, indent=2)

    return EvaluationResult(eval_summary=eval_summary, per_scene_metrics=per_scene, run_artifacts=per_run_payload)
