from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from controller.harness_protocol import EVALUATION_PROTOCOL_VERSION, EVALUATION_SCENE_TASK_MAPPING
from proposer.archive_reader import aggregate_by_harness, read_manual_runs
from proposer.registry import HarnessRegistry


PARETO_MINIMIZE_KEYS = [
    "collision_count_avg",
    "near_miss_count_avg",
    "completion_time_sec_avg",
    "llm_call_count_avg",
]


@dataclass(frozen=True)
class EvaluationResult:
    eval_summary: Dict
    per_scene_metrics: Dict


def dominates(a: Dict, b: Dict, keys: Iterable[str]) -> bool:
    keys = list(keys)
    return all(float(a[k]) <= float(b[k]) for k in keys) and any(float(a[k]) < float(b[k]) for k in keys)


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


def evaluate_candidate_offline(
    repo_root: Path,
    harness_id: str,
    archive_root: Path,
    manual_debug_jsonl: Path,
) -> EvaluationResult:
    """
    MVP evaluator: uses fixed protocol and existing run archive as the source-of-truth metrics.
    For new candidates without executed runs, summary is emitted with pending status.
    """
    repo_root = Path(repo_root)
    archive_root = Path(archive_root)
    data = aggregate_by_harness(read_manual_runs(Path(manual_debug_jsonl)))
    item = data.get(harness_id)

    if item is None:
        eval_summary = {
            "harness_id": harness_id,
            "status": "pending_execution",
            "evaluation_protocol": {
                "version": EVALUATION_PROTOCOL_VERSION,
                "scene_to_task_zone": dict(EVALUATION_SCENE_TASK_MAPPING),
            },
            "metrics": {},
        }
        per_scene = {}
    else:
        eval_summary = {
            "harness_id": harness_id,
            "status": "evaluated_from_archive",
            "evaluation_protocol": item["evaluation_protocol"],
            "total_runs": item["total_runs"],
            "metrics": item["metrics"],
        }
        per_scene = item["per_scene_metrics"]

    target = archive_root / ("baselines" if harness_id.startswith("baseline") else "candidates") / harness_id
    (target / "code_or_spec").mkdir(parents=True, exist_ok=True)
    (target / "traces").mkdir(parents=True, exist_ok=True)

    harness_dir = HarnessRegistry(repo_root).get(harness_id).dir_path
    for name in ["spec.json", "state_encoder.py", "trigger_policy.py", "prompt_builder.py", "proposer_note.txt"]:
        src = harness_dir / name
        if src.exists():
            shutil.copy2(src, target / "code_or_spec" / name)

    with (target / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    with (target / "per_scene_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_scene, f, ensure_ascii=False, indent=2)

    # trace pointers (full traces stay in original manual archive)
    trace_ptr = {
        "source": "proposer_archive/manual_runs/runs",
        "note": "raw traces preserved in original archive for auditability",
    }
    with (target / "traces" / "trace_pointers.json").open("w", encoding="utf-8") as f:
        json.dump(trace_ptr, f, ensure_ascii=False, indent=2)

    return EvaluationResult(eval_summary=eval_summary, per_scene_metrics=per_scene)
