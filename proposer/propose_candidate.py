from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from controller.harness_protocol import EVALUATION_PROTOCOL_SEQUENCE, EVALUATION_PROTOCOL_VERSION, TOTAL_EVAL_RUNS
from proposer.evaluate_candidate import mark_pareto
from proposer.registry import HarnessRegistry, validate_candidate_boundary


def _next_candidate_id(candidates_dir: Path) -> str:
    ids: List[int] = []
    for d in candidates_dir.glob("candidate_*"):
        try:
            ids.append(int(str(d.name).split("_")[-1]))
        except Exception:
            continue
    return f"candidate_{(max(ids) + 1) if ids else 1:04d}"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def propose_next_candidate(repo_root: Path, note: str = "") -> Path:
    repo_root = Path(repo_root)
    reg = HarnessRegistry(repo_root)
    baselines = reg.list_baselines()
    if not baselines:
        raise RuntimeError("No baselines found in harnesses/")

    parent = sorted(baselines, key=lambda x: x.harness_id, reverse=True)[0]
    parent_spec = dict(parent.spec)

    candidate_id = _next_candidate_id(reg.candidates_dir)
    candidate_dir = reg.candidates_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)

    parent_trigger = dict(parent_spec.get("trigger_policy") or {})
    threshold = parent_trigger.get("threshold")
    if threshold is None:
        threshold = 0.55
    else:
        threshold = max(0.35, float(threshold) - 0.05)

    parent_spec["id"] = candidate_id
    parent_spec["kind"] = "candidate"
    parent_spec["parent"] = parent.harness_id
    parent_spec["lineage"] = {
        "parent_id": parent.harness_id,
        "parent_kind": "baseline" if parent.harness_id.startswith("baseline") else "candidate",
        "derived_from": parent.harness_id,
    }
    parent_spec.setdefault("mutation", {})
    parent_spec["mutation"]["type"] = "heuristic_threshold_and_prompt_order"
    parent_spec["trigger_policy"]["type"] = "hybrid"
    parent_spec["trigger_policy"]["threshold"] = float(threshold)
    parent_spec["trigger_policy"]["heartbeat_seconds"] = 5.0
    parent_spec["trigger_policy"]["consecutive_high_risk"] = 2
    parent_spec["trigger_policy"]["hysteresis"] = 0.05
    parent_spec["prompt_builder"]["paragraph_order"] = ["opening", "task", "runtime", "risk", "examples", "output"]

    for name in ["state_encoder.py", "trigger_policy.py", "prompt_builder.py"]:
        shutil.copy2(parent.dir_path / name, candidate_dir / name)

    (candidate_dir / "spec.json").write_text(json.dumps(parent_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "proposer_note.txt").write_text(
        (
            note
            or (
                "Hypothesis: baseline3 has good risk sensitivity, but adding hybrid trigger\n"
                "(heartbeat + risk threshold + hysteresis) may reduce delayed replans while\n"
                "containing unnecessary LLM calls under stable low-risk intervals."
            )
        ),
        encoding="utf-8",
    )

    validate_candidate_boundary(candidate_dir)
    return candidate_dir


def rebuild_index(archive_root: Path) -> Dict:
    archive_root = Path(archive_root)
    index_path = archive_root / "index.json"

    entries: List[Dict] = []
    for bucket in ["baselines", "candidates"]:
        base = archive_root / bucket
        if not base.exists():
            continue
        for harness_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            eval_path = harness_dir / "eval_summary.json"
            per_scene_path = harness_dir / "per_scene_metrics.json"
            if not eval_path.exists():
                continue
            eval_summary = _load_json(eval_path)
            per_scene = _load_json(per_scene_path) if per_scene_path.exists() else {}
            run_dirs = sorted([p for p in (harness_dir / "runs").glob("run_*")]) if (harness_dir / "runs").exists() else []

            parent_id = eval_summary.get("parent_id")
            parent_kind = eval_summary.get("parent_kind")
            derived_from = eval_summary.get("derived_from")
            if (not parent_id):
                spec_path = harness_dir / "code_or_spec" / "spec.json"
                if spec_path.exists():
                    try:
                        spec_payload = _load_json(spec_path)
                        parent_id = spec_payload.get("parent") or ((spec_payload.get("lineage") or {}).get("parent_id"))
                        parent_kind = (spec_payload.get("lineage") or {}).get("parent_kind")
                        if (not parent_kind) and parent_id:
                            parent_kind = "baseline" if str(parent_id).startswith("baseline") else "candidate"
                        derived_from = (spec_payload.get("lineage") or {}).get("derived_from") or parent_id
                    except Exception:
                        pass
            if (not parent_kind) and parent_id:
                parent_kind = "baseline" if str(parent_id).startswith("baseline") else "candidate"
            entries.append(
                {
                    "candidate_id": str(eval_summary.get("harness_id") or harness_dir.name),
                    "kind": "baseline" if bucket == "baselines" else "candidate",
                    "parent_id": parent_id,
                    "parent_kind": parent_kind,
                    "derived_from": derived_from,
                    "path": str(harness_dir.as_posix()),
                    "total_runs": int(eval_summary.get("total_runs") or len(run_dirs)),
                    "metrics": dict(eval_summary.get("metrics") or {}),
                    "status": str(eval_summary.get("status") or "unknown"),
                    "per_scene_metrics_path": str(per_scene_path.as_posix()) if per_scene_path.exists() else None,
                    "eval_summary_path": str(eval_path.as_posix()),
                    "per_scene_metrics": per_scene,
                    "trace_locations": {
                        "runs_dir": str((harness_dir / "runs").as_posix()),
                        "run_count": len(run_dirs),
                    },
                }
            )

    pareto_ready = []
    for e in entries:
        m = e.get("metrics") or {}
        if {"collision_count_avg", "near_miss_count_avg", "completion_time_mission_sec_avg", "llm_call_count_avg"}.issubset(m.keys()):
            pareto_ready.append(
                {
                    "candidate_id": e["candidate_id"],
                    "metrics": {
                        "collision_count_avg": m["collision_count_avg"],
                        "near_miss_count_avg": m["near_miss_count_avg"],
                        "completion_time_mission_sec_avg": m["completion_time_mission_sec_avg"],
                        "llm_call_count_avg": m["llm_call_count_avg"],
                    },
                }
            )
    # adapt to evaluator's key name for pareto function
    for p in pareto_ready:
        p["metrics"]["completion_time_sec_avg"] = p["metrics"].pop("completion_time_mission_sec_avg")

    pareto_map = {
        e["candidate_id"]: e for e in mark_pareto(
            [{"harness_id": x["candidate_id"], "metrics": x["metrics"]} for x in pareto_ready]
        )
    }
    for e in entries:
        e["pareto_frontier"] = bool(pareto_map.get(e["candidate_id"], {}).get("pareto_frontier", False))

    index = {
        "archive_version": "proposer_archive_v2",
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "pairs": EVALUATION_PROTOCOL_SEQUENCE,
            "total_runs": TOTAL_EVAL_RUNS,
        },
        "entries": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index
