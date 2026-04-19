from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

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

    # MVP heuristic: start from the strongest prior baseline by ID order (baseline3 first),
    # then make a targeted trigger+prompt edit while keeping mutation boundary strict.
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
            entries.append(
                {
                    "harness_id": str(eval_summary.get("harness_id") or harness_dir.name),
                    "kind": "baseline" if bucket == "baselines" else "candidate",
                    "path": str(harness_dir.as_posix()),
                    "metrics": dict(eval_summary.get("metrics") or {}),
                    "status": str(eval_summary.get("status") or "unknown"),
                    "per_scene_metrics": per_scene,
                    "trace_dir": str((harness_dir / "traces").as_posix()),
                }
            )

    pareto_ready = [e for e in entries if e.get("metrics")]
    pareto_map = {e["harness_id"]: e for e in mark_pareto(pareto_ready)}
    for e in entries:
        e["pareto_frontier"] = bool(pareto_map.get(e["harness_id"], {}).get("pareto_frontier", False))

    index = {
        "archive_version": "proposer_archive_v2",
        "evaluation_protocol": {
            "version": "uav_search_v1",
            "scene_to_task_zone": {
                "SCENE1": "zoneA",
                "SCENE2": "zoneB",
                "SCENE3": "zoneC",
            },
        },
        "entries": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index
