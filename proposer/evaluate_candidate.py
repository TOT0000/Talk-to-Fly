from __future__ import annotations

import json
import subprocess
import shutil
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from controller.harness_protocol import (
    EVALUATION_PROTOCOLS,
    get_evaluation_protocol,
)
from proposer.live_benchmark_runner import LiveBenchmarkRunner, RunArtifact
from proposer.registry import HarnessRegistry
from proposer.contract_validator import validate_candidate_contract
from proposer.runtime_verifier import verify_runtime_artifact, TRACE_SCHEMA_VERSION
from proposer.candidate_manifest import build_provenance_bundle
from proposer.evaluation_pipeline import FormalEvaluator, build_failure_dossier, write_dossier
from proposer.archive_manager import persist_evidence_bundle


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


def _summary_file_name(stage: str) -> str:
    return f"eval_summary_{stage}.json"


def _scene_file_name(stage: str) -> str:
    return f"per_scene_metrics_{stage}.json"


def _run_file_name(stage: str) -> str:
    return f"per_run_metrics_{stage}.json"




def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_manual_runs_xlsx(*, xlsx_path: Path, baseline_id: str, evaluation_protocol: Dict) -> List[Dict]:
    if not xlsx_path.exists():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def _cell_value(cell) -> str | None:
        ctype = cell.get("t")
        if ctype == "inlineStr":
            return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
        v = cell.find("a:v", ns)
        return None if v is None else str(v.text)

    try:
        with zipfile.ZipFile(xlsx_path) as zf:
            sheet_xml = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    except Exception:
        return []

    rows = sheet_xml.findall(".//a:sheetData/a:row", ns)
    if not rows:
        return []

    header_cells = rows[0].findall("a:c", ns)
    headers = [_cell_value(c) or "" for c in header_cells]
    wanted = {
        "run_id",
        "scene_id",
        "selected_baseline_id",
        "run_status",
        "task_success",
        "completion_time_mission_sec",
        "total_llm_call_count",
        "collision_count",
        "near_miss_count",
    }
    if not wanted.issubset(set(headers)):
        return []

    col_to_idx = {h: i for i, h in enumerate(headers)}
    scene_zone = {str(p.get("scene_id")): str(p.get("task_zone")) for p in list(evaluation_protocol.get("pairs") or [])}

    per_scene_seed: Dict[str, int] = {}
    parsed: List[Dict] = []
    for row in rows[1:]:
        values = [_cell_value(c) for c in row.findall("a:c", ns)]
        if not values:
            continue
        selected_baseline = values[col_to_idx["selected_baseline_id"]] if col_to_idx["selected_baseline_id"] < len(values) else None
        if str(selected_baseline or "") != str(baseline_id):
            continue
        scene_id = str(values[col_to_idx["scene_id"]] if col_to_idx["scene_id"] < len(values) else "")
        if scene_id not in scene_zone:
            continue
        seed = per_scene_seed.get(scene_id, 0)
        per_scene_seed[scene_id] = seed + 1

        success_raw = values[col_to_idx["task_success"]] if col_to_idx["task_success"] < len(values) else "0"
        mission_success = str(success_raw or "0").strip() in {"1", "true", "True"}
        completion_raw = values[col_to_idx["completion_time_mission_sec"]] if col_to_idx["completion_time_mission_sec"] < len(values) else None
        llm_raw = values[col_to_idx["total_llm_call_count"]] if col_to_idx["total_llm_call_count"] < len(values) else "0"
        collision_raw = values[col_to_idx["collision_count"]] if col_to_idx["collision_count"] < len(values) else "0"
        near_miss_raw = values[col_to_idx["near_miss_count"]] if col_to_idx["near_miss_count"] < len(values) else "0"
        parsed.append(
            {
                "run_id": str(values[col_to_idx["run_id"]] if col_to_idx["run_id"] < len(values) else f"manual_{scene_id}_{seed}"),
                "scene_id": scene_id,
                "task_zone": scene_zone[scene_id],
                "run_status": str(values[col_to_idx["run_status"]] if col_to_idx["run_status"] < len(values) else "unknown"),
                "mission_success": mission_success,
                "completion_time_mission_sec": (None if completion_raw in {None, ""} else float(completion_raw)),
                "llm_call_count": int(float(llm_raw or 0)),
                "collision_count": int(float(collision_raw or 0)),
                "near_miss_count": int(float(near_miss_raw or 0)),
                "seed": seed,
            }
        )
    return parsed


def _load_baseline_runs_for_formal(*, archive_root: Path, baseline_id: str, evaluation_protocol: Dict) -> List[Dict]:
    if not baseline_id:
        return []
    baseline_root = archive_root / ("baselines" if baseline_id.startswith("baseline") else "candidates") / baseline_id
    formal_path = baseline_root / "per_run_metrics_formal.json"
    if formal_path.exists():
        return _load_json(formal_path)

    legacy_path = baseline_root / "per_run_metrics.json"
    if legacy_path.exists():
        rows = _load_json(legacy_path)
        _write_json(formal_path, rows)
        return rows

    manual_xlsx = archive_root.parent / "proposer_archive" / "manual_runs" / "task_runs.xlsx"
    rows = _parse_manual_runs_xlsx(xlsx_path=manual_xlsx, baseline_id=baseline_id, evaluation_protocol=evaluation_protocol)
    if rows:
        _write_json(formal_path, rows)
    return rows

def evaluate_candidate_live(
    repo_root: Path,
    harness_id: str,
    archive_root: Path,
    evaluation_mode: str | None = None,
) -> EvaluationResult:
    repo_root = Path(repo_root)
    archive_root = Path(archive_root)
    harness_entry = HarnessRegistry(repo_root).get(harness_id)

    target = archive_root / ("baselines" if harness_id.startswith("baseline") else "candidates") / harness_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "code_or_spec").mkdir(parents=True, exist_ok=True)

    evaluation_protocol = get_evaluation_protocol(kind=harness_entry.kind, requested_mode=evaluation_mode)
    stage = str(evaluation_protocol.get("mode") or "formal")

    runner = LiveBenchmarkRunner(
        repo_root=repo_root,
        output_root=target,
        harness_id=harness_id,
        evaluation_protocol=evaluation_protocol,
    )
    runs = runner.run()
    commit_hash = "unknown"
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        pass

    spec_payload = harness_entry.spec
    manifest = dict(spec_payload.get("candidate_manifest") or spec_payload.get("proposal_contract") or {})
    if harness_entry.kind == "candidate" and manifest:
        parent_id = str(spec_payload.get("parent") or "")
        if parent_id:
            parent_dir = HarnessRegistry(repo_root).get(parent_id).dir_path
            validate_candidate_contract(candidate_dir=harness_entry.dir_path, parent_dir=parent_dir)

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
        "parent_diff.patch",
        "proposer_tool_audit.json",
    ]:
        src = harness_entry.dir_path / name
        if src.exists():
            shutil.copy2(src, target / "code_or_spec" / name)

    by_scene: Dict[str, List[RunArtifact]] = {}
    for r in runs:
        by_scene.setdefault(r.scene_id, []).append(r)

    scene_rows = {}
    for pair in list(evaluation_protocol.get("pairs") or []):
        scene = str(pair["scene_id"])
        zone = str(pair["task_zone"])
        cnt = int(pair["runs"])
        scene_rows[scene] = _build_scene_metrics(by_scene.get(scene, []), scene, zone, cnt)

    per_scene = {
        "evaluation_stage": stage,
        "evaluation_protocol": {
            "name": evaluation_protocol.get("name"),
            "version": evaluation_protocol.get("version"),
            "runs_per_scene": int(evaluation_protocol.get("runs_per_scene") or 0),
            "total_runs_expected": int(evaluation_protocol.get("total_runs") or 0),
        },
        "scenes": scene_rows,
    }

    success_total = sum(1 for r in runs if r.mission_success)
    overall_completion_avg = _avg_completion_success_only(runs)
    formal_exists = (target / _summary_file_name("formal")).exists()
    promoted_to_formal = bool(formal_exists or (stage == "formal"))
    eval_summary = {
        "harness_id": harness_id,
        "kind": harness_entry.kind,
        "status": "evaluated_live",
        "evaluation_stage": stage,
        "parent_id": harness_entry.spec.get("parent"),
        "parent_kind": ("baseline" if str(harness_entry.spec.get("parent", "")).startswith("baseline") else "candidate") if harness_entry.spec.get("parent") else None,
        "derived_from": harness_entry.spec.get("parent"),
        "evaluation_protocol": {
            "name": evaluation_protocol.get("name"),
            "version": evaluation_protocol.get("version"),
            "pairs": list(evaluation_protocol.get("pairs") or []),
            "runs_per_scene": int(evaluation_protocol.get("runs_per_scene") or 0),
            "total_runs": int(evaluation_protocol.get("total_runs") or 0),
        },
        "total_runs": len(runs),
        "total_runs_expected": int(evaluation_protocol.get("total_runs") or 0),
        "total_runs_completed": len(runs),
        "stage_complete": len(runs) == int(evaluation_protocol.get("total_runs") or 0),
        "promoted_to_formal": promoted_to_formal,
        "available_evaluation_stages": sorted(
            [k for k in EVALUATION_PROTOCOLS.keys() if (target / _summary_file_name(k)).exists()] + [stage]
        ),
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
    dossiers: List[Dict] = []
    for row in per_run_payload:
        metadata_path = Path(str(row.get("metadata_path") or ""))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        prompt_source = dict(metadata.get("evaluate_prompt_source") or {})
        prompt_asset = str(prompt_source.get("selected_prompt_asset_path") or "")
        prompt_asset_text = Path(prompt_asset).read_text(encoding="utf-8") if prompt_asset and Path(prompt_asset).exists() else ""
        rendered_prompt = str(prompt_source.get("rendered_prompt") or "")
        provenance = build_provenance_bundle(
            candidate_id=harness_id,
            commit_hash=commit_hash,
            config_payload=spec_payload,
            prompt_asset_text=prompt_asset_text,
            rendered_prompt_text=rendered_prompt,
            state_schema_version=str(spec_payload.get("state_schema_version") or "state_schema_v1"),
            trigger_policy_version=str(spec_payload.get("trigger_policy_version") or "trigger_policy_v1"),
            benchmark_pack_id=str(evaluation_protocol.get("version") or "benchmark_unknown"),
            scene_id=str(row.get("scene_id") or ""),
            zone_id=str(row.get("task_zone") or ""),
            seed=int(row.get("seed") or 0),
            evaluator_version=f"{stage}_v1",
            trace_schema_version=TRACE_SCHEMA_VERSION,
        )
        metadata["provenance"] = provenance
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        verification = verify_runtime_artifact(candidate_manifest=manifest, run_artifact=row, metadata=metadata)
        row["runtime_verification"] = {"passed": verification.passed, "checks": verification.checks, "errors": verification.errors}
        if not verification.passed or (not bool(row.get("mission_success"))):
            dossier = build_failure_dossier(
                candidate_id=harness_id,
                baseline_id=str(spec_payload.get("parent") or "baseline1"),
                run=row,
                metadata=metadata,
                verification=verification.checks,
            )
            dossier_path = target / "runs" / str(row.get("run_id")) / "failure_dossier.json"
            write_dossier(dossier_path, dossier)
            dossiers.append(dossier)
    evidence_path = persist_evidence_bundle(
        archive_root=archive_root,
        candidate_id=harness_id,
        payload={
            "candidate_manifest": manifest,
            "artifact_hashes": {},
            "state_schema_version": str(spec_payload.get("state_schema_version") or "state_schema_v1"),
            "trigger_policy_version": str(spec_payload.get("trigger_policy_version") or "trigger_policy_v1"),
            "lineage": {"parent": spec_payload.get("parent")},
            "failure_dossiers_count": len(dossiers),
        },
    )
    eval_summary["evidence_bundle_path"] = evidence_path.as_posix()
    with (target / _run_file_name(stage)).open("w", encoding="utf-8") as f:
        json.dump(per_run_payload, f, ensure_ascii=False, indent=2)
    with (target / _summary_file_name(stage)).open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    with (target / _scene_file_name(stage)).open("w", encoding="utf-8") as f:
        json.dump(per_scene, f, ensure_ascii=False, indent=2)
    if stage == "formal":
        # Keep formal outputs in legacy paths for backward compatibility.
        with (target / "per_run_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_run_payload, f, ensure_ascii=False, indent=2)
        with (target / "eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        with (target / "per_scene_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_scene, f, ensure_ascii=False, indent=2)

        baseline_id = str(spec_payload.get("parent") or "")
        baseline_runs = _load_baseline_runs_for_formal(
            archive_root=archive_root,
            baseline_id=baseline_id,
            evaluation_protocol=evaluation_protocol,
        )
        if baseline_runs:
            formal = FormalEvaluator().evaluate(
                candidate_id=harness_id,
                baseline_id=baseline_id,
                candidate_runs=per_run_payload,
                baseline_runs=baseline_runs,
            )
            _write_json(target / "formal_summary.json", formal["formal_summary"])
            _write_json(target / "formal_pairwise_deltas.json", formal["formal_pairwise_deltas"])
            _write_json(target / "formal_safety_report.json", formal["formal_safety_report"])
            _write_json(target / "formal_dossier.json", formal["formal_dossier"])
            eval_summary["formal_decision"] = formal["formal_summary"].get("decision")
            eval_summary["formal_decision_rationale"] = formal["formal_summary"].get("decision_rationale")
            eval_summary["formal_artifacts"] = {
                "formal_summary": (target / "formal_summary.json").as_posix(),
                "formal_pairwise_deltas": (target / "formal_pairwise_deltas.json").as_posix(),
                "formal_safety_report": (target / "formal_safety_report.json").as_posix(),
                "formal_dossier": (target / "formal_dossier.json").as_posix(),
            }
            with (target / _summary_file_name(stage)).open("w", encoding="utf-8") as f:
                json.dump(eval_summary, f, ensure_ascii=False, indent=2)
            with (target / "eval_summary.json").open("w", encoding="utf-8") as f:
                json.dump(eval_summary, f, ensure_ascii=False, indent=2)

    return EvaluationResult(eval_summary=eval_summary, per_scene_metrics=per_scene, run_artifacts=per_run_payload)
