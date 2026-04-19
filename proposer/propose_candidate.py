from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

from controller.harness_protocol import EVALUATION_PROTOCOL_SEQUENCE, EVALUATION_PROTOCOL_VERSION, TOTAL_EVAL_RUNS
from proposer.archive_reader import summarize_archive_for_proposer
from proposer.evaluate_candidate import mark_pareto
from proposer.prompts import OUTPUT_CONTRACT, build_iteration_prompt
from proposer.registry import ALLOWED_MUTATION_FILES, HarnessRegistry, validate_candidate_boundary


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


def _extract_json_object(text: str) -> Dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_]*\n", "", raw)
        raw = raw.rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def _llm_json(llm, model_name: str, prompt: str) -> Dict:
    raw = llm.request(prompt=prompt, model_name=model_name, stream=False)
    return _extract_json_object(str(raw or ""))


def _build_file_generation_prompt(
    *,
    parent_harness_id: str,
    parent_spec: Dict,
    parent_file_name: str,
    parent_file_content: str,
    proposal: Dict,
) -> str:
    return (
        "You are generating one bounded harness file for UAV harness optimization.\n"
        "Allowed harness boundary files: spec.json, state_encoder.py, trigger_policy.py, prompt_builder.py, proposer_note.txt, README.md\n"
        "You must output ONLY the full content of the requested file, no markdown fences.\n"
        "Do not modify simulator/PX4/controller/executor/collision math/checkpoint rules.\n\n"
        f"Parent harness: {parent_harness_id}\n"
        f"Parent spec:\n{json.dumps(parent_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Proposal contract:\n{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
        f"Requested file: {parent_file_name}\n"
        "Current parent file content:\n"
        f"{parent_file_content}\n\n"
        "Generate improved content aligned with the proposal hypothesis and weakness.\n"
        "If requested file is spec.json, ensure valid JSON and keep candidate lineage metadata."
    )


def propose_next_candidate(
    repo_root: Path,
    note: str = "",
    focus_text: str = "Improve safety-aware replan timing while avoiding unnecessary detours.",
    allow_fallback_heuristic: bool = False,
) -> Path:
    repo_root = Path(repo_root)
    reg = HarnessRegistry(repo_root)
    baselines = reg.list_baselines()
    if not baselines:
        raise RuntimeError("No baselines found in harnesses/")

    archive_summary = summarize_archive_for_proposer(repo_root, repo_root / "proposer_archive_v2")
    prompt = build_iteration_prompt(
        baseline_list=json.dumps(archive_summary.get("baseline_list", []), ensure_ascii=False),
        candidate_list=json.dumps(archive_summary.get("candidate_list", []), ensure_ascii=False),
        pareto_list=json.dumps(archive_summary.get("pareto_list", []), ensure_ascii=False),
        latest_harness=str(archive_summary.get("latest_harness", "(none)")),
        focus_text=focus_text,
        archive_evidence=json.dumps(
            {
                "entries": archive_summary.get("entries", []),
                "trace_snippets": archive_summary.get("trace_snippets", []),
                "output_contract": OUTPUT_CONTRACT,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    from controller.llm_wrapper import LLMWrapper, MODEL_NAME

    llm = LLMWrapper(temperature=0.1)
    try:
        proposal = _llm_json(llm, MODEL_NAME, prompt)
    except Exception:
        if not allow_fallback_heuristic:
            raise
        # conservative fallback path (explicitly marked)
        proposal = {
            "parent_harness": "baseline3",
            "candidate_id": "",
            "one_sentence_hypothesis": "Conservative fallback due to proposer LLM failure.",
            "weakness_being_addressed": "LLM unavailable during proposal call",
            "expected_tradeoff": "Minimal structured change",
            "files_to_create_or_modify": ["spec.json", "proposer_note.txt"],
            "proposer_note_text": "Fallback proposal generated because LLM proposer call failed.",
        }

    parent_id = str(proposal.get("parent_harness") or "").strip()
    if not parent_id:
        raise ValueError("LLM proposer output missing parent_harness")

    parent_entry = reg.get(parent_id)

    asked_candidate_id = str(proposal.get("candidate_id") or "").strip()
    if re.fullmatch(r"candidate_\d{4}", asked_candidate_id) and not (reg.candidates_dir / asked_candidate_id).exists():
        candidate_id = asked_candidate_id
    else:
        candidate_id = _next_candidate_id(reg.candidates_dir)

    candidate_dir = reg.candidates_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)

    files_to_modify = [str(v) for v in list(proposal.get("files_to_create_or_modify") or [])]
    if not files_to_modify:
        files_to_modify = ["spec.json", "trigger_policy.py", "proposer_note.txt"]

    # Start from parent snapshot for deterministic bounded edits.
    for name in ["spec.json", "state_encoder.py", "trigger_policy.py", "prompt_builder.py"]:
        src = parent_entry.dir_path / name
        if src.exists():
            shutil.copy2(src, candidate_dir / name)

    parent_spec = _load_json(parent_entry.dir_path / "spec.json")

    normalized_target_files = []
    for name in files_to_modify:
        base = Path(name).name
        if base in ALLOWED_MUTATION_FILES:
            normalized_target_files.append(base)

    for target_file in normalized_target_files:
        if target_file in {"proposer_note.txt", "README.md"}:
            continue
        parent_file_content = (candidate_dir / target_file).read_text(encoding="utf-8") if (candidate_dir / target_file).exists() else ""
        file_prompt = _build_file_generation_prompt(
            parent_harness_id=parent_id,
            parent_spec=parent_spec,
            parent_file_name=target_file,
            parent_file_content=parent_file_content,
            proposal=proposal,
        )
        generated = str(llm.request(prompt=file_prompt, model_name=MODEL_NAME, stream=False) or "").strip()
        if target_file == "spec.json":
            spec_obj = _extract_json_object(generated)
            spec_obj["id"] = candidate_id
            spec_obj["kind"] = "candidate"
            spec_obj["parent"] = parent_id
            lineage = dict(spec_obj.get("lineage") or {})
            lineage.update(
                {
                    "parent_id": parent_id,
                    "parent_kind": "baseline" if parent_id.startswith("baseline") else "candidate",
                    "derived_from": parent_id,
                }
            )
            spec_obj["lineage"] = lineage
            spec_obj.setdefault("mutation", {})
            spec_obj["mutation"]["type"] = "llm_agent_driven"
            (candidate_dir / target_file).write_text(json.dumps(spec_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            (candidate_dir / target_file).write_text(generated + "\n", encoding="utf-8")

    proposer_note_text = str(proposal.get("proposer_note_text") or "").strip()
    final_note = note or proposer_note_text or "LLM proposer generated candidate without additional note."
    (candidate_dir / "proposer_note.txt").write_text(final_note + "\n", encoding="utf-8")

    # Ensure mandatory spec metadata even if LLM skipped it.
    spec = _load_json(candidate_dir / "spec.json")
    spec["id"] = candidate_id
    spec["kind"] = "candidate"
    spec["parent"] = parent_id
    spec.setdefault("lineage", {})
    spec["lineage"]["parent_id"] = parent_id
    spec["lineage"]["parent_kind"] = "baseline" if parent_id.startswith("baseline") else "candidate"
    spec["lineage"]["derived_from"] = parent_id
    spec.setdefault("proposal_contract", {})
    spec["proposal_contract"] = {
        "one_sentence_hypothesis": str(proposal.get("one_sentence_hypothesis") or ""),
        "weakness_being_addressed": str(proposal.get("weakness_being_addressed") or ""),
        "expected_tradeoff": str(proposal.get("expected_tradeoff") or ""),
        "files_to_create_or_modify": normalized_target_files,
    }
    (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

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
