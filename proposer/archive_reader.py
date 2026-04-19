from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def read_archive_index(archive_root: Path) -> Dict:
    archive_root = Path(archive_root)
    return _read_json(archive_root / "index.json", {"entries": []})


def summarize_archive_for_proposer(repo_root: Path, archive_root: Path, max_entries: int = 12) -> Dict:
    repo_root = Path(repo_root)
    archive_root = Path(archive_root)
    idx = read_archive_index(archive_root)
    entries = list(idx.get("entries", []))

    baselines = [e.get("candidate_id") for e in entries if e.get("kind") == "baseline"]
    candidates = [e.get("candidate_id") for e in entries if e.get("kind") == "candidate"]
    pareto = [e.get("candidate_id") for e in entries if bool(e.get("pareto_frontier"))]
    latest = entries[-1].get("candidate_id") if entries else "(none)"

    selected_entries = list(entries[-max_entries:])
    snippets = collect_representative_trace_snippets(repo_root=repo_root, archive_root=archive_root, max_traces=6)

    selector = _load_archive_selector(repo_root, entries)
    if selector is not None:
        selected_entries = _apply_selector(selector, "select_entries", selected_entries, max_entries)
        snippets = _apply_selector(selector, "select_trace_snippets", snippets, 6)

    compact_entries: List[Dict] = []
    for e in selected_entries:
        compact_entries.append(
            {
                "candidate_id": e.get("candidate_id"),
                "kind": e.get("kind"),
                "parent_id": e.get("parent_id"),
                "status": e.get("status"),
                "total_runs": e.get("total_runs"),
                "metrics": e.get("metrics", {}),
                "pareto_frontier": bool(e.get("pareto_frontier")),
                "per_scene_metrics_path": e.get("per_scene_metrics_path"),
                "trace_locations": e.get("trace_locations", {}),
            }
        )

    return {
        "baseline_list": baselines,
        "candidate_list": candidates,
        "pareto_list": pareto,
        "latest_harness": latest,
        "entries": compact_entries,
        "trace_snippets": snippets,
    }


def _load_archive_selector(repo_root: Path, entries: List[Dict]):
    if not entries:
        return None
    latest_id = str(entries[-1].get("candidate_id") or "")
    if not latest_id:
        return None
    if latest_id.startswith("baseline"):
        selector_path = repo_root / "harnesses" / latest_id / "archive_selector.py"
    else:
        selector_path = repo_root / "harnesses" / "candidates" / latest_id / "archive_selector.py"
    if not selector_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("harness_archive_selector", str(selector_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _apply_selector(mod, fn_name: str, payload: List[Dict], max_count: int) -> List[Dict]:
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return payload
    try:
        out = fn(list(payload), int(max_count))
        if isinstance(out, list):
            return out
    except Exception:
        return payload
    return payload


def collect_representative_trace_snippets(repo_root: Path, archive_root: Path, max_traces: int = 6) -> List[Dict]:
    repo_root = Path(repo_root)
    archive_root = Path(archive_root)
    out: List[Dict] = []

    run_dirs = sorted(archive_root.glob("**/runs/run_*"))
    for run_dir in run_dirs[:max_traces]:
        runtime = run_dir / "runtime_trace.jsonl"
        planning = run_dir / "planning_trace.jsonl"
        metadata = run_dir / "metadata.json"
        item = {"run_dir": run_dir.as_posix()}

        if metadata.exists():
            item["metadata"] = _read_json(metadata, {})

        if runtime.exists():
            try:
                first_line = runtime.read_text(encoding="utf-8").splitlines()[:1]
                item["runtime_head"] = first_line
            except Exception:
                pass

        if planning.exists():
            try:
                first_line = planning.read_text(encoding="utf-8").splitlines()[:1]
                item["planning_head"] = first_line
            except Exception:
                pass

        out.append(item)

    # fallback to legacy manual archive if live runs are absent
    if not out:
        legacy_runs = sorted((repo_root / "proposer_archive/manual_runs/runs").glob("run_*"))
        for run_dir in legacy_runs[:max_traces]:
            runtime_files = sorted(run_dir.glob("*_runtime_trace.jsonl"))
            planning_files = sorted(run_dir.glob("*_planning_trace.jsonl"))
            item = {"run_dir": run_dir.as_posix()}
            if runtime_files:
                item["runtime_path"] = runtime_files[0].as_posix()
            if planning_files:
                item["planning_path"] = planning_files[0].as_posix()
            out.append(item)

    return out
