from __future__ import annotations

import difflib
import json
import py_compile
from pathlib import Path
from typing import Dict, List, Tuple

from proposer.archive_reader import read_archive_index
from proposer.consistency import validate_candidate_contract_alignment
from proposer.registry import ALLOWED_MUTATION_FILES, HarnessRegistry


class ProposerToolbox:
    """Controlled read/write helper APIs used by the coding-agent proposer."""

    def __init__(self, repo_root: Path, archive_root: Path):
        self.repo_root = Path(repo_root)
        self.archive_root = Path(archive_root)
        self.registry = HarnessRegistry(self.repo_root)
        self.audit_log: List[Dict] = []

    def _record(self, tool: str, **payload) -> None:
        event = {"tool": tool}
        event.update(payload)
        self.audit_log.append(event)

    def _read_json(self, path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _index_entry_for_harness(self, harness_id: str) -> Dict:
        idx = read_archive_index(self.archive_root)
        for entry in list(idx.get("entries", [])):
            if str(entry.get("candidate_id")) == str(harness_id):
                return dict(entry)
        return {}

    def _trace_files_for_run(self, run_dir: Path, kind: str) -> List[Path]:
        names = [f"{kind}_trace.jsonl"]
        names.extend(sorted([p.name for p in run_dir.glob(f"*_{kind}_trace.jsonl")]))
        out: List[Path] = []
        for name in names:
            path = run_dir / name
            if path.exists():
                out.append(path)
        return out

    def _extract_harness_markers(self, run_dir: Path, max_lines: int = 6) -> List[str]:
        markers: List[str] = []
        metadata = run_dir / "metadata.json"
        if metadata.exists():
            payload = self._read_json(metadata, {})
            for key in ["candidate_id", "harness_id", "baseline_id", "selected_baseline_id", "selected_harness_id"]:
                val = payload.get(key)
                if val:
                    markers.append(str(val))
        for trace_path in self._trace_files_for_run(run_dir, "planning") + self._trace_files_for_run(run_dir, "runtime"):
            try:
                lines = trace_path.read_text(encoding="utf-8").splitlines()[: max(1, int(max_lines))]
            except Exception:
                continue
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for key in [
                    "candidate_id",
                    "harness_id",
                    "baseline_id",
                    "selected_baseline_id",
                    "selected_harness_id",
                    "selected_candidate_id",
                ]:
                    val = obj.get(key)
                    if val:
                        markers.append(str(val))
        return markers

    def _resolve_evidence_sources(self, harness_id: str) -> List[Dict]:
        entry = self._index_entry_for_harness(harness_id)
        kind = "baseline" if str(harness_id).startswith("baseline") else "candidate"
        bucket = "baselines" if kind == "baseline" else "candidates"
        harness_root = self.archive_root / bucket / str(harness_id)

        candidates: List[Tuple[Path, str, bool]] = []
        index_runs_dir = Path(str(((entry.get("trace_locations") or {}).get("runs_dir") or "")))
        if str(index_runs_dir):
            candidates.append((index_runs_dir, "archive_index_runs_dir", True))
        candidates.append((harness_root / "runs", "archive_harness_runs", True))

        pointer_path = harness_root / "traces" / "trace_pointers.json"
        if pointer_path.exists():
            pointer = self._read_json(pointer_path, {})
            src = str(pointer.get("source") or "").strip()
            if src:
                src_path = Path(src)
                if not src_path.is_absolute():
                    candidates.append(((self.repo_root / src_path), "trace_pointer_source", False))
                    candidates.append(((self.archive_root / src_path), "trace_pointer_source_archive_relative", False))

        candidates.append((self.repo_root / "proposer_archive" / "manual_runs" / "runs", "legacy_manual_runs_default", False))

        out: List[Dict] = []
        seen = set()
        for path, source_type, scoped in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append({"path": path, "source_type": source_type, "scoped_to_harness": bool(scoped)})
        return out

    def _collect_runs(self, harness_id: str, limit: int = 12) -> Tuple[List[Dict], Dict]:
        runs: List[Dict] = []
        debug = {"harness_id": harness_id, "searched_sources": [], "resolution": "not_found"}
        for src in self._resolve_evidence_sources(harness_id):
            source_path = Path(src["path"])
            item = {
                "source_type": src["source_type"],
                "path": source_path.as_posix(),
                "scoped_to_harness": bool(src["scoped_to_harness"]),
                "exists": source_path.exists(),
                "matched_runs": 0,
                "status": "path_not_found",
            }
            if not source_path.exists():
                debug["searched_sources"].append(item)
                continue

            run_dirs = sorted(source_path.glob("run_*"))
            item["available_runs"] = len(run_dirs)
            matched_in_source = 0
            for run_dir in run_dirs:
                if not run_dir.is_dir():
                    continue
                markers = self._extract_harness_markers(run_dir)
                is_match = bool(src["scoped_to_harness"]) or (str(harness_id) in markers)
                if not is_match:
                    continue
                metadata = self.read_run_metadata(run_dir.as_posix())
                runtime_files = [p.name for p in self._trace_files_for_run(run_dir, "runtime")]
                planning_files = [p.name for p in self._trace_files_for_run(run_dir, "planning")]
                runs.append(
                    {
                        "run_id": run_dir.name,
                        "run_dir": run_dir.as_posix(),
                        "source": source_path.as_posix(),
                        "source_type": src["source_type"],
                        "available_files": {
                            "metadata": bool(metadata),
                            "runtime_traces": runtime_files,
                            "planning_traces": planning_files,
                        },
                        "scene": metadata.get("scene"),
                        "task": metadata.get("task"),
                        "status": metadata.get("status") or metadata.get("run_status"),
                    }
                )
                matched_in_source += 1
                if len(runs) >= max(1, int(limit)):
                    break
            item["matched_runs"] = matched_in_source
            item["status"] = "ok" if matched_in_source > 0 else ("no_runs_in_source" if item.get("available_runs", 0) == 0 else "no_matching_runs")
            debug["searched_sources"].append(item)
            if matched_in_source > 0:
                debug["resolution"] = "ok"
                break
            if len(runs) >= max(1, int(limit)):
                break
        if not runs:
            debug["reason"] = "no_runs_truly_exist_or_unmapped"
        return runs[: max(1, int(limit))], debug

    def list_harnesses(self, kind: str = "all") -> List[Dict]:
        entries = []
        if kind in {"all", "baseline"}:
            entries.extend(self.registry.list_baselines())
        if kind in {"all", "candidate"}:
            entries.extend(self.registry.list_candidates())
        out = [
            {
                "harness_id": e.harness_id,
                "kind": e.kind,
                "parent": e.spec.get("parent"),
                "path": e.dir_path.as_posix(),
            }
            for e in entries
        ]
        self._record("list_harnesses", kind=kind, count=len(out))
        return out

    def read_harness_spec(self, harness_id: str) -> Dict:
        entry = self.registry.get(harness_id)
        out = dict(entry.spec)
        self._record("read_harness_spec", harness_id=harness_id)
        return out

    def read_harness_code(self, harness_id: str, file_name: str) -> str:
        entry = self.registry.get(harness_id)
        base = Path(str(file_name)).name
        if base not in ALLOWED_MUTATION_FILES:
            raise ValueError(f"file not allowed: {file_name}")
        path = entry.dir_path / base
        if not path.exists():
            return ""
        out = path.read_text(encoding="utf-8")
        self._record("read_harness_code", harness_id=harness_id, file_name=base, bytes=len(out))
        return out

    def diff_harnesses(self, parent_harness: str, candidate_harness: str, file_name: str) -> str:
        base = Path(str(file_name)).name
        left = self.read_harness_code(parent_harness, base).splitlines(keepends=True)
        right = self.read_harness_code(candidate_harness, base).splitlines(keepends=True)
        diff = difflib.unified_diff(left, right, fromfile=f"{parent_harness}/{base}", tofile=f"{candidate_harness}/{base}")
        out = "".join(diff)
        self._record("diff_harnesses", parent_harness=parent_harness, candidate_harness=candidate_harness, file_name=base, bytes=len(out))
        return out

    def list_runs(self, harness_id: str, limit: int = 12) -> List[Dict]:
        out, debug = self._collect_runs(harness_id=harness_id, limit=limit)
        self._record("list_runs", harness_id=harness_id, count=len(out), limit=int(limit), debug=debug)
        return out

    def read_run_metadata(self, run_dir: str) -> Dict:
        path = Path(str(run_dir)) / "metadata.json"
        if not path.exists():
            return {}
        out = json.loads(path.read_text(encoding="utf-8"))
        self._record("read_run_metadata", run_dir=str(run_dir), has_metadata=bool(out))
        return out

    def search_traces(self, harness_id: str, needle: str, max_hits: int = 12) -> List[Dict]:
        hits: List[Dict] = []
        runs, debug = self._collect_runs(harness_id=harness_id, limit=64)
        searched_paths: List[str] = []
        for run in runs:
            run_dir = Path(run["run_dir"])
            metadata = self.read_run_metadata(run_dir.as_posix())
            if metadata:
                meta_blob = json.dumps(metadata, ensure_ascii=False)
                if needle.lower() in meta_blob.lower():
                    hits.append(
                        {
                            "run_id": run["run_id"],
                            "trace": (run_dir / "metadata.json").as_posix(),
                            "line_no": 1,
                            "line": meta_blob[:600],
                            "source_type": run.get("source_type"),
                            "match_source": "metadata",
                        }
                    )
                    if len(hits) >= int(max_hits):
                        self._record(
                            "search_traces",
                            harness_id=harness_id,
                            needle=needle,
                            hits=len(hits),
                            max_hits=int(max_hits),
                            searched_paths=searched_paths,
                            debug=debug,
                        )
                        return hits
            for path in self._trace_files_for_run(run_dir, "runtime") + self._trace_files_for_run(run_dir, "planning"):
                searched_paths.append(path.as_posix())
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if needle.lower() in line.lower():
                        hits.append(
                            {
                                "run_id": run["run_id"],
                                "trace": path.as_posix(),
                                "line_no": i,
                                "line": line[:600],
                                "source_type": run.get("source_type"),
                                "match_source": "trace_line",
                            }
                        )
                        if len(hits) >= int(max_hits):
                            self._record(
                                "search_traces",
                                harness_id=harness_id,
                                needle=needle,
                                hits=len(hits),
                                max_hits=int(max_hits),
                                searched_paths=searched_paths,
                                debug=debug,
                            )
                            return hits
        self._record(
            "search_traces",
            harness_id=harness_id,
            needle=needle,
            hits=len(hits),
            max_hits=int(max_hits),
            searched_paths=searched_paths,
            debug=debug,
            reason=("no_runs_truly_exist" if not runs else "trace_or_metadata_absent"),
        )
        return hits

    def read_trace_snippet(self, trace_path: str, line_no: int, window: int = 2) -> List[str]:
        path = Path(str(trace_path))
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        center = max(1, int(line_no))
        start = max(1, center - int(window))
        end = min(len(lines), center + int(window))
        out = lines[start - 1:end]
        self._record("read_trace_snippet", trace_path=str(trace_path), line_no=int(line_no), window=int(window), lines=len(out))
        return out

    def smoke_check_candidate(self, candidate_dir: str) -> Dict:
        path = Path(str(candidate_dir))
        compiled = []
        for py in sorted(path.glob("*.py")):
            py_compile.compile(str(py), doraise=True)
            compiled.append(py.name)
        out = {"ok": True, "compiled": compiled}
        self._record("smoke_check_candidate", candidate_dir=str(candidate_dir), compiled=len(compiled))
        return out

    def validate_candidate(self, candidate_dir: str, parent_dir: str) -> Dict:
        result = validate_candidate_contract_alignment(Path(candidate_dir), parent_dir=Path(parent_dir))
        out = {"ok": True, "result": result}
        self._record("validate_candidate", candidate_dir=str(candidate_dir), parent_dir=str(parent_dir))
        return out

    def export_audit(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps({"events": self.audit_log}, ensure_ascii=False, indent=2), encoding="utf-8")
