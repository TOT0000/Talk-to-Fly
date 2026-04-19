from __future__ import annotations

import difflib
import json
import py_compile
from pathlib import Path
from typing import Dict, List

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
        idx = read_archive_index(self.archive_root)
        out: List[Dict] = []
        for entry in list(idx.get("entries", [])):
            if str(entry.get("candidate_id")) != str(harness_id):
                continue
            runs_dir = Path(str(((entry.get("trace_locations") or {}).get("runs_dir") or "")))
            if not runs_dir.exists():
                continue
            for run_dir in sorted(runs_dir.glob("run_*"))[: max(1, int(limit))]:
                out.append({"run_id": run_dir.name, "run_dir": run_dir.as_posix()})
        self._record("list_runs", harness_id=harness_id, count=len(out), limit=int(limit))
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
        for run in self.list_runs(harness_id, limit=64):
            run_dir = Path(run["run_dir"])
            for trace_name in ["runtime_trace.jsonl", "planning_trace.jsonl"]:
                path = run_dir / trace_name
                if not path.exists():
                    continue
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if needle.lower() in line.lower():
                        hits.append(
                            {
                                "run_id": run["run_id"],
                                "trace": path.as_posix(),
                                "line_no": i,
                                "line": line[:600],
                            }
                        )
                        if len(hits) >= int(max_hits):
                            self._record("search_traces", harness_id=harness_id, needle=needle, hits=len(hits), max_hits=int(max_hits))
                            return hits
        self._record("search_traces", harness_id=harness_id, needle=needle, hits=len(hits), max_hits=int(max_hits))
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
