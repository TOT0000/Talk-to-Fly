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

    def list_harnesses(self, kind: str = "all") -> List[Dict]:
        entries = []
        if kind in {"all", "baseline"}:
            entries.extend(self.registry.list_baselines())
        if kind in {"all", "candidate"}:
            entries.extend(self.registry.list_candidates())
        return [
            {
                "harness_id": e.harness_id,
                "kind": e.kind,
                "parent": e.spec.get("parent"),
                "path": e.dir_path.as_posix(),
            }
            for e in entries
        ]

    def read_harness_spec(self, harness_id: str) -> Dict:
        entry = self.registry.get(harness_id)
        return dict(entry.spec)

    def read_harness_code(self, harness_id: str, file_name: str) -> str:
        entry = self.registry.get(harness_id)
        base = Path(str(file_name)).name
        if base not in ALLOWED_MUTATION_FILES:
            raise ValueError(f"file not allowed: {file_name}")
        path = entry.dir_path / base
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def diff_harnesses(self, parent_harness: str, candidate_harness: str, file_name: str) -> str:
        base = Path(str(file_name)).name
        left = self.read_harness_code(parent_harness, base).splitlines(keepends=True)
        right = self.read_harness_code(candidate_harness, base).splitlines(keepends=True)
        diff = difflib.unified_diff(left, right, fromfile=f"{parent_harness}/{base}", tofile=f"{candidate_harness}/{base}")
        return "".join(diff)

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
        return out

    def read_run_metadata(self, run_dir: str) -> Dict:
        path = Path(str(run_dir)) / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

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
                            return hits
        return hits

    def read_trace_snippet(self, trace_path: str, line_no: int, window: int = 2) -> List[str]:
        path = Path(str(trace_path))
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        center = max(1, int(line_no))
        start = max(1, center - int(window))
        end = min(len(lines), center + int(window))
        return lines[start - 1:end]

    def smoke_check_candidate(self, candidate_dir: str) -> Dict:
        path = Path(str(candidate_dir))
        compiled = []
        for py in sorted(path.glob("*.py")):
            py_compile.compile(str(py), doraise=True)
            compiled.append(py.name)
        return {"ok": True, "compiled": compiled}

    def validate_candidate(self, candidate_dir: str, parent_dir: str) -> Dict:
        result = validate_candidate_contract_alignment(Path(candidate_dir), parent_dir=Path(parent_dir))
        return {"ok": True, "result": result}
