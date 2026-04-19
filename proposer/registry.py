from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ALLOWED_MUTATION_FILES = {
    "spec.json",
    "state_encoder.py",
    "trigger_policy.py",
    "prompt_builder.py",
    "README.md",
    "proposer_note.txt",
}


@dataclass(frozen=True)
class HarnessEntry:
    harness_id: str
    kind: str
    dir_path: Path
    spec: Dict


class HarnessRegistry:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.harnesses_dir = self.repo_root / "harnesses"
        self.candidates_dir = self.harnesses_dir / "candidates"

    def _read_spec(self, harness_dir: Path) -> Dict:
        with (harness_dir / "spec.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_baselines(self) -> List[HarnessEntry]:
        out: List[HarnessEntry] = []
        for path in sorted(self.harnesses_dir.glob("baseline*/spec.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            out.append(HarnessEntry(harness_id=spec["id"], kind="baseline", dir_path=path.parent, spec=spec))
        return out

    def list_candidates(self) -> List[HarnessEntry]:
        out: List[HarnessEntry] = []
        if not self.candidates_dir.exists():
            return out
        for path in sorted(self.candidates_dir.glob("candidate_*/spec.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            out.append(HarnessEntry(harness_id=spec["id"], kind="candidate", dir_path=path.parent, spec=spec))
        return out

    def get(self, harness_id: str) -> HarnessEntry:
        for item in (self.list_baselines() + self.list_candidates()):
            if item.harness_id == harness_id:
                return item
        raise KeyError(f"Unknown harness_id: {harness_id}")


def validate_candidate_boundary(candidate_dir: Path) -> None:
    candidate_dir = Path(candidate_dir)
    for path in candidate_dir.iterdir():
        if path.is_file() and path.name not in ALLOWED_MUTATION_FILES:
            raise ValueError(
                f"Candidate boundary violation: {path.name} is not allowed. "
                f"Allowed files: {sorted(ALLOWED_MUTATION_FILES)}"
            )
