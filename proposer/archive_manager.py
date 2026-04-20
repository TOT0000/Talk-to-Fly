from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def persist_evidence_bundle(archive_root: Path, *, candidate_id: str, payload: Dict) -> Path:
    out = Path(archive_root) / "evidence" / candidate_id / "evidence_bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
