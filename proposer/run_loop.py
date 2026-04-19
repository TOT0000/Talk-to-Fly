from __future__ import annotations

from pathlib import Path

from proposer.evaluate_candidate import evaluate_candidate_live
from proposer.propose_candidate import propose_next_candidate, rebuild_index
from proposer.registry import HarnessRegistry


def run_once(
    repo_root: Path,
    evaluate_baselines: bool = False,
    focus_text: str = "Improve safety-aware replan timing while avoiding unnecessary detours.",
    allow_fallback_heuristic: bool = False,
) -> str:
    repo_root = Path(repo_root)
    archive_v2 = repo_root / "proposer_archive_v2"

    reg = HarnessRegistry(repo_root)

    if evaluate_baselines:
        for baseline in reg.list_baselines():
            evaluate_candidate_live(
                repo_root=repo_root,
                harness_id=baseline.harness_id,
                archive_root=archive_v2,
            )

    candidate_dir = propose_next_candidate(
        repo_root,
        focus_text=focus_text,
        allow_fallback_heuristic=allow_fallback_heuristic,
    )
    candidate_id = candidate_dir.name
    evaluate_candidate_live(
        repo_root=repo_root,
        harness_id=candidate_id,
        archive_root=archive_v2,
        evaluation_mode="screening",
    )
    rebuild_index(archive_v2)
    return candidate_id


if __name__ == "__main__":
    cid = run_once(Path(__file__).resolve().parents[1], evaluate_baselines=False)
    print(cid)
