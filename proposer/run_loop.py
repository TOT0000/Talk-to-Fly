from __future__ import annotations

from pathlib import Path

from proposer.evaluate_candidate import evaluate_candidate_offline
from proposer.propose_candidate import propose_next_candidate, rebuild_index
from proposer.registry import HarnessRegistry


def run_once(repo_root: Path) -> str:
    repo_root = Path(repo_root)
    archive_v2 = repo_root / "proposer_archive_v2"
    debug_jsonl = repo_root / "proposer_archive/manual_runs/task_runs_debug.jsonl"

    reg = HarnessRegistry(repo_root)

    # Ensure baselines are materialized in archive v2 first.
    for baseline in reg.list_baselines():
        evaluate_candidate_offline(
            repo_root=repo_root,
            harness_id=baseline.harness_id,
            archive_root=archive_v2,
            manual_debug_jsonl=debug_jsonl,
        )

    candidate_dir = propose_next_candidate(repo_root)
    candidate_id = candidate_dir.name
    evaluate_candidate_offline(
        repo_root=repo_root,
        harness_id=candidate_id,
        archive_root=archive_v2,
        manual_debug_jsonl=debug_jsonl,
    )
    rebuild_index(archive_v2)
    return candidate_id


if __name__ == "__main__":
    cid = run_once(Path(__file__).resolve().parents[1])
    print(cid)
