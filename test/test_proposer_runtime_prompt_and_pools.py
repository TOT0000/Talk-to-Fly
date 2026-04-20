import json
from pathlib import Path

from proposer.archive_reader import summarize_archive_for_proposer
from proposer.registry import HarnessRegistry
from proposer.prompts import AGENT_SYSTEM_PROMPT, AGENT_TOOL_POLICY_PROMPT, AGENT_NEXT_ACTION_PROMPT
from proposer.live_benchmark_runner import LiveBenchmarkRunner


def _write_spec(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_candidate_0001_removed_from_default_archive_summary_pool(tmp_path):
    repo = tmp_path / "repo"
    archive = repo / "proposer_archive_v2"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "index.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"candidate_id": "baseline1", "kind": "baseline"},
                    {"candidate_id": "candidate_0001", "kind": "candidate"},
                    {"candidate_id": "candidate_0002", "kind": "candidate"},
                ]
            }
        ),
        encoding="utf-8",
    )

    out = summarize_archive_for_proposer(repo, archive)
    assert "candidate_0001" not in out["candidate_list"]
    assert "candidate_0002" in out["candidate_list"]
    assert "candidate_0001" in out["archival_candidate_list"]


def test_candidate_0001_removed_from_default_parent_candidate_pool(tmp_path):
    repo = tmp_path / "repo"
    _write_spec(repo / "harnesses" / "baseline1" / "spec.json", {"id": "baseline1", "kind": "baseline"})
    _write_spec(repo / "harnesses" / "candidates" / "candidate_0001" / "spec.json", {"id": "candidate_0001", "kind": "candidate", "parent": "baseline1"})
    _write_spec(repo / "harnesses" / "candidates" / "candidate_0002" / "spec.json", {"id": "candidate_0002", "kind": "candidate", "parent": "baseline1"})

    reg = HarnessRegistry(repo)
    default_pool = {x.harness_id for x in reg.list_candidates(include_excluded_for_proposer=False)}
    assert "candidate_0001" not in default_pool
    assert "candidate_0002" in default_pool
    manual_pool = {x.harness_id for x in reg.list_candidates(include_excluded_for_proposer=True)}
    assert "candidate_0001" in manual_pool


def test_runtime_prompts_updated_to_new_policy_text():
    assert "You are the proposer agent for a UAV mission-planning harness optimization system." in AGENT_SYSTEM_PROMPT
    assert "Evidence-first retrieval policy" in AGENT_SYSTEM_PROMPT
    assert "If runtime prompt assets or prompt text are available" in AGENT_TOOL_POLICY_PROMPT
    assert "If prompt modifications are proposed" in AGENT_NEXT_ACTION_PROMPT


def test_extract_prompt_source_evidence_from_planning_trace(tmp_path):
    trace = tmp_path / "planning_trace.jsonl"
    trace.write_text(
        json.dumps({"foo": "bar"}) + "\n" + json.dumps({"evaluate_prompt_source": {"selected_prompt_asset_path": "/tmp/a.txt", "rendered_prompt_hash_sha256": "abc"}}) + "\n",
        encoding="utf-8",
    )
    evidence = LiveBenchmarkRunner._extract_prompt_source_evidence(trace)
    assert evidence["selected_prompt_asset_path"] == "/tmp/a.txt"
    assert evidence["rendered_prompt_hash_sha256"] == "abc"
