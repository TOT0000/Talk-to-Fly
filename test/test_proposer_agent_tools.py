import json
from pathlib import Path

from proposer.agent_tools import ProposerToolbox


def _write_harness(harness_dir: Path, harness_id: str, kind: str, parent: str | None = None):
    harness_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": harness_id,
        "kind": kind,
        "parent": parent,
        "sandbox": {
            "state_features": {"enabled": True, "module": "state_features.py"},
            "trigger_logic": {"enabled": True, "module": "trigger_logic.py"},
            "prompt_composer": {"enabled": True, "module": "prompt_composer.py"},
        },
    }
    (harness_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (harness_dir / "state_features.py").write_text("def encode_state_features(snapshot, spec):\n    return {'risk': snapshot.get('predicted_collision_probability')}\n", encoding="utf-8")
    (harness_dir / "trigger_logic.py").write_text("def should_trigger_replan(state, memory, spec):\n    return (bool(state.get('predicted_collision_probability')), 'risk')\n", encoding="utf-8")
    (harness_dir / "prompt_composer.py").write_text("def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n    return f\"risk={encoded_state.get('risk')}\"\n", encoding="utf-8")


def test_toolbox_list_read_search_and_snippet(tmp_path):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    candidate = repo / "harnesses" / "candidates" / "candidate_0001"
    archive = repo / "proposer_archive_v2"
    _write_harness(baseline, "baseline3", "baseline")
    _write_harness(candidate, "candidate_0001", "candidate", parent="baseline3")

    run_dir = archive / "candidates" / "candidate_0001" / "runs" / "run_0001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps({"mission_success": True}), encoding="utf-8")
    (run_dir / "runtime_trace.jsonl").write_text('{"risk":"near_miss"}\n{"risk":"ok"}\n', encoding="utf-8")
    (run_dir / "planning_trace.jsonl").write_text('{"event":"replan"}\n', encoding="utf-8")
    (archive / "index.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "candidate_id": "candidate_0001",
                        "trace_locations": {"runs_dir": (archive / "candidates" / "candidate_0001" / "runs").as_posix()},
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tools = ProposerToolbox(repo_root=repo, archive_root=archive)
    harnesses = tools.list_harnesses()
    assert {h["harness_id"] for h in harnesses} == {"baseline3", "candidate_0001"}
    assert tools.read_harness_spec("candidate_0001")["parent"] == "baseline3"
    assert "encode_state_features" in tools.read_harness_code("baseline3", "state_features.py")
    assert tools.list_runs("candidate_0001")[0]["run_id"] == "run_0001"
    assert tools.read_run_metadata(str(run_dir))["mission_success"] is True
    hits = tools.search_traces("candidate_0001", "near_miss", max_hits=1)
    assert len(hits) == 1
    snippet = tools.read_trace_snippet(hits[0]["trace"], hits[0]["line_no"], window=1)
    assert snippet
    smoke = tools.smoke_check_candidate(str(candidate))
    assert smoke["ok"] is True
