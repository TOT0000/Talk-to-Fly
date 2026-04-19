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
    audit_path = candidate / "proposer_tool_audit.json"
    tools.export_audit(audit_path)
    exported = json.loads(audit_path.read_text(encoding="utf-8"))
    assert any(e.get("tool") == "search_traces" for e in exported.get("events", []))


def test_list_runs_baseline_falls_back_to_legacy_manual_archive(tmp_path):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline3"
    archive = repo / "proposer_archive_v2"
    _write_harness(baseline, "baseline3", "baseline")

    baseline_archive = archive / "baselines" / "baseline3"
    (baseline_archive / "traces").mkdir(parents=True, exist_ok=True)
    (baseline_archive / "traces" / "trace_pointers.json").write_text(
        json.dumps({"source": "proposer_archive/manual_runs/runs"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (archive / "index.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "candidate_id": "baseline3",
                        "kind": "baseline",
                        "trace_locations": {"runs_dir": (baseline_archive / "runs").as_posix()},
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    legacy_run = repo / "proposer_archive" / "manual_runs" / "runs" / "run_legacy_1"
    legacy_run.mkdir(parents=True, exist_ok=True)
    (legacy_run / "run_legacy_1_planning_trace.jsonl").write_text(
        '{"selected_baseline_id":"baseline3","event":"near_miss"}\n',
        encoding="utf-8",
    )
    (legacy_run / "run_legacy_1_runtime_trace.jsonl").write_text(
        '{"risk":"near_miss"}\n',
        encoding="utf-8",
    )

    tools = ProposerToolbox(repo_root=repo, archive_root=archive)
    runs = tools.list_runs("baseline3", limit=5)
    assert len(runs) == 1
    assert runs[0]["source_type"] in {"trace_pointer_source", "legacy_manual_runs_default"}
    hits = tools.search_traces("baseline3", "near_miss", max_hits=2)
    assert hits


def test_list_runs_candidate_prefers_archive_v2_runs(tmp_path):
    repo = tmp_path / "repo"
    candidate = repo / "harnesses" / "candidates" / "candidate_0002"
    archive = repo / "proposer_archive_v2"
    _write_harness(candidate, "candidate_0002", "candidate", parent="baseline3")

    run_dir = archive / "candidates" / "candidate_0002" / "runs" / "run_0009"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"harness_id": "candidate_0002", "scene": "SCENE1", "task": "zoneA", "status": "ok"}),
        encoding="utf-8",
    )
    (run_dir / "runtime_trace.jsonl").write_text('{"event":"ok"}\n', encoding="utf-8")
    (archive / "index.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "candidate_id": "candidate_0002",
                        "kind": "candidate",
                        "trace_locations": {"runs_dir": (archive / "candidates" / "candidate_0002" / "runs").as_posix()},
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tools = ProposerToolbox(repo_root=repo, archive_root=archive)
    runs = tools.list_runs("candidate_0002", limit=5)
    assert len(runs) == 1
    assert runs[0]["source_type"] in {"archive_index_runs_dir", "archive_harness_runs"}
    assert runs[0]["scene"] == "SCENE1"


def test_absent_evidence_is_reported_honestly(tmp_path):
    repo = tmp_path / "repo"
    baseline = repo / "harnesses" / "baseline1"
    archive = repo / "proposer_archive_v2"
    _write_harness(baseline, "baseline1", "baseline")
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "index.json").write_text(
        json.dumps({"entries": [{"candidate_id": "baseline1", "trace_locations": {"runs_dir": str(archive / "baselines" / "baseline1" / "runs")}}]}),
        encoding="utf-8",
    )
    tools = ProposerToolbox(repo_root=repo, archive_root=archive)
    assert tools.list_runs("baseline1") == []
    assert tools.search_traces("baseline1", "near_miss", max_hits=3) == []
    assert any(
        e.get("tool") == "search_traces" and e.get("reason") == "no_runs_truly_exist"
        for e in tools.audit_log
    )
