from __future__ import annotations

import json
from pathlib import Path

from controller.harness_sandbox import load_harness_sandbox_profile
from proposer.live_benchmark_runner import LiveBenchmarkRunner


class _DummyLogger:
    def __init__(self, run_id: str):
        self._run_id = run_id

    def get_pending_run_summary(self):
        return {
            "run_id": self._run_id,
            "run_status": "error",
            "mission_success": False,
            "collision_count": 0,
            "near_miss_count": 0,
            "replan_count": 0,
        }

    def save_pending_run(self):
        return None


def test_prompt_composer_wrapper_accepts_kwargs_for_positional_only_fn(tmp_path: Path):
    harness_dir = tmp_path / "candidate_0001"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "prompt_composer.py").write_text(
        "def compose_prompt_context(stage, task_description, encoded_state, snapshot, spec):\n"
        "    return f'{stage}:{task_description}:{bool(spec)}'\n",
        encoding="utf-8",
    )
    (harness_dir / "spec.json").write_text(
        json.dumps(
            {
                "id": "candidate_0001",
                "sandbox": {
                    "prompt_composer": {"enabled": True, "module": "prompt_composer.py"}
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profile = load_harness_sandbox_profile((harness_dir / "spec.json").as_posix())
    fn = profile["prompt_composer"]["fn"]
    out = fn(
        stage="initial",
        task_description="do mission",
        encoded_state={},
        snapshot={},
        spec={"k": "v"},
    )
    assert out == "initial:do mission:True"


def test_capture_latest_saved_run_handles_missing_planning_trace(tmp_path: Path):
    output_root = tmp_path / "archive" / "candidates" / "candidate_0001"
    protocol = {"mode": "screening", "name": "candidate_screening_v1", "version": "v1", "pairs": [], "runs_per_scene": 2, "total_runs": 6}
    runner = LiveBenchmarkRunner(repo_root=tmp_path, output_root=output_root, harness_id="candidate_0001", evaluation_protocol=protocol)

    run_id = "run_0001"
    (runner._log_root / f"{run_id}_runtime_trace.jsonl").write_text("{}\n", encoding="utf-8")
    (runner._log_root / f"{run_id}_summary.json").write_text("{}\n", encoding="utf-8")
    (runner._log_root / f"{run_id}_debug.json").write_text("{}\n", encoding="utf-8")

    art = runner._capture_latest_saved_run(logger=_DummyLogger(run_id), scene_id="SCENE1", zone="zoneA")
    assert Path(art.runtime_trace_path).exists()
    assert Path(art.planning_trace_path).exists()
    assert Path(art.planning_trace_path).read_text(encoding="utf-8") == ""
