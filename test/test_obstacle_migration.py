from pathlib import Path

ACTIVE_TEXT_PATHS = [
    Path("controller"),
    Path("serving/webui"),
]

PROMPT_PATHS = [Path("controller/assets"), Path("controller/prompts")]

FORBIDDEN = ["worker_positions", "min_uav_worker_distance_m", "dominant_risky_worker", "per_worker_collision_probabilities"]


def _iter_text_files(base: Path):
    if not base.exists():
        return
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".txt", ".md", ".json"}:
            yield p


def test_active_prompts_use_obstacle_terms_only():
    banned = ["worker", "workers", "worker_positions", "worker_1"]
    for base in PROMPT_PATHS:
        for path in _iter_text_files(base):
            text = path.read_text(encoding="utf-8").lower()
            for token in banned:
                assert token not in text, f"{token} found in {path}"


def test_no_active_worker_naming_in_runtime_paths_except_legacy_helper():
    for base in ACTIVE_TEXT_PATHS:
        for path in _iter_text_files(base):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                assert token not in text, f"{token} found in {path}"


def test_heartbeat_prompt_template_uses_obstacle_positions():
    p = Path("controller/assets/tello/agent_heartbeat_soft_prompt.txt")
    text = p.read_text(encoding="utf-8")
    assert "{obstacle_positions}" in text
    assert "{worker_positions}" not in text


def test_runtime_snapshot_and_summary_schema_migrated_to_obstacles():
    src = Path("controller/llm_controller.py").read_text(encoding="utf-8")
    assert '"obstacles"' in src
    assert 'obstacle_positions_summary' in src
    assert '"min_uav_obstacle_distance_m"' in src
    assert '"workers"' not in src


def test_safety_context_schema_uses_obstacles():
    src = Path("controller/safety_context.py").read_text(encoding="utf-8") + "\n" + Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert "per_obstacle_collision_probabilities" in src
    assert "dominant_risky_obstacle" in src or "dominant_threat_id" in src
    assert "per_worker_collision_probabilities" not in src


def test_task_run_logger_schema_uses_obstacles():
    src = Path("controller/task_run_logger.py").read_text(encoding="utf-8")
    assert "min_uav_obstacle_distance_m" in src
    assert "obstacles" in src
    assert "per_obstacle_predicted_collision_probability" in src
    assert "min_uav_worker_distance_m" not in src
