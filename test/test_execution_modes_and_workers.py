from pathlib import Path


def test_modes_defined_in_controller_source():
    source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    for mode in [
        'typefly-oneshot',
        'typefly-threshold-replan',
        'agent-heartbeat-soft',
        'agent-heartbeat-hardgate',
    ]:
        assert mode in source


def test_agent_heartbeat_fixed_timer_only_in_source():
    source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    layout_source = Path('controller/benchmark_layout.py').read_text(encoding='utf-8')
    assert 'AGENT_HEARTBEAT_INTERVAL_SECONDS' in source
    assert 'AGENT_HEARTBEAT_INTERVAL_SECONDS = 5.0' in layout_source
    assert '_maybe_run_agent_heartbeat' in source


def test_worker_scenario_paths_and_shared_speed_defined():
    layout_source = Path('controller/benchmark_layout.py').read_text(encoding='utf-8')
    scene_source = Path('controller/baseline_scenes.py').read_text(encoding='utf-8')
    assert 'WORKER_DEFAULT_SPEED_MPS = 0.4' in layout_source
    assert 'zoneA=patrol zoneB=bottleneck zoneC=cross_traffic speed=0.4' in scene_source
    assert 'worker_1' in scene_source and 'worker_2' in scene_source and 'worker_3' in scene_source


def test_fixed_w13_manual_w2_scene_exists_with_expected_positions():
    scene_source = Path('controller/baseline_scenes.py').read_text(encoding='utf-8')
    assert 'SCENE_FIXED_W13_MANUAL_W2' in scene_source
    assert 'StaticObstacle("worker_1", 3.0, 4.0' in scene_source
    assert 'StaticObstacle("worker_3", 7.0, 3.7' in scene_source
    assert 'StaticObstacle("worker_2", 8.5, 7.8' in scene_source


def test_ui_mode_switch_options_exist():
    source = Path('serving/webui/typefly.py').read_text(encoding='utf-8')
    assert 'MODE_TYPEFLY_ONESHOT' in source
    assert 'MODE_TYPEFLY_THRESHOLD_REPLAN' in source
    assert 'MODE_AGENT_HEARTBEAT_SOFT' in source
    assert 'MODE_AGENT_HEARTBEAT_HARDGATE' in source


def test_gc_no_longer_replans_on_no_progress_tiny_residual_or_max_iterations():
    source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    assert 'no_progress_fail_safe' not in source
    assert 'tiny_residual_vector' not in source
    assert 'return summary, should_request_replan' in source


def test_llm_wrapper_provider_model_mapping_and_logs_exist():
    source = Path('controller/llm_wrapper.py').read_text(encoding='utf-8')
    assert 'OPENAI_DEFAULT_MODEL' in source
    assert 'GEMINI_DEFAULT_MODEL' in source
    assert 'LMSTUDIO_DEFAULT_MODEL' in source
    assert 'if self.provider == "gemini" and selected_model.lower().startswith("gpt-")' in source
    assert 'elif self.provider == "openai" and selected_model.lower().startswith("gemini-")' in source
    assert '[LLM] provider=' in source
    assert '[LLM] base_url=' in source
    assert '[LLM] model_name=' in source
    assert '[LLM] key_source=' in source


def test_four_experiment_pipelines_do_not_depend_on_removed_graph_runner():
    registry_source = Path('controller/pipeline_registry.py').read_text(encoding='utf-8')
    controller_source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    planner_source = Path('controller/llm_planner.py').read_text(encoding='utf-8')
    graph_token = 'lang' + 'graph'

    for pipeline_id in ['baseline1', 'baseline2', 'agent', 'baseline3']:
        assert f'id="{pipeline_id}"' in registry_source
    assert 'base_mode="agent-heartbeat-soft"' in registry_source
    assert 'base_mode="typefly-threshold-replan"' in registry_source
    assert 'plan_agent_heartbeat' in planner_source
    assert 'evaluate_agent_replan_record' in planner_source
    assert graph_token not in controller_source.lower()
    assert graph_token not in planner_source.lower()


def test_should_trigger_auto_replan_is_bool_only_without_heartbeat_status_reference():
    source = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    start = source.index('def _should_trigger_auto_replan')
    end = source.index('def stop_controller')
    fn_source = source[start:end]
    assert 'hb_status' not in fn_source
    assert 'return "none"' not in fn_source
    assert 'return "request_started"' not in fn_source
    assert 'return True,' not in fn_source
    assert 'return False,' not in fn_source
