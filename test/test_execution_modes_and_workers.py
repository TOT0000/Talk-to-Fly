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
    assert 'AGENT_HEARTBEAT_INTERVAL_SECONDS' in source
    assert '_maybe_run_agent_heartbeat' in source


def test_worker_scenario_paths_and_shared_speed_defined():
    layout_source = Path('controller/benchmark_layout.py').read_text(encoding='utf-8')
    scene_source = Path('controller/baseline_scenes.py').read_text(encoding='utf-8')
    assert 'WORKER_DEFAULT_SPEED_MPS = 0.4' in layout_source
    assert 'zoneA=patrol zoneB=bottleneck zoneC=cross_traffic speed=0.4' in scene_source
    assert 'worker_1' in scene_source and 'worker_2' in scene_source and 'worker_3' in scene_source


def test_ui_mode_switch_options_exist():
    source = Path('serving/webui/typefly.py').read_text(encoding='utf-8')
    assert 'MODE_TYPEFLY_ONESHOT' in source
    assert 'MODE_TYPEFLY_THRESHOLD_REPLAN' in source
    assert 'MODE_AGENT_HEARTBEAT_SOFT' in source
    assert 'MODE_AGENT_HEARTBEAT_HARDGATE' in source
