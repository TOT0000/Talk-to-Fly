import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.llm_wrapper import LLMWrapper
from controller.llm_controller import LLMController


def test_wrapper_empty_model_routes_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    route = LLMWrapper()._resolve_request_route("")
    assert route["provider"] == "openai"
    assert route["model"] == "gpt-4o"


def test_wrapper_nonempty_model_routes_lmstudio(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    route = LLMWrapper()._resolve_request_route("google/gemma-4-e4b")
    assert route["provider"] == "lmstudio"
    assert route["model"] == "google/gemma-4-e4b"


def test_wrapper_ignores_llm_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    assert LLMWrapper()._resolve_request_route("")["provider"] == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert LLMWrapper()._resolve_request_route("google/gemma-4-e4b")["provider"] == "lmstudio"


def test_manual_agent_mixed_routing_resolution(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
    c = object.__new__(LLMController)
    c._run_model_lock_active = False
    c._run_locked_models = None

    class DummyPlanner:
        def set_model(self, model_name):
            self.model_name = model_name

        def set_agent_model_names(self, heartbeat_model_name=None, evaluator_model_name=None):
            self.heartbeat_model_name = heartbeat_model_name
            self.evaluator_model_name = evaluator_model_name

    c.planner = DummyPlanner()

    selected = LLMController.set_manual_agent_models(c, "", "google/gemma-4-e2b")
    assert selected["planner_resolved_provider"] == "openai"
    assert selected["planner_resolved_model"] == "gpt-4o"
    assert selected["evaluator_resolved_provider"] == "lmstudio"
    assert selected["evaluator_resolved_model"] == "google/gemma-4-e2b"

    selected = LLMController.set_manual_agent_models(c, "google/gemma-4-e4b", "")
    assert selected["planner_resolved_provider"] == "lmstudio"
    assert selected["planner_resolved_model"] == "google/gemma-4-e4b"
    assert selected["evaluator_resolved_provider"] == "openai"
    assert selected["evaluator_resolved_model"] == "gpt-4o"
