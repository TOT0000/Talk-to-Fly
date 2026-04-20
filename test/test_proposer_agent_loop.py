import inspect

import pytest

import proposer.propose_candidate as pc


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def request(self, prompt, model_name, stream=False):
        self.calls += 1
        if not self._responses:
            return '{"action":"final_proposal","proposal":{}}'
        return self._responses.pop(0)


class _FakeTools:
    def list_harnesses(self, kind="all"):
        return [{"harness_id": "baseline2", "kind": "baseline"}]

    def read_harness_spec(self, harness_id):
        return {"id": harness_id}

    def read_harness_code(self, harness_id, file_name):
        return ""

    def diff_harnesses(self, parent_harness, candidate_harness, file_name):
        return ""

    def list_runs(self, harness_id, limit=12):
        return []

    def read_run_metadata(self, run_dir):
        return {}

    def search_traces(self, harness_id, needle, max_hits=12):
        return []

    def read_trace_snippet(self, trace_path, line_no, window=2):
        return []


    def list_runtime_prompt_assets(self, harness_id):
        return []

    def read_runtime_prompt_asset(self, harness_id, asset_name=None, stage=None):
        return {}

    def diff_runtime_prompt_assets(self, harness_a, harness_b, stage="initial"):
        return {}
    def validate_candidate(self, candidate_dir, parent_dir):
        return {"ok": True}

    def smoke_check_candidate(self, candidate_dir):
        return {"ok": True}


def _final_action_json():
    return """{
      "action": "final_proposal",
      "proposal": {
        "parent_harness": "baseline2",
        "candidate_id": "candidate_9999",
        "one_sentence_hypothesis": "h",
        "weakness_being_addressed": "w",
        "expected_tradeoff": "t",
        "expected_runtime_effect": "r",
        "sandbox_modules_to_modify": ["trigger_logic.py"],
        "files_to_create_or_modify": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
        "changed_files": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
        "runtime_wiring_plan": {},
        "smoke_test_evidence_to_check": {},
        "proposer_note_text": "n",
        "implementation_contract": {"trigger_policy": {}, "state_encoder": {}, "prompt_builder": {}},
        "invariants": ["i"]
      }
    }"""


def test_agent_loop_requires_tool_and_run_evidence_before_final():
    llm = _FakeLLM(
        [
            _final_action_json(),
            '{"action":"tool_call","tool_name":"list_harnesses","tool_args":{"kind":"all"}}',
            _final_action_json(),
            '{"action":"tool_call","tool_name":"list_runs","tool_args":{"harness_id":"baseline2","limit":2}}',
            _final_action_json(),
        ]
    )
    out = pc._run_proposer_agent_loop(
        llm=llm,
        proposer_model="dummy",
        focus_text="focus",
        archive_summary={"baseline_list": ["baseline2"], "candidate_list": [], "pareto_list": []},
        tools=_FakeTools(),
        max_steps=8,
    )
    assert out["agent_meta"]["tool_steps"] >= 2
    assert out["agent_meta"]["run_evidence_tool_steps"] >= 1
    assert llm.calls >= 5


def test_agent_loop_has_max_step_limit():
    llm = _FakeLLM(['{"action":"unknown"}', '{"action":"unknown"}'])
    with pytest.raises(RuntimeError, match="exceeded max steps"):
        pc._run_proposer_agent_loop(
            llm=llm,
            proposer_model="dummy",
            focus_text="focus",
            archive_summary={},
            tools=_FakeTools(),
            max_steps=2,
        )


def test_agent_loop_marks_limited_evidence_when_run_hits_absent():
    llm = _FakeLLM(
        [
            '{"action":"tool_call","tool_name":"list_harnesses","tool_args":{"kind":"all"}}',
            '{"action":"tool_call","tool_name":"list_runs","tool_args":{"harness_id":"baseline2","limit":2}}',
            _final_action_json(),
        ]
    )
    out = pc._run_proposer_agent_loop(
        llm=llm,
        proposer_model="dummy",
        focus_text="focus",
        archive_summary={},
        tools=_FakeTools(),
        max_steps=5,
    )
    limited = out["proposal"]["smoke_test_evidence_to_check"]["evidence_limitations"]
    assert "run evidence limited" in limited


def test_propose_next_candidate_uses_agent_loop_not_single_shot_prompt():
    src = inspect.getsource(pc.propose_next_candidate)
    assert "_run_proposer_agent_loop(" in src
    assert "build_iteration_prompt(" not in src
