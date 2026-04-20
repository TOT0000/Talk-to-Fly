import json

from proposer import prompts
from proposer.propose_candidate import _build_file_generation_prompt, _prepare_proposal_contract


def test_system_prompt_contains_new_alignment_constraints_and_no_axis_bias():
    text = prompts.SYSTEM_PROMPT
    required = [
        "Do not assume any one axis is preferred in advance.",
        "Primary hypothesis modules must be actual runtime-loaded modules or runtime-used prompt assets.",
        "Do not claim a prompt change if only metadata changes but the rendered runtime prompt text remains effectively unchanged.",
        "Do not claim a trigger change if the proposal uses configuration keys that are not actually read by the runtime-loaded trigger module.",
        "how the claimed change will be verified in actual evaluation runtime",
    ]
    for phrase in required:
        assert phrase in text


def test_output_contract_contains_expanded_runtime_wiring_and_prompt_source_fields():
    text = prompts.OUTPUT_CONTRACT
    for field in [
        "primary_runtime_entrypoints",
        "runtime_prompt_source_plan",
        "config_key_alignment_plan",
        "evaluate_prompt_source_evidence",
    ]:
        assert field in text


def test_self_review_contract_includes_new_default_to_revise_rules():
    text = prompts.SELF_REVIEW_CONTRACT
    for phrase in [
        "the proposal names a legacy wrapper as a primary hypothesis module but runtime does not actually load it",
        "the proposal claims a prompt change but evaluation would still use the old baseline prompt source",
        "the proposal claims a trigger change but uses config keys not read by the runtime-loaded trigger module",
    ]:
        assert phrase in text


def test_file_generation_prompt_includes_prompt_and_trigger_alignment_rules():
    generated = _build_file_generation_prompt(
        parent_harness_id="baseline3",
        parent_spec={"id": "baseline3", "trigger_policy": {"type": "threshold"}},
        parent_file_name="trigger_logic.py",
        parent_file_content="def should_trigger(*args, **kwargs):\n    return False\n",
        proposal={
            "candidate_id": "candidate_9999",
            "runtime_wiring_plan": {"legacy_sync_plan": "none"},
        },
    )
    assert "ensure the changed prompt is the one evaluation runtime will actually render and send" in generated
    assert "ensure the proposed config keys are actually read by the runtime-loaded trigger module" in generated


def test_prepare_contract_accepts_minimal_fallback_with_hypothesis_modules():
    payload = {
        "parent_harness": "baseline3",
        "candidate_id": "",
        "one_sentence_hypothesis": "fallback",
        "weakness_being_addressed": "llm unavailable",
        "expected_tradeoff": "minimal change",
        "expected_runtime_effect": "stable runtime",
        "hypothesis_target_modules": ["trigger_logic.py"],
        "files_to_create_or_modify": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
        "proposer_note_text": "fallback",
        "sandbox_modules_to_modify": ["trigger_logic.py"],
        "changed_files": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
        "implementation_contract": {"trigger_policy": {}, "state_encoder": {}, "prompt_builder": {}},
        "runtime_wiring_plan": {
            "sandbox_modules_changed": ["trigger_logic.py"],
            "runtime_load_path_or_entrypoint": "controller.harness_sandbox runtime sandbox loader",
            "spec_manifest_loader_alignment": "aligned",
            "legacy_sync_plan": "none",
            "primary_runtime_entrypoints": ["controller.harness_sandbox.load_harness_sandbox_profile"],
            "runtime_prompt_source_plan": "inherited",
            "config_key_alignment_plan": "no new keys",
        },
        "smoke_test_evidence_to_check": {
            "trigger_logic_evidence": "changed",
            "state_features_evidence": "unchanged",
            "prompt_composer_evidence": "unchanged",
            "evidence_limitations": "fallback",
            "evaluate_prompt_source_evidence": "inherited",
        },
        "invariants": ["contract and runtime metadata align"],
    }
    out = _prepare_proposal_contract(payload)
    assert "trigger_logic.py" in out["hypothesis_target_modules"]


def test_proposer_prompts_doc_syncs_with_runtime_contract_keywords():
    doc = open("proposer/PROPOSER_PROMPTS.md", "r", encoding="utf-8").read()
    runtime = "\n".join(
        [
            prompts.AGENT_SYSTEM_PROMPT,
            prompts.AGENT_TOOL_POLICY_PROMPT,
            prompts.AGENT_NEXT_ACTION_PROMPT,
            prompts.OUTPUT_CONTRACT,
            prompts.SELF_REVIEW_CONTRACT,
        ]
    )
    shared_keywords = [
        "runtime_wiring_plan",
        "smoke_test_evidence_to_check",
        "hypothesis_target_modules",
        "primary_runtime_entrypoints",
        "runtime_prompt_source_plan",
        "config_key_alignment_plan",
        "evaluate_prompt_source_evidence",
    ]
    for keyword in shared_keywords:
        assert keyword in doc
        assert keyword in runtime

    assert "15" in doc
    assert "JSON object" in prompts.OUTPUT_CONTRACT
    json.dumps({"ok": True})
