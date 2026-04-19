import json

from proposer import prompts
from proposer.propose_candidate import _build_file_generation_prompt


def test_system_prompt_uses_sandbox_worldview_and_runtime_alignment_language():
    text = prompts.SYSTEM_PROMPT
    for required in [
        "state_features.py",
        "trigger_logic.py",
        "prompt_composer.py",
        "archive_selector.py",
        "validator_rules.py",
        "runtime wiring alignment",
    ]:
        assert required in text
    assert "State Encoder" not in text
    assert "Trigger Policy" not in text
    assert "Prompt Builder" not in text


def test_output_contract_contains_runtime_wiring_and_smoke_evidence_fields():
    text = prompts.OUTPUT_CONTRACT
    assert "runtime_wiring_plan" in text
    assert "smoke_test_evidence_to_check" in text
    assert "trigger_logic_evidence" in text
    assert "state_features_evidence" in text
    assert "prompt_composer_evidence" in text


def test_self_review_contract_is_runtime_first():
    text = prompts.SELF_REVIEW_CONTRACT
    assert "runtime can load changed sandbox modules" in text
    assert "spec/manifest/loader alignment" in text
    assert "=> revise" in text


def test_file_generation_prompt_is_sandbox_first_and_blocks_ambiguous_mixing():
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
    assert "Primary editable targets are sandbox modules" in generated
    assert "Legacy files (state_encoder.py, trigger_policy.py, prompt_builder.py) are compatibility wrappers" in generated
    assert "forbidden example: editing trigger_logic.py while spec/loader still routes to trigger_policy.py" in generated


def test_proposer_prompts_doc_syncs_with_runtime_contract_keywords():
    doc = open("proposer/PROPOSER_PROMPTS.md", "r", encoding="utf-8").read()
    runtime = "\n".join(
        [prompts.AGENT_SYSTEM_PROMPT, prompts.AGENT_TOOL_POLICY_PROMPT, prompts.AGENT_NEXT_ACTION_PROMPT, prompts.OUTPUT_CONTRACT, prompts.SELF_REVIEW_CONTRACT]
    )
    shared_keywords = [
        "runtime_wiring_plan",
        "smoke_test_evidence_to_check",
        "state_features.py",
        "trigger_logic.py",
        "prompt_composer.py",
        "archive_selector.py",
        "validator_rules.py",
    ]
    for keyword in shared_keywords:
        assert keyword in doc
        assert keyword in runtime

    # Ensure doc references the exact 14-key contract change.
    assert "14" in doc

    # Basic parse sanity: output contract still describes JSON object.
    assert "JSON object" in prompts.OUTPUT_CONTRACT
    json.dumps({"ok": True})
