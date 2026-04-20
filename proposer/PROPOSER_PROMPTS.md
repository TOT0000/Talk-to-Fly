# Proposer Runtime Prompts (Source-of-truth mirror)

This file mirrors the runtime constants in `proposer/prompts.py`.

## Active runtime prompt blocks

- `AGENT_SYSTEM_PROMPT`
- `AGENT_TOOL_POLICY_PROMPT`
- `AGENT_NEXT_ACTION_PROMPT`
- `FINAL_PROPOSAL_CONTRACT`
- `SELF_REVIEW_CONTRACT`

## Key runtime behavior enforced

- Evidence-first multi-round tool loop (tool calls before final proposal).
- No pre-selected optimization axis; trigger/state/prompt are all evidence-driven options.
- Primary hypothesis modules must be runtime-loaded modules or runtime-used prompt assets.
- Prompt-change claims must align to actual evaluation prompt source.
- Trigger-change claims must align to runtime-loaded trigger module config keys.
- Runtime-first self-review defaults to revise on wiring/prompt-source/config-key misalignment.

## Current contract keys (15)

- `parent_harness`
- `candidate_id`
- `one_sentence_hypothesis`
- `weakness_being_addressed`
- `expected_tradeoff`
- `expected_runtime_effect`
- `hypothesis_target_modules`
- `sandbox_modules_to_modify`
- `files_to_create_or_modify`
- `changed_files`
- `runtime_wiring_plan`
- `smoke_test_evidence_to_check`
- `proposer_note_text`
- `implementation_contract`
- `invariants`

Keyword parity: `runtime_wiring_plan`, `smoke_test_evidence_to_check`, `hypothesis_target_modules`, `primary_runtime_entrypoints`, `runtime_prompt_source_plan`, `config_key_alignment_plan`, `evaluate_prompt_source_evidence`.

## Sync policy

If this markdown diverges from `proposer/prompts.py`, runtime code is the source of truth.
