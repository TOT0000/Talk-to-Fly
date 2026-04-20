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
- Baseline-first evidence when candidate evidence is absent; baseline+candidate evidence when available.
- Runtime prompt assets are valid evidence source (not only template metadata).
- Proposal contract includes `runtime_wiring_plan`, `smoke_test_evidence_to_check`, and `hypothesis_target_modules`.
- Runtime-first self-review defaults to `revise` when wiring mismatch / unsupported prompt-change claim occurs.

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

## Sync policy

If this markdown diverges from `proposer/prompts.py`, runtime code is the source of truth.
