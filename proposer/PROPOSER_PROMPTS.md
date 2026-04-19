# Proposer Prompts (Multi-round Tool-Using Agent)

This document mirrors runtime prompts used by:
- `proposer/prompts.py`
- `proposer/propose_candidate.py::_run_proposer_agent_loop`

## Prompt roles

1. `AGENT_SYSTEM_PROMPT`
   - Defines optimization objective, safety-first priorities, immutable system boundaries, and sandbox runtime module worldview.
   - Canonical sandbox modules: `state_features.py`, `trigger_logic.py`, `prompt_composer.py`, `archive_selector.py`, `validator_rules.py`.

2. `AGENT_TOOL_POLICY_PROMPT`
   - Defines tool-use order and policy.
   - Requires multi-round behavior (not single-shot).
   - Requires prioritizing run-evidence tools (`list_runs`, `search_traces`, `read_run_metadata`, `read_trace_snippet`) before code-only diagnosis.

3. `AGENT_NEXT_ACTION_PROMPT`
   - Per-step protocol: model returns either `tool_call` or `final_proposal` JSON.
   - Enforces at least one tool step before final proposal.

4. `FINAL_PROPOSAL_CONTRACT`
   - Structured proposal schema only after retrieval/diagnosis.
   - Includes runtime wiring and smoke-evidence requirements.

5. `SELF_REVIEW_CONTRACT` + `build_self_review_prompt(...)`
   - Runtime-first review contract for pass/revise decisions after guardrails/smoke outcomes.

## Agent step protocol (runtime)

Each loop step returns one JSON action:

- Tool step
```json
{"action":"tool_call","tool_name":"list_runs","tool_args":{"harness_id":"baseline2","limit":4}}
```

- Final proposal
```json
{"action":"final_proposal","proposal":{ "...FINAL_PROPOSAL_CONTRACT...": "..." }}
```

Runtime enforcements:
- No final proposal before at least one tool call.
- No final proposal before at least one run-evidence tool call.
- Agent loop has max step limit (fail-fast on overflow).
- If run evidence is weak/absent, proposal must mark `smoke_test_evidence_to_check.evidence_limitations`.

## Final proposal contract keys

Required proposal keys (14 total):
- `parent_harness`
- `candidate_id`
- `one_sentence_hypothesis`
- `weakness_being_addressed`
- `expected_tradeoff`
- `expected_runtime_effect`
- `sandbox_modules_to_modify`
- `files_to_create_or_modify`
- `changed_files`
- `runtime_wiring_plan`
- `smoke_test_evidence_to_check`
- `proposer_note_text`
- `implementation_contract`
- `invariants`

## Workflow integration

The multi-round proposer workflow is:
1. agent retrieval/diagnosis loop (`tool_call` steps),
2. final proposal output,
3. candidate code generation,
4. hard guardrails + smoke/import checks,
5. self-review/revise loop (bounded rounds),
6. finalize candidate only when checks pass.

## Sync policy

If this markdown and runtime constants diverge, runtime code is source of truth and this file must be updated in the same commit.
