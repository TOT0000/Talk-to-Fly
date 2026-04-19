# Proposer Prompts (Runtime Wiring Alignment Edition)

This document mirrors the **runtime-used** proposer prompts in `proposer/prompts.py`.

## Runtime-used prompt constants/templates

- `SYSTEM_PROMPT`
- `ITERATION_TASK_TEMPLATE`
- `OUTPUT_CONTRACT`
- `SELF_REVIEW_CONTRACT`
- `build_iteration_prompt(...)`
- `build_self_review_prompt(...)`

## 1) SYSTEM_PROMPT (design summary)

Key changes now enforced:

- Uses **sandbox module worldview only** as canonical runtime language:
  - `state_features.py`
  - `trigger_logic.py`
  - `prompt_composer.py`
  - `archive_selector.py`
  - `validator_rules.py`
- Explicitly marks legacy files (`state_encoder.py`, `trigger_policy.py`, `prompt_builder.py`) as compatibility wrapper / metadata mirror, not primary runtime-effect targets.
- Elevates **runtime wiring alignment** to top hard requirement:
  - the changed module must be what runtime actually executes,
  - spec/manifest/loader must be aligned,
  - smoke evidence must support the claim.
- Requires explicit honesty when archive evidence is limited.

## 2) ITERATION_TASK_TEMPLATE (design summary)

The task template now requires:

- sandbox-first mutation planning,
- no legacy/sandbox mixed ambiguous routing,
- required `runtime_wiring_plan` and `smoke_test_evidence_to_check`,
- explicit limited-evidence disclosure when evidence is thin.

## 3) OUTPUT_CONTRACT (runtime schema)

Runtime prompt requires exactly 14 keys:

1. `parent_harness`
2. `candidate_id`
3. `one_sentence_hypothesis`
4. `weakness_being_addressed`
5. `expected_tradeoff`
6. `expected_runtime_effect`
7. `sandbox_modules_to_modify`
8. `files_to_create_or_modify`
9. `changed_files`
10. `runtime_wiring_plan`
11. `smoke_test_evidence_to_check`
12. `proposer_note_text`
13. `implementation_contract`
14. `invariants`

Additional required structure:

- `runtime_wiring_plan` includes:
  - `sandbox_modules_changed`
  - `runtime_load_path_or_entrypoint`
  - `spec_manifest_loader_alignment`
  - `legacy_sync_plan`
- `smoke_test_evidence_to_check` includes:
  - `trigger_logic_evidence`
  - `state_features_evidence`
  - `prompt_composer_evidence`
  - `evidence_limitations`

## 4) SELF_REVIEW_CONTRACT (runtime-first)

Self-review priority order is now explicit:

1. runtime will load changed sandbox modules,
2. changed files truly include runtime-effect edits,
3. spec/manifest/loader alignment,
4. smoke evidence supports runtime claims,
5. any wiring ambiguity => `revise`.

Narrative wording/style nits are explicitly de-prioritized.

## 5) File-generation prompt behavior (runtime-used)

`proposer/propose_candidate.py::_build_file_generation_prompt(...)` now states:

- sandbox modules are primary editable targets,
- legacy modules are compatibility-only,
- forbidden ambiguous case: changed sandbox module but loader/spec still points to legacy route,
- if legacy sync is needed, it must be explicit in spec/manifest/runtime wiring plan.

## 6) Sync policy

If this document and runtime constants diverge, runtime constants in `proposer/prompts.py` and `proposer/propose_candidate.py` are source of truth and this document must be updated in the same change.
