# Restricted Meta-Harness MVP (Talk-to-Fly)

This proposer loop limits mutation boundary to harness-level modules only, with **sandbox runtime-effect modules as primary targets**:

- Primary runtime-effect modules:
  - `state_features.py`
  - `trigger_logic.py`
  - `prompt_composer.py`
  - `archive_selector.py`
  - `validator_rules.py`
- Compatibility / metadata modules:
  - `state_encoder.py`
  - `trigger_policy.py`
  - `prompt_builder.py`
- Required contract/meta files:
  - `spec.json`

## Two-stage evaluation protocol

- **Baseline formal protocol (unchanged)**:
  - `SCENE1 -> zoneA`, 8 runs
  - `SCENE2 -> zoneB`, 8 runs
  - `SCENE3 -> zoneC`, 8 runs
  - total **24** runs
- **Candidate screening protocol (default for candidates)**:
  - `SCENE1 -> zoneA`, 2 runs
  - `SCENE2 -> zoneB`, 2 runs
  - `SCENE3 -> zoneC`, 2 runs
  - total **6** runs

Screening is a low-cost triage stage only. Formal 24-run candidate evaluation is the only stage used for direct baseline-level comparison.

## Agent-driven proposer

`proposer/propose_candidate.py` now drives proposal creation with LLM calls via `controller.llm_wrapper.LLMWrapper`.

- It uses controlled coding-agent tools (`proposer/agent_tools.py`) to list/read/diff harnesses and inspect runs/traces.
- It now uses a **multi-round agent loop** (tool step -> observation -> next step) instead of single-shot proposal prompting.
- It uses `AGENT_SYSTEM_PROMPT` + `AGENT_TOOL_POLICY_PROMPT` + `AGENT_NEXT_ACTION_PROMPT` + `FINAL_PROPOSAL_CONTRACT` in `proposer/prompts.py`.
- It proposes exactly one candidate each invocation.
- It keeps candidate edits bounded by `proposer/registry.py` allowed files.
- It now enforces contract/spec/code consistency via `proposer/consistency.py` before a candidate is accepted.
- Proposer model defaults to `gpt-4.1` and is configurable via `TYPEFLY_PROPOSER_MODEL`.
- If `TYPEFLY_PROPOSER_MODEL` is a GPT model and `OPENAI_API_KEY` is missing, proposer fails with a clear error (unless explicit fallback is enabled).

### Multi-round agent step loop (runtime)

`propose_next_candidate()` runs a bounded step loop (default max 10 steps):
1. ask model for next action (`tool_call` or `final_proposal`),
2. execute tool calls and append observations,
3. require at least one tool call and at least one run-evidence tool call before accepting `final_proposal`,
4. fail fast if step limit exceeded.

Run-evidence tools are prioritized in policy and enforced before finalize:
- `list_runs`
- `search_traces`
- `read_run_metadata`
- `read_trace_snippet`

### Run evidence sources used by proposer tools

`proposer/agent_tools.py` now resolves run/trace evidence with explicit source normalization and fallback:

1. `proposer_archive_v2` index `trace_locations.runs_dir` (primary, schema-aligned).
2. `proposer_archive_v2/<baselines|candidates>/<harness_id>/runs` (primary local harness folder).
3. `trace_pointers.json` declared source path under each harness archive entry (legacy bridge).
4. `proposer_archive/manual_runs/runs` (legacy/manual fallback).

For legacy shared folders, runs are mapped to a harness by reading markers from metadata/trace JSON lines
(e.g. `selected_baseline_id`, `harness_id`, `candidate_id`) instead of hardcoded harness IDs.

Tool behaviors:

- `list_runs(harness_id)` returns structured run rows (`run_id`, `run_dir`, `source`, `source_type`, file availability, optional scene/task/status).
- `search_traces(...)` searches across runtime/planning traces and run metadata in resolved sources.
- If no evidence exists, tools return empty hits/runs **with audit/debug reason metadata** (not silent path bugs).

The proposer now also ranks harnesses by available run evidence before selecting the sample set, so it prefers
evidence-rich baselines/candidates and only falls back to spec/code-first behavior when run evidence is truly absent.

## Runtime-effect sandbox modules

- `state_features.py`: encodes runtime snapshot into state features that are injected into planning context.
- `trigger_logic.py`: optional runtime trigger decision hook used by heartbeat/threshold replan checks.
- `prompt_composer.py`: composes additional prompt context appended to planner task text at runtime.
- `archive_selector.py`: controls which archive entries/trace snippets are fed to proposer summary (latest harness selector).
- `validator_rules.py`: validator extension hook (`runtime_effect_modules()`) used by consistency checks.

## Editable options audit (kept / deprecated / rewired)

- **Kept (runtime-effect):**
  - `state_features.py`, `trigger_logic.py`, `prompt_composer.py`
  - (compatibility legacy names) `state_encoder.py`, `trigger_policy.py`, `prompt_builder.py`
- **Deprecated (metadata-only / fake options):**
  - `prompt_builder.paragraph_order`
  - `prompt_builder.stages`
  - `state_encoder.summary_style` (kept for compatibility metadata, not trusted as runtime-effect signal)
- **Rewired to runtime checks:**
  - Runtime-effect detection now verifies actual changed modules and can be extended by `validator_rules.runtime_effect_modules()`.
  - Candidate contract must include expected `changed_files`, and final accepted spec stores actual changed files + diff metadata.

## Contract alignment guardrails (new)

`propose_next_candidate()` now applies **minimal hard guardrails + bounded self-review revise loop**.

- Hard guardrails focus only on:
  - boundary safety,
  - required artifacts (`spec`, `manifest`, `runtime_metadata`, `proposer_note`, diff/audit files),
  - py_compile + import checks,
  - structured runtime wiring smoke verification artifact,
  - runtime wiring consistency for sandbox modules,
  - at least one runtime-effect module changed.
- High-level narrative/string semantics are intentionally not hard-failed by system validator.
- If guardrails/smoke fail, proposer enters bounded revise loop (default up to 2 revisions) and asks the coding agent to self-fix files, then re-check.

If any check fails, proposer **fails fast** and removes the just-created candidate directory instead of silently accepting an inconsistent candidate.

### Structured runtime wiring smoke verification

After candidate generation and before final acceptance, proposer now writes:

- `runtime_wiring_verification.json`

with fields including:
- loaded runtime module/function per line:
  - `loaded_trigger_module` / `loaded_trigger_function`
  - `loaded_state_module` / `loaded_state_function`
  - `loaded_prompt_module` / `loaded_prompt_function`
- candidate claims:
  - `candidate_trigger_module_claim`
  - `candidate_state_module_claim`
  - `candidate_prompt_module_claim`
- per-line alignment checks:
  - `trigger_alignment_ok`
  - `state_alignment_ok`
  - `prompt_alignment_ok`
- final result:
  - `passed`
  - `notes`

`spec.runtime_metadata` now records:
- `runtime_wiring_verification_path`
- `runtime_wiring_verification_passed`

If wiring verification fails for any claimed/changed line, proposer treats it as smoke/guardrail failure and pushes the candidate into self-review/revise (bounded rounds) or final reject.

## Diff-safe / branch-safe / test-safe metadata

Accepted candidates now persist the following in `spec.json`:
- `manifest` (active sandbox modules + evidence pointers)
- `runtime_metadata.parent_commit`
- `runtime_metadata.changed_files`
- `runtime_metadata.candidate_branch_hint`
- `runtime_metadata.diff_path`
- `runtime_metadata.proposer_tool_audit_path`
- `runtime_metadata.proposer_tool_event_count`
- `runtime_metadata.hypothesis_target_modules`
- `runtime_metadata.runtime_effect_changed_files`
- `runtime_metadata.supporting_generated_files`
- `runtime_metadata.full_diff_files`

And each accepted candidate writes:
- `parent_diff.patch` (code diff)
- `proposer_tool_audit.json` (actual tool-call audit trail, including trace search/snippet reads)

for rollback/review without auto-push or auto-merge.

### Metadata semantics for research analysis (primary vs supporting)

To avoid ambiguity between hypothesis scope and packaging artifacts:

- `proposal_contract.hypothesis_target_modules`:
  - primary runtime-effect module lines this candidate intends to test.
- `proposal_contract.runtime_effect_changed_files`:
  - actually changed files that belong to runtime-effect module set **and** are inside `hypothesis_target_modules` (primary claim scope).
- `proposal_contract.supporting_generated_files`:
  - changed files produced for packaging/metadata (e.g. `spec.json`, `proposer_note.txt`) rather than primary hypothesis line.
- `proposal_contract.full_diff_files`:
  - full parent->candidate changed file set.

Runtime metadata mirrors the same semantics:
- `runtime_metadata.hypothesis_target_modules`
- `runtime_metadata.runtime_effect_changed_files`
- `runtime_metadata.supporting_generated_files`
- `runtime_metadata.full_diff_files`

Legacy/compatibility field interpretation:
- `files_to_create_or_modify`: generation-time requested/allowed write target list (not the primary hypothesis scope by itself).
- `runtime_metadata.changed_files`: backward-compatible changed file list; prefer `runtime_effect_changed_files` + `supporting_generated_files` for analysis.

## 如何確認 proposer 真的有調閱紀錄檔

1. 打開 `harnesses/candidates/<candidate_id>/proposer_tool_audit.json`，確認至少有：
   - `search_traces`
   - `read_trace_snippet`
   - `read_run_metadata`
2. 打開同目錄的 `spec.json`，確認：
   - `runtime_metadata.proposer_tool_audit_path == "proposer_tool_audit.json"`
   - `runtime_metadata.proposer_tool_event_count > 0`
3. 對照 `parent_diff.patch` 與 `proposal_contract`，檢查 candidate 不只是改 spec，而是改到 runtime-effect sandbox code。

## Live benchmark loop

1. Proposer creates `harnesses/candidates/candidate_xxxx/`.
2. Evaluator uses `LLMController` live execution by stage:
   - candidate screening: 6 runs
   - candidate formal: 24 runs
   - baselines: 24 runs
3. Per-run outputs are copied to:
   - `runs/run_xxx/runtime_trace.jsonl`
   - `runs/run_xxx/planning_trace.jsonl`
   - `runs/run_xxx/metadata.json`
4. Aggregation writes:
   - stage-specific files: `per_scene_metrics_screening.json` / `eval_summary_screening.json`
   - stage-specific files: `per_scene_metrics_formal.json` / `eval_summary_formal.json`
   - legacy aliases kept for formal: `per_scene_metrics.json` / `eval_summary.json`
5. Index rebuild updates lineage + Pareto frontier.

## Runtime mode is now spec-driven (baseline + candidate)

Runtime execution mode is selected from each harness `spec.json` trigger policy, not only from hardcoded baseline IDs:

- `trigger_policy.type in {periodic, heartbeat, hybrid}` -> `agent-heartbeat-soft`
- `trigger_policy.type in {event_predicted_collision_probability, threshold, event...}` -> `typefly-threshold-replan`
- otherwise fallback -> `typefly-oneshot`

`heartbeat_seconds`, `threshold`, and related trigger params are loaded from the same spec and propagated into runtime run context.

You can verify applied config in:

- terminal `[MODE] ...` line during execution
- per-run `metadata.json` (contains `run_summary`/`debug_summary`)
- planning trace rows (`planning_trace.jsonl`) with trigger evidence fields

Heartbeat observability fields now include:
- `heartbeat_mode_enabled`
- `heartbeat_due`
- `heartbeat_result` (`continue` / `replan` / `skipped` / `blocked` / `failed`)
- `heartbeat_reason`
- `trigger_reason`
- `replan_applied`
- `replan_skip_reason`

Completion observability fields now use active-zone scope:
- `active_task_zone`
- `active_zone_checkpoints`
- `completed_active_checkpoints`
- `remaining_active_checkpoints`
- `mission_success_reason`
- `completion_scope=zone_scoped`

## CLI

```bash
python -m proposer.cli list-baselines
python -m proposer.cli list-candidates
python -m proposer.cli show-candidate-summary candidate_0001
python -m proposer.cli top-k --metric collision_count_avg -k 5
python -m proposer.cli diff baseline2 baseline3
python -m proposer.cli propose --focus-text "improve risk timing"
python -m proposer.cli evaluate candidate_0001
python -m proposer.cli evaluate candidate_0001 --mode formal
python -m proposer.cli evaluate baseline1
python -m proposer.cli reindex
python -m proposer.cli run-iteration --focus-text "improve risk timing"
```

`evaluate <candidate_id>` defaults to screening mode. Use `--mode formal` after screening to promote a candidate into full 24-run formal evaluation.

`top-k` defaults to formal-only entries; add `--include-screening` to include screening-stage candidates.

> If no LLM key/provider is configured, propose/run-iteration may fail unless `--allow-fallback-heuristic` is provided.

## Default proposer pool exclusions (anchoring guard)

- `candidate_0001` remains in archive/history and can still be manually inspected.
- But it is removed from default proposer candidate retrieval/parent pool:
  - default `list_harnesses(kind="candidate"|"all")` excludes `candidate_0001`
  - archive summary candidate pool excludes `candidate_0001`
  - proposer rejects `parent_harness=candidate_0001` unless `TYPEFLY_ALLOW_EXCLUDED_PARENT=1`

Manual inspection is still available through direct lookup APIs (`read_harness_spec("candidate_0001")`, CLI summary, direct file reads).

## Runtime prompt asset tools (proposer toolbox)

New proposer tools:

- `list_runtime_prompt_assets(harness_id)`
- `read_runtime_prompt_asset(harness_id, asset_name=None, stage=None)`
- `diff_runtime_prompt_assets(harness_a, harness_b, stage="initial")`

These tools read real runtime prompt files under `controller/assets/tello/` using the same stage/variant routing rules as planner runtime.

## Evaluate prompt-source evidence

Planning traces now include prompt-source evidence fields:

- `selected_prompt_module`
- `selected_prompt_module_path`
- `selected_prompt_asset_path`
- `selected_prompt_asset_name`
- `prompt_hash_sha256`
- `rendered_prompt_source`
- `evaluate_prompt_source` (summary block)

Run metadata copies `evaluate_prompt_source`, so prompt-source alignment is verifiable during candidate evaluation.
