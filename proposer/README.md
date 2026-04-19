# Restricted Meta-Harness MVP (Talk-to-Fly)

This proposer loop limits mutation boundary to harness-level modules only:

- `state_encoder.py`
- `trigger_policy.py`
- `prompt_builder.py`
- `spec.json`
- `state_features.py`
- `trigger_logic.py`
- `prompt_composer.py`
- `archive_selector.py`
- `validator_rules.py`

## Fixed evaluation protocol (strict)

- `SCENE1 -> zoneA`, repeated 8 runs
- `SCENE2 -> zoneB`, repeated 8 runs
- `SCENE3 -> zoneC`, repeated 8 runs

Total runs per evaluated harness: **24**.

## Agent-driven proposer

`proposer/propose_candidate.py` now drives proposal creation with LLM calls via `controller.llm_wrapper.LLMWrapper`.

- It uses controlled coding-agent tools (`proposer/agent_tools.py`) to list/read/diff harnesses and inspect runs/traces.
- It uses the system prompt + iteration prompt + output contract in `proposer/prompts.py`.
- It proposes exactly one candidate each invocation.
- It keeps candidate edits bounded by `proposer/registry.py` allowed files.
- It now enforces contract/spec/code consistency via `proposer/consistency.py` before a candidate is accepted.
- Proposer model defaults to `gpt-4.1` and is configurable via `TYPEFLY_PROPOSER_MODEL`.
- If `TYPEFLY_PROPOSER_MODEL` is a GPT model and `OPENAI_API_KEY` is missing, proposer fails with a clear error (unless explicit fallback is enabled).

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
  - runtime wiring consistency for sandbox modules,
  - at least one runtime-effect module changed.
- High-level narrative/string semantics are intentionally not hard-failed by system validator.
- If guardrails/smoke fail, proposer enters bounded revise loop (default up to 2 revisions) and asks the coding agent to self-fix files, then re-check.

If any check fails, proposer **fails fast** and removes the just-created candidate directory instead of silently accepting an inconsistent candidate.

## Diff-safe / branch-safe / test-safe metadata

Accepted candidates now persist the following in `spec.json`:
- `manifest` (active sandbox modules + evidence pointers)
- `runtime_metadata.parent_commit`
- `runtime_metadata.changed_files`
- `runtime_metadata.candidate_branch_hint`
- `runtime_metadata.diff_path`
- `runtime_metadata.proposer_tool_audit_path`
- `runtime_metadata.proposer_tool_event_count`

And each accepted candidate writes:
- `parent_diff.patch` (code diff)
- `proposer_tool_audit.json` (actual tool-call audit trail, including trace search/snippet reads)

for rollback/review without auto-push or auto-merge.

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
2. Evaluator uses `LLMController` live execution for 24 runs.
3. Per-run outputs are copied to:
   - `runs/run_xxx/runtime_trace.jsonl`
   - `runs/run_xxx/planning_trace.jsonl`
   - `runs/run_xxx/metadata.json`
4. Aggregation writes:
   - `per_scene_metrics.json`
   - `eval_summary.json`
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

## CLI

```bash
python -m proposer.cli list-baselines
python -m proposer.cli list-candidates
python -m proposer.cli show-candidate-summary candidate_0001
python -m proposer.cli top-k --metric collision_count_avg -k 5
python -m proposer.cli diff baseline2 baseline3
python -m proposer.cli propose --focus-text "improve risk timing"
python -m proposer.cli evaluate candidate_0001
python -m proposer.cli reindex
python -m proposer.cli run-iteration --focus-text "improve risk timing"
```

> If no LLM key/provider is configured, propose/run-iteration may fail unless `--allow-fallback-heuristic` is provided.
