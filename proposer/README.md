# Restricted Meta-Harness MVP (Talk-to-Fly)

This proposer loop limits mutation boundary to harness-level modules only:

- `state_encoder.py`
- `trigger_policy.py`
- `prompt_builder.py`
- `spec.json`

## Fixed evaluation protocol (strict)

- `SCENE1 -> zoneA`, repeated 8 runs
- `SCENE2 -> zoneB`, repeated 8 runs
- `SCENE3 -> zoneC`, repeated 8 runs

Total runs per evaluated harness: **24**.

## Agent-driven proposer

`proposer/propose_candidate.py` now drives proposal creation with LLM calls via `controller.llm_wrapper.LLMWrapper`.

- It reads archive/index + representative trace snippets.
- It uses the system prompt + iteration prompt + output contract in `proposer/prompts.py`.
- It proposes exactly one candidate each invocation.
- It keeps candidate edits bounded by `proposer/registry.py` allowed files.

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
