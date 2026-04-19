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

This protocol is written into:

- `controller/harness_protocol.py`
- each evaluated harness `eval_summary.json`
- `proposer_archive_v2/index.json`

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

## CLI

```bash
python -m proposer.cli list-baselines
python -m proposer.cli list-candidates
python -m proposer.cli show-candidate-summary candidate_0001
python -m proposer.cli top-k --metric collision_count_avg -k 5
python -m proposer.cli diff baseline2 baseline3
python -m proposer.cli propose
python -m proposer.cli evaluate candidate_0001
python -m proposer.cli reindex
python -m proposer.cli run-iteration
```
