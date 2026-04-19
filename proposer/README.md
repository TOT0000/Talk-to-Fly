# Restricted Meta-Harness MVP (Talk-to-Fly)

This proposer loop intentionally limits code mutation boundary to harness-level modules only:

- `state_encoder.py`
- `trigger_policy.py`
- `prompt_builder.py`
- `spec.json`

## Fixed evaluation protocol

The evaluator stores and enforces a fixed scene-task mapping:

- `SCENE1 -> zoneA`
- `SCENE2 -> zoneB`
- `SCENE3 -> zoneC`

This mapping is written into `proposer_archive_v2/index.json` and each candidate's `eval_summary.json`.

## Loop (MVP)

1. Materialize baseline harnesses into `proposer_archive_v2/baselines/*`.
2. Proposer reads prior specs + metrics + trace pointers from archive.
3. Proposer writes a new `harnesses/candidates/candidate_xxxx/`.
4. Evaluator writes `eval_summary.json`, `per_scene_metrics.json`, and trace pointers.
5. Index updater recomputes Pareto frontier flags.

## CLI

```bash
python -m proposer.cli list-baselines
python -m proposer.cli list-candidates
python -m proposer.cli top-k --metric collision_count_avg -k 5
python -m proposer.cli diff baseline2 baseline3
python -m proposer.cli propose
python -m proposer.cli evaluate candidate_0001
python -m proposer.cli reindex
```
