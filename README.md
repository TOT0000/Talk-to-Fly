# Talk-to-Fly

## Model Grid Experiment Mode (Planner × Evaluator)

This repository now includes a resumable batch experiment script for comparing planner/evaluator model combinations on a fixed setup:

- pipeline: `Agent-Feedback-Eval` (`agent`)
- scene: `SCENE3`
- objective: `zone_C` only
- model grid: 7 × 7 (planner × evaluator)
- repeats: 10
- total runs: 490

### Default model list (centralized)

The default model IDs are centralized in `controller/model_grid_config.py` (`DEFAULT_MODEL_GRID_IDS`).

> Important: model IDs must exactly match what your LM Studio OpenAI-compatible endpoint exposes (e.g., `/v1/models`). If your local endpoint uses different IDs, update `DEFAULT_MODEL_GRID_IDS` in that file only.

### Run the full 490-run experiment

```bash
python tools/run_model_grid_experiment.py
```

For LM Studio manual model-loading workflow, run by blocks:

```bash
python tools/run_model_grid_experiment.py --block-by planner
python tools/run_model_grid_experiment.py --block-by evaluator
```

Default outputs:

- `~/typefly_logs/model_grid_results.csv`
- `~/typefly_logs/model_grid_results.xlsx` (if `openpyxl` is available)

### Resume behavior

The script deduplicates by tuple:

- `(planner_model, evaluator_model, repeat_idx)`

If a tuple already exists in the CSV, that run is skipped. This supports interruption and resume without rewriting finished rows.

In block mode, the script pauses at each block transition and asks for Enter confirmation after you manually load the fixed block model in LM Studio.

### What is intentionally *not* archived in experiment mode

The script sets controller archive mode to off and discards each pending run after extracting final summary fields, so no formal task-run archive is persisted for these batch runs.

Concretely, the batch mode does **not** save:

- runtime trace JSONL archive
- planning trace JSONL archive
- per-run archive artifacts under the task-run archive directory

### Planner/Evaluator model routing in experiment mode

For each run tuple:

- `planner.model_name = planner_model`
- `planner.heartbeat_model_name = planner_model`
- `planner.evaluator_model_name = evaluator_model`

This enforces planner-side consistency (default planner + heartbeat aligned) while keeping evaluator independent.
