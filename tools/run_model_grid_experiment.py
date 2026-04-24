#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import queue
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from controller.experiment_result_logger import (
    ExperimentKey,
    ExperimentResultLogger,
    normalize_checkpoint_list,
)
from controller.model_grid_config import (
    DEFAULT_EXPERIMENT_TAG,
    DEFAULT_MODEL_GRID_IDS,
    DEFAULT_PIPELINE_ID,
    DEFAULT_REPEAT_COUNT,
    DEFAULT_SCENE_ID,
    DEFAULT_ZONE_ID,
    ZONE_TO_CHECKPOINTS,
)


def _build_zone_objective(zone_id: str) -> dict:
    zone_key = str(zone_id or "").strip()
    if zone_key not in ZONE_TO_CHECKPOINTS:
        raise ValueError(f"Unsupported zone_id: {zone_key}")
    return {
        "active_zone_ids": [zone_key],
        "active_checkpoint_ids": list(ZONE_TO_CHECKPOINTS[zone_key]),
        "source": f"batch_experiment_{zone_key.lower()}",
    }


def _task_text_for_zone(zone_id: str) -> str:
    return f"Agent benchmark run: inspect {zone_id} only."


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run(args: argparse.Namespace):
    from controller.abs.robot_wrapper import RobotType
    from controller.llm_controller import LLMController

    logger = ExperimentResultLogger(csv_path=args.output_csv, xlsx_path=args.output_xlsx)
    done_keys = logger.load_completed_keys()

    controller = LLMController(
        robot_type=RobotType.PX4_SIM,
        virtual_queue=queue.Queue(maxsize=500),
        use_http=False,
        message_queue=None,
        enable_video=False,
    )
    controller.start_robot()
    controller.set_archive_enabled(False)
    controller.set_selected_pipeline(args.pipeline_id)
    controller.set_baseline_scene(args.scene_id)
    controller.apply_baseline_scene()
    zone_objective = _build_zone_objective(args.zone_id)

    try:
        all_pairs = list(itertools.product(args.models, args.models))
        for planner_model, evaluator_model in all_pairs:
            for repeat_idx in range(1, args.repeats + 1):
                key = ExperimentKey(planner_model=planner_model, evaluator_model=evaluator_model, repeat_idx=repeat_idx)
                if key in done_keys:
                    print(f"[SKIP] completed: planner={planner_model} evaluator={evaluator_model} repeat={repeat_idx}")
                    continue

                controller.planner.set_model(planner_model)
                # Planner-side consistency: heartbeat == planner model.
                controller.planner.set_agent_model_names(
                    heartbeat_model_name=planner_model,
                    evaluator_model_name=evaluator_model,
                )
                controller.set_selected_pipeline(args.pipeline_id)
                controller.set_baseline_scene(args.scene_id)
                controller.apply_baseline_scene()
                controller.set_active_objective_set_override(zone_objective)
                controller._reset_benchmark_progress_tracking()
                controller.task_run_logger.discard_pending_run()

                print(
                    f"[RUN] planner={planner_model} evaluator={evaluator_model} repeat={repeat_idx} "
                    f"pipeline={args.pipeline_id} scene={args.scene_id} zone={args.zone_id}"
                )
                controller.execute_task_description(
                    task_description=_task_text_for_zone(args.zone_id),
                    framework_mode="agent-heartbeat-soft",
                )

                summary = controller.task_run_logger.get_pending_run_summary()
                controller.task_run_logger.discard_pending_run()
                final_summary = dict(controller.final_mission_summary or {})

                completed = summary.get("true_completed_checkpoints") or final_summary.get("final_true_completed_checkpoints") or []
                remaining = summary.get("true_remaining_checkpoints") or final_summary.get("final_true_remaining_checkpoints") or []

                row = {
                    "experiment_tag": args.experiment_tag,
                    "pipeline_id": args.pipeline_id,
                    "scenario_id": args.scene_id,
                    "zone_id": args.zone_id,
                    "repeat_idx": repeat_idx,
                    "planner_model": planner_model,
                    "evaluator_model": evaluator_model,
                    "run_status": summary.get("run_status") or final_summary.get("final_run_status") or "",
                    "mission_success": _safe_bool(summary.get("mission_success", final_summary.get("mission_success"))),
                    "termination_reason": summary.get("termination_reason") or final_summary.get("termination_reason") or "",
                    "failure_reason": summary.get("failure_reason") or "",
                    "completion_time_mission_sec": summary.get("completion_time_mission_sec") or final_summary.get("completion_time_mission_sec"),
                    "replan_count": summary.get("replan_count") or final_summary.get("replan_count", 0),
                    "collision_count": summary.get("collision_count") or final_summary.get("collision_count", 0),
                    "near_miss_count": summary.get("near_miss_count") or final_summary.get("near_miss_count", 0),
                    "min_uav_worker_distance_m": summary.get("min_uav_worker_distance_m") or final_summary.get("min_uav_worker_distance_m"),
                    "completion_ratio": summary.get("completion_ratio") or final_summary.get("completion_ratio"),
                    "completed_checkpoints": normalize_checkpoint_list(completed),
                    "remaining_checkpoints": normalize_checkpoint_list(remaining),
                    "run_id": summary.get("run_id", ""),
                    "task_id": summary.get("task_id", ""),
                }
                logger.append_result(row)
                done_keys.add(key)
    finally:
        controller.clear_active_objective_set_override()
        controller.stop_robot()


def parse_args() -> argparse.Namespace:
    default_output = os.path.expanduser("~/typefly_logs/model_grid_results.csv")
    parser = argparse.ArgumentParser(description="Run planner/evaluator model grid experiments (resumable).")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_GRID_IDS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--pipeline-id", default=DEFAULT_PIPELINE_ID)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--zone-id", default=DEFAULT_ZONE_ID)
    parser.add_argument("--experiment-tag", default=DEFAULT_EXPERIMENT_TAG)
    parser.add_argument("--output-csv", default=default_output)
    parser.add_argument("--output-xlsx", default=os.path.splitext(default_output)[0] + ".xlsx")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
