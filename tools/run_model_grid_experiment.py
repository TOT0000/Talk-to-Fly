#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import urllib.error
import urllib.request
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


def _fetch_lmstudio_visible_models(base_url: str, timeout_sec: float) -> list[str]:
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    request = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"LM Studio server is not reachable at {endpoint}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LM Studio /v1/models returned non-JSON response: {e}") from e

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id", "")).strip()
        if model_id:
            model_ids.append(model_id)
    return sorted(set(model_ids))


def _wait_for_block_confirmation(
    *,
    block_by: str,
    block_model: str,
    block_index: int,
    total_blocks: int,
    pending_pairs: list[tuple[str, str]],
):
    preview = ", ".join(f"({p} × {e})" for p, e in pending_pairs[:8])
    if len(pending_pairs) > 8:
        preview += f", ... (+{len(pending_pairs) - 8} more)"

    if block_by == "planner":
        print(f"[BLOCK] block_by=planner fixed_planner={block_model} block_index={block_index}/{total_blocks}")
        print(f"[BLOCK] evaluators_in_this_block={[e for _, e in pending_pairs]}")
    else:
        print(f"[BLOCK] block_by=evaluator fixed_evaluator={block_model} block_index={block_index}/{total_blocks}")
        print(f"[BLOCK] planners_in_this_block={[p for p, _ in pending_pairs]}")
    print(f"[BLOCK] pending_pair_count={len(pending_pairs)} preview={preview}")
    print(
        "[ACTION REQUIRED] Please load the fixed block model in LM Studio. "
        "After LM Studio /v1/models shows it as visible, press Enter to continue."
    )
    try:
        input("")
    except EOFError as e:
        raise RuntimeError(
            "Interactive confirmation required for block mode, but stdin is not available."
        ) from e


def _build_blocks(block_by: str, models: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
    blocks: list[tuple[str, list[tuple[str, str]]]] = []
    if block_by == "planner":
        for planner_model in models:
            blocks.append((planner_model, [(planner_model, ev) for ev in models]))
    else:
        for evaluator_model in models:
            blocks.append((evaluator_model, [(pl, evaluator_model) for pl in models]))
    return blocks


def _classify_failure(exc: Exception) -> str:
    text = str(exc or "").strip()
    lowered = text.lower()
    if "reasoning_only_empty_content" in lowered:
        return "reasoning_only_empty_content"
    if "context size has been exceeded" in lowered:
        return "context_size_exceeded"
    if "context_length_exceeded" in lowered:
        return "context_size_exceeded"
    return "runtime_exception"


def _run_single_attempt(
    *,
    args: argparse.Namespace,
    planner_model: str,
    evaluator_model: str,
    block_model: str,
    block_idx: int,
    zone_objective: dict,
) -> dict:
    from controller.abs.robot_wrapper import RobotType
    from controller.llm_controller import LLMController

    controller = None
    try:
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
        controller.planner.set_model(planner_model)
        controller.planner.set_agent_model_names(
            heartbeat_model_name=planner_model,
            evaluator_model_name=evaluator_model,
        )
        controller.set_active_objective_set_override(zone_objective)
        controller._reset_benchmark_progress_tracking()
        controller.task_run_logger.discard_pending_run()
        controller.execute_task_description(
            task_description=_task_text_for_zone(args.zone_id),
            framework_mode="agent-heartbeat-soft",
        )
        summary = controller.task_run_logger.get_pending_run_summary()
        controller.task_run_logger.discard_pending_run()
        final_summary = dict(controller.final_mission_summary or {})
        completed = summary.get("true_completed_checkpoints") or final_summary.get("final_true_completed_checkpoints") or []
        remaining = summary.get("true_remaining_checkpoints") or final_summary.get("final_true_remaining_checkpoints") or []
        failure_reason = str(summary.get("failure_reason") or "").strip()
        run_status = str(summary.get("run_status") or final_summary.get("final_run_status") or "")
        if (not failure_reason) and run_status.lower() == "failed":
            failure_reason = "runtime_exception"
        return {
            "run_status": run_status,
            "mission_success": _safe_bool(summary.get("mission_success", final_summary.get("mission_success"))),
            "termination_reason": summary.get("termination_reason") or final_summary.get("termination_reason") or "",
            "failure_reason": failure_reason,
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
            "block_by": args.block_by,
            "block_model": block_model,
            "block_index": block_idx,
        }
    except Exception as exc:
        return {
            "run_status": "failed",
            "mission_success": False,
            "termination_reason": "",
            "failure_reason": _classify_failure(exc),
            "completion_time_mission_sec": None,
            "replan_count": 0,
            "collision_count": 0,
            "near_miss_count": 0,
            "min_uav_worker_distance_m": None,
            "completion_ratio": 0.0,
            "completed_checkpoints": "",
            "remaining_checkpoints": "",
            "run_id": "",
            "task_id": "",
            "block_by": args.block_by,
            "block_model": block_model,
            "block_index": block_idx,
        }
    finally:
        if controller is not None:
            try:
                controller.clear_active_objective_set_override()
            except Exception:
                pass
            try:
                controller.stop_robot()
            except Exception:
                pass


def run(args: argparse.Namespace):
    logger = ExperimentResultLogger(csv_path=args.output_csv, xlsx_path=args.output_xlsx)
    done_keys = logger.load_completed_keys()
    selected_model_count = len(list(args.models))
    print(
        f"[MODE] single-endpoint mode = {'ON' if args.single_endpoint_mode else 'OFF'}"
    )
    if args.single_endpoint_mode:
        print(
            "[MODE] Single LM Studio endpoint can only reliably support planner_model == evaluator_model. "
            "Off-diagonal combinations will be skipped."
        )
    all_pairs = [(p, e) for p in args.models for e in args.models]
    allowed_pairs = [
        (p, e)
        for p, e in all_pairs
        if (not args.single_endpoint_mode) or (p == e)
    ]
    planned_runs = len(allowed_pairs) * int(args.repeats)
    if args.single_endpoint_mode:
        print(f"[PLAN] selected model count = {selected_model_count}")
        print(f"[PLAN] planned diagonal runs = {planned_runs}")
    else:
        print(
            f"[PLAN] total_pairs={len(all_pairs)} allowed_pairs={len(allowed_pairs)} repeats={args.repeats} "
            f"planned_runs={planned_runs}"
        )

    zone_objective = _build_zone_objective(args.zone_id)

    blocks = _build_blocks(args.block_by, list(args.models))
    for block_idx, (block_model, pair_list) in enumerate(blocks, start=1):
        pending_pairs = []
        for planner_model, evaluator_model in pair_list:
            if args.single_endpoint_mode and planner_model != evaluator_model:
                print(
                    "[SKIP] single-endpoint mode requires planner_model == evaluator_model "
                    f"(planner={planner_model}, evaluator={evaluator_model})"
                )
                continue
            has_remaining = False
            for repeat_idx in range(1, args.repeats + 1):
                key = ExperimentKey(planner_model=planner_model, evaluator_model=evaluator_model, repeat_idx=repeat_idx)
                if key not in done_keys:
                    has_remaining = True
                    break
            if has_remaining:
                pending_pairs.append((planner_model, evaluator_model))
        if not pending_pairs:
            print(f"[BLOCK-SKIP] block_by={args.block_by} block_model={block_model} all repeats already completed.")
            continue

        _wait_for_block_confirmation(
            block_by=args.block_by,
            block_model=block_model,
            block_index=block_idx,
            total_blocks=len(blocks),
            pending_pairs=pending_pairs,
        )

        visible_models = _fetch_lmstudio_visible_models(args.lmstudio_base_url, timeout_sec=args.lmstudio_timeout_sec)
        print(f"[CHECK] LM Studio models visible: {visible_models}")
        if block_model not in visible_models:
            raise RuntimeError(
                f"[STOP] required model: {block_model} | visible models: {visible_models}"
            )

        for planner_model, evaluator_model in pair_list:
            if args.single_endpoint_mode and planner_model != evaluator_model:
                continue
            for repeat_idx in range(1, args.repeats + 1):
                key = ExperimentKey(planner_model=planner_model, evaluator_model=evaluator_model, repeat_idx=repeat_idx)
                if key in done_keys:
                    print(f"[SKIP] completed: planner={planner_model} evaluator={evaluator_model} repeat={repeat_idx}")
                    continue

                if args.strict_run_model_check:
                    visible_now = _fetch_lmstudio_visible_models(args.lmstudio_base_url, timeout_sec=args.lmstudio_timeout_sec)
                    missing = [m for m in (planner_model, evaluator_model) if m not in visible_now]
                    if missing:
                        raise RuntimeError(
                            f"[STOP] required model: {missing} | visible models: {visible_now}"
                        )

                print(
                    f"[RUN] planner={planner_model} evaluator={evaluator_model} repeat={repeat_idx} "
                    f"pipeline={args.pipeline_id} scene={args.scene_id} zone={args.zone_id} "
                    f"block_by={args.block_by} block_model={block_model} block_index={block_idx}"
                )
                run_payload = _run_single_attempt(
                    args=args,
                    planner_model=planner_model,
                    evaluator_model=evaluator_model,
                    block_model=block_model,
                    block_idx=block_idx,
                    zone_objective=zone_objective,
                )
                row = {
                    "experiment_tag": args.experiment_tag,
                    "pipeline_id": args.pipeline_id,
                    "scenario_id": args.scene_id,
                    "zone_id": args.zone_id,
                    "repeat_idx": repeat_idx,
                    "planner_model": planner_model,
                    "evaluator_model": evaluator_model,
                    **run_payload,
                }
                logger.append_result(row)
                done_keys.add(key)


def parse_args() -> argparse.Namespace:
    default_output = os.path.expanduser("~/typefly_logs/model_grid_results.csv")
    parser = argparse.ArgumentParser(description="Run planner/evaluator model grid experiments (resumable).")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_GRID_IDS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--block-by", choices=["planner", "evaluator"], default="planner")
    parser.add_argument("--pipeline-id", default=DEFAULT_PIPELINE_ID)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--zone-id", default=DEFAULT_ZONE_ID)
    parser.add_argument("--experiment-tag", default=DEFAULT_EXPERIMENT_TAG)
    parser.add_argument("--output-csv", default=default_output)
    parser.add_argument("--output-xlsx", default=os.path.splitext(default_output)[0] + ".xlsx")
    parser.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--lmstudio-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--strict-run-model-check",
        action="store_true",
        help="Before each run, require both planner/evaluator model IDs to be visible in /v1/models.",
    )
    parser.add_argument(
        "--single-endpoint-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), only planner_model == evaluator_model runs are allowed "
            "for single LM Studio endpoint environments."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
