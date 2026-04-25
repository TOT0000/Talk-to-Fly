#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    DEFAULT_EVALUATOR_MODEL_IDS,
    DEFAULT_EXPERIMENT_TAG,
    DEFAULT_FIXED_EVALUATOR_MODEL,
    DEFAULT_FIXED_PLANNER_MODEL,
    DEFAULT_PIPELINE_ID,
    DEFAULT_PLANNER_MODEL_IDS,
    DEFAULT_REPEAT_COUNT,
    DEFAULT_SCENE_ID,
    DEFAULT_ZONE_ID,
    ZONE_TO_CHECKPOINTS,
)


@dataclass(frozen=True)
class PairSpec:
    pair_index: int
    planner_model: str
    evaluator_model: str
    phase: str

    @property
    def pair_label(self) -> str:
        return f"{self.planner_model}__{self.evaluator_model}"


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


def _resolve_visible_model_id(required_model_id: str, visible_model_ids: list[str]) -> str:
    required = str(required_model_id or "").strip()
    if required in visible_model_ids:
        return required
    lowered_map: dict[str, list[str]] = {}
    for model_id in visible_model_ids:
        lowered_map.setdefault(model_id.lower(), []).append(model_id)
    hits = lowered_map.get(required.lower(), [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(
            f"[STOP] ambiguous model id mapping for '{required}' in visible ids: {hits}"
        )
    raise RuntimeError(
        f"[STOP] required model id not visible: required={required} visible_model_ids={visible_model_ids}"
    )


def _build_pair_specs(
    planner_models: list[str],
    evaluator_models: list[str],
    fixed_planner_model: str,
    fixed_evaluator_model: str,
) -> list[PairSpec]:
    planner_candidates = [str(m).strip() for m in planner_models if str(m).strip()]
    evaluator_candidates = [str(m).strip() for m in evaluator_models if str(m).strip()]
    fixed_planner = str(fixed_planner_model).strip()
    fixed_evaluator = str(fixed_evaluator_model).strip()

    if fixed_planner not in planner_candidates:
        raise ValueError(f"fixed planner model is not in planner candidates: {fixed_planner}")
    if fixed_evaluator not in evaluator_candidates:
        raise ValueError(f"fixed evaluator model is not in evaluator candidates: {fixed_evaluator}")

    phase1_pairs = [
        (fixed_planner, evaluator, "phase1_fixed_planner_sweep_evaluator")
        for evaluator in evaluator_candidates
    ]
    phase2_pairs = [
        (planner, fixed_evaluator, "phase2_fixed_evaluator_sweep_planner")
        for planner in planner_candidates
    ]
    raw_pairs: list[tuple[str, str, str]] = [*phase1_pairs, *phase2_pairs]

    seen: set[tuple[str, str]] = set()
    pair_specs: list[PairSpec] = []
    for planner_model, evaluator_model, phase in raw_pairs:
        key = (planner_model, evaluator_model)
        if key in seen:
            continue
        seen.add(key)
        pair_specs.append(
            PairSpec(
                pair_index=len(pair_specs) + 1,
                planner_model=planner_model,
                evaluator_model=evaluator_model,
                phase=phase,
            )
        )
    return pair_specs


def _wait_for_pair_confirmation(pair: PairSpec, total_pairs: int, repeats: int):
    print(
        f"[PAIR] pair_index={pair.pair_index}/{total_pairs} planner={pair.planner_model} "
        f"evaluator={pair.evaluator_model} phase={pair.phase} repeats={repeats}"
    )
    print(
        "[ACTION REQUIRED] Please load BOTH models in LM Studio for this pair. "
        "After /v1/models shows both IDs, press Enter to continue."
    )
    try:
        input("")
    except EOFError as e:
        raise RuntimeError("Interactive confirmation required, but stdin is not available.") from e


def _classify_failure(exc: Exception) -> str:
    text = str(exc or "").strip()
    lowered = text.lower()
    if "provider_not_lmstudio" in lowered:
        return "provider_not_lmstudio"
    if "invalid_authorization_header" in lowered:
        return "invalid_authorization_header"
    if "reasoning_only_empty_content" in lowered:
        return "reasoning_only_empty_content"
    if "context size has been exceeded" in lowered:
        return "context_size_exceeded"
    if "context_length_exceeded" in lowered:
        return "context_size_exceeded"
    return "runtime_exception"


def _is_fatal_failure_reason(reason: str) -> bool:
    value = str(reason or "").strip().lower()
    return value in {
        "provider_not_lmstudio",
        "invalid_authorization_header",
        "robot_not_ready",
        "offboard_takeoff_failed",
        "ros_executor_lifecycle_failure",
    } or value.startswith("run_cleanup_failed:")


def _configure_lmstudio_runtime_or_raise(lmstudio_base_url: str):
    normalized = str(lmstudio_base_url or "").strip().rstrip("/")
    if not normalized:
        raise RuntimeError("provider_not_lmstudio: empty_lmstudio_base_url")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    os.environ["LLM_PROVIDER"] = "lmstudio"
    os.environ["LMSTUDIO_BASE_URL"] = normalized
    os.environ.setdefault("LMSTUDIO_API_KEY", "lmstudio")
    os.environ["TYPEFLY_ENFORCE_LMSTUDIO"] = "1"

    from controller.llm_wrapper import resolve_runtime_provider_config

    runtime = resolve_runtime_provider_config()
    provider = str(runtime.get("provider") or "")
    base_url = str(runtime.get("base_url") or "")
    if provider != "lmstudio":
        raise RuntimeError(f"provider_not_lmstudio: provider={provider}")
    if base_url != normalized:
        raise RuntimeError(
            f"provider_not_lmstudio: base_url_mismatch provider={provider} base_url={base_url} expected={normalized}"
        )
    print(f"[LLM-RUNTIME] provider={provider} base_url={base_url}")


def _run_single_attempt(
    *,
    args: argparse.Namespace,
    planner_model: str,
    evaluator_model: str,
    pair: PairSpec,
    zone_objective: dict,
) -> dict:
    from controller.abs.robot_wrapper import RobotType
    from controller.llm_controller import LLMController

    controller = None
    run_payload: dict | None = None
    cleanup_error = ""
    cleanup_issues: list[str] = []
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
        ready, ready_reason = controller.check_robot_ready_for_task()
        if not ready:
            mapped_reason = "offboard_takeoff_failed" if "offboard_not_ready" in str(ready_reason) else "robot_not_ready"
            run_payload = {
                "run_status": "failed",
                "mission_success": False,
                "termination_reason": "preflight_robot_not_ready",
                "failure_reason": mapped_reason,
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
                "pair_index": pair.pair_index,
                "pair_label": pair.pair_label,
                "phase": pair.phase,
            }
            run_payload["_abort_batch"] = True
            run_payload["_cleanup_issues"] = f"robot_not_ready:{ready_reason}"
            return run_payload
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
            execution_source="batch_model_grid",
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
        run_payload = {
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
            "pair_index": pair.pair_index,
            "pair_label": pair.pair_label,
            "phase": pair.phase,
        }
        return run_payload
    except Exception as exc:
        run_payload = {
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
            "pair_index": pair.pair_index,
            "pair_label": pair.pair_label,
            "phase": pair.phase,
        }
        return run_payload
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
            try:
                controller.shutdown_for_run_end()
            except Exception as cleanup_exc:
                cleanup_error = str(cleanup_exc or cleanup_error)
            try:
                clean_ok, clean_issues = controller.verify_no_active_run_artifacts()
                if not clean_ok:
                    cleanup_issues = list(clean_issues or [])
            except Exception as verify_exc:
                cleanup_error = str(verify_exc or cleanup_error or "cleanup_verification_failed")
        if cleanup_error or cleanup_issues:
            issue_text = cleanup_error or ",".join(cleanup_issues) or "unknown_cleanup_failure"
            if run_payload is None:
                run_payload = {
                    "run_status": "failed",
                    "mission_success": False,
                    "termination_reason": "",
                    "failure_reason": f"run_cleanup_failed:{issue_text}",
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
                    "pair_index": pair.pair_index,
                    "pair_label": pair.pair_label,
                    "phase": pair.phase,
                }
            else:
                run_payload["run_status"] = "failed"
                run_payload["mission_success"] = False
                run_payload["failure_reason"] = f"run_cleanup_failed:{issue_text}"
            run_payload["_abort_batch"] = True
            run_payload["_cleanup_issues"] = f"ros_executor_lifecycle_failure:{issue_text}"


def _print_plan_summary(pair_specs: list[PairSpec], repeats: int):
    print("[PLAN] pair-based experiment mode = ON")
    print(f"[PLAN] total unique pairs = {len(pair_specs)}")
    print(f"[PLAN] repeats per pair = {repeats}")
    print(f"[PLAN] planned runs total = {len(pair_specs) * int(repeats)}")
    for pair in pair_specs:
        print(
            f"[PLAN-PAIR] index={pair.pair_index} planner={pair.planner_model} "
            f"evaluator={pair.evaluator_model} phase={pair.phase}"
        )


def run(args: argparse.Namespace):
    _configure_lmstudio_runtime_or_raise(args.lmstudio_base_url)
    logger = ExperimentResultLogger(csv_path=args.output_csv, xlsx_path=args.output_xlsx)
    done_keys = logger.load_completed_keys()

    pair_specs = _build_pair_specs(
        planner_models=list(args.planner_models),
        evaluator_models=list(args.evaluator_models),
        fixed_planner_model=args.fixed_planner_model,
        fixed_evaluator_model=args.fixed_evaluator_model,
    )
    _print_plan_summary(pair_specs=pair_specs, repeats=args.repeats)

    zone_objective = _build_zone_objective(args.zone_id)
    total_pairs = len(pair_specs)

    for pair in pair_specs:
        pending_repeats = []
        for repeat_idx in range(1, args.repeats + 1):
            key = ExperimentKey(
                planner_model=pair.planner_model,
                evaluator_model=pair.evaluator_model,
                repeat_idx=repeat_idx,
            )
            if key not in done_keys:
                pending_repeats.append(repeat_idx)
        if not pending_repeats:
            print(
                f"[PAIR-SKIP] pair_index={pair.pair_index} planner={pair.planner_model} "
                f"evaluator={pair.evaluator_model} all repeats already completed"
            )
            continue

        _wait_for_pair_confirmation(pair=pair, total_pairs=total_pairs, repeats=args.repeats)
        visible_models = _fetch_lmstudio_visible_models(
            args.lmstudio_base_url,
            timeout_sec=args.lmstudio_timeout_sec,
        )
        planner_model_id = _resolve_visible_model_id(pair.planner_model, visible_models)
        evaluator_model_id = _resolve_visible_model_id(pair.evaluator_model, visible_models)
        print(f"[CHECK] visible model ids = {visible_models}")
        print(
            f"[CHECK] resolved planner model id = {planner_model_id} | "
            f"resolved evaluator model id = {evaluator_model_id}"
        )

        for repeat_idx in range(1, args.repeats + 1):
            key = ExperimentKey(
                planner_model=pair.planner_model,
                evaluator_model=pair.evaluator_model,
                repeat_idx=repeat_idx,
            )
            if key in done_keys:
                print(
                    f"[SKIP] completed: pair_index={pair.pair_index} planner={pair.planner_model} "
                    f"evaluator={pair.evaluator_model} repeat={repeat_idx}"
                )
                continue

            if args.strict_run_model_check:
                visible_now = _fetch_lmstudio_visible_models(
                    args.lmstudio_base_url,
                    timeout_sec=args.lmstudio_timeout_sec,
                )
                planner_model_id = _resolve_visible_model_id(pair.planner_model, visible_now)
                evaluator_model_id = _resolve_visible_model_id(pair.evaluator_model, visible_now)

            print(
                f"[RUN] pair_index={pair.pair_index}/{total_pairs} repeat={repeat_idx}/{args.repeats} "
                f"provider=lmstudio base_url={os.environ.get('LMSTUDIO_BASE_URL')} "
                f"planner={planner_model_id} evaluator={evaluator_model_id} "
                f"phase={pair.phase} pipeline={args.pipeline_id} scene={args.scene_id} zone={args.zone_id}"
            )
            run_payload = _run_single_attempt(
                args=args,
                planner_model=planner_model_id,
                evaluator_model=evaluator_model_id,
                pair=pair,
                zone_objective=zone_objective,
            )
            abort_batch = bool(run_payload.get("_abort_batch"))
            cleanup_issue_text = str(run_payload.get("_cleanup_issues") or "")
            payload_for_log = {k: v for k, v in run_payload.items() if not str(k).startswith("_")}
            row = {
                "experiment_tag": args.experiment_tag,
                "pipeline_id": args.pipeline_id,
                "scenario_id": args.scene_id,
                "zone_id": args.zone_id,
                "repeat_idx": repeat_idx,
                "pair_index": pair.pair_index,
                "pair_label": pair.pair_label,
                "phase": pair.phase,
                "planner_model": planner_model_id,
                "evaluator_model": evaluator_model_id,
                **payload_for_log,
            }
            logger.append_result(row)
            done_keys.add(key)
            failure_reason = str(payload_for_log.get("failure_reason") or "")
            if _is_fatal_failure_reason(failure_reason):
                raise RuntimeError(
                    f"[STOP] fatal run failure: pair_index={pair.pair_index} repeat={repeat_idx} "
                    f"planner={planner_model_id} evaluator={evaluator_model_id} reason={failure_reason}"
                )
            if abort_batch:
                raise RuntimeError(
                    f"[STOP] cleanup verification failed after run "
                    f"(pair_index={pair.pair_index}, planner={planner_model_id}, "
                    f"evaluator={evaluator_model_id}, repeat={repeat_idx}): {cleanup_issue_text}"
                )


def parse_args() -> argparse.Namespace:
    default_output = os.path.expanduser("~/typefly_logs/model_grid_results.csv")
    parser = argparse.ArgumentParser(description="Run planner/evaluator pair experiments (resumable).")
    parser.add_argument("--planner-models", nargs="+", default=list(DEFAULT_PLANNER_MODEL_IDS))
    parser.add_argument("--evaluator-models", nargs="+", default=list(DEFAULT_EVALUATOR_MODEL_IDS))
    parser.add_argument("--fixed-planner-model", default=DEFAULT_FIXED_PLANNER_MODEL)
    parser.add_argument("--fixed-evaluator-model", default=DEFAULT_FIXED_EVALUATOR_MODEL)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEAT_COUNT)
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
        help="Before each repeat, require both planner/evaluator model IDs to be visible in /v1/models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
