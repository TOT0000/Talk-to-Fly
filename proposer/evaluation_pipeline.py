from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from proposer.error_classifier import classify_failure


class ScreeningEvaluator:
    name = "screening"
    evaluator_version = "screening_v1"

    def should_promote(self, candidate_summary: Dict, baseline_summary: Dict) -> bool:
        cand = dict(candidate_summary.get("metrics") or {})
        base = dict(baseline_summary.get("metrics") or {})
        # safety-first gate: collision and near-miss cannot be worse than baseline.
        return float(cand.get("collision_count_avg", 9999)) <= float(base.get("collision_count_avg", 9999)) and float(
            cand.get("near_miss_count_avg", 9999)
        ) <= float(base.get("near_miss_count_avg", 9999))


class FormalEvaluator:
    name = "formal"
    evaluator_version = "formal_v2"

    SAFETY_METRICS = ("collision_count", "near_miss_count")
    EFFECTIVENESS_METRICS = ("mission_success", "completion_time_mission_sec", "llm_call_count")

    def evaluate(self, *, candidate_id: str, baseline_id: str, candidate_runs: List[Dict], baseline_runs: List[Dict]) -> Dict:
        cand_scene = self._group_summary(candidate_runs, key="scene_id")
        base_scene = self._group_summary(baseline_runs, key="scene_id")
        cand_seed = self._group_summary(candidate_runs, key="seed")
        base_seed = self._group_summary(baseline_runs, key="seed")

        pairwise = self._pairwise_deltas(candidate_runs, baseline_runs)
        safety_report = self._safety_report(pairwise)
        dossier = self._failure_dossier(candidate_runs)

        decision, rationale = self._final_decision(safety_report=safety_report, pairwise=pairwise)

        summary = {
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
            "evaluator_version": self.evaluator_version,
            "decision": decision,
            "decision_rationale": rationale,
            "per_scene_metrics_summary": {
                "candidate": cand_scene,
                "baseline": base_scene,
            },
            "per_seed_metrics_summary": {
                "candidate": cand_seed,
                "baseline": base_seed,
            },
            "paired_win_loss_tie": pairwise["win_loss_tie"],
            "unsafe_vs_faster_tradeoff_summary": pairwise["unsafe_vs_faster_tradeoff_summary"],
            "improvements_concentration_markers": pairwise["improvements_concentration_markers"],
        }

        return {
            "formal_summary": summary,
            "formal_pairwise_deltas": pairwise,
            "formal_safety_report": safety_report,
            "formal_dossier": dossier,
        }

    def _group_summary(self, rows: List[Dict], *, key: str) -> Dict:
        buckets: Dict[str, List[Dict]] = {}
        for r in rows:
            k = str(r.get(key))
            buckets.setdefault(k, []).append(r)
        out = {}
        for k, group in buckets.items():
            out[k] = {
                "runs": len(group),
                "success_rate": self._mean_bool(group, "mission_success"),
                "collision_count_avg": self._mean_num(group, "collision_count"),
                "near_miss_count_avg": self._mean_num(group, "near_miss_count"),
                "completion_time_mission_sec_avg": self._mean_num(group, "completion_time_mission_sec", success_only=True),
                "llm_call_count_avg": self._mean_num(group, "llm_call_count"),
                "distribution": self._distribution(group),
            }
        return out

    def _distribution(self, rows: List[Dict]) -> Dict:
        dist = {}
        for metric in ("collision_count", "near_miss_count", "completion_time_mission_sec", "llm_call_count"):
            vals = [float(r.get(metric)) for r in rows if r.get(metric) is not None and (metric != "completion_time_mission_sec" or bool(r.get("mission_success")))]
            dist[metric] = self._stats(vals)
        successes = [1.0 if bool(r.get("mission_success")) else 0.0 for r in rows]
        dist["mission_success"] = self._stats(successes)
        return dist

    def _pairwise_deltas(self, candidate_runs: List[Dict], baseline_runs: List[Dict]) -> Dict:
        cand_map = {(str(r.get("scene_id")), int(r.get("seed", 0))): r for r in candidate_runs}
        base_map = {(str(r.get("scene_id")), int(r.get("seed", 0))): r for r in baseline_runs}
        keys = sorted(set(cand_map) & set(base_map))

        per_pair = []
        wins = losses = ties = 0
        unsafe_faster_cases: List[Dict] = []
        seed_rollup: Dict[int, List[Dict]] = {}
        scene_rollup: Dict[str, List[Dict]] = {}

        for scene, seed in keys:
            c = cand_map[(scene, seed)]
            b = base_map[(scene, seed)]
            delta = {
                "scene_id": scene,
                "seed": seed,
                "collision_delta": float(c.get("collision_count", 0)) - float(b.get("collision_count", 0)),
                "near_miss_delta": float(c.get("near_miss_count", 0)) - float(b.get("near_miss_count", 0)),
                "success_delta": (1.0 if bool(c.get("mission_success")) else 0.0) - (1.0 if bool(b.get("mission_success")) else 0.0),
                "completion_time_delta": self._num_delta(c.get("completion_time_mission_sec"), b.get("completion_time_mission_sec")),
                "llm_calls_delta": float(c.get("llm_call_count", 0)) - float(b.get("llm_call_count", 0)),
            }
            safety_better_or_equal = delta["collision_delta"] <= 0 and delta["near_miss_delta"] <= 0
            perf_better = (delta["success_delta"] > 0) or (delta["completion_time_delta"] is not None and delta["completion_time_delta"] < 0)
            if safety_better_or_equal and perf_better:
                wins += 1
                result = "win"
            elif (delta["collision_delta"] > 0) or (delta["near_miss_delta"] > 0):
                losses += 1
                result = "loss"
            else:
                ties += 1
                result = "tie"
            delta["paired_result"] = result
            per_pair.append(delta)
            seed_rollup.setdefault(seed, []).append(delta)
            scene_rollup.setdefault(scene, []).append(delta)

            faster = delta["completion_time_delta"] is not None and delta["completion_time_delta"] < 0
            unsafe = delta["collision_delta"] > 0 or delta["near_miss_delta"] > 0
            if faster and unsafe:
                unsafe_faster_cases.append({"scene_id": scene, "seed": seed, "delta": delta})

        return {
            "pairs_compared": len(keys),
            "pairing_key": ["scene_id", "seed"],
            "per_seed_delta_report": {str(k): self._delta_rollup(v) for k, v in seed_rollup.items()},
            "per_scene_delta_report": {k: self._delta_rollup(v) for k, v in scene_rollup.items()},
            "paired_rows": per_pair,
            "win_loss_tie": {"win": wins, "loss": losses, "tie": ties},
            "unsafe_vs_faster_tradeoff_summary": {
                "unsafe_but_faster_pair_count": len(unsafe_faster_cases),
                "unsafe_but_faster_pairs": unsafe_faster_cases,
            },
            "improvements_concentration_markers": self._concentration_markers(seed_rollup, scene_rollup),
        }

    def _concentration_markers(self, seed_rollup: Dict[int, List[Dict]], scene_rollup: Dict[str, List[Dict]]) -> Dict:
        seed_only = []
        for seed, rows in seed_rollup.items():
            if any((r["success_delta"] > 0 or (r["completion_time_delta"] is not None and r["completion_time_delta"] < 0)) for r in rows):
                if all((r["success_delta"] <= 0 and (r["completion_time_delta"] is None or r["completion_time_delta"] >= 0)) for s, rr in seed_rollup.items() if s != seed for r in rr):
                    seed_only.append(seed)

        scene_only = []
        for scene, rows in scene_rollup.items():
            if any((r["success_delta"] > 0 or (r["completion_time_delta"] is not None and r["completion_time_delta"] < 0)) for r in rows):
                if all((r["success_delta"] <= 0 and (r["completion_time_delta"] is None or r["completion_time_delta"] >= 0)) for sc, rr in scene_rollup.items() if sc != scene for r in rr):
                    scene_only.append(scene)

        return {
            "improvement_only_in_seeds": sorted(seed_only),
            "improvement_only_in_scenes": sorted(scene_only),
        }

    def _safety_report(self, pairwise: Dict) -> Dict:
        rows = list(pairwise.get("paired_rows") or [])
        unsafe_rows = [r for r in rows if r["collision_delta"] > 0 or r["near_miss_delta"] > 0]
        return {
            "pairs_compared": len(rows),
            "unsafe_regression_pairs": unsafe_rows,
            "unsafe_regression_pair_count": len(unsafe_rows),
            "safety_first_pass": len(unsafe_rows) == 0,
        }

    def _failure_dossier(self, candidate_runs: List[Dict]) -> Dict:
        failed = [r for r in candidate_runs if not bool(r.get("mission_success")) or bool((r.get("runtime_verification") or {}).get("passed") is False)]
        by_class: Dict[str, List[Dict]] = {}
        for r in failed:
            checks = dict(((r.get("runtime_verification") or {}).get("checks") or {}))
            err = classify_failure(verification=checks, run=r, metadata={})
            by_class.setdefault(err, []).append(
                {
                    "run_id": r.get("run_id"),
                    "scene_id": r.get("scene_id"),
                    "seed": r.get("seed"),
                    "run_status": r.get("run_status"),
                    "mission_success": r.get("mission_success"),
                }
            )
        return {
            "failure_total": len(failed),
            "by_error_class": {k: {"count": len(v), "examples": v[:5]} for k, v in sorted(by_class.items())},
            "scene_failure_counts": self._count_by_field(failed, "scene_id"),
            "seed_failure_counts": self._count_by_field(failed, "seed"),
        }

    def _final_decision(self, *, safety_report: Dict, pairwise: Dict) -> Tuple[str, Dict]:
        reasons: List[str] = []
        if not bool(safety_report.get("safety_first_pass")):
            reasons.append("safety_regression_detected")

        wins = int((pairwise.get("win_loss_tie") or {}).get("win", 0))
        losses = int((pairwise.get("win_loss_tie") or {}).get("loss", 0))
        if losses > 0:
            reasons.append("paired_losses_detected")
        if wins == 0:
            reasons.append("no_clear_paired_win")

        decision = "formal_pass" if not reasons else "formal_fail"
        return decision, {
            "safety_first_pass": bool(safety_report.get("safety_first_pass")),
            "wins": wins,
            "losses": losses,
            "blocking_reasons": reasons,
        }

    @staticmethod
    def _num_delta(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    @staticmethod
    def _mean_bool(rows: List[Dict], key: str) -> float:
        if not rows:
            return 0.0
        return mean(1.0 if bool(r.get(key)) else 0.0 for r in rows)

    @staticmethod
    def _mean_num(rows: List[Dict], key: str, *, success_only: bool = False) -> float | None:
        filtered = rows
        if success_only:
            filtered = [r for r in rows if bool(r.get("mission_success"))]
        vals = [float(r.get(key)) for r in filtered if r.get(key) is not None]
        if not vals:
            return None
        return mean(vals)

    @staticmethod
    def _stats(values: List[float]) -> Dict:
        if not values:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None}
        ordered = sorted(values)
        p90_idx = max(0, min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1)))))
        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": ordered[0],
            "max": ordered[-1],
            "p90": ordered[p90_idx],
        }

    @staticmethod
    def _delta_rollup(rows: List[Dict]) -> Dict:
        return {
            "pairs": len(rows),
            "collision_delta_avg": mean(r["collision_delta"] for r in rows) if rows else 0.0,
            "near_miss_delta_avg": mean(r["near_miss_delta"] for r in rows) if rows else 0.0,
            "success_delta_avg": mean(r["success_delta"] for r in rows) if rows else 0.0,
            "completion_time_delta_avg": mean([r["completion_time_delta"] for r in rows if r["completion_time_delta"] is not None]) if any(r["completion_time_delta"] is not None for r in rows) else None,
            "llm_calls_delta_avg": mean(r["llm_calls_delta"] for r in rows) if rows else 0.0,
            "wins": sum(1 for r in rows if r["paired_result"] == "win"),
            "losses": sum(1 for r in rows if r["paired_result"] == "loss"),
            "ties": sum(1 for r in rows if r["paired_result"] == "tie"),
        }

    @staticmethod
    def _count_by_field(rows: List[Dict], field: str) -> Dict:
        out: Dict[str, int] = {}
        for r in rows:
            key = str(r.get(field))
            out[key] = out.get(key, 0) + 1
        return out


def build_failure_dossier(*, candidate_id: str, baseline_id: str, run: Dict, metadata: Dict, verification: Dict) -> Dict:
    classification = classify_failure(verification=verification, run=run, metadata=metadata)
    report = dict(metadata.get("evaluate_error_report") or {})
    return {
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "scene": run.get("scene_id"),
        "zone": run.get("task_zone"),
        "seed": metadata.get("provenance", {}).get("seed", 0),
        "termination_reason": report.get("termination_reason") or report.get("failure_reason") or run.get("run_status"),
        "metrics": {
            "collision_count": run.get("collision_count"),
            "near_miss_count": run.get("near_miss_count"),
            "mission_success": run.get("mission_success"),
            "completion_time_mission_sec": run.get("completion_time_mission_sec"),
            "replan_count": run.get("replan_count"),
            "llm_calls": run.get("llm_call_count"),
        },
        "key_trace_paths": {
            "runtime": run.get("runtime_trace_path"),
            "planning": run.get("planning_trace_path"),
            "metadata": run.get("metadata_path"),
        },
        "error_classification": classification,
        "harness_vs_system_reason": (
            "runtime/semantics/trace failure indicates harness-system integration issue"
            if classification in {"runtime_wiring_error", "benchmark_semantics_error", "metric_or_benchmark_semantics_error", "trace_incomplete_error"}
            else "task-level mission failure with complete trace; likely genuine harness regression"
        ),
    }


def write_dossier(path: Path, dossier: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
