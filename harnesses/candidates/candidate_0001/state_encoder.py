from __future__ import annotations


def encode_state(snapshot: dict, spec: dict) -> dict:
    cfg = dict((spec or {}).get("state_encoder") or {})
    include = list(cfg.get("include_fields") or [])
    out = {k: snapshot.get(k) for k in include}

    if cfg.get("include_risk_related"):
        out["predicted_collision_probability"] = snapshot.get("predicted_collision_probability")
        out["risk_summary"] = snapshot.get("risk_summary")

    if cfg.get("include_targets"):
        progress = dict(snapshot.get("benchmark_progress") or {})
        out["current_target_checkpoint"] = progress.get("current_target")
        out["remaining_checkpoints"] = progress.get("remaining")

    if cfg.get("include_geometry_flags"):
        out["blocked_workers_for_subgoal"] = snapshot.get("blocked_workers_for_subgoal")
        out["path_geometry_flags"] = snapshot.get("path_geometry_flags")

    style = str(cfg.get("summary_style") or "structured")
    out["summary_style"] = style
    return out
