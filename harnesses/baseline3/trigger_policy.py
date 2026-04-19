from __future__ import annotations


def should_trigger_replan(state: dict, memory: dict, spec: dict) -> tuple[bool, str]:
    cfg = dict((spec or {}).get("trigger_policy") or {})
    trigger_type = str(cfg.get("type") or "periodic")
    risk = float(state.get("predicted_collision_probability") or 0.0)

    if trigger_type == "event_predicted_collision_probability":
        threshold = float(cfg.get("threshold", 0.5))
        strictly_greater = bool(cfg.get("strictly_greater", True))
        hit = (risk > threshold) if strictly_greater else (risk >= threshold)
        return (hit, f"risk_{risk:.3f}_threshold_{threshold:.3f}")

    # periodic/hybrid behaviors are orchestrated by controller heartbeat loop;
    # here we just expose declarative policy metadata.
    return (False, "periodic_controller_driven")
