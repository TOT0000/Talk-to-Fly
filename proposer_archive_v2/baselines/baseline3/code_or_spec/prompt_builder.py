from __future__ import annotations


def build_prompt(stage: str, task_description: str, encoded_state: dict, spec: dict) -> dict:
    cfg = dict((spec or {}).get("prompt_builder") or {})
    return {
        "stage": stage,
        "task_description": task_description,
        "template_family": cfg.get("template_family"),
        "paragraph_order": cfg.get("paragraph_order", []),
        "include_example": bool(cfg.get("include_example", True)),
        "example_family": cfg.get("example_family"),
        "encoded_state": dict(encoded_state or {}),
    }
