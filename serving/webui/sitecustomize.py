"""Startup fixes for WebUI trajectory-history retention.

When `serving/webui/typefly.py` is executed as a script, this directory is on
`sys.path`, so Python imports this module before running the WebUI.  The repair
below is intentionally narrow: it only expands the trajectory history window so
long response-driven runs do not lose the beginning of the UAV path.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_FULL_TRAJECTORY_POINTS = int(os.getenv("TYPEFLY_TRAJECTORY_HISTORY_MAX_POINTS", "100000"))


def _replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1) if old in text else text


def _repair_typefly_history_limit(repo_root: Path) -> None:
    target = repo_root / "serving" / "webui" / "typefly.py"
    if not target.exists():
        return
    try:
        text = target.read_text(encoding="utf-8")
    except Exception:
        return
    original = text
    text = _replace_once(
        text,
        "TRAJECTORY_HISTORY_MAX_POINTS = 5000",
        "TRAJECTORY_HISTORY_MAX_POINTS = int(os.getenv(\"TYPEFLY_TRAJECTORY_HISTORY_MAX_POINTS\", \"100000\"))",
    )
    if text != original:
        try:
            target.write_text(text, encoding="utf-8")
            print(f"[sitecustomize] expanded WebUI trajectory history limit in {target}")
        except Exception:
            pass


def _repair_controller_sampler_limit(repo_root: Path) -> None:
    target = repo_root / "controller" / "llm_controller.py"
    if not target.exists():
        return
    try:
        text = target.read_text(encoding="utf-8")
    except Exception:
        return
    original = text
    text = _replace_once(
        text,
        "        self._uav_trajectory_sampler_interval_sec: float = 0.1\n        self._uav_trajectory_sampler_active_during_run: bool = False",
        "        self._uav_trajectory_sampler_interval_sec: float = 0.1\n        self._uav_trajectory_history_max_points: int = int(os.getenv(\"TYPEFLY_TRAJECTORY_HISTORY_MAX_POINTS\", \"100000\"))\n        self._uav_trajectory_sampler_active_during_run: bool = False",
    )
    text = _replace_once(
        text,
        "                        if len(self._latest_uav_trajectory_points) > 5000:\n                            self._latest_uav_trajectory_points = self._latest_uav_trajectory_points[-5000:]",
        "                        history_limit = int(getattr(self, \"_uav_trajectory_history_max_points\", 100000) or 100000)\n                        if history_limit > 0 and len(self._latest_uav_trajectory_points) > history_limit:\n                            self._latest_uav_trajectory_points = self._latest_uav_trajectory_points[-history_limit:]",
    )
    if text != original:
        try:
            target.write_text(text, encoding="utf-8")
            print(f"[sitecustomize] expanded controller trajectory sampler limit in {target}")
        except Exception:
            pass


def _main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _repair_typefly_history_limit(repo_root)
    _repair_controller_sampler_limit(repo_root)


_main()
