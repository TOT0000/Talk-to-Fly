from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import subprocess
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Dict, List, Set

from controller.harness_sandbox import load_harness_sandbox_profile
from controller.harness_protocol import EVALUATION_PROTOCOLS, get_evaluation_protocol
from proposer.agent_tools import ProposerToolbox
from proposer.archive_reader import summarize_archive_for_proposer
from proposer.consistency import validate_candidate_contract_alignment
from proposer.evaluate_candidate import mark_pareto
from proposer.prompts import build_agent_next_action_prompt, build_self_review_prompt
from proposer.registry import ALLOWED_MUTATION_FILES, DEFAULT_EXCLUDED_PROPOSER_CANDIDATES, TRACKED_CONTRACT_FILES, HarnessRegistry, validate_candidate_boundary


def _next_candidate_id(candidates_dir: Path) -> str:
    ids: List[int] = []
    for d in candidates_dir.glob("candidate_*"):
        try:
            ids.append(int(str(d.name).split("_")[-1]))
        except Exception:
            continue
    return f"candidate_{(max(ids) + 1) if ids else 1:04d}"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_object(text: str) -> Dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_]*\n", "", raw)
        raw = raw.rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def _llm_json(llm, model_name: str, prompt: str) -> Dict:
    raw = llm.request(prompt=prompt, model_name=model_name, stream=False)
    return _extract_json_object(str(raw or ""))


def _available_agent_tools() -> List[Dict]:
    return [
        {"name": "list_harnesses", "args": {"kind": "all|baseline|candidate"}},
        {"name": "read_harness_spec", "args": {"harness_id": "str"}},
        {"name": "read_harness_code", "args": {"harness_id": "str", "file_name": "str"}},
        {"name": "diff_harnesses", "args": {"parent_harness": "str", "candidate_harness": "str", "file_name": "str"}},
        {"name": "list_runs", "args": {"harness_id": "str", "limit": "int"}},
        {"name": "read_run_metadata", "args": {"run_dir": "str"}},
        {"name": "read_evaluate_error_report", "args": {"run_dir": "str"}},
        {"name": "search_traces", "args": {"harness_id": "str", "needle": "str", "max_hits": "int"}},
        {"name": "read_trace_snippet", "args": {"trace_path": "str", "line_no": "int", "window": "int"}},
        {"name": "list_runtime_prompt_assets", "args": {"harness_id": "str"}},
        {"name": "read_runtime_prompt_asset", "args": {"harness_id": "str", "asset_name": "str?", "stage": "initial|replan|heartbeat?"}},
        {"name": "diff_runtime_prompt_assets", "args": {"harness_a": "str", "harness_b": "str", "stage": "initial|replan|heartbeat"}},
        {"name": "validate_candidate", "args": {"candidate_dir": "str", "parent_dir": "str"}},
        {"name": "smoke_check_candidate", "args": {"candidate_dir": "str"}},
    ]


def _execute_agent_tool(tools: ProposerToolbox, tool_name: str, tool_args: Dict) -> Dict:
    args = dict(tool_args or {})
    if tool_name == "list_harnesses":
        return {"result": tools.list_harnesses(kind=str(args.get("kind") or "all"), include_archived=bool(args.get("include_archived", False)))}
    if tool_name == "read_harness_spec":
        return {"result": tools.read_harness_spec(str(args.get("harness_id") or ""))}
    if tool_name == "read_harness_code":
        return {"result": tools.read_harness_code(str(args.get("harness_id") or ""), str(args.get("file_name") or ""))}
    if tool_name == "diff_harnesses":
        return {
            "result": tools.diff_harnesses(
                parent_harness=str(args.get("parent_harness") or ""),
                candidate_harness=str(args.get("candidate_harness") or ""),
                file_name=str(args.get("file_name") or ""),
            )
        }
    if tool_name == "list_runs":
        return {"result": tools.list_runs(harness_id=str(args.get("harness_id") or ""), limit=int(args.get("limit") or 12))}
    if tool_name == "read_run_metadata":
        return {"result": tools.read_run_metadata(run_dir=str(args.get("run_dir") or ""))}
    if tool_name == "read_evaluate_error_report":
        return {"result": tools.read_evaluate_error_report(run_dir=str(args.get("run_dir") or ""))}
    if tool_name == "search_traces":
        return {
            "result": tools.search_traces(
                harness_id=str(args.get("harness_id") or ""),
                needle=str(args.get("needle") or ""),
                max_hits=int(args.get("max_hits") or 12),
            )
        }
    if tool_name == "read_trace_snippet":
        return {
            "result": tools.read_trace_snippet(
                trace_path=str(args.get("trace_path") or ""),
                line_no=int(args.get("line_no") or 1),
                window=int(args.get("window") or 2),
            )
        }
    if tool_name == "list_runtime_prompt_assets":
        return {"result": tools.list_runtime_prompt_assets(harness_id=str(args.get("harness_id") or ""))}
    if tool_name == "read_runtime_prompt_asset":
        return {
            "result": tools.read_runtime_prompt_asset(
                harness_id=str(args.get("harness_id") or ""),
                asset_name=(str(args.get("asset_name")) if args.get("asset_name") else None),
                stage=(str(args.get("stage")) if args.get("stage") else None),
            )
        }
    if tool_name == "diff_runtime_prompt_assets":
        return {
            "result": tools.diff_runtime_prompt_assets(
                harness_a=str(args.get("harness_a") or ""),
                harness_b=str(args.get("harness_b") or ""),
                stage=str(args.get("stage") or "initial"),
            )
        }
    if tool_name == "validate_candidate":
        return {
            "result": tools.validate_candidate(
                candidate_dir=str(args.get("candidate_dir") or ""),
                parent_dir=str(args.get("parent_dir") or ""),
            )
        }
    if tool_name == "smoke_check_candidate":
        return {"result": tools.smoke_check_candidate(candidate_dir=str(args.get("candidate_dir") or ""))}
    raise ValueError(f"unknown tool: {tool_name}")


def _run_proposer_agent_loop(
    *,
    llm,
    proposer_model: str,
    focus_text: str,
    archive_summary: Dict,
    tools: ProposerToolbox,
    max_steps: int,
) -> Dict:
    transcript: List[Dict] = []
    tool_steps = 0
    run_evidence_tool_steps = 0
    run_evidence_hits = 0
    available_tools = _available_agent_tools()
    run_evidence_tools = {"list_runs", "search_traces", "read_run_metadata", "read_evaluate_error_report", "read_trace_snippet"}
    for step_idx in range(1, max(1, int(max_steps)) + 1):
        prompt = build_agent_next_action_prompt(
            step_idx=step_idx,
            max_steps=max(1, int(max_steps)),
            focus_text=focus_text,
            available_tools_json=json.dumps(available_tools, ensure_ascii=False, indent=2),
            archive_overview_json=json.dumps(
                {
                    "baseline_list": archive_summary.get("baseline_list", []),
                    "candidate_list": archive_summary.get("candidate_list", []),
                    "pareto_list": archive_summary.get("pareto_list", []),
                    "latest_harness": archive_summary.get("latest_harness"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            transcript_json=json.dumps(transcript[-12:], ensure_ascii=False, indent=2),
        )
        action_obj = _extract_json_object(str(llm.request(prompt=prompt, model_name=proposer_model, stream=False) or "{}"))
        action = str(action_obj.get("action") or "").strip().lower()

        if action == "tool_call":
            tool_name = str(action_obj.get("tool_name") or "").strip()
            tool_args = dict(action_obj.get("tool_args") or {})
            try:
                result = _execute_agent_tool(tools, tool_name, tool_args)
            except Exception as exc:
                result = {"error": str(exc)}
            transcript.append({"step": step_idx, "action": "tool_call", "tool_name": tool_name, "tool_args": tool_args, "observation": result})
            tool_steps += 1
            if tool_name in run_evidence_tools:
                run_evidence_tool_steps += 1
                payload = result.get("result")
                if isinstance(payload, list):
                    run_evidence_hits += len(payload)
                elif isinstance(payload, dict) and payload:
                    run_evidence_hits += 1
            continue

        if action == "final_proposal":
            if tool_steps <= 0:
                transcript.append({"step": step_idx, "action": "feedback", "message": "At least one tool_call is required before final_proposal."})
                continue
            if run_evidence_tool_steps <= 0:
                transcript.append({"step": step_idx, "action": "feedback", "message": "Use run evidence tools before final_proposal."})
                continue
            proposal = dict(action_obj.get("proposal") or {})
            smoke_block = dict(proposal.get("smoke_test_evidence_to_check") or {})
            limitation = str(smoke_block.get("evidence_limitations") or "").strip()
            if run_evidence_hits <= 0:
                smoke_block["evidence_limitations"] = limitation or "run evidence limited; diagnosis falls back to conservative spec/code-level hypothesis"
            else:
                smoke_block["evidence_limitations"] = limitation or "none"
            proposal["smoke_test_evidence_to_check"] = smoke_block
            return {
                "proposal": proposal,
                "agent_meta": {
                    "steps_used": step_idx,
                    "tool_steps": tool_steps,
                    "run_evidence_tool_steps": run_evidence_tool_steps,
                    "run_evidence_hits": run_evidence_hits,
                },
            }

        transcript.append({"step": step_idx, "action": "feedback", "message": f"Invalid action: {action}. Use tool_call or final_proposal."})

    raise RuntimeError(f"proposer agent exceeded max steps without valid final proposal: max_steps={max_steps}")


def _normalize_files_to_modify(proposal: Dict) -> List[str]:
    normalized_target_files: List[str] = []
    for name in list(proposal.get("files_to_create_or_modify") or []):
        base = Path(str(name)).name
        if base in ALLOWED_MUTATION_FILES and base not in normalized_target_files:
            normalized_target_files.append(base)
    return normalized_target_files


def _prepare_proposal_contract(proposal: Dict) -> Dict:
    required_keys = [
        "parent_harness",
        "candidate_id",
        "one_sentence_hypothesis",
        "weakness_being_addressed",
        "expected_tradeoff",
        "expected_runtime_effect",
        "hypothesis_target_modules",
        "proposer_note_text",
        "implementation_contract",
        "invariants",
        "sandbox_modules_to_modify",
        "changed_files",
        "runtime_wiring_plan",
        "smoke_test_evidence_to_check",
    ]
    for key in required_keys:
        if key not in proposal:
            raise ValueError(f"LLM proposer output missing required key: {key}")

    normalized_target_files = _normalize_files_to_modify(proposal)
    if "spec.json" not in normalized_target_files:
        normalized_target_files.append("spec.json")
    if "proposer_note.txt" not in normalized_target_files:
        normalized_target_files.append("proposer_note.txt")
    if not normalized_target_files:
        raise ValueError("LLM proposer output missing files_to_create_or_modify")
    if not any(name.endswith(".py") and name != "proposer_note.txt" for name in normalized_target_files):
        raise ValueError("Proposal must modify at least one sandbox code module (.py), not only spec/note.")

    implementation_contract = dict(proposal.get("implementation_contract") or {})
    for section in ["trigger_policy", "state_encoder", "prompt_builder"]:
        if section not in implementation_contract:
            raise ValueError(f"implementation_contract missing section: {section}")

    runtime_wiring_plan = dict(proposal.get("runtime_wiring_plan") or {})
    for key in [
        "sandbox_modules_changed",
        "runtime_load_path_or_entrypoint",
        "spec_manifest_loader_alignment",
        "legacy_sync_plan",
        "primary_runtime_entrypoints",
        "runtime_prompt_source_plan",
        "config_key_alignment_plan",
    ]:
        if key not in runtime_wiring_plan:
            raise ValueError(f"runtime_wiring_plan missing key: {key}")

    smoke_evidence = dict(proposal.get("smoke_test_evidence_to_check") or {})
    for key in [
        "trigger_logic_evidence",
        "state_features_evidence",
        "prompt_composer_evidence",
        "evidence_limitations",
        "evaluate_prompt_source_evidence",
    ]:
        if key not in smoke_evidence:
            raise ValueError(f"smoke_test_evidence_to_check missing key: {key}")

    invariants = list(proposal.get("invariants") or [])
    if not invariants:
        raise ValueError("LLM proposer output missing invariants")

    normalized = dict(proposal)
    normalized["files_to_create_or_modify"] = normalized_target_files
    normalized["implementation_contract"] = implementation_contract
    normalized["invariants"] = invariants
    normalized["sandbox_modules_to_modify"] = [Path(str(v)).name for v in list(proposal.get("sandbox_modules_to_modify") or [])]
    normalized["hypothesis_target_modules"] = [Path(str(v)).name for v in list(proposal.get("hypothesis_target_modules") or [])]
    if not normalized["hypothesis_target_modules"]:
        normalized["hypothesis_target_modules"] = list(normalized["sandbox_modules_to_modify"])
    normalized["changed_files"] = [Path(str(v)).name for v in list(proposal.get("changed_files") or [])]
    normalized["runtime_wiring_plan"] = runtime_wiring_plan
    normalized["smoke_test_evidence_to_check"] = smoke_evidence
    return normalized


def _git_parent_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out
    except Exception:
        return ""


def _build_parent_diff(parent_dir: Path, candidate_dir: Path, changed_files: Set[str]) -> str:
    chunks: List[str] = []
    for name in sorted(changed_files):
        left = (parent_dir / name).read_text(encoding="utf-8").splitlines(keepends=True) if (parent_dir / name).exists() else []
        right = (candidate_dir / name).read_text(encoding="utf-8").splitlines(keepends=True) if (candidate_dir / name).exists() else []
        chunks.extend(
            unified_diff(
                left,
                right,
                fromfile=f"parent/{name}",
                tofile=f"candidate/{name}",
            )
        )
    return "".join(chunks)


def _detect_changed_files(candidate_dir: Path, parent_dir: Path) -> Set[str]:
    changed: Set[str] = set()
    for name in TRACKED_CONTRACT_FILES:
        cp = candidate_dir / name
        pp = parent_dir / name
        if cp.exists() and not pp.exists():
            changed.add(name)
            continue
        if cp.exists() and pp.exists():
            if cp.read_text(encoding="utf-8") != pp.read_text(encoding="utf-8"):
                changed.add(name)
    return changed


def _build_grounded_note(*, contract: Dict, spec: Dict, changed_files: Set[str]) -> str:
    trigger = dict(spec.get("trigger_policy") or {})
    state = dict(spec.get("state_encoder") or {})
    prompt_cfg = dict(spec.get("prompt_builder") or {})
    return "\n".join(
        [
            f"Parent harness: {contract.get('parent_harness')}",
            f"Hypothesis: {contract.get('one_sentence_hypothesis')}",
            f"Weakness addressed: {contract.get('weakness_being_addressed')}",
            f"Expected tradeoff: {contract.get('expected_tradeoff')}",
            f"Expected runtime effect: {contract.get('expected_runtime_effect')}",
            (
                "Implemented trigger_policy: "
                f"type={trigger.get('type')}, heartbeat_seconds={trigger.get('heartbeat_seconds')}, threshold={trigger.get('threshold')}"
            ),
            (
                "Implemented state_encoder: "
                f"summary_style={state.get('summary_style')}, include_risk_related={state.get('include_risk_related')}"
            ),
            (
                "Implemented prompt_builder: "
                f"template_family={prompt_cfg.get('template_family')}, include_example={prompt_cfg.get('include_example')}"
            ),
            f"Changed files: {sorted(changed_files)}",
            f"Hypothesis target modules: {contract.get('hypothesis_target_modules')}",
            f"Runtime-effect changed files: {contract.get('runtime_effect_changed_files')}",
            f"Supporting generated files: {contract.get('supporting_generated_files')}",
            f"Contract invariants: {contract.get('invariants')}",
            f"Sandbox modules: {contract.get('sandbox_modules_to_modify')}",
        ]
    )


def _default_sandbox_file_content(name: str) -> str:
    templates = {
        "state_features.py": (
            "from __future__ import annotations\n\n"
            "def encode_state_features(snapshot: dict, spec: dict) -> dict:\n"
            "    cfg = dict((spec or {}).get('state_encoder') or {})\n"
            "    include = list(cfg.get('include_fields') or [])\n"
            "    return {k: snapshot.get(k) for k in include}\n"
        ),
        "trigger_logic.py": (
            "from __future__ import annotations\n\n"
            "def should_trigger_replan(state: dict, memory: dict, spec: dict) -> tuple[bool, str]:\n"
            "    cfg = dict((spec or {}).get('trigger_policy') or {})\n"
            "    threshold = cfg.get('threshold')\n"
            "    risk = float(state.get('predicted_collision_probability') or 0.0)\n"
            "    if threshold is None:\n"
            "        return (False, 'sandbox_no_threshold')\n"
            "    hit = risk >= float(threshold)\n"
            "    return (hit, f'risk_{risk:.3f}_threshold_{float(threshold):.3f}')\n"
        ),
        "prompt_composer.py": (
            "from __future__ import annotations\n\n"
            "def compose_prompt_context(stage: str, task_description: str, encoded_state: dict, snapshot: dict, spec: dict) -> str:\n"
            "    return f'stage={stage}; encoded_state={encoded_state}'\n"
        ),
        "archive_selector.py": (
            "from __future__ import annotations\n\n"
            "def select_entries(entries: list[dict], max_entries: int) -> list[dict]:\n"
            "    return list(entries)[-int(max_entries):]\n\n"
            "def select_trace_snippets(snippets: list[dict], max_traces: int) -> list[dict]:\n"
            "    return list(snippets)[:int(max_traces)]\n"
        ),
        "validator_rules.py": (
            "from __future__ import annotations\n\n"
            "def runtime_effect_modules() -> list[str]:\n"
            "    return ['state_features.py', 'trigger_logic.py', 'prompt_composer.py']\n"
        ),
    }
    return templates.get(name, "")


def _run_candidate_smoke_checks(candidate_dir: Path) -> None:
    for path in sorted(candidate_dir.glob('*.py')):
        py_compile.compile(str(path), doraise=True)


def _run_import_checks(candidate_dir: Path) -> None:
    import importlib.util

    for path in sorted(candidate_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"candidate_mod_{path.stem}", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"import loader unavailable: {path.name}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _runtime_effect_modules_for_candidate(candidate_dir: Path) -> Set[str]:
    default = {"state_features.py", "trigger_logic.py", "prompt_composer.py", "state_encoder.py", "trigger_policy.py", "prompt_builder.py"}
    rules_path = candidate_dir / "validator_rules.py"
    if not rules_path.exists():
        return default
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate_validator_rules_meta", str(rules_path))
        if spec is None or spec.loader is None:
            return default
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "runtime_effect_modules", None)
        if callable(fn):
            out = {Path(str(x)).name for x in list(fn() or [])}
            if out:
                return out
    except Exception:
        return default
    return default


def _build_change_semantics(*, proposal_contract: Dict, changed_files: Set[str], runtime_effect_modules: Set[str]) -> Dict:
    changed = {Path(str(x)).name for x in set(changed_files or set())}
    primary_claims = [Path(str(x)).name for x in list(proposal_contract.get("hypothesis_target_modules") or proposal_contract.get("sandbox_modules_to_modify") or [])]
    primary_claims = sorted([x for x in primary_claims if x])
    runtime_effect_changed = sorted(changed.intersection(set(runtime_effect_modules)).intersection(set(primary_claims)))
    supporting = sorted(changed - set(runtime_effect_changed))
    return {
        "hypothesis_target_modules": primary_claims,
        "runtime_effect_changed_files": runtime_effect_changed,
        "supporting_generated_files": supporting,
        "full_diff_files": sorted(changed),
        "editable_allowed_files": sorted(ALLOWED_MUTATION_FILES),
    }


def _runtime_wiring_smoke_verification(*, candidate_dir: Path, proposal_contract: Dict, changed_files: Set[str]) -> Dict:
    try:
        profile = load_harness_sandbox_profile(str((candidate_dir / "spec.json").as_posix()))
    except Exception as exc:
        changed = {Path(str(x)).name for x in set(changed_files or set())}
        claimed = {Path(str(x)).name for x in set(proposal_contract.get("sandbox_modules_to_modify") or [])}
        return {
            "passed": False,
            "loader_entrypoint": "controller.harness_sandbox.load_harness_sandbox_profile",
            "loaded_trigger_module": "",
            "loaded_trigger_function": "",
            "loaded_state_module": "",
            "loaded_state_function": "",
            "loaded_prompt_module": "",
            "loaded_prompt_function": "",
            "candidate_trigger_module_claim": "trigger_logic.py" if "trigger_logic.py" in claimed else "",
            "candidate_state_module_claim": "state_features.py" if "state_features.py" in claimed else "",
            "candidate_prompt_module_claim": "prompt_composer.py" if "prompt_composer.py" in claimed else "",
            "trigger_alignment_ok": False if "trigger_logic.py" in changed else None,
            "state_alignment_ok": False if "state_features.py" in changed else None,
            "prompt_alignment_ok": False if "prompt_composer.py" in changed else None,
            "changed_files": sorted(changed),
            "manifest_active_sandbox_modules": [],
            "notes": [f"runtime wiring loader failed: {exc}"],
        }
    trigger = dict(profile.get("trigger_logic") or {})
    state = dict(profile.get("state_features") or {})
    prompt = dict(profile.get("prompt_composer") or {})
    claimed = {
        Path(str(x)).name
        for x in list(proposal_contract.get("hypothesis_target_modules") or proposal_contract.get("sandbox_modules_to_modify") or [])
    }
    changed = {Path(str(x)).name for x in set(changed_files or set())}
    manifest_active = {Path(str(x)).name for x in list(((profile.get("spec") or {}).get("manifest") or {}).get("active_sandbox_modules") or [])}

    def _fn_name(obj) -> str:
        return getattr(obj, "__name__", "") if callable(obj) else ""

    checks = {
        "trigger": {
            "module": "trigger_logic.py",
            "loaded_module": Path(str(trigger.get("module") or "")).name,
            "loaded_function": _fn_name(trigger.get("fn")),
            "claim": "trigger_logic.py" if "trigger_logic.py" in claimed else "",
        },
        "state": {
            "module": "state_features.py",
            "loaded_module": Path(str(state.get("module") or "")).name,
            "loaded_function": _fn_name(state.get("fn")),
            "claim": "state_features.py" if "state_features.py" in claimed else "",
        },
        "prompt": {
            "module": "prompt_composer.py",
            "loaded_module": Path(str(prompt.get("module") or "")).name,
            "loaded_function": _fn_name(prompt.get("fn")),
            "claim": "prompt_composer.py" if "prompt_composer.py" in claimed else "",
        },
    }

    notes: List[str] = []

    def _alignment(item: Dict) -> bool | None:
        expected = item["module"]
        is_claimed = bool(item["claim"])
        if not is_claimed:
            if expected in changed:
                notes.append(f"{expected}: runtime-effect file changed but not in primary hypothesis_target_modules")
            notes.append(f"{expected}: not in primary claim; skipped strict alignment check")
            return None
        module_ok = item["loaded_module"] == expected
        fn_ok = bool(item["loaded_function"])
        active_ok = expected in manifest_active if manifest_active else True
        changed_ok = expected in changed
        if not module_ok:
            notes.append(f"{expected}: runtime loaded {item['loaded_module']} (expected {expected})")
        if not fn_ok:
            notes.append(f"{expected}: runtime function not found")
        if not active_ok:
            notes.append(f"{expected}: manifest.active_sandbox_modules missing expected module")
        if not changed_ok:
            notes.append(f"{expected}: module was claimed but not detected in changed_files")
        return bool(module_ok and fn_ok and active_ok and changed_ok)

    trigger_ok = _alignment(checks["trigger"])
    state_ok = _alignment(checks["state"])
    prompt_ok = _alignment(checks["prompt"])
    all_flags = [x for x in [trigger_ok, state_ok, prompt_ok] if x is not None]
    passed = all(all_flags) if all_flags else True
    if passed and all_flags:
        notes.append("runtime wiring alignment passed for all claimed/changed sandbox lines")

    verification = {
        "passed": bool(passed),
        "loader_entrypoint": "controller.harness_sandbox.load_harness_sandbox_profile",
        "loaded_trigger_module": checks["trigger"]["loaded_module"],
        "loaded_trigger_function": checks["trigger"]["loaded_function"],
        "loaded_state_module": checks["state"]["loaded_module"],
        "loaded_state_function": checks["state"]["loaded_function"],
        "loaded_prompt_module": checks["prompt"]["loaded_module"],
        "loaded_prompt_function": checks["prompt"]["loaded_function"],
        "candidate_trigger_module_claim": checks["trigger"]["claim"],
        "candidate_state_module_claim": checks["state"]["claim"],
        "candidate_prompt_module_claim": checks["prompt"]["claim"],
        "trigger_alignment_ok": trigger_ok,
        "state_alignment_ok": state_ok,
        "prompt_alignment_ok": prompt_ok,
        "changed_files": sorted(changed),
        "manifest_active_sandbox_modules": sorted(manifest_active),
        "notes": notes,
    }
    return verification


def _normalize_generated_text(text: str) -> str:
    out = str(text or "").strip()
    if out.startswith("```"):
        lines = out.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        out = "\n".join(lines).strip()
    return out


def _guess_error_file_from_text(text: str) -> str:
    raw = str(text or "")
    m = re.search(r"([A-Za-z0-9_./-]+\.py)", raw)
    if not m:
        return ""
    return Path(m.group(1)).name


def _build_file_generation_prompt(
    *,
    parent_harness_id: str,
    parent_spec: Dict,
    parent_file_name: str,
    parent_file_content: str,
    proposal: Dict,
    external_feedback_context: Dict | None = None,
) -> str:
    feedback_block = ""
    if external_feedback_context:
        feedback_block = (
            "\n\nExternal structured feedback context (must be addressed in revisions):\n"
            f"{json.dumps(external_feedback_context, ensure_ascii=False, indent=2)}\n"
        )
    return (
        "You are generating one bounded harness file for UAV harness optimization.\n"
        "Allowed harness boundary files: spec.json, state_features.py, trigger_logic.py, prompt_composer.py, archive_selector.py, validator_rules.py, state_encoder.py, trigger_policy.py, prompt_builder.py, proposer_note.txt, README.md\n"
        "Runtime-first rule: Prioritize sandbox runtime-effect modules, actual runtime prompt assets, and wiring alignment.\n"
        "Primary editable targets are:\n"
        "- state_features.py\n"
        "- trigger_logic.py\n"
        "- prompt_composer.py\n"
        "- archive_selector.py\n"
        "- validator_rules.py\n"
        "Legacy files (state_encoder.py, trigger_policy.py, prompt_builder.py) are compatibility wrappers/metadata mirrors, not primary targets.\n"
        "Do not create legacy-vs-sandbox ambiguity.\n"
        "Do not generate a candidate that is only a near-duplicate of prior candidates unless the proposal contract explicitly justifies that narrow direction with evidence.\n"
        "If the proposal targets planning prompt content:\n"
        "- make the prompt change concrete and behaviorally meaningful\n"
        "- do not only change template metadata\n"
        "- ensure the changed prompt is the one evaluation runtime will actually render and send\n"
        "If the proposal targets trigger behavior:\n"
        "- ensure the proposed config keys are actually read by the runtime-loaded trigger module\n"
        "- do not declare a trigger improvement that depends on unused config keys\n"
        "You must output ONLY the full content of the requested file, with no markdown fences.\n"
        "Do not modify simulator / PX4 / controller / executor / collision math / checkpoint rules.\n\n"
        f"Parent harness: {parent_harness_id}\n"
        f"Parent spec:\n{json.dumps(parent_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Proposal contract:\n{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
        f"Requested file: {parent_file_name}\n"
        "Current parent file content:\n"
        f"{parent_file_content}\n\n"
        f"{feedback_block}"
        "Generate improved content aligned with:\n"
        "- the proposal hypothesis\n"
        "- the identified failure mode in baselines and prior candidates\n"
        "- runtime wiring alignment\n"
        "- the declared primary hypothesis modules\n"
        "- the declared supporting/generated artifacts\n"
        "- the actual evaluation prompt / trigger / state execution path\n"
        "If requested file is spec.json, ensure valid JSON and keep candidate lineage metadata."
    )


def propose_next_candidate(
    repo_root: Path,
    note: str = "",
    focus_text: str = "Improve safety-aware replan timing while avoiding unnecessary detours.",
    allow_fallback_heuristic: bool = False,
    max_revision_rounds: int = 0,
    max_agent_steps: int = 10,
    parent_harness_override: str | None = None,
    external_feedback_context: Dict | None = None,
    proposal_iteration_id: str | None = None,
    proposal_revision_round: int = 0,
    validator_round_index: int = 0,
    evaluator_round_index: int = 0,
    proposer_loop_status: str = "proposed",
) -> Path:
    repo_root = Path(repo_root)
    reg = HarnessRegistry(repo_root)
    baselines = reg.list_baselines()
    if not baselines:
        raise RuntimeError("No baselines found in harnesses/")

    archive_summary = summarize_archive_for_proposer(repo_root, repo_root / "proposer_archive_v2")
    tools = ProposerToolbox(repo_root=repo_root, archive_root=repo_root / "proposer_archive_v2")

    from controller.llm_wrapper import LLMWrapper, MODEL_NAME

    llm = LLMWrapper(temperature=0.1)
    proposer_model = str(os.getenv("TYPEFLY_PROPOSER_MODEL", "gpt-4.1")).strip() or "gpt-4.1"
    agent_meta: Dict = {}
    if proposer_model.lower().startswith("gpt-") and (not os.getenv("OPENAI_API_KEY")) and (not allow_fallback_heuristic):
        raise RuntimeError(
            "Proposer requires OpenAI provider for GPT models. "
            "Set OPENAI_API_KEY or override TYPEFLY_PROPOSER_MODEL to a provider-compatible model."
        )
    try:
        agent_output = _run_proposer_agent_loop(
            llm=llm,
            proposer_model=proposer_model or MODEL_NAME,
            focus_text=focus_text,
            archive_summary=archive_summary,
            tools=tools,
            max_steps=max_agent_steps,
        )
        agent_meta = dict(agent_output.get("agent_meta") or {})
        proposal = _prepare_proposal_contract(dict(agent_output.get("proposal") or {}))
    except Exception:
        if not allow_fallback_heuristic:
            raise
        # conservative fallback path (explicitly marked)
        proposal = _prepare_proposal_contract({
            "parent_harness": "baseline3",
            "candidate_id": "",
            "one_sentence_hypothesis": "Conservative fallback due to proposer LLM failure.",
            "weakness_being_addressed": "LLM unavailable during proposal call",
            "expected_tradeoff": "Minimal structured change",
            "expected_runtime_effect": "Preserve baseline runtime behavior while keeping proposer alive.",
            "hypothesis_target_modules": ["trigger_logic.py"],
            "files_to_create_or_modify": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
            "proposer_note_text": "Fallback proposal generated because LLM proposer call failed.",
            "sandbox_modules_to_modify": ["trigger_logic.py"],
            "changed_files": ["spec.json", "trigger_logic.py", "proposer_note.txt"],
            "implementation_contract": {
                "trigger_policy": {},
                "state_encoder": {},
                "prompt_builder": {},
            },
            "runtime_wiring_plan": {
                "sandbox_modules_changed": ["trigger_logic.py"],
                "runtime_load_path_or_entrypoint": "controller.harness_sandbox runtime sandbox loader",
                "spec_manifest_loader_alignment": "spec.sandbox + spec.manifest.active_sandbox_modules include trigger_logic.py",
                "legacy_sync_plan": "none",
                "primary_runtime_entrypoints": ["controller.harness_sandbox.load_harness_sandbox_profile"],
                "runtime_prompt_source_plan": "prompt assets remain inherited from parent; no prompt module claim in this fallback",
                "config_key_alignment_plan": "trigger logic fallback does not introduce new trigger config keys",
            },
            "smoke_test_evidence_to_check": {
                "trigger_logic_evidence": "smoke/import checks and runtime_metadata.changed_files include trigger_logic.py",
                "state_features_evidence": "not changed in this fallback candidate",
                "prompt_composer_evidence": "not changed in this fallback candidate",
                "evidence_limitations": "fallback path due to proposer LLM failure",
                "evaluate_prompt_source_evidence": "planning trace evaluate_prompt_source should continue to report inherited baseline prompt source",
            },
            "invariants": [
                "proposal_contract files must match actual changed files",
                "spec trigger policy must align with trigger_policy.py behavior",
            ],
        })
        agent_meta = {"fallback_used": True, "reason": "agent_loop_or_llm_failure"}

    parent_id = str(parent_harness_override or proposal.get("parent_harness") or "").strip()
    if not parent_id:
        raise ValueError("LLM proposer output missing parent_harness")
    if parent_id in DEFAULT_EXCLUDED_PROPOSER_CANDIDATES and os.getenv("TYPEFLY_ALLOW_EXCLUDED_PARENT", "0").strip() != "1":
        raise ValueError(f"parent_harness {parent_id} is excluded from default proposer parent pool")

    parent_entry = reg.get(parent_id)

    asked_candidate_id = str(proposal.get("candidate_id") or "").strip()
    if re.fullmatch(r"candidate_\d{4}", asked_candidate_id) and not (reg.candidates_dir / asked_candidate_id).exists():
        candidate_id = asked_candidate_id
    else:
        candidate_id = _next_candidate_id(reg.candidates_dir)

    candidate_dir = reg.candidates_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)

    files_to_modify = _normalize_files_to_modify(proposal)
    if not files_to_modify:
        raise ValueError("proposal_contract.files_to_create_or_modify must be non-empty")

    # Start from parent snapshot for deterministic bounded edits.
    for name in [
        "spec.json",
        "state_encoder.py",
        "trigger_policy.py",
        "prompt_builder.py",
        "state_features.py",
        "trigger_logic.py",
        "prompt_composer.py",
        "archive_selector.py",
        "validator_rules.py",
    ]:
        src = parent_entry.dir_path / name
        if src.exists():
            shutil.copy2(src, candidate_dir / name)
    # Ensure runtime-effect sandbox modules always exist for wiring checks,
    # even when parent harness is legacy-only (state_encoder/trigger_policy/prompt_builder).
    scaffolded_runtime_modules: List[str] = []
    for required in ["state_features.py", "trigger_logic.py", "prompt_composer.py"]:
        path = candidate_dir / required
        if not path.exists():
            fallback = _default_sandbox_file_content(required)
            if fallback:
                path.write_text(fallback, encoding="utf-8")
                scaffolded_runtime_modules.append(required)

    if scaffolded_runtime_modules:
        existing_declared = {Path(str(x)).name for x in list(proposal.get("files_to_create_or_modify") or [])}
        proposal["files_to_create_or_modify"] = list(existing_declared.union(set(scaffolded_runtime_modules)))
        existing_changed = {Path(str(x)).name for x in list(proposal.get("changed_files") or [])}
        proposal["changed_files"] = list(existing_changed.union(set(scaffolded_runtime_modules)))

    parent_spec = _load_json(parent_entry.dir_path / "spec.json")

    normalized_target_files = [Path(name).name for name in files_to_modify if Path(name).name in ALLOWED_MUTATION_FILES]
    for scaffold_name in scaffolded_runtime_modules:
        if scaffold_name not in normalized_target_files:
            normalized_target_files.append(scaffold_name)

    for target_file in normalized_target_files:
        if target_file in {"proposer_note.txt", "README.md"}:
            continue
        if not (candidate_dir / target_file).exists():
            fallback = _default_sandbox_file_content(target_file)
            if fallback:
                (candidate_dir / target_file).write_text(fallback, encoding="utf-8")
        parent_file_content = (candidate_dir / target_file).read_text(encoding="utf-8") if (candidate_dir / target_file).exists() else ""
        file_prompt = _build_file_generation_prompt(
            parent_harness_id=parent_id,
            parent_spec=parent_spec,
            parent_file_name=target_file,
            parent_file_content=parent_file_content,
            proposal=proposal,
            external_feedback_context=external_feedback_context,
        )
        generated = _normalize_generated_text(str(llm.request(prompt=file_prompt, model_name=proposer_model or MODEL_NAME, stream=False) or "").strip())
        if target_file == "spec.json":
            spec_obj = _extract_json_object(generated)
            spec_obj["id"] = candidate_id
            spec_obj["kind"] = "candidate"
            spec_obj["parent"] = parent_id
            lineage = dict(spec_obj.get("lineage") or {})
            lineage.update(
                {
                    "parent_id": parent_id,
                    "parent_kind": "baseline" if parent_id.startswith("baseline") else "candidate",
                    "derived_from": parent_id,
                }
            )
            spec_obj["lineage"] = lineage
            spec_obj.setdefault("mutation", {})
            spec_obj["mutation"]["type"] = "llm_agent_driven"
            (candidate_dir / target_file).write_text(json.dumps(spec_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            (candidate_dir / target_file).write_text(generated + "\n", encoding="utf-8")

    try:
        # Ensure mandatory spec metadata even if LLM skipped it.
        spec = _load_json(candidate_dir / "spec.json")
        spec["id"] = candidate_id
        spec["kind"] = "candidate"
        spec["parent"] = parent_id
        spec.setdefault("lineage", {})
        spec["lineage"]["parent_id"] = parent_id
        spec["lineage"]["parent_kind"] = "baseline" if parent_id.startswith("baseline") else "candidate"
        spec["lineage"]["derived_from"] = parent_id
        spec.setdefault("proposal_contract", {})
        spec.setdefault("sandbox", {})
        spec["sandbox"] = {
            "state_features": {"module": "state_features.py", "enabled": True},
            "trigger_logic": {"module": "trigger_logic.py", "enabled": True},
            "prompt_composer": {"module": "prompt_composer.py", "enabled": True},
            "archive_selector": {"module": "archive_selector.py", "enabled": True},
            "validator_rules": {"module": "validator_rules.py", "enabled": True},
            "deprecated_options": {
                "prompt_builder.paragraph_order": "deprecated_runtime_no_effect",
                "prompt_builder.stages": "deprecated_runtime_no_effect",
                "state_encoder.summary_style": "deprecated_metadata_only",
            },
        }
        spec.setdefault("manifest", {})
        spec["manifest"] = {
            "lineage": dict(spec.get("lineage") or {}),
            "active_sandbox_modules": [
                "state_features.py",
                "trigger_logic.py",
                "prompt_composer.py",
                "archive_selector.py",
                "validator_rules.py",
            ],
            "evidence_pointers": {
                "archive_index": "proposer_archive_v2/index.json",
                "trace_snippet_count": len(archive_summary.get("trace_snippets", [])),
            },
        }
        spec["proposal_contract"] = {
            "parent_harness": parent_id,
            "one_sentence_hypothesis": str(proposal.get("one_sentence_hypothesis") or ""),
            "weakness_being_addressed": str(proposal.get("weakness_being_addressed") or ""),
            "expected_tradeoff": str(proposal.get("expected_tradeoff") or ""),
            "expected_runtime_effect": str(proposal.get("expected_runtime_effect") or ""),
            "sandbox_modules_to_modify": list(proposal.get("sandbox_modules_to_modify") or []),
            "hypothesis_target_modules": list(proposal.get("hypothesis_target_modules") or proposal.get("sandbox_modules_to_modify") or []),
            "files_to_create_or_modify": normalized_target_files,
            "changed_files": list(proposal.get("changed_files") or []),
            "runtime_wiring_plan": dict(proposal.get("runtime_wiring_plan") or {}),
            "smoke_test_evidence_to_check": dict(proposal.get("smoke_test_evidence_to_check") or {}),
            "implementation_contract": dict(proposal.get("implementation_contract") or {}),
            "invariants": list(proposal.get("invariants") or []),
        }
        (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

        proposer_note_text = str(proposal.get("proposer_note_text") or "").strip()
        grounded_note = _build_grounded_note(contract=spec["proposal_contract"], spec=spec, changed_files=set())
        final_note = note or proposer_note_text or grounded_note
        if not note:
            final_note = grounded_note
        (candidate_dir / "proposer_note.txt").write_text(final_note + "\n", encoding="utf-8")

        changed_files = _detect_changed_files(candidate_dir, parent_entry.dir_path)
        wiring_verification = _runtime_wiring_smoke_verification(
            candidate_dir=candidate_dir,
            proposal_contract=_load_json(candidate_dir / "spec.json").get("proposal_contract", {}),
            changed_files=changed_files,
        )
        (candidate_dir / "runtime_wiring_verification.json").write_text(
            json.dumps(wiring_verification, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        spec = _load_json(candidate_dir / "spec.json")
        change_semantics = _build_change_semantics(
            proposal_contract=dict(spec.get("proposal_contract") or {}),
            changed_files=changed_files,
            runtime_effect_modules=_runtime_effect_modules_for_candidate(candidate_dir),
        )
        spec["proposal_contract"]["full_diff_files"] = list(change_semantics["full_diff_files"])
        spec["proposal_contract"]["runtime_effect_changed_files"] = list(change_semantics["runtime_effect_changed_files"])
        spec["proposal_contract"]["supporting_generated_files"] = list(change_semantics["supporting_generated_files"])
        spec["proposal_contract"]["hypothesis_target_modules"] = list(change_semantics["hypothesis_target_modules"])
        parent_commit = _git_parent_commit(repo_root)
        spec["runtime_metadata"] = {
            "parent_harness": parent_id,
            "parent_commit": parent_commit,
            "candidate_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "changed_files": sorted(changed_files),
            "full_diff_files": list(change_semantics["full_diff_files"]),
            "runtime_effect_changed_files": list(change_semantics["runtime_effect_changed_files"]),
            "supporting_generated_files": list(change_semantics["supporting_generated_files"]),
            "hypothesis_target_modules": list(change_semantics["hypothesis_target_modules"]),
            "diff_path": "parent_diff.patch",
            "candidate_branch_hint": f"candidate/{candidate_id}",
            "agent_loop_meta": dict(agent_meta or {}),
            "runtime_wiring_verification_path": "runtime_wiring_verification.json",
            "runtime_wiring_verification_passed": bool(wiring_verification.get("passed")),
            "proposal_iteration_id": str(proposal_iteration_id or ""),
            "proposal_revision_round": int(proposal_revision_round),
            "validator_round_index": int(validator_round_index),
            "evaluator_round_index": int(evaluator_round_index),
            "proposer_loop_status": str(proposer_loop_status or "proposed"),
        }
        (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        (candidate_dir / "parent_diff.patch").write_text(
            _build_parent_diff(parent_entry.dir_path, candidate_dir, changed_files),
            encoding="utf-8",
        )
        (candidate_dir / "proposer_tool_audit.json").write_text(
            json.dumps({"events": tools.audit_log}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        spec = _load_json(candidate_dir / "spec.json")
        spec["runtime_metadata"]["proposer_tool_audit_path"] = "proposer_tool_audit.json"
        spec["runtime_metadata"]["proposer_tool_event_count"] = len(tools.audit_log)
        (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        (candidate_dir / "parent_diff.patch").write_text(
            _build_parent_diff(parent_entry.dir_path, candidate_dir, changed_files),
            encoding="utf-8",
        )
        (candidate_dir / "proposer_tool_audit.json").write_text(
            json.dumps({"events": tools.audit_log}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        spec = _load_json(candidate_dir / "spec.json")
        spec["runtime_metadata"]["proposer_tool_audit_path"] = "proposer_tool_audit.json"
        spec["runtime_metadata"]["proposer_tool_event_count"] = len(tools.audit_log)
        (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        (candidate_dir / "parent_diff.patch").write_text(
            _build_parent_diff(parent_entry.dir_path, candidate_dir, changed_files),
            encoding="utf-8",
        )

        if not note:
            refreshed_note = _build_grounded_note(contract=spec["proposal_contract"], spec=spec, changed_files=changed_files)
            (candidate_dir / "proposer_note.txt").write_text(refreshed_note + "\n", encoding="utf-8")
            final_note = refreshed_note

        last_error = ""
        for _round in range(max(0, int(max_revision_rounds)) + 1):
            changed_files = _detect_changed_files(candidate_dir, parent_entry.dir_path)
            wiring_verification = _runtime_wiring_smoke_verification(
                candidate_dir=candidate_dir,
                proposal_contract=_load_json(candidate_dir / "spec.json").get("proposal_contract", {}),
                changed_files=changed_files,
            )
            (candidate_dir / "runtime_wiring_verification.json").write_text(
                json.dumps(wiring_verification, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            spec = _load_json(candidate_dir / "spec.json")
            change_semantics = _build_change_semantics(
                proposal_contract=dict(spec.get("proposal_contract") or {}),
                changed_files=changed_files,
                runtime_effect_modules=_runtime_effect_modules_for_candidate(candidate_dir),
            )
            spec["proposal_contract"]["full_diff_files"] = list(change_semantics["full_diff_files"])
            spec["proposal_contract"]["runtime_effect_changed_files"] = list(change_semantics["runtime_effect_changed_files"])
            spec["proposal_contract"]["supporting_generated_files"] = list(change_semantics["supporting_generated_files"])
            spec["proposal_contract"]["hypothesis_target_modules"] = list(change_semantics["hypothesis_target_modules"])
            parent_commit = _git_parent_commit(repo_root)
            spec["runtime_metadata"] = {
                "parent_harness": parent_id,
                "parent_commit": parent_commit,
                "candidate_created_at_utc": datetime.now(timezone.utc).isoformat(),
                "changed_files": sorted(changed_files),
                "full_diff_files": list(change_semantics["full_diff_files"]),
                "runtime_effect_changed_files": list(change_semantics["runtime_effect_changed_files"]),
                "supporting_generated_files": list(change_semantics["supporting_generated_files"]),
                "hypothesis_target_modules": list(change_semantics["hypothesis_target_modules"]),
                "diff_path": "parent_diff.patch",
                "candidate_branch_hint": f"candidate/{candidate_id}",
                "agent_loop_meta": dict(agent_meta or {}),
                "runtime_wiring_verification_path": "runtime_wiring_verification.json",
                "runtime_wiring_verification_passed": bool(wiring_verification.get("passed")),
            }
            (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            (candidate_dir / "parent_diff.patch").write_text(
                _build_parent_diff(parent_entry.dir_path, candidate_dir, changed_files),
                encoding="utf-8",
            )
            (candidate_dir / "proposer_tool_audit.json").write_text(
                json.dumps({"events": tools.audit_log}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            spec = _load_json(candidate_dir / "spec.json")
            spec["runtime_metadata"]["proposer_tool_audit_path"] = "proposer_tool_audit.json"
            spec["runtime_metadata"]["proposer_tool_event_count"] = len(tools.audit_log)
            (candidate_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

            if not note:
                refreshed_note = _build_grounded_note(contract=spec["proposal_contract"], spec=spec, changed_files=changed_files)
                (candidate_dir / "proposer_note.txt").write_text(refreshed_note + "\n", encoding="utf-8")
                final_note = refreshed_note

            try:
                validate_candidate_boundary(candidate_dir)
                _run_candidate_smoke_checks(candidate_dir)
                _run_import_checks(candidate_dir)
                validate_candidate_contract_alignment(candidate_dir, parent_dir=parent_entry.dir_path, proposal_contract=spec["proposal_contract"])
                latest_wiring = _load_json(candidate_dir / "runtime_wiring_verification.json") if (candidate_dir / "runtime_wiring_verification.json").exists() else {}
                if not bool(latest_wiring.get("passed", False)):
                    raise RuntimeError(f"runtime wiring verification failed: {json.dumps(latest_wiring, ensure_ascii=False)}")
            except Exception as exc:
                last_error = str(exc)
            else:
                last_error = ""

            review_prompt = build_self_review_prompt(
                proposal_contract_json=json.dumps(spec.get("proposal_contract", {}), ensure_ascii=False, indent=2),
                candidate_spec_json=json.dumps(spec, ensure_ascii=False, indent=2),
                changed_files_json=json.dumps(sorted(changed_files), ensure_ascii=False),
                runtime_wiring_verification_json=json.dumps(
                    _load_json(candidate_dir / "runtime_wiring_verification.json") if (candidate_dir / "runtime_wiring_verification.json").exists() else {},
                    ensure_ascii=False,
                    indent=2,
                ),
                last_error=last_error or "(none)",
            )
            review = _extract_json_object(str(llm.request(prompt=review_prompt, model_name=proposer_model or MODEL_NAME, stream=False) or "{}"))
            status = str(review.get("status") or "").strip().lower()
            files_to_revise = [Path(str(x)).name for x in list(review.get("files_to_modify") or []) if Path(str(x)).name in ALLOWED_MUTATION_FILES]
            error_file = _guess_error_file_from_text(last_error)
            if error_file and error_file in ALLOWED_MUTATION_FILES and error_file not in files_to_revise:
                files_to_revise.insert(0, error_file)
            if not last_error and status == "pass":
                return candidate_dir
            if _round >= max(0, int(max_revision_rounds)):
                raise RuntimeError(f"candidate failed hard guardrails/self-review after {_round + 1} attempts: {last_error or review}")
            if not files_to_revise:
                files_to_revise = ["spec.json"] + ([next(iter(changed_files - {"spec.json", "proposer_note.txt"}), "trigger_logic.py")] if changed_files else ["trigger_logic.py"])
            issue_text = "; ".join([str(x) for x in list(review.get("issues") or [])][:4])
            if last_error:
                issue_text = f"{issue_text}; hard_guardrail_error={last_error}" if issue_text else f"hard_guardrail_error={last_error}"
            for target_file in files_to_revise:
                if target_file in {"proposer_note.txt", "README.md", "parent_diff.patch", "proposer_tool_audit.json"}:
                    continue
                parent_file_content = (candidate_dir / target_file).read_text(encoding="utf-8") if (candidate_dir / target_file).exists() else ""
                file_prompt = _build_file_generation_prompt(
                    parent_harness_id=parent_id,
                    parent_spec=parent_spec,
                    parent_file_name=target_file,
                    parent_file_content=parent_file_content,
                    proposal=proposal,
                ) + f"\n\nSelf-review revision issues to fix:\n{issue_text}\n"
                generated = _normalize_generated_text(str(llm.request(prompt=file_prompt, model_name=proposer_model or MODEL_NAME, stream=False) or "").strip())
                if target_file == "spec.json":
                    spec_obj = _extract_json_object(generated)
                    spec_obj["id"] = candidate_id
                    spec_obj["kind"] = "candidate"
                    spec_obj["parent"] = parent_id
                    prior = _load_json(candidate_dir / "spec.json")
                    if "proposal_contract" not in spec_obj:
                        spec_obj["proposal_contract"] = dict(prior.get("proposal_contract") or {})
                    if "sandbox" not in spec_obj:
                        spec_obj["sandbox"] = dict(prior.get("sandbox") or {})
                    if "manifest" not in spec_obj:
                        spec_obj["manifest"] = dict(prior.get("manifest") or {})
                    (candidate_dir / "spec.json").write_text(json.dumps(spec_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    (candidate_dir / target_file).write_text(generated + "\n", encoding="utf-8")
        raise RuntimeError("unreachable: revision loop ended without decision")
    except Exception:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        raise


def rebuild_index(archive_root: Path) -> Dict:
    archive_root = Path(archive_root)
    index_path = archive_root / "index.json"

    entries: List[Dict] = []
    for bucket in ["baselines", "candidates"]:
        base = archive_root / bucket
        if not base.exists():
            continue
        for harness_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            kind = "baseline" if bucket == "baselines" else "candidate"
            stage_summaries: Dict[str, Dict] = {}
            stage_scene_paths: Dict[str, str | None] = {}
            for stage in EVALUATION_PROTOCOLS.keys():
                stage_eval_path = harness_dir / f"eval_summary_{stage}.json"
                stage_scene_path = harness_dir / f"per_scene_metrics_{stage}.json"
                if stage_eval_path.exists():
                    stage_summaries[stage] = _load_json(stage_eval_path)
                    stage_scene_paths[stage] = str(stage_scene_path.as_posix()) if stage_scene_path.exists() else None
            # Backward compatibility for older formal-only archives.
            legacy_eval_path = harness_dir / "eval_summary.json"
            legacy_scene_path = harness_dir / "per_scene_metrics.json"
            if "formal" not in stage_summaries and legacy_eval_path.exists():
                stage_summaries["formal"] = _load_json(legacy_eval_path)
                stage_scene_paths["formal"] = str(legacy_scene_path.as_posix()) if legacy_scene_path.exists() else None
            if not stage_summaries:
                continue
            eval_summary = stage_summaries.get("formal") or stage_summaries.get("screening") or next(iter(stage_summaries.values()))
            active_stage = "formal" if "formal" in stage_summaries else "screening"
            active_scene_path = stage_scene_paths.get(active_stage)
            per_scene = _load_json(Path(active_scene_path)) if active_scene_path else {}
            run_dirs = sorted([p for p in (harness_dir / "runs").glob("run_*")]) if (harness_dir / "runs").exists() else []

            parent_id = eval_summary.get("parent_id")
            parent_kind = eval_summary.get("parent_kind")
            derived_from = eval_summary.get("derived_from")
            if (not parent_id):
                spec_path = harness_dir / "code_or_spec" / "spec.json"
                if spec_path.exists():
                    try:
                        spec_payload = _load_json(spec_path)
                        parent_id = spec_payload.get("parent") or ((spec_payload.get("lineage") or {}).get("parent_id"))
                        parent_kind = (spec_payload.get("lineage") or {}).get("parent_kind")
                        if (not parent_kind) and parent_id:
                            parent_kind = "baseline" if str(parent_id).startswith("baseline") else "candidate"
                        derived_from = (spec_payload.get("lineage") or {}).get("derived_from") or parent_id
                    except Exception:
                        pass
            if (not parent_kind) and parent_id:
                parent_kind = "baseline" if str(parent_id).startswith("baseline") else "candidate"
            evaluation_protocol = dict(eval_summary.get("evaluation_protocol") or get_evaluation_protocol(kind=kind, requested_mode=active_stage))
            entries.append(
                {
                    "candidate_id": str(eval_summary.get("harness_id") or harness_dir.name),
                    "kind": kind,
                    "parent_id": parent_id,
                    "parent_kind": parent_kind,
                    "derived_from": derived_from,
                    "path": str(harness_dir.as_posix()),
                    "total_runs": int(eval_summary.get("total_runs") or len(run_dirs)),
                    "total_runs_expected": int(eval_summary.get("total_runs_expected") or evaluation_protocol.get("total_runs") or 0),
                    "total_runs_completed": int(eval_summary.get("total_runs_completed") or eval_summary.get("total_runs") or len(run_dirs)),
                    "metrics": dict(eval_summary.get("metrics") or {}),
                    "status": str(eval_summary.get("status") or "unknown"),
                    "evaluation_stage": active_stage,
                    "promoted_to_formal": bool("formal" in stage_summaries and "screening" in stage_summaries),
                    "evaluation_protocol": evaluation_protocol,
                    "stage_summaries": {
                        stage: {
                            "eval_summary_path": str((harness_dir / f"eval_summary_{stage}.json").as_posix()) if (harness_dir / f"eval_summary_{stage}.json").exists() else None,
                            "per_scene_metrics_path": stage_scene_paths.get(stage),
                        }
                        for stage in sorted(stage_summaries.keys())
                    },
                    "per_scene_metrics_path": active_scene_path,
                    "eval_summary_path": str((harness_dir / f"eval_summary_{active_stage}.json").as_posix()) if (harness_dir / f"eval_summary_{active_stage}.json").exists() else str((harness_dir / "eval_summary.json").as_posix()),
                    "per_scene_metrics": per_scene,
                    "trace_locations": {
                        "runs_dir": str((harness_dir / "runs").as_posix()),
                        "run_count": len(run_dirs),
                    },
                }
            )

    pareto_ready = []
    for e in entries:
        if str(e.get("evaluation_stage") or "") != "formal":
            continue
        m = e.get("metrics") or {}
        if {"collision_count_avg", "near_miss_count_avg", "completion_time_mission_sec_avg", "llm_call_count_avg"}.issubset(m.keys()):
            pareto_ready.append(
                {
                    "candidate_id": e["candidate_id"],
                    "metrics": {
                        "collision_count_avg": m["collision_count_avg"],
                        "near_miss_count_avg": m["near_miss_count_avg"],
                        "completion_time_mission_sec_avg": m["completion_time_mission_sec_avg"],
                        "llm_call_count_avg": m["llm_call_count_avg"],
                    },
                }
            )
    for p in pareto_ready:
        p["metrics"]["completion_time_sec_avg"] = p["metrics"].pop("completion_time_mission_sec_avg")

    pareto_map = {}
    for e in mark_pareto([{"harness_id": x["candidate_id"], "metrics": x["metrics"]} for x in pareto_ready]):
        key = str(e.get("candidate_id") or e.get("harness_id") or "")
        if key:
            pareto_map[key] = e
    for e in entries:
        e["pareto_frontier"] = bool(pareto_map.get(e["candidate_id"], {}).get("pareto_frontier", False))

    index = {
        "archive_version": "proposer_archive_v2",
        "evaluation_protocols": dict(EVALUATION_PROTOCOLS),
        "entries": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index
