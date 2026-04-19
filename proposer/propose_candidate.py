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

from controller.harness_protocol import EVALUATION_PROTOCOL_SEQUENCE, EVALUATION_PROTOCOL_VERSION, TOTAL_EVAL_RUNS
from proposer.agent_tools import ProposerToolbox
from proposer.archive_reader import summarize_archive_for_proposer
from proposer.consistency import validate_candidate_contract_alignment
from proposer.evaluate_candidate import mark_pareto
from proposer.prompts import build_agent_next_action_prompt, build_self_review_prompt
from proposer.registry import ALLOWED_MUTATION_FILES, TRACKED_CONTRACT_FILES, HarnessRegistry, validate_candidate_boundary


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
        {"name": "search_traces", "args": {"harness_id": "str", "needle": "str", "max_hits": "int"}},
        {"name": "read_trace_snippet", "args": {"trace_path": "str", "line_no": "int", "window": "int"}},
        {"name": "validate_candidate", "args": {"candidate_dir": "str", "parent_dir": "str"}},
        {"name": "smoke_check_candidate", "args": {"candidate_dir": "str"}},
    ]


def _execute_agent_tool(tools: ProposerToolbox, tool_name: str, tool_args: Dict) -> Dict:
    args = dict(tool_args or {})
    if tool_name == "list_harnesses":
        return {"result": tools.list_harnesses(kind=str(args.get("kind") or "all"))}
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
    run_evidence_tools = {"list_runs", "search_traces", "read_run_metadata", "read_trace_snippet"}
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
        "one_sentence_hypothesis",
        "weakness_being_addressed",
        "expected_tradeoff",
        "expected_runtime_effect",
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

    invariants = list(proposal.get("invariants") or [])
    if not invariants:
        raise ValueError("LLM proposer output missing invariants")

    normalized = dict(proposal)
    normalized["files_to_create_or_modify"] = normalized_target_files
    normalized["implementation_contract"] = implementation_contract
    normalized["invariants"] = invariants
    normalized["sandbox_modules_to_modify"] = [Path(str(v)).name for v in list(proposal.get("sandbox_modules_to_modify") or [])]
    normalized["changed_files"] = [Path(str(v)).name for v in list(proposal.get("changed_files") or [])]
    normalized["runtime_wiring_plan"] = dict(proposal.get("runtime_wiring_plan") or {})
    normalized["smoke_test_evidence_to_check"] = dict(proposal.get("smoke_test_evidence_to_check") or {})
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


def _normalize_generated_text(text: str) -> str:
    out = str(text or "")
    while "\\n" in out:
        out = out.replace("\\n", "\n")
    return out


def _build_file_generation_prompt(
    *,
    parent_harness_id: str,
    parent_spec: Dict,
    parent_file_name: str,
    parent_file_content: str,
    proposal: Dict,
) -> str:
    return (
        "You are generating one bounded harness file for UAV harness optimization.\n"
        "Allowed harness boundary files: spec.json, state_features.py, trigger_logic.py, prompt_composer.py, archive_selector.py, validator_rules.py, state_encoder.py, trigger_policy.py, prompt_builder.py, proposer_note.txt, README.md\n"
        "Runtime-first rule: prioritize sandbox runtime-effect modules and wiring alignment.\n"
        "Primary editable targets are sandbox modules: state_features.py, trigger_logic.py, prompt_composer.py, archive_selector.py, validator_rules.py.\n"
        "Legacy files (state_encoder.py, trigger_policy.py, prompt_builder.py) are compatibility wrappers/metadata mirrors, not primary targets.\n"
        "Do not create legacy-vs-sandbox ambiguity; forbidden example: editing trigger_logic.py while spec/loader still routes to trigger_policy.py.\n"
        "If legacy sync is required for compatibility, keep it explicitly consistent with spec/manifest/runtime_wiring_plan.\n"
        "You must output ONLY the full content of the requested file, no markdown fences.\n"
        "Do not modify simulator/PX4/controller/executor/collision math/checkpoint rules.\n\n"
        f"Parent harness: {parent_harness_id}\n"
        f"Parent spec:\n{json.dumps(parent_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Proposal contract:\n{json.dumps(proposal, ensure_ascii=False, indent=2)}\n\n"
        f"Requested file: {parent_file_name}\n"
        "Current parent file content:\n"
        f"{parent_file_content}\n\n"
        "Generate improved content aligned with the proposal hypothesis and weakness.\n"
        "If requested file is spec.json, ensure valid JSON and keep candidate lineage metadata."
    )


def propose_next_candidate(
    repo_root: Path,
    note: str = "",
    focus_text: str = "Improve safety-aware replan timing while avoiding unnecessary detours.",
    allow_fallback_heuristic: bool = False,
    max_revision_rounds: int = 2,
    max_agent_steps: int = 10,
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
            },
            "smoke_test_evidence_to_check": {
                "trigger_logic_evidence": "smoke/import checks and runtime_metadata.changed_files include trigger_logic.py",
                "state_features_evidence": "not changed in this fallback candidate",
                "prompt_composer_evidence": "not changed in this fallback candidate",
                "evidence_limitations": "fallback path due to proposer LLM failure",
            },
            "invariants": [
                "proposal_contract files must match actual changed files",
                "spec trigger policy must align with trigger_policy.py behavior",
            ],
        })
        agent_meta = {"fallback_used": True, "reason": "agent_loop_or_llm_failure"}

    parent_id = str(proposal.get("parent_harness") or "").strip()
    if not parent_id:
        raise ValueError("LLM proposer output missing parent_harness")

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
    for required in ["state_features.py", "trigger_logic.py", "prompt_composer.py"]:
        path = candidate_dir / required
        if not path.exists():
            fallback = _default_sandbox_file_content(required)
            if fallback:
                path.write_text(fallback, encoding="utf-8")

    parent_spec = _load_json(parent_entry.dir_path / "spec.json")

    normalized_target_files = [Path(name).name for name in files_to_modify if Path(name).name in ALLOWED_MUTATION_FILES]

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
        spec = _load_json(candidate_dir / "spec.json")
        spec["proposal_contract"]["files_to_create_or_modify"] = sorted(changed_files)
        parent_commit = _git_parent_commit(repo_root)
        spec["runtime_metadata"] = {
            "parent_harness": parent_id,
            "parent_commit": parent_commit,
            "candidate_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "changed_files": sorted(changed_files),
            "diff_path": "parent_diff.patch",
            "candidate_branch_hint": f"candidate/{candidate_id}",
            "agent_loop_meta": dict(agent_meta or {}),
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
            spec = _load_json(candidate_dir / "spec.json")
            spec["proposal_contract"]["files_to_create_or_modify"] = sorted(changed_files)
            parent_commit = _git_parent_commit(repo_root)
            spec["runtime_metadata"] = {
                "parent_harness": parent_id,
                "parent_commit": parent_commit,
                "candidate_created_at_utc": datetime.now(timezone.utc).isoformat(),
                "changed_files": sorted(changed_files),
                "diff_path": "parent_diff.patch",
                "candidate_branch_hint": f"candidate/{candidate_id}",
                "agent_loop_meta": dict(agent_meta or {}),
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
            except Exception as exc:
                last_error = str(exc)
            else:
                last_error = ""

            review_prompt = build_self_review_prompt(
                proposal_contract_json=json.dumps(spec.get("proposal_contract", {}), ensure_ascii=False, indent=2),
                candidate_spec_json=json.dumps(spec, ensure_ascii=False, indent=2),
                changed_files_json=json.dumps(sorted(changed_files), ensure_ascii=False),
                last_error=last_error or "(none)",
            )
            review = _extract_json_object(str(llm.request(prompt=review_prompt, model_name=proposer_model or MODEL_NAME, stream=False) or "{}"))
            status = str(review.get("status") or "").strip().lower()
            files_to_revise = [Path(str(x)).name for x in list(review.get("files_to_modify") or []) if Path(str(x)).name in ALLOWED_MUTATION_FILES]
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
            eval_path = harness_dir / "eval_summary.json"
            per_scene_path = harness_dir / "per_scene_metrics.json"
            if not eval_path.exists():
                continue
            eval_summary = _load_json(eval_path)
            per_scene = _load_json(per_scene_path) if per_scene_path.exists() else {}
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
            entries.append(
                {
                    "candidate_id": str(eval_summary.get("harness_id") or harness_dir.name),
                    "kind": "baseline" if bucket == "baselines" else "candidate",
                    "parent_id": parent_id,
                    "parent_kind": parent_kind,
                    "derived_from": derived_from,
                    "path": str(harness_dir.as_posix()),
                    "total_runs": int(eval_summary.get("total_runs") or len(run_dirs)),
                    "metrics": dict(eval_summary.get("metrics") or {}),
                    "status": str(eval_summary.get("status") or "unknown"),
                    "per_scene_metrics_path": str(per_scene_path.as_posix()) if per_scene_path.exists() else None,
                    "eval_summary_path": str(eval_path.as_posix()),
                    "per_scene_metrics": per_scene,
                    "trace_locations": {
                        "runs_dir": str((harness_dir / "runs").as_posix()),
                        "run_count": len(run_dirs),
                    },
                }
            )

    pareto_ready = []
    for e in entries:
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

    pareto_map = {
        e["candidate_id"]: e for e in mark_pareto(
            [{"harness_id": x["candidate_id"], "metrics": x["metrics"]} for x in pareto_ready]
        )
    }
    for e in entries:
        e["pareto_frontier"] = bool(pareto_map.get(e["candidate_id"], {}).get("pareto_frontier", False))

    index = {
        "archive_version": "proposer_archive_v2",
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "pairs": EVALUATION_PROTOCOL_SEQUENCE,
            "total_runs": TOTAL_EVAL_RUNS,
        },
        "entries": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index
