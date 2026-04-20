from __future__ import annotations

import argparse
import json
from pathlib import Path

from proposer.propose_candidate import propose_next_candidate, rebuild_index
from proposer.registry import HarnessRegistry
from proposer.evaluate_candidate import evaluate_candidate_live
from proposer.run_loop import run_once
from proposer.agent_tools import ProposerToolbox


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_index() -> dict:
    p = _repo_root() / "proposer_archive_v2/index.json"
    if not p.exists():
        return {"entries": []}
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_list_baselines(_: argparse.Namespace) -> int:
    reg = HarnessRegistry(_repo_root())
    for b in reg.list_baselines():
        print(f"{b.harness_id}\t{b.spec.get('name')}\t{b.dir_path}")
    return 0


def cmd_list_candidates(_: argparse.Namespace) -> int:
    reg = HarnessRegistry(_repo_root())
    for c in reg.list_candidates():
        print(f"{c.harness_id}\tparent={c.spec.get('parent')}\t{c.dir_path}")
    return 0


def cmd_show_summary(args: argparse.Namespace) -> int:
    idx = _load_index()
    cid = args.candidate_id
    for e in idx.get("entries", []):
        if e.get("candidate_id") == cid:
            print(json.dumps(e, ensure_ascii=False, indent=2))
            return 0
    raise SystemExit(f"Candidate not found in index: {cid}")


def cmd_topk(args: argparse.Namespace) -> int:
    idx = _load_index()
    metric = args.metric
    rows = []
    for e in idx.get("entries", []):
        if (not args.include_screening) and str(e.get("evaluation_stage") or "") != "formal":
            continue
        m = dict(e.get("metrics") or {})
        if metric in m and m[metric] is not None:
            rows.append((e.get("candidate_id"), float(m[metric]), m))
    rows.sort(key=lambda x: x[1], reverse=args.desc)
    for hid, score, m in rows[: args.k]:
        print(f"{hid}\t{metric}={score:.4f}\t{json.dumps(m, ensure_ascii=False)}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    reg = HarnessRegistry(_repo_root())
    a = reg.get(args.a).spec
    b = reg.get(args.b).spec
    print(json.dumps({"a": a, "b": b}, ensure_ascii=False, indent=2))
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    p = propose_next_candidate(
        _repo_root(),
        note=args.note or "",
        focus_text=args.focus_text,
        allow_fallback_heuristic=bool(args.allow_fallback_heuristic),
    )
    print(p.as_posix())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    repo = _repo_root()
    out = evaluate_candidate_live(
        repo_root=repo,
        harness_id=args.harness_id,
        archive_root=repo / "proposer_archive_v2",
        evaluation_mode=args.mode,
    )
    print(json.dumps({"eval_summary": out.eval_summary, "per_scene_metrics": out.per_scene_metrics}, ensure_ascii=False, indent=2))
    return 0


def cmd_reindex(_: argparse.Namespace) -> int:
    index = rebuild_index(_repo_root() / "proposer_archive_v2")
    print(f"index_entries={len(index.get('entries', []))}")
    return 0


def cmd_run_iteration(args: argparse.Namespace) -> int:
    cid = run_once(
        _repo_root(),
        evaluate_baselines=bool(args.evaluate_baselines),
        focus_text=args.focus_text,
        allow_fallback_heuristic=bool(args.allow_fallback_heuristic),
    )
    print(cid)
    return 0




def cmd_list_runtime_prompt_assets(args: argparse.Namespace) -> int:
    tools = ProposerToolbox(repo_root=_repo_root(), archive_root=_repo_root() / "proposer_archive_v2")
    print(json.dumps(tools.list_runtime_prompt_assets(args.harness_id), ensure_ascii=False, indent=2))
    return 0


def cmd_read_runtime_prompt_asset(args: argparse.Namespace) -> int:
    tools = ProposerToolbox(repo_root=_repo_root(), archive_root=_repo_root() / "proposer_archive_v2")
    print(
        json.dumps(
            tools.read_runtime_prompt_asset(args.harness_id, asset_name=args.asset_name, stage=args.stage),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_diff_runtime_prompt_assets(args: argparse.Namespace) -> int:
    tools = ProposerToolbox(repo_root=_repo_root(), archive_root=_repo_root() / "proposer_archive_v2")
    print(
        json.dumps(
            tools.diff_runtime_prompt_assets(args.harness_a, args.harness_b, stage=args.stage),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restricted proposer MVP CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("list-baselines")
    s.set_defaults(func=cmd_list_baselines)

    s = sp.add_parser("list-candidates")
    s.set_defaults(func=cmd_list_candidates)

    s = sp.add_parser("show-candidate-summary")
    s.add_argument("candidate_id")
    s.set_defaults(func=cmd_show_summary)

    s = sp.add_parser("top-k")
    s.add_argument("--metric", default="success_rate")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--desc", action="store_true")
    s.add_argument("--include-screening", action="store_true", help="Include screening-stage candidates in ranking output.")
    s.set_defaults(func=cmd_topk)

    s = sp.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_diff)

    s = sp.add_parser("propose")
    s.add_argument("--note", default="")
    s.add_argument("--focus-text", default="Improve safety-aware replan timing while avoiding unnecessary detours.")
    s.add_argument("--allow-fallback-heuristic", action="store_true")
    s.set_defaults(func=cmd_propose)

    s = sp.add_parser("evaluate")
    s.add_argument("harness_id")
    s.add_argument("--mode", choices=["screening", "formal"], default=None)
    s.set_defaults(func=cmd_evaluate)

    s = sp.add_parser("reindex")
    s.set_defaults(func=cmd_reindex)


    s = sp.add_parser("list-runtime-prompt-assets")
    s.add_argument("harness_id")
    s.set_defaults(func=cmd_list_runtime_prompt_assets)

    s = sp.add_parser("read-runtime-prompt-asset")
    s.add_argument("harness_id")
    s.add_argument("--asset-name", default=None)
    s.add_argument("--stage", choices=["initial", "replan", "heartbeat"], default=None)
    s.set_defaults(func=cmd_read_runtime_prompt_asset)

    s = sp.add_parser("diff-runtime-prompt-assets")
    s.add_argument("harness_a")
    s.add_argument("harness_b")
    s.add_argument("--stage", choices=["initial", "replan", "heartbeat"], default="initial")
    s.set_defaults(func=cmd_diff_runtime_prompt_assets)

    s = sp.add_parser("run-iteration")
    s.add_argument("--evaluate-baselines", action="store_true")
    s.add_argument("--focus-text", default="Improve safety-aware replan timing while avoiding unnecessary detours.")
    s.add_argument("--allow-fallback-heuristic", action="store_true")
    s.set_defaults(func=cmd_run_iteration)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
