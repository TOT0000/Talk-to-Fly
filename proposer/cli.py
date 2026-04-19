from __future__ import annotations

import argparse
import json
from pathlib import Path

from proposer.archive_reader import aggregate_by_harness, read_manual_runs
from proposer.evaluate_candidate import evaluate_candidate_offline
from proposer.propose_candidate import propose_next_candidate, rebuild_index
from proposer.registry import HarnessRegistry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def cmd_topk(args: argparse.Namespace) -> int:
    data = aggregate_by_harness(read_manual_runs(_repo_root() / "proposer_archive/manual_runs/task_runs_debug.jsonl"))
    metric = args.metric
    rows = []
    for hid, v in data.items():
        m = v.get("metrics", {})
        if metric in m:
            rows.append((hid, float(m[metric]), m))
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
    p = propose_next_candidate(_repo_root(), note=args.note or "")
    print(p.as_posix())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    repo = _repo_root()
    out = evaluate_candidate_offline(
        repo_root=repo,
        harness_id=args.harness_id,
        archive_root=repo / "proposer_archive_v2",
        manual_debug_jsonl=repo / "proposer_archive/manual_runs/task_runs_debug.jsonl",
    )
    print(json.dumps({"eval_summary": out.eval_summary, "per_scene_metrics": out.per_scene_metrics}, ensure_ascii=False, indent=2))
    return 0


def cmd_reindex(_: argparse.Namespace) -> int:
    index = rebuild_index(_repo_root() / "proposer_archive_v2")
    print(f"index_entries={len(index.get('entries', []))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restricted proposer MVP CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("list-baselines")
    s.set_defaults(func=cmd_list_baselines)

    s = sp.add_parser("list-candidates")
    s.set_defaults(func=cmd_list_candidates)

    s = sp.add_parser("top-k")
    s.add_argument("--metric", default="mission_success_rate")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--desc", action="store_true")
    s.set_defaults(func=cmd_topk)

    s = sp.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_diff)

    s = sp.add_parser("propose")
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_propose)

    s = sp.add_parser("evaluate")
    s.add_argument("harness_id")
    s.set_defaults(func=cmd_evaluate)

    s = sp.add_parser("reindex")
    s.set_defaults(func=cmd_reindex)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
