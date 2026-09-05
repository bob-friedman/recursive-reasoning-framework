#!/usr/bin/env python3
"""run_sample.py — paste-and-run any single ARC-1 style sample + human-readable summary.

Usage:
  python3 scripts/run_sample.py samples/<name>.json [--timeout 300] [--keep-results]

Sample format (see samples/_example_fill.json):
  {"rule": "fill"|"recolor"|"move",
   "demos": [{"input": G, "output"?: G}...],
   "test": {"input": G, "output"?: G}}

Flow: validate via the env (outputs checked against the rule when supplied) ->
run the agent harness on it -> print a human summary AND write SUMMARY.md
(a short human-readable file next to the results). Raw JSON stays local (gitignored).
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from plugins.arc_static_env import ArcStaticEnv

PALETTE = "·█▓░◆●▲■×+*"


def render(grid):
    return "\n".join(" ".join(PALETTE[v % len(PALETTE)] if v else "·" for v in row)
                     for row in grid)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one pasted ARC-1 sample with summary")
    ap.add_argument("sample", help="samples/<name>.json")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keep-results", action="store_true",
                    help="keep results JSON (default: removed after summary)")
    args = ap.parse_args()

    sample = Path(args.sample)
    if not sample.exists():
        print(f"No such sample: {sample}\nSee samples/_example_fill.json for format.")
        return 2
    task_id = sample.stem

    # 1. validate (raises with a human message on bad format / rule contradiction)
    env = ArcStaticEnv()
    try:
        rule, demo_ins, test_in = env._spec(task_id)
    except ValueError as e:
        print(f"Sample rejected: {e}")
        return 2
    expected = env._rule_apply(rule, test_in)
    print(f"Sample {task_id}: rule={rule}, {len(demo_ins)} demos, "
          f"test {len(test_in)}x{len(test_in[0])} — validated.")
    print("--- test input ---")
    print(render(test_in))

    # 2. run the agent harness on it
    results = BASE / f"results_sample_{task_id}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "core.harness", "--env", "arc_static_env",
         "--tasks", task_id, "--results", str(results), "--timeout", str(args.timeout)],
        cwd=BASE, capture_output=True, text=True)
    if results.exists():
        doc = json.loads(results.read_text())
        rec = doc["results"][0]
    else:
        print("Harness produced no results file; stdout tail:")
        print(proc.stdout[-1500:])
        return 1

    # 3. human summary (console + SUMMARY file)
    won = rec["solved"]
    steps = rec["actions_taken"]
    secs = rec["duration_sec"]
    preds = [s["payload"].get("predicted_outcome", "") for s in rec["outcome_log"]]
    head = f"{'SOLVED' if won else 'FAILED'} in {steps} step(s), {secs:.1f}s"
    print(f"=== {task_id}: {head} ===")
    for i, p in enumerate(preds, 1):
        print(f"  step {i}: {p[:160]}")
    if won:
        print("--- solved grid ---")
        final = env._rule_apply(rule, test_in)
        print(render(final))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = (f"# Sample run: {task_id} ({stamp})\n\n"
               f"Rule: {rule} | {len(demo_ins)} demos | test {len(test_in)}x{len(test_in[0])}\n\n"
               f"Result: **{head}**\n\n"
               f"## Predictions\n\n" +
               "".join(f"{i}. {p}\n" for i, p in enumerate(preds, 1) or ["(no steps taken)"]) +
               f"\n## Replay\n\n`python3 scripts/run_sample.py samples/{task_id}.json`\n")
    (BASE / f"SUMMARY_{task_id}.md").write_text(summary, encoding="utf-8")
    print(f"Summary written to SUMMARY_{task_id}.md")
    if not args.keep_results:
        results.unlink(missing_ok=True)
    return 0 if won else 1


if __name__ == "__main__":
    sys.exit(main())
