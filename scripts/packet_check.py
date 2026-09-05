#!/usr/bin/env python3
"""packet_check.py — machine guardrails for the autonomous op pipeline (Plan §10).

Runs without humans: DSL self-test, win-path for EVERY level, single-op insufficiency
spot-checks for invention levels, engine-freeze check. Exit 0 = packet may file for vote.
Exit nonzero = stop, fix, re-run. Usage: python3 scripts/packet_check.py
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core.grid_algebra import (  # noqa: E402
    count_color, demo, find_objects, recolor, reflect_h, rotate_cw, translate,
)
from plugins.arc_grid_env import ArcGridEnv  # noqa: E402
from plugins.arc_static_env import ArcStaticEnv  # noqa: E402

# ARC-1 static tasks -> winning action sequence (exact-match adapter)
STATIC_PATHS = {
    "arc1_fill_1": [("FILL", {})],
    "arc1_recolor_1": [("RECOLOR", {})],
    "arc1_move_1": [("RIGHT", {})],
}

# task -> winning action sequence [(action, extra_payload)]
WIN_PATHS = {
    "arc_move_1": [("RIGHT", {})],
    "arc_move_2": [("LEFT", {}), ("LEFT", {})],
    "arc_recolor_1": [("RECOLOR", {})],
    "arc_recolor_2": [("RECOLOR", {})],
    "arc_rotate_1": [("ROTATE", {})],
    "arc_contact_1": [("RIGHT", {}), ("RIGHT", {}), ("RIGHT", {})],
    "arc_reflect_1": [("REFLECT", {})],
    "arc_fill_1": [("FILL", {})],
    "arc_fill_2": [("FILL", {})],
    "arc_symmetry_1": [("MIRROR", {})],
    "arc_symmetry_2": [("MIRROR", {"axis": "horizontal"})],
    "arc_tile_1": [("TILE", {})],
    "arc_tile_2": [("TILE", {})],
}

# sim-family levels derive win-paths from the env's own SIM_LEVELS table
# (single source of truth: adding a sim level auto-extends this gate)
SIM_ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "ROTATE", "REFLECT", "MIRROR", "TILE",
               "HOLLOW", "COVER", "RANK", "REPEAT", "MREPEAT", "UPSCALE", "DOWNSCALE",
               "CROP", "CENTER", "FALL", "ALIGN", "SETTLE", "SWAP", "TRIM", "FRAME",
               "SORTX", "TOPK", "BEAT", "STAMP", "FILL", "RECOLOR"]
SIM_PAYLOADS = {"COVER": {"color": 1}, "RANK": {"rank": 1}, "SWAP": {"a": 1, "b": 4},
                "MIRROR": {"axis": "vertical"}, "TRIM": {"min": 2, "max": 3},
                "TOPK": {"k": 2}, "STAMP": {"dx": 3, "dy": 0}}

# invention levels -> single ops that must ALL fail (admission gate)
INSUFFICIENCY = {
    "arc_fill_1": ["translate", "recolor", "rotate", "reflect"],
    "arc_fill_2": ["translate", "recolor", "rotate", "reflect"],
    "arc_symmetry_1": ["translate", "recolor", "rotate", "reflect"],
    "arc_symmetry_2": ["translate", "recolor", "rotate", "reflect"],
    "arc_tile_1": ["translate", "recolor", "rotate", "reflect"],
    "arc_tile_2": ["translate", "recolor", "rotate", "reflect"],
}


def single_op(grid, name):
    if name == "translate":
        return translate(grid, 1, 0, color=1)
    if name == "recolor":
        return recolor(grid, {0: 1})
    if name == "rotate":
        return rotate_cw(grid)
    if name == "reflect":
        return reflect_h(grid)
    raise ValueError(name)


def main() -> int:
    demo()  # raises on any admitted-op self-test failure
    e = ArcGridEnv()
    for tid, acts in WIN_PATHS.items():
        e.start(tid)
        assert not e._obs()["is_win"], f"{tid} starts solved!"
        for a, p in acts:
            r = e.step({"action": a, **p, "predicted_outcome": "packet_check"})
        assert r[0]["is_win"], f"{tid} win-path failed"
        print(f"ok  {tid} ({len(acts)} steps)")
    for tid, ops in INSUFFICIENCY.items():
        g = e.start(tid)["grid"]
        for name in ops:
            assert not e._is_win(single_op(g, name), tid), f"{tid}: {name} reaches goal!"
        print(f"ok  {tid} insufficiency ({len(ops)} single-ops fail)")
    # sim levels: designated win-path (1+ steps) + FULL cross-action exclusivity.
    # Single-step levels: no other single action may reach the goal.
    # Multi-step levels: no single action may reach it (sequence required by design).
    n_sim = 0
    for tid, steps in ArcGridEnv.SIM_LEVELS.items():
        e.start(tid)
        assert not e._obs()["is_win"], f"{tid} starts solved!"
        for a, p in steps:
            r = e.step({"action": a, **p, "predicted_outcome": "packet_check"})
        assert r[0]["is_win"], f"{tid} sim win-path failed"
        win_action = steps[0][0] if len(steps) == 1 else None
        for a in SIM_ACTIONS:
            if a == win_action:
                continue
            e.start(tid)
            try:
                r = e.step({"action": a, **SIM_PAYLOADS.get(a, {}),
                            "predicted_outcome": "packet_check"})
            except ValueError:
                continue
            assert not r[0]["is_win"], f"LEAK: {tid} winnable by {a}"
        n_sim += 1
        print(f"ok  {tid} sim win ({len(steps)} steps) + exclusive")
    frozen = subprocess.run(
        ["git", "diff", "--name-only", "--", "core/harness.py", "core/environment.py"],
        capture_output=True, text=True, cwd=BASE,
    ).stdout.strip()
    assert not frozen, f"engine freeze violated: {frozen}"
    print("ok  engine freeze (harness.py, environment.py untouched)")
    s = ArcStaticEnv()
    for tid, acts in STATIC_PATHS.items():
        s.start(tid)
        assert not s._obs()["is_win"], f"{tid} starts solved!"
        assert len(s.demos) == 2, f"{tid} must show 2 demo pairs"
        for a, p in acts:
            r = s.step({"action": a, **p, "predicted_outcome": "packet_check"})
        assert r[0]["is_win"], f"{tid} static win-path failed"
        print(f"ok  {tid} static ({len(acts)} steps, exact-match)")
    print(f"ALL GREEN: {len(WIN_PATHS)} win-paths + {n_sim} sim-exclusive + "
          f"{len(INSUFFICIENCY)} insufficiency gates + {len(STATIC_PATHS)} static")
    return 0


if __name__ == "__main__":
    sys.exit(main())
