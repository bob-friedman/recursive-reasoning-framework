"""arc_static_env.py — ARC-1 style static I/O adapter (no new physics).

Each task shows demo input/output pairs plus a test input. The agent transforms
current_grid (starting = test input) with the SAME action set as arc_grid_env.
Win when current_grid exactly equals the expected output (exact-match verifier).

Tasks (rule families already admitted):
- arc1_fill_1:    fill enclosed interior (flood_fill_interior)
- arc1_recolor_1: recolor smallest object to 3
- arc1_move_1:    slide block RIGHT until wall-adjacent

Ground truth = the admitted op applied to the input. Demos vary size/position.
"""
from typing import Any, Dict, List
from core.environment import BaseEnvironment
from core.grid_algebra import compress, find_objects, flood_fill_interior, smallest
from plugins.arc_grid_env import ArcGridEnv as _GridEnv

MOVABLE = (1, 4)


class ArcStaticEnv(BaseEnvironment):
    domain = "arc_static"
    description = "ARC-1 static I/O adapter: demo pairs + test input, exact-match win"

    def __init__(self):
        self.task_id = ""
        self.input_grid = []
        self.current = []
        self.expected = []
        self.demos = []
        self._tmp = _GridEnv()

    def get_valid_actions(self) -> List[str]:
        return ["LEFT", "RIGHT", "UP", "DOWN", "ROTATE", "REFLECT", "MIRROR", "TILE",
                "HOLLOW", "COVER", "RANK", "REPEAT", "MREPEAT", "UPSCALE", "DOWNSCALE",
                "CROP", "CENTER", "FALL", "ALIGN", "SETTLE", "SWAP", "TRIM", "FRAME",
                "SORTX", "TOPK", "BEAT", "STAMP", "FILL", "RECOLOR", "RESET"]

    # --- custom samples: samples/<task_id>.json (paste-and-run, no code) ---
    # Format: {"rule": "fill"|"recolor"|"move", "demos": [{"input": G, "output": G?}...],
    #          "test": {"input": G, "output": G?}}. Outputs optional: omitted outputs are
    # derived from the named rule (same ground-truth fns as built-ins).
    def _load_custom(self, task_id: str):
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "samples" / f"{task_id}.json"
        if not p.exists():
            return None
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Cannot read samples/{task_id}.json: {e}")
        rule = str(doc.get("rule", "")).lower()
        if rule not in ("fill", "recolor", "move"):
            raise ValueError(f"samples/{task_id}.json: rule must be fill|recolor|move")
        demos = doc.get("demos", [])
        if not isinstance(demos, list) or not demos:
            raise ValueError(f"samples/{task_id}.json: need >=1 demo with input")
        demo_ins = []
        for i, d in enumerate(demos):
            if not isinstance(d, dict) or "input" not in d:
                raise ValueError(f"samples/{task_id}.json: demo {i} needs 'input'")
            gi = self._coerce_grid(d["input"], f"demo {i} input")
            demo_ins.append(gi)
            if "output" in d:  # verify supplied output matches the rule (else reject)
                go = self._coerce_grid(d["output"], f"demo {i} output")
                if self._rule_apply(rule, gi) != go:
                    raise ValueError(
                        f"samples/{task_id}.json: demo {i} output contradicts rule '{rule}'")
        test = doc.get("test", {})
        if not isinstance(test, dict) or "input" not in test:
            raise ValueError(f"samples/{task_id}.json: need test.input")
        test_in = self._coerce_grid(test["input"], "test input")
        return rule, demo_ins, test_in

    @staticmethod
    def _coerce_grid(g, what: str):
        if (not isinstance(g, list) or not g or
                not all(isinstance(r, list) and r for r in g)):
            raise ValueError(f"invalid grid for {what}: need non-empty rows")
        w = len(g[0])
        if any(len(r) != w for r in g):
            raise ValueError(f"invalid grid for {what}: ragged rows")
        if any(not isinstance(v, int) or v < 0 or v > 9 for r in g for v in r):
            raise ValueError(f"invalid grid for {what}: values must be ints 0-9")
        if len(g) > 30 or w > 30:
            raise ValueError(f"invalid grid for {what}: max 30x30")
        return [[int(v) for v in r] for r in g]

    def _spec(self, task_id: str):
        custom = self._load_custom(task_id)
        if custom is not None:
            return custom
        return self._spec_builtin(task_id)

    def _spec_builtin(self, task_id: str):
        if task_id == "arc1_fill_1":
            d1 = [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
            d2 = [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 2, 0, 2, 0],
                [0, 0, 2, 0, 2, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
            test = [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
            return "fill", [d1, d2], test
        if task_id == "arc1_recolor_1":
            d1 = [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0],
            ]
            d2 = [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 2],
                [0, 1, 1, 0, 0, 0],
            ]
            test = [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 2],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
            return "recolor", [d1, d2], test
        if task_id == "arc1_move_1":
            d1 = [
                [0, 0, 0, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 0, 0, 0, 2],
            ]
            d2 = [
                [0, 0, 0, 0, 0, 2],
                [0, 0, 1, 0, 0, 2],
                [0, 0, 1, 0, 0, 2],
                [0, 0, 0, 0, 0, 2],
            ]
            test = [
                [0, 0, 0, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 0, 0, 0, 2],
            ]
            return "move", [d1, d2], test
        raise ValueError(f"Unknown static task '{task_id}'")

    def _rule_apply(self, rule: str, grid):
        if rule == "fill":
            out, _ = flood_fill_interior(grid, color=1)
            return out
        if rule == "recolor":
            return self._recolor_smallest(grid)
        if rule == "move":
            g = [row[:] for row in grid]
            for _ in range(20):  # slide RIGHT to wall (bounded)
                nxt = self._tmp._slide(g, "RIGHT", color=1)
                if nxt == g:
                    break
                g = nxt
            return g
        raise ValueError(rule)

    def _recolor_smallest(self, grid, to_color=3):
        objs = [o for o in find_objects(grid) if o["color"] not in (0, 2)]
        if not objs:
            return [row[:] for row in grid]
        tgt = smallest(objs)
        out = [row[:] for row in grid]
        for x, y in tgt["cells"]:
            out[y][x] = to_color
        return out

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        rule, demo_ins, test_in = self._spec(task_id)
        self.demos = [{"input": d, "output": self._rule_apply(rule, d)} for d in demo_ins]
        self.input_grid = [row[:] for row in test_in]
        self.expected = self._rule_apply(rule, test_in)
        assert self.expected != self.input_grid, f"{task_id}: test must require action"
        self.current = [row[:] for row in self.input_grid]
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        err = self.validate_action(action_payload)
        if err:
            raise ValueError(err)
        action = str(action_payload.get("action", "")).upper()
        if action == "RESET":
            self.current = [row[:] for row in self.input_grid]
        elif action in ("LEFT", "RIGHT", "UP", "DOWN"):
            try:
                color = int(action_payload.get("color", 1))
            except (TypeError, ValueError):
                raise ValueError("Invalid 'color'; must be integer")
            if color not in MOVABLE:
                raise ValueError(f"Color {color} immovable. Movable: {list(MOVABLE)}")
            self.current = self._tmp._slide(self.current, action, color=color)
        elif action == "RECOLOR":
            self.current = self._recolor_smallest(self.current, 3)
        elif action == "MIRROR":
            from core.grid_algebra import symmetry_complete
            axis = str(action_payload.get("axis", "vertical")).lower()
            if axis not in ("vertical", "horizontal"):
                raise ValueError("Invalid 'axis'")
            self.current, _ = symmetry_complete(self.current, axis=axis, color=1)
        elif action in ("HOLLOW", "COVER", "RANK", "REPEAT", "MREPEAT", "UPSCALE",
                        "DOWNSCALE", "CROP", "CENTER", "FALL", "ALIGN", "SETTLE", "SWAP",
                        "TRIM", "FRAME", "SORTX", "TOPK", "BEAT", "STAMP", "ROTATE",
                        "REFLECT", "TILE", "FILL"):
            # delegate every DSL action through the grid env's generic dispatcher
            if action == "ROTATE":
                from core.grid_algebra import rotate_object
                self.current, _ = rotate_object(self.current, color=1)
            elif action == "REFLECT":
                from core.grid_algebra import reflect_object
                self.current, _ = reflect_object(self.current, color=1)
            elif action == "TILE":
                from core.grid_algebra import tile_pattern
                self.current, _ = tile_pattern(self.current, color=1)
            elif action == "FILL":
                self.current, _ = flood_fill_interior(self.current, color=1)
            else:
                self.current = self._tmp._batch_action(self.current, action, dict(
                    {k: v for k, v in action_payload.items() if k != "action"}))
        obs = self._obs()
        return obs, bool(obs["is_win"]), bool(obs["is_win"])

    def _obs(self):
        win = self.current == self.expected
        return {"task_id": self.task_id, "input_grid": self.input_grid,
                "current_grid": self.current,
                "demo_pairs": self.demos,
                "algebra": compress(self.current), "is_win": win}
