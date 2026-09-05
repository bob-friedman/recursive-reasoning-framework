"""arc_grid_env.py — minimal interactive ARC-style grid game for RRF.

Two demo tasks, same generic physics (no per-task hardcoding in agent):
- arc_move_1:   push color-1 block RIGHT until it touches color-2 wall -> win.
- arc_recolor_1: recolor smallest object to color 3 via RECOLOR action -> win.
Obs includes raw grid + compressed algebra (core.grid_algebra.compress)
so a small LLM can reason over symbols, backtest in Python, and act once.
"""
from typing import Any, Dict, List
from core.environment import BaseEnvironment
from core.grid_algebra import compress, find_objects, flood_fill_interior, in_contact, is_adjacent, reflect_object, rotate_object, smallest

MOVABLE = (1, 4)  # generic: any listed color translates rigidly; walls (2) never move


class ArcGridEnv(BaseEnvironment):
    domain = "arc_grid"
    description = "Interactive ARC-style grid demo: move/recolor objects with algebra obs"

    def __init__(self):
        self.grid = []
        self.task_id = ""
        self._init_grid = []
        self._fill_mask = set()  # initially-enclosed cells (fill tasks only)

    def get_valid_actions(self) -> List[str]:
        return ["LEFT", "RIGHT", "UP", "DOWN", "ROTATE", "REFLECT", "FILL", "RECOLOR", "RESET"]

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.grid = [row[:] for row in self._level(task_id)]
        self._init_grid = [row[:] for row in self.grid]
        self._fill_mask = set()
        if "fill" in task_id:  # interior = background cells not border-connected at start
            h, w = len(self.grid), len(self.grid[0])
            painted, _ = flood_fill_interior(self.grid, color=-1)
            self._fill_mask = {(x, y) for y in range(h) for x in range(w) if painted[y][x] == -1}
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        err = self.validate_action(action_payload)
        if err:
            raise ValueError(err)
        action = str(action_payload.get("action", "")).upper()
        if action == "RESET":
            self.grid = [row[:] for row in self._level(self.task_id)]
        elif action in ("LEFT", "RIGHT", "UP", "DOWN"):
            try:
                color = int(action_payload.get("color", 1))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid 'color' {action_payload.get('color')!r}; must be one of {list(MOVABLE)}")
            if color not in MOVABLE:
                raise ValueError(f"Color {color} is immovable (walls/goals never move). Movable: {list(MOVABLE)}")
            self.grid = self._slide(self.grid, action, color=color)
        elif action == "ROTATE":
            self.grid, _ = rotate_object(self.grid, color=1)
        elif action == "REFLECT":
            self.grid, _ = reflect_object(self.grid, color=1)
        elif action == "FILL":
            self.grid, _ = flood_fill_interior(self.grid, color=1)
        elif action == "RECOLOR":
            self.grid = self._recolor_smallest(self.grid, 3)
        obs = self._obs()
        return obs, bool(obs["is_win"]), bool(obs["is_win"])

    # --- levels (env-side ground truth; agent must discover, not read) ---
    # Public: arc_move_1 (wall right), arc_recolor_1.
    # Holdout (reviewer-added, agent-blind): arc_move_2 (wall LEFT, wider grid,
    #   block starts right, must go LEFT), arc_recolor_2 (wider, two objects).
    def _level(self, task_id: str):
        if task_id == "arc_fill_1":  # 7x7 ring, 3x3 interior -> FILL wins
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_fill_2":  # 8x6 offset ring, 3x2 interior -> FILL wins
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 0],
                [0, 0, 2, 0, 0, 0, 2, 0],
                [0, 0, 2, 0, 0, 0, 2, 0],
                [0, 0, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_reflect_1":  # L spine-right/foot-left -> REFLECT -> foot-right
            return [
                [0, 0, 0, 0, 2],
                [0, 0, 1, 0, 2],
                [0, 0, 1, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 0, 0, 0, 2],
            ]
        if task_id == "arc_contact_1":  # 2x1 pusher -> 3x RIGHT through gap, touch color-4
            return [
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0],
                [1, 1, 0, 0, 0, 4, 0],
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0],
            ]
        if task_id == "arc_rotate_1":  # tall 1x3 bar -> win when wider than tall
            return [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_move_2":
            return [
                [2, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 1, 1, 0, 0],
                [2, 0, 0, 1, 1, 0, 0],
                [2, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_recolor_2":
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 2],
                [0, 1, 1, 1, 0, 2],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        if "recolor" in task_id:
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 2],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        return [  # arc_move_1 default
            [0, 0, 0, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 2],
        ]

    def _slide(self, grid, action, color=1):
        # Rigid-body translation: all color cells move together or not at all.
        # Blocked iff any target is out-of-bounds or hits a wall (color 2).
        # Self-overlap is allowed (ignored), so wide objects never split.
        dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}[action]
        h, w = len(grid), len(grid[0])
        cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
        if not cells:
            return [row[:] for row in grid]
        for x, y in cells:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                return [row[:] for row in grid]  # edge blocks whole object
            if grid[ny][nx] == 2:
                return [row[:] for row in grid]  # wall blocks whole object
        out = [row[:] for row in grid]
        for x, y in cells:
            out[y][x] = 0
        for x, y in cells:
            out[y + dy][x + dx] = color
        return out

    def _recolor_smallest(self, grid, to_color):
        objs = [o for o in find_objects(grid) if o["color"] not in (0, 2)]
        if not objs:
            return grid
        tgt = smallest(objs)
        out = [row[:] for row in grid]
        for x, y in tgt["cells"]:
            out[y][x] = to_color
        return out

    def _obs(self):
        win = self._is_win(self.grid, self.task_id)
        return {"task_id": self.task_id, "grid": self.grid,
                "algebra": compress(self.grid), "is_win": win}

    def _is_win(self, grid, task_id):
        if "recolor" in task_id:
            return any(o["color"] == 3 for o in find_objects(grid))
        if "fill" in task_id:
            # win iff exactly the initially-enclosed cells changed, all to color 1
            # (recolor-all outsiders fail: outdoor cells must be untouched)
            h, w = len(grid), len(grid[0])
            for y in range(h):
                for x in range(w):
                    if (x, y) in self._fill_mask:
                        if grid[y][x] != 1:
                            return False
                    elif grid[y][x] != self._init_grid[y][x]:
                        return False
            return len(self._fill_mask) > 0
        if "contact" in task_id:
            # win when any color-1 object contacts any color-4 object (DSL op, not bespoke)
            objs = find_objects(grid)
            ones = [o for o in objs if o["color"] == 1]
            fours = [o for o in objs if o["color"] == 4]
            return any(in_contact(a, b) for a in ones for b in fours)
        if "rotate" in task_id:
            # win when the color-1 object lies wider than tall (tall bar rotated flat)
            objs1 = [o for o in find_objects(grid) if o["color"] == 1]
            if not objs1:
                return False
            x0, y0, x1, y1 = objs1[0]["bbox"]
            return (x1 - x0) > (y1 - y0)
        if "reflect" in task_id:
            # win when the foot overhangs right: bottom-row max-x exceeds top-row max-x
            objs1 = [o for o in find_objects(grid) if o["color"] == 1]
            if not objs1:
                return False
            cells = objs1[0]["cells"]
            y0 = min(y for _, y in cells)
            y1 = max(y for _, y in cells)
            top_max = max(x for x, y in cells if y == y0)
            bot_max = max(x for x, y in cells if y == y1)
            return bot_max > top_max
        # move tasks: color-1 object adjacent to any color-2 wall (any direction)
        objs1 = [o for o in find_objects(grid) if o["color"] == 1]
        return any(is_adjacent(grid, o, 2) for o in objs1)
