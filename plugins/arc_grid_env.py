"""arc_grid_env.py — minimal interactive ARC-style grid game for RRF.

Two demo tasks, same generic physics (no per-task hardcoding in agent):
- arc_move_1:   push color-1 block RIGHT until it touches color-2 wall -> win.
- arc_recolor_1: recolor smallest object to color 3 via RECOLOR action -> win.
Obs includes raw grid + compressed algebra (core.grid_algebra.compress)
so a small LLM can reason over symbols, backtest in Python, and act once.
"""
from typing import Any, Dict, List
from core.environment import BaseEnvironment
from core.grid_algebra import (align_tops, beat_mark, center_object, compress, cover_color,
    crop_to_bbox, downscale_integer, erase_by_size, fall_rigid, find_objects, flood_fill_interior,
    frame_merged_bbox, hollow_object, in_contact, is_adjacent, is_symmetric, mirror_repeat,
    paint_object_at, reflect_object, recolor_by_rank, repeat_until_wall, rotate_object,
    settle_sand, smallest, sort_row_by_area, stamp_copy, swap_colors, symmetry_complete,
    tile_pattern, topk_keep, upscale_integer)

MOVABLE = (1, 4)  # generic: any listed color translates rigidly; walls (2) never move


class ArcGridEnv(BaseEnvironment):
    domain = "arc_grid"
    description = "Interactive ARC-style grid demo: move/recolor objects with algebra obs"

    def __init__(self):
        self.grid = []
        self.task_id = ""
        self._init_grid = []
        self._tile_expected = []
        self._sim_expected = []
        self._fill_mask = set()  # initially-enclosed cells (fill tasks only)

    def get_valid_actions(self) -> List[str]:
        return ["LEFT", "RIGHT", "UP", "DOWN", "ROTATE", "REFLECT", "MIRROR", "TILE",
                "HOLLOW", "COVER", "RANK", "REPEAT", "MREPEAT", "UPSCALE", "DOWNSCALE",
                "CROP", "CENTER", "FALL", "ALIGN", "SETTLE", "SWAP", "TRIM", "FRAME",
                "SORTX", "TOPK", "BEAT", "STAMP", "FILL", "RECOLOR", "RESET"]

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.grid = [row[:] for row in self._level(task_id)]
        self._init_grid = [row[:] for row in self.grid]
        self._fill_mask = set()
        if "fill" in task_id:  # interior = background cells not border-connected at start
            h, w = len(self.grid), len(self.grid[0])
            painted, _ = flood_fill_interior(self.grid, color=-1)
            self._fill_mask = {(x, y) for y in range(h) for x in range(w) if painted[y][x] == -1}
        self._tile_expected = []
        if "tile" in task_id:  # expected = full beat simulation on the initial grid
            self._tile_expected, added = tile_pattern(self.grid, color=1)
            assert added > 0, f"{task_id}: level must require tiling"
        self._sim_expected = []
        if task_id in self.SIM_LEVELS:  # expected = winning sequence applied to initial grid
            self._sim_expected = self._apply_sim(self.grid, self.SIM_LEVELS[task_id])
            assert self._sim_expected != self.grid, f"{task_id}: level must require action"
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
        elif action == "MIRROR":
            axis = str(action_payload.get("axis", "vertical")).lower()
            if axis not in ("vertical", "horizontal"):
                raise ValueError(f"Invalid 'axis' {axis!r}; must be vertical|horizontal")
            self.grid, _ = symmetry_complete(self.grid, axis=axis, color=1)
        elif action == "FILL":
            self.grid, _ = flood_fill_interior(self.grid, color=1)
        elif action == "TILE":
            self.grid, _ = tile_pattern(self.grid, color=1)
        elif action in ("HOLLOW", "COVER", "RANK", "REPEAT", "MREPEAT", "UPSCALE",
                        "DOWNSCALE", "CROP", "CENTER", "FALL", "ALIGN", "SETTLE", "SWAP",
                        "TRIM", "FRAME", "SORTX", "TOPK", "BEAT", "STAMP"):
            self.grid = self._batch_action(self.grid, action, action_payload)
        elif action == "RECOLOR":
            self.grid = self._recolor_smallest(self.grid, 3)
        obs = self._obs()
        return obs, bool(obs["is_win"]), bool(obs["is_win"])

    # --- Batch family (simulation-win levels; expected precomputed at start) ---
    def _int_payload(self, payload: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(payload.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid '{key}'; must be an integer")

    def _batch_action(self, grid, action: str, payload: Dict[str, Any]):
        """One generic dispatcher for batch-family actions. Payload-validated, no task branches."""
        if action == "HOLLOW":
            color = self._int_payload(payload, "color", 1)
            out, _ = hollow_object(grid, color=color)
            return out
        if action == "COVER":
            color = self._int_payload(payload, "color", 1)
            out, _ = cover_color(grid, color=color)
            return out
        if action == "RANK":
            rank = self._int_payload(payload, "rank", 0)
            out, _ = recolor_by_rank(grid, rank=rank)
            return out
        if action == "REPEAT":
            out, _ = repeat_until_wall(grid, color=1)
            return out
        if action == "MREPEAT":
            out, _ = mirror_repeat(grid, color=1)
            return out
        if action == "UPSCALE":
            out = upscale_integer(grid, factor=2)
            return out if out is not None else [row[:] for row in grid]
        if action == "DOWNSCALE":
            out = downscale_integer(grid, factor=2)
            return out if out is not None else [row[:] for row in grid]
        if action == "CROP":
            out, _ = crop_to_bbox(grid)
            return out
        if action == "CENTER":
            out, _ = center_object(grid, color=1)
            return out
        if action == "FALL":
            out, _ = fall_rigid(grid, color=1)
            return out
        if action == "ALIGN":
            out, _ = align_tops(grid)
            return out
        if action == "SETTLE":
            out, _ = settle_sand(grid)
            return out
        if action == "SWAP":
            a = self._int_payload(payload, "a", 1)
            b = self._int_payload(payload, "b", 4)
            return swap_colors(grid, color_a=a, color_b=b)
        if action == "TRIM":
            lo = self._int_payload(payload, "min", 2)
            hi = self._int_payload(payload, "max", 3)
            out, _ = erase_by_size(grid, min_area=lo, max_area=hi)
            return out
        if action == "FRAME":
            out, _ = frame_merged_bbox(grid)
            return out
        if action == "SORTX":
            out, _ = sort_row_by_area(grid)
            return out
        if action == "TOPK":
            k = self._int_payload(payload, "k", 2)
            out, _ = topk_keep(grid, k=k)
            return out
        if action == "BEAT":
            out, _ = beat_mark(grid, color=1)
            return out
        if action == "STAMP":
            dx = self._int_payload(payload, "dx", 3)
            dy = self._int_payload(payload, "dy", 0)
            out, _ = stamp_copy(grid, dx=dx, dy=dy, color=1)
            return out
        raise ValueError(f"Unknown batch action {action}")

    # task -> winning step sequence [(action, payload)]; expected precomputed at start.
    # Simulation-exact win: grid must equal the full sequence applied to the initial grid.
    SIM_LEVELS = {
        "arc_hollow_1": [("HOLLOW", {})],
        "arc_cover_1": [("COVER", {"color": 1})],
        "arc_rank_1": [("RANK", {"rank": 1})],
        "arc_repeat_1": [("REPEAT", {})],
        "arc_mrepeat_1": [("MREPEAT", {})],
        "arc_up_1": [("UPSCALE", {})],
        "arc_down_1": [("DOWNSCALE", {})],
        "arc_crop_1": [("CROP", {})],
        "arc_center_1": [("CENTER", {})],
        "arc_fall_1": [("FALL", {})],
        "arc_align_1": [("ALIGN", {})],
        "arc_settle_1": [("SETTLE", {})],
        "arc_swap_1": [("SWAP", {"a": 1, "b": 4})],
        "arc_size_1": [("TRIM", {"min": 2, "max": 3})],
        "arc_merge_1": [("FRAME", {})],
        "arc_nest_1": [("RIGHT", {"color": 4}), ("RIGHT", {"color": 4})],
        "arc_edge_1": [("LEFT", {}), ("LEFT", {})],
        "arc_count_1": [("COVER", {"color": 4})],
        "arc_sortx_1": [("SORTX", {})],
        "arc_topk_1": [("TOPK", {"k": 2})],
        "arc_beat_1": [("BEAT", {})],
        "arc_rot4_1": [("COVER", {"color": 4})],
        "arc_stamp_1": [("STAMP", {"dx": 3, "dy": 0})],
    }

    @staticmethod
    def _apply_sim(grid, steps):
        """Apply a winning step sequence (slide actions honor color payload)."""
        g = [row[:] for row in grid]
        tmp = ArcGridEnv()
        for action, payload in steps:
            if action in ("LEFT", "RIGHT", "UP", "DOWN"):
                color = int(dict(payload).get("color", 1))
                g = tmp._slide(g, action, color=color)
            else:
                g = tmp._batch_action(g, action, dict(payload))
        return g
    # Public: arc_move_1 (wall right), arc_recolor_1.
    # Holdout (reviewer-added, agent-blind): arc_move_2 (wall LEFT, wider grid,
    #   block starts right, must go LEFT), arc_recolor_2 (wider, two objects).
    # --- levels (env-side ground truth; agent must discover, not read) ---
    def _level(self, task_id: str):
        if task_id == "arc_size_1":  # areas 1,2,4 -> TRIM 2-3 keeps the domino
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ]
        if task_id == "arc_merge_1":  # two dots -> FRAME paints merged bbox border
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_nest_1":  # 4-dot outside gapped ring -> 2x RIGHT nests it
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [4, 0, 0, 0, 0, 2, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_edge_1":  # center block -> 2x LEFT touches border
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_count_1":  # 2 ones vs 5 fours (+1 stray one) -> COVER 4 flips
            return [
                [0, 0, 0, 0, 0, 0, 1],
                [1, 1, 4, 4, 4, 4, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_sortx_1":  # big-left small-right -> SORTX orders by area
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_topk_1":  # areas 1,2,4 -> TOPK 2 erases the single
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ]
        if task_id == "arc_beat_1":  # domino seed + lone dot -> BEAT marks origin only
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_rot4_1":  # plus + extra 4-cell -> COVER 4 restores order-4
            return [
                [4, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        if task_id == "arc_stamp_1":  # 2x2 block -> STAMP dx=3 copies it right
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_hollow_1":  # solid 3x3 -> HOLLOW rings it
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_cover_1":  # 2x2 block + dot -> COVER color-1 spares nothing? (dot is color-4)
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 4],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_rank_1":  # areas 1,2,4 -> RANK 1 recolors the domino
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ]
        if task_id == "arc_repeat_1":  # lone seed -> REPEAT fills to wall
            return [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_mrepeat_1":  # asymmetric seed -> MREPEAT skips where TILE fills
            return [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_up_1":  # 3x3 motif in 5x5 -> UPSCALE to 10x10 (bound-exact)
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_down_1":  # 6x6 uniform 2x2 blocks -> DOWNSCALE to 3x3
            return [
                [1, 1, 0, 0, 2, 2],
                [1, 1, 0, 0, 2, 2],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_crop_1":  # offset object -> CROP to bbox
            return [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_center_1":  # corner block -> CENTER
            return [
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_fall_1":  # overhang shape: rigid keeps form, sand shears it
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2],
            ]
        if task_id == "arc_align_1":  # staggered blocks -> ALIGN tops
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_settle_1":  # scattered sand over floor -> SETTLE compacts
            return [
                [0, 1, 0, 4, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 4],
                [0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2],
            ]
        if task_id == "arc_swap_1":  # 1s and 4 -> SWAP exchanges them
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_tile_1":  # period-2 seed run -> TILE extends to wall
            return [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_tile_2":  # period-3 seed, offset -> TILE extends to wall
            return [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        if task_id == "arc_symmetry_1":  # half-L -> MIRROR vertical wins
            return [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        if task_id == "arc_symmetry_2":  # top-heavy -> MIRROR horizontal wins
            return [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
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
        if task_id in self.SIM_LEVELS:
            # simulation-exact win: grid must equal the precomputed single application
            return bool(self._sim_expected) and grid == self._sim_expected and grid != self._init_grid
        if "tile" in task_id:
            # win iff grid exactly equals the full-beat simulation (no extras, no gaps)
            return bool(self._tile_expected) and grid == self._tile_expected and grid != self._init_grid
        if "symmetry" in task_id:
            # axis from level family: symmetry_2 completes downward, else vertical.
            # Non-triviality required: uniform paint (recolor-all) is symmetric but wins nothing.
            axis = "horizontal" if "symmetry_2" in task_id else "vertical"
            return is_symmetric(grid, axis=axis) and any(
                v == 0 for row in grid for v in row) and any(
                v == 1 for row in grid for v in row)
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
