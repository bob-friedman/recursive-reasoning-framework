from typing import Any, Dict, List, Tuple
from core.environment import BaseEnvironment


# Microban Levels 1-2 — modular test (David Skinner, 2000)
# ; 1
# ####
# # .#
# #  ###
# #*@  #
# #  $ #
# #  ###
# ####
# ; 2
# ######
# #    #
# # #@ #
# # $* #
# # .* #
# #    #
# ######
MICROBAN_LEVEL_1 = [
    "####",
    "# .#",
    "#  ###",
    "#*@  #",
    "#  $ #",
    "#  ###",
    "####",
]

MICROBAN_LEVEL_2 = [
    "######",
    "#    #",
    "# #@ #",
    "# $* #",
    "# .* #",
    "#    #",
    "######",
]

# Expand LEVELS dict for cross-level testing — no core edits required
LEVELS = {
    "microban_1": MICROBAN_LEVEL_1,
    "task_1": MICROBAN_LEVEL_1,
    "level_1": MICROBAN_LEVEL_1,
    "sokoban_1": MICROBAN_LEVEL_1,
    "microban_2": MICROBAN_LEVEL_2,
    "level_2": MICROBAN_LEVEL_2,
    "sokoban_2": MICROBAN_LEVEL_2,
}


class SokobanEnvironment(BaseEnvironment):
    """
    Sokoban plugin — reuses RRF for puzzle discovery.
    Single-level test: Microban #1. No core edits required.
    Rules:
    - UP/DOWN/LEFT/RIGHT: move player; push box if adjacent and beyond is free
    - '#': wall, ' ': floor, '.': goal, '$': box, '*': box on goal,
      '@': player, '+': player on goal (handled internally as floor+overlay)
    Win when all boxes on goals.
    """

    domain = "sokoban"
    description = "Microban #1-2 Sokoban — modular push puzzle (UP/DOWN/LEFT/RIGHT), cross-level transfer test"

    def __init__(self):
        self.task_id: str = ""
        self.width: int = 0
        self.height: int = 0
        self.walls: set[Tuple[int, int]] = set()
        self.goals: set[Tuple[int, int]] = set()
        self.boxes: set[Tuple[int, int]] = set()
        self.player: Tuple[int, int] = (0, 0)
        self.initial_boxes: set[Tuple[int, int]] = set()
        self.initial_player: Tuple[int, int] = (0, 0)
        self.steps: int = 0

    def get_valid_actions(self) -> List[str]:
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "player": {"type": "array"},
                "boxes": {"type": "array"},
                "goals": {"type": "array"},
                "walls": {"type": "array"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "steps": {"type": "integer"},
                "is_win": {"type": "boolean"},
                "render": {"type": "string"},
            },
        }

    def _load_level(self, raw: List[str]):
        # Normalize to max width with padding as floor outside walls treated as wall
        self.height = len(raw)
        self.width = max(len(row) for row in raw)
        self.walls.clear()
        self.goals.clear()
        self.boxes.clear()
        # Parse
        for y, row in enumerate(raw):
            for x in range(self.width):
                ch = row[x] if x < len(row) else " "
                if ch == "#":
                    self.walls.add((x, y))
                elif ch == ".":
                    self.goals.add((x, y))
                elif ch == "$":
                    self.boxes.add((x, y))
                elif ch == "*":
                    self.boxes.add((x, y))
                    self.goals.add((x, y))
                elif ch == "@":
                    self.player = (x, y)
                elif ch == "+":
                    self.player = (x, y)
                    self.goals.add((x, y))
                elif ch == " ":
                    pass  # floor
                else:
                    pass
        # Any missing cell outside walls is implicitly wall for collision, but not stored
        self.initial_boxes = set(self.boxes)
        self.initial_player = self.player
        self.steps = 0

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        # Select level by task_id, default to microban_1
        key = task_id.lower()
        raw = LEVELS.get(key, LEVELS.get("microban_1"))
        # Also handle numeric suffix like "level_1" already covered; fallback to microban_1
        if raw is None:
            raw = MICROBAN_LEVEL_1
        self._load_level(raw)
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        err = self.validate_action(action_payload)
        if err:
            raise ValueError(err)
        action = str(action_payload.get("action", "")).upper()
        # Also accept WASD aliases
        alias = {"W": "UP", "A": "LEFT", "S": "DOWN", "D": "RIGHT"}
        action = alias.get(action, action)

        dirs = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        dx, dy = dirs[action]

        px, py = self.player
        nx, ny = px + dx, py + dy

        # Helper: is wall (including out-of-bounds implied wall)
        def is_wall(x, y):
            if (x, y) in self.walls:
                return True
            # Treat outside bounding box as wall if beyond level width/height or missing row
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return True
            return False

        # Check next cell
        if is_wall(nx, ny):
            # blocked, no state change
            obs = self._obs()
            return obs, False, self._is_win()

        if (nx, ny) in self.boxes:
            # Need to push box
            bx, by = nx + dx, ny + dy
            if is_wall(bx, by) or (bx, by) in self.boxes:
                obs = self._obs()
                return obs, False, self._is_win()
            # Push
            self.boxes.remove((nx, ny))
            self.boxes.add((bx, by))
            self.player = (nx, ny)
        else:
            # Free move (floor or goal)
            self.player = (nx, ny)

        self.steps += 1
        is_win = self._is_win()
        done = is_win
        return self._obs(), done, is_win

    def _is_win(self) -> bool:
        # Win when every box is on a goal and every goal has a box (for this single-level, boxes == goals count)
        # Simplified: all boxes on goals
        return len(self.boxes) > 0 and self.boxes.issubset(self.goals)

    def _render(self) -> str:
        rows = []
        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                if (x, y) in self.walls:
                    row_chars.append("#")
                elif (x, y) == self.player and (x, y) in self.goals:
                    row_chars.append("+")
                elif (x, y) == self.player:
                    row_chars.append("@")
                elif (x, y) in self.boxes and (x, y) in self.goals:
                    row_chars.append("*")
                elif (x, y) in self.boxes:
                    row_chars.append("$")
                elif (x, y) in self.goals:
                    row_chars.append(".")
                else:
                    row_chars.append(" ")
            # rstrip to keep inet display tidy but keep internal
            rows.append("".join(row_chars).rstrip())
        return "\n".join(rows)

    def _obs(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "player": list(self.player),
            "boxes": sorted(list(self.boxes)),
            "goals": sorted(list(self.goals)),
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "is_win": self._is_win(),
            "render": self._render(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {"steps": self.steps, "is_win": self._is_win()}
