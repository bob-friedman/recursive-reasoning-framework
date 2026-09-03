"""
plugins/tictactoe_env.py — Simple Boardgame Exemplar for RRF.

Minimal 3x3 Tic-Tac-Toe vs deterministic opponent (first-empty).
Designed as the "simple boardgame" amplifier demo requested — no core edits,
fully backtestable via logs, BFS-solvable, and cross-task memory transferable.

Board indexing (PLACE_1..PLACE_9):
  1|2|3
  4|5|6
  7|8|9   -> indices 0..8 row-major

Rules (deterministic, knowable + testable):
- Agent is X (1), opponent is O (2). Agent moves first.
- Legal: PLACE_1..PLACE_9 on empty cell; illegal keeps board unchanged, last_valid=false.
- After agent move, if X wins (3-in-row) -> is_win=true, done=true.
- Else if board full -> draw (is_win=false, done=true).
- Else opponent auto-plays first empty cell (lowest index) as O, then checks O win/draw.
- Win lines (8): 3 rows, 3 cols, 2 diags — stored to memory as high-confidence after solve.

Why this suits RRF amplifier pattern:
- Sokoban required BFS for optimal 33-move path; here BFS/minimax (≤9 ply) finds forced win.
- Eleusis required segmented backtest for phase-shift; here stationary rules allow whole-log backtest.
- Deterministic opponent makes predicted_outcome verifiable before POST (README 17: deterministic rigor).
- Cross-task: tictactoe_1 -> tictactoe_2 inherits win_lines + opponent policy zero-shot (no rediscovery).

No core edits: auto-discovered via core/harness.py importlib (plugins.tictactoe_env).
"""
from typing import Any, Dict, List, Optional
from core.environment import BaseEnvironment

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]

class TicTacToeEnvironment(BaseEnvironment):
    """
    Tic-Tac-Toe 3x3 — deterministic first-empty opponent, win 3-in-row.
    Simple boardgame plugin: agent must discover win-lines and optimal play via BFS.
    """
    domain = "tictactoe"
    description = "Tic-Tac-Toe 3x3 — deterministic opponent (first-empty), win 3-in-row, BFS/minimax exemplar"

    def __init__(self):
        self.board: List[int] = [0]*9
        self.task_id: str = ""
        self.steps: int = 0
        self.last_move: Optional[int] = None
        self.last_valid: bool = True
        self.last_player: str = "-"
        self.winner: Optional[str] = None
        self.done: bool = False

    def get_valid_actions(self) -> List[str]:
        return [f"PLACE_{i}" for i in range(1, 10)]

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "board": {"type": "array", "items": {"type": "integer"}},
                "board_render": {"type": "string"},
                "valid_moves": {"type": "array", "items": {"type": "string"}},
                "last_move": {"type": ["integer", "null"]},
                "last_valid": {"type": "boolean"},
                "last_player": {"type": "string"},
                "winner": {"type": ["string", "null"]},
                "is_win": {"type": "boolean"},
                "is_draw": {"type": "boolean"},
                "done": {"type": "boolean"},
                "steps": {"type": "integer"},
            },
        }

    def _check_winner(self, board: List[int]) -> Optional[str]:
        for a, b, c in WIN_LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return "X" if board[a] == 1 else "O"
        return None

    def _render(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(3):
            rows.append("|".join(sym[self.board[r*3 + c]] for c in range(3)))
        return "\n-----\n".join(rows)

    def _valid_moves(self) -> List[str]:
        return [f"PLACE_{i+1}" for i, v in enumerate(self.board) if v == 0]

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.board = [0]*9
        self.steps = 0
        self.last_move = None
        self.last_valid = True
        self.last_player = "-"
        self.winner = None
        self.done = False
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        if self.done:
            return self._obs(), True, self.winner == "X"

        # Parse action: PLACE_X or {"pos": int} or {"value": int}
        val = action_payload.get("pos", action_payload.get("value"))
        if val is None:
            action = str(action_payload.get("action", "")).upper()
            err = self.validate_action(action_payload)
            if err:
                raise ValueError(err)
            try:
                val = int(action.split("_")[1])
            except Exception as e:
                raise ValueError(f"Invalid PLACE action '{action}': need PLACE_1..9") from e
        else:
            try:
                val = int(val)
            except Exception:
                raise ValueError(f"Invalid pos '{val}': must be 1..9")
        if not 1 <= val <= 9:
            raise ValueError(f"pos {val} out of range 1..9")
        idx = val - 1

        # Illegal: occupied cell -> no state change, last_valid=false
        if self.board[idx] != 0:
            self.last_move = val
            self.last_valid = False
            self.last_player = "X(illegal)"
            # steps not incremented for illegal? increment to count attempt like eleusis
            self.steps += 1
            obs = self._obs()
            return obs, obs["done"], obs["is_win"]

        # Legal agent move
        self.board[idx] = 1
        self.last_move = val
        self.last_valid = True
        self.last_player = "X"
        self.steps += 1

        w = self._check_winner(self.board)
        if w == "X":
            self.winner = "X"
            self.done = True
            obs = self._obs()
            return obs, True, True
        if all(v != 0 for v in self.board):
            self.winner = None
            self.done = True
            obs = self._obs()
            return obs, True, False

        # Opponent auto-move: first empty (deterministic, testable)
        opp_idx = next((i for i, v in enumerate(self.board) if v == 0), None)
        if opp_idx is not None:
            self.board[opp_idx] = 2
            w2 = self._check_winner(self.board)
            if w2 == "O":
                self.winner = "O"
                self.done = True
                self.last_player = "O"
                obs = self._obs()
                return obs, True, False
            if all(v != 0 for v in self.board):
                self.winner = None
                self.done = True
                self.last_player = "O"
                obs = self._obs()
                return obs, True, False
            self.last_player = "O"

        obs = self._obs()
        is_win = self.winner == "X"
        return obs, obs["done"], is_win

    def _obs(self) -> Dict[str, Any]:
        w = self._check_winner(self.board)
        # winner already tracked for done states; but compute draw
        is_win = (w == "X") or (self.winner == "X")
        is_draw = self.done and self.winner is None and all(v != 0 for v in self.board)
        # expose done consistently
        done = self.done or is_win or (w == "O") or is_draw
        return {
            "task_id": self.task_id,
            "board": list(self.board),
            "board_render": self._render(),
            "valid_moves": self._valid_moves(),
            "last_move": self.last_move,
            "last_valid": self.last_valid,
            "last_player": self.last_player,
            "winner": self.winner if self.done else w,
            "is_win": is_win,
            "is_draw": is_draw,
            "done": done,
            "steps": self.steps,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {"steps": self.steps, "winner": self.winner, "is_win": self.winner == "X"}
