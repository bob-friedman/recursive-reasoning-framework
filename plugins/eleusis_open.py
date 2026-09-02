"""
plugins/eleusis_open.py — Self-Recursive Open Exploration (Future Direction #3)

Variant of eleusis_env.py with undisclosed twist for open-ended discovery:
- Hidden flip interval = 4 (not 5), phase 1 even->> odd-><, phase2 inverted
- Hidden win = len>=12 and final odd (not 10 even)
- phase_hidden NOT exposed in observation (forces pure log inference)
- Used to test README #3 Self-Recursive Exploration: agent must discover all predicates without idea.txt

No core edits required.
"""
from typing import Any, Dict, List
import random
from core.environment import BaseEnvironment

class EleusisOpenEnvironment(BaseEnvironment):
    domain = "inductive_logic_open"
    description = "Eleusis Open — undisclosed 4-step inversion, win len>=12 odd, no phase_hidden"

    def __init__(self):
        self.sequence: List[int] = [5]
        self.phase: int = 1
        self.task_id: str = ""
        self.steps: int = 0
        self.last_valid: bool = True
        self.last_attempt: Any = None
        self.flip_interval: int = 4
        self.win_len: int = 12

    def get_valid_actions(self) -> List[str]:
        return [f"PLAY_{i}" for i in range(1, 11)]

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "sequence": {"type": "array"},
                "length": {"type": "integer"},
                # phase_hidden intentionally omitted for #3
                "last_attempt": {"type": "integer"},
                "last_valid": {"type": "boolean"},
                "is_win": {"type": "boolean"},
                "steps": {"type": "integer"},
            },
        }

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.sequence = [5]
        self.phase = 1
        self.steps = 0
        self.last_valid = True
        self.last_attempt = None
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        val = action_payload.get("value")
        if val is None:
            action = str(action_payload.get("action", "")).upper()
            err = self.validate_action(action_payload)
            if err:
                raise ValueError(err)
            try:
                val = int(action.split("_")[1])
            except Exception as e:
                raise ValueError(f"Invalid PLAY action '{action}': need PLAY_1..10") from e
        else:
            try:
                val = int(val)
            except Exception:
                raise ValueError(f"Invalid value '{val}'")
            if not 1 <= val <= 10:
                raise ValueError(f"Value {val} out of range 1..10")

        prev = self.sequence[-1]
        length = len(self.sequence)
        is_valid = False
        if self.phase == 1:
            is_valid = (val > prev) if length % 2 == 0 else (val < prev)
        else:
            is_valid = (val < prev) if length % 2 == 0 else (val > prev)

        self.last_attempt = val
        self.last_valid = bool(is_valid)
        self.steps += 1

        if is_valid:
            self.sequence.append(val)
            if len(self.sequence) % self.flip_interval == 0:
                self.phase = 2 if self.phase == 1 else 1

        is_win = (len(self.sequence) >= self.win_len and self.sequence[-1] % 2 == 1)
        done = is_win
        return self._obs(), done, is_win

    def _obs(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sequence": list(self.sequence),
            "length": len(self.sequence),
            "last_attempt": self.last_attempt,
            "last_valid": self.last_valid,
            "is_win": len(self.sequence) >= self.win_len and self.sequence[-1] % 2 == 1,
            "steps": self.steps,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {"length": len(self.sequence), "phase": self.phase, "is_win": len(self.sequence) >= self.win_len and self.sequence[-1] % 2 == 1}
