"""
plugins/eleusis_env.py — Inductive Sequence (Eleusis) Environment.

Implements idea.txt Parity-Shift Rule with terminal goal, as v1.1
additive plugin. No core edits required (core/harness.py:396 importlib).

Satisfies README Future Direction #2 Contradiction Handling:
non-stationary rule change every 5 successes forces segmented backtest
and confidence downgrade, while preserving deterministic rigor for
existing plugins (state_machine, sokoban, counter, die).

Rule (hidden from agent):
- Phase 1: if len(sequence) even → next > prev else next < prev
- Phase 2: inverted (even → <, odd → >)
- Twist: every 5 successful appends phase flips (1↔2)
- Win: len>=10 and final even

Discovery via RRF workflow: hypothesize → python backtest against
logs/*_env.log (must split at len%5==0) → predict → step → verify →
targeted BFS search for even terminal.
"""
from typing import Any, Dict, List
from core.environment import BaseEnvironment


class EleusisEnvironment(BaseEnvironment):
    """
    Inductive Sequence (Eleusis) — phase-shifting parity rule.
    High-stress test for contradiction handling: agent must detect
    the 5-step phase inversion via deterministic log replay and
    downgrade prior high-confidence rules.
    """

    domain = "inductive_logic"
    description = "Inductive Sequence (Eleusis) — phase-shifting parity (5-step inversion), win len>=10 even"

    def __init__(self):
        self.sequence: List[int] = [5]
        self.phase: int = 1
        self.task_id: str = ""
        self.steps: int = 0
        self.last_valid: bool = True
        self.last_attempt: Any = None

    def get_valid_actions(self) -> List[str]:
        # PLAY_1..10 canonical; harness validate_action checks this
        return [f"PLAY_{i}" for i in range(1, 11)]

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "sequence": {"type": "array"},
                "length": {"type": "integer"},
                "phase_hidden": {"type": "integer", "description": "hidden, not exposed to agent in ideal run — for harness debugging only"},
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
        # Accept both PLAY_X and {"value": int} for flexibility
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
            # also validate range even when using value field
            try:
                val = int(val)
            except Exception:
                raise ValueError(f"Invalid value '{val}': must be int 1..10")
            if not 1 <= val <= 10:
                raise ValueError(f"Value {val} out of range 1..10")

        prev = self.sequence[-1]
        length = len(self.sequence)

        # Hidden logic per idea.txt:42-45
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
            if len(self.sequence) % 5 == 0:
                self.phase = 2 if self.phase == 1 else 1

        is_win = (len(self.sequence) >= 10 and self.sequence[-1] % 2 == 0)
        done = is_win
        return self._obs(), done, is_win

    def _obs(self) -> Dict[str, Any]:
        # Note: phase_hidden is included for harness logging/debugging;
        # agent should infer it via log analysis, not trust it blindly.
        return {
            "task_id": self.task_id,
            "sequence": list(self.sequence),
            "length": len(self.sequence),
            "phase_hidden": self.phase,
            "last_attempt": self.last_attempt,
            "last_valid": self.last_valid,
            "is_win": len(self.sequence) >= 10 and self.sequence[-1] % 2 == 0,
            "steps": self.steps,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {"length": len(self.sequence), "phase": self.phase, "is_win": len(self.sequence) >= 10 and self.sequence[-1] % 2 == 0}
