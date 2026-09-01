from typing import Any, Dict, List
from core.environment import BaseEnvironment

class StateMachineEnvironment(BaseEnvironment):
    """
    Cryptic State-Machine Proof-of-Concept Environment.
    Rules:
    - PRESS_A: Resets color to White and clears history sequence.
    - PRESS_B: Appends 'PRESS_B' to sequence history.
    - PRESS_C: If preceded immediately by 'PRESS_B', color transitions to Blue.
    - PRESS_E: If current color is Blue, color transitions to Green.
    - PRESS_D: Neutral state step.
    """

    domain = "state_machine"
    description = "Cryptic sequential state-machine: B->C=Blue, Blue+E=Green"

    def __init__(self):
        self.valid_actions = ["PRESS_A", "PRESS_B", "PRESS_C", "PRESS_D", "PRESS_E"]
        self.color = "White"
        self.history = []
        self.task_id = ""

    def get_valid_actions(self) -> List[str]:
        return self.valid_actions

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.color = "White"
        self.history = []
        return self._get_observation()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        action_name = str(action_payload.get("action", "")).upper()
        if action_name not in self.valid_actions:
            raise ValueError(f"Invalid action '{action_name}'. Must be one of {self.valid_actions}")

        if action_name == "PRESS_A":
            self.color = "White"
            self.history.clear()
        elif action_name == "PRESS_B":
            self.history.append("PRESS_B")
        elif action_name == "PRESS_C":
            if self.history and self.history[-1] == "PRESS_B":
                self.color = "Blue"
            self.history.append("PRESS_C")
        elif action_name == "PRESS_E":
            if self.color == "Blue":
                self.color = "Green"
            self.history.append("PRESS_E")
        elif action_name == "PRESS_D":
            self.history.append("PRESS_D")

        is_win = False
        if "green" in self.task_id.lower() or "task_2" in self.task_id.lower():
            is_win = (self.color == "Green")
        else:
            is_win = (self.color == "Blue")

        done = is_win
        return self._get_observation(), done, is_win

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "color": self.color,
            "sequence_length": len(self.history),
            "recent_history": self.history[-5:],
        }

