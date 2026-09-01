from typing import Any, Dict, List
from core.environment import BaseEnvironment


class CounterEnvironment(BaseEnvironment):
    """
    Example reusable plugin: Counter.
    Demonstrates how RRF can be reused for other domains beyond the
    cryptic state-machine (e.g., arithmetic reasoning, tutoring).
    Rules:
    - INCREMENT: counter += 1
    - DECREMENT: counter -= 1
    - RESET: counter = 0
    Goal is encoded in task_id: e.g., task_counter_5 => target 5,
    task_counter_-3 => target -3. Default target is 3.
    """

    domain = "counter"
    description = "Simple counter domain for arithmetic/tutoring reuse demo"

    def __init__(self):
        self.counter = 0
        self.target = 3
        self.task_id = ""

    def get_valid_actions(self) -> List[str]:
        return ["INCREMENT", "DECREMENT", "RESET"]

    def _parse_target(self, task_id: str) -> int:
        # Extract trailing integer after last '_' or default 3
        import re
        m = re.search(r"(-?\d+)$", task_id)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return 3

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.counter = 0
        self.target = self._parse_target(task_id)
        return self._obs()

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        err = self.validate_action(action_payload)
        if err:
            raise ValueError(err)
        action = str(action_payload.get("action", "")).upper()
        if action == "INCREMENT":
            self.counter += 1
        elif action == "DECREMENT":
            self.counter -= 1
        elif action == "RESET":
            self.counter = 0
        is_win = (self.counter == self.target)
        done = is_win
        return self._obs(), done, is_win

    def _obs(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "counter": self.counter,
            "target": self.target,
            "distance": self.target - self.counter,
        }

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "counter": {"type": "integer"},
                "target": {"type": "integer"},
                "distance": {"type": "integer"},
            },
        }
