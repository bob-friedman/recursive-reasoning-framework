"""
plugins/die_env.py — Stochastic discovery plugin for 6-sided die.

Demonstrates RRF extension to stochastic events per README §5
"Stochastic Discovery & AI/Human Equivalence (Insight)":

- step() returns sample 1..6 + distribution/observation with sample
  count and chi-square proxy (variance_proxy) so the LLM can fit
  a uniform distribution via scipy.stats.chisquare backtest
  against logs/die_env.log.
- get_observation_schema() exposes distribution fields for
  hypothesis logging.
- Memory rules written with confidence: medium (per
  README.md:175 contradiction handling) since single-roll
  prediction cannot beat 1/6 chance; downgrades to low on
  contradiction vs expected.
- No core edits (modularity holds: importlib auto-discovers via
  core/harness.py:386; git diff core/ empty).
"""
from typing import Any, Dict, List
from core.environment import BaseEnvironment


DIE_FACES = 6


class DieEnvironment(BaseEnvironment):
    """
    Stochastic 6-sided die plugin. Solvable as a distribution
    discovery puzzle, not single-roll prediction.

    Mechanics:
    - ROLL: returns sampled 1..6, increments sample_count, records
      history for chi-square fit; win condition when sample_count
      >= target_rolls and distribution p-values within tolerance
      of uniform 1/6 (chisquare test).
    - task_id encodes target_rolls: e.g., die_30 => 30 rolls
      required to claim is_win=True.
    - RESET: clears history (for re-experiments).
    """

    domain = "die"
    description = "Stochastic 6-sided die — distribution discovery (scipy.stats.chisquare backtest, confidence: medium)"

    def __init__(self):
        self.task_id: str = ""
        self.target_rolls: int = 30
        self.history: List[int] = []
        self.last_roll: int = 0
        self.distribution: Dict[int, int] = {i: 0 for i in range(1, DIE_FACES + 1)}
        self.variance_proxy: float = 0.0  # chi-square statistic proxy

    def get_valid_actions(self) -> List[str]:
        return ["ROLL", "RESET"]

    def get_observation_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "last_roll": {"type": "integer"},
                "sample_count": {"type": "integer"},
                "target_rolls": {"type": "integer"},
                "distribution": {"type": "object"},
                "expected_uniform": {"type": "number"},
                "variance_proxy": {"type": "number"},
                "is_win": {"type": "boolean"},
                "history_tail": {"type": "array"},
            },
        }

    def _parse_target(self, task_id: str) -> int:
        import re
        m = re.search(r"(-?\d+)$", task_id)
        if m:
            try:
                return max(1, int(m.group(1)))
            except ValueError:
                pass
        return 30

    def start(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.target_rolls = self._parse_target(task_id)
        self.history = []
        self.distribution = {i: 0 for i in range(1, DIE_FACES + 1)}
        self.last_roll = 0
        self.variance_proxy = 0.0
        return self._obs()

    def _chi_square(self) -> float:
        """Compute chi-square statistic vs uniform 1/DIE_FACES."""
        if not self.history:
            return 0.0
        n = len(self.history)
        expected = n / DIE_FACES
        if expected == 0:
            return 0.0
        chi = 0.0
        for face in range(1, DIE_FACES + 1):
            obs = self.distribution[face]
            chi += (obs - expected) ** 2 / expected
        return chi

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        err = self.validate_action(action_payload)
        if err:
            raise ValueError(err)
        action = str(action_payload.get("action", "")).upper()
        if action == "RESET":
            self.history = []
            self.distribution = {i: 0 for i in range(1, DIE_FACES + 1)}
            self.last_roll = 0
            self.variance_proxy = 0.0
        elif action == "ROLL":
            import random
            roll = random.randint(1, DIE_FACES)
            self.last_roll = roll
            self.history.append(roll)
            self.distribution[roll] += 1
            self.variance_proxy = self._chi_square()
        is_win = self._is_win()
        done = is_win
        return self._obs(), done, is_win

    def _is_win(self) -> bool:
        # Win when sample count reaches target AND distribution is
        # within chi-square critical value for 5 df at 0.05 (~11.07)
        if len(self.history) < self.target_rolls:
            return False
        return self.variance_proxy <= 11.07

    def _obs(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "last_roll": self.last_roll,
            "sample_count": len(self.history),
            "target_rolls": self.target_rolls,
            "distribution": dict(self.distribution),
            "expected_uniform": 1.0 / DIE_FACES,
            "variance_proxy": self.variance_proxy,
            "is_win": self._is_win(),
            "history_tail": self.history[-10:],
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "sample_count": len(self.history),
            "variance_proxy": self.variance_proxy,
            "is_win": self._is_win(),
        }
