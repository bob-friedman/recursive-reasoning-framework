from typing import Any, Dict, List

class BaseEnvironment:
    """
    Domain-agnostic Base Environment Contract.
    Any custom environment plugin must implement these methods.
    """

    def get_valid_actions(self) -> List[str]:
        raise NotImplementedError

    def start(self, task_id: str) -> Dict[str, Any]:
        """Initialize the environment and return the first observation."""
        raise NotImplementedError

    def step(self, action_payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool, bool]:
        """
        Accept an arbitrary JSON dictionary action.
        Returns: (observation_dict, is_done, is_win)
        """
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        return {}
